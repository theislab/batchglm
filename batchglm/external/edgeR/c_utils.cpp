#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>


const double one_tenthousandth=std::pow(10, -4.0);
const double mildly_low_value=std::pow(10, -8.0);
const double one_million=std::pow(10, 6);

/* All functions are taken from the C++ backend of edgeR.
 * The R wrappers were replaced with according numpy / python C-API.
 */

double compute_unit_nb_deviance (double y, double mu, double phi) {
    y+=mildly_low_value;
    mu+=mildly_low_value;

    /* Calculating the deviance using either the Poisson (small phi*mu), the Gamma (large) or NB (everything else).
     * Some additional work is put in to make the transitions between families smooth.
     */
    if (phi < one_tenthousandth) {
        const double resid = y - mu;
        return 2 * ( y * std::log(y/mu) - resid - 0.5*resid*resid*phi*(1+phi*(2/3*resid-y)) );
    } else {
        const double product=mu*phi;
        if (product > one_million) {
            return 2 * ( (y - mu)/mu - std::log(y/mu) ) * mu/(1+product);
        } else {
            const double invphi=1/phi;
            return 2 * (y * std::log( y/mu ) + (y + invphi) * std::log( (mu + invphi)/(y + invphi) ) );
        }
    }
};


static PyObject *loess_by_col(PyObject *self, PyObject *args) {

    const double low_value = std::pow(10.0, -10.0);
    PyArrayObject *x, *y;
    int span;
    PyArg_ParseTuple(args, "OOi", &x, &y, &span);
    if (PyErr_Occurred()) {
        return NULL;
    }
    if (!(PyArray_Check(x)) && !(PyArray_Check(y))) {
        PyErr_SetString(PyExc_TypeError, "First two arguments must be numpy arrays.");
        return NULL;
    }
    int total = PyArray_SIZE(x);
    npy_intp *x_dims = PyArray_DIMS(x);
    npy_intp *y_dims = PyArray_DIMS(y);

    int ncols = y_dims[1];

    double *x_ptr;
    double **y_ptrs;
    PyArray_AsCArray((PyObject **)&x, &x_ptr, x_dims, 1, PyArray_DescrFromType(NPY_DOUBLE));
    if (PyErr_Occurred()){
        return NULL;
    }
    PyArray_AsCArray((PyObject **)&y, &y_ptrs, y_dims, 2, PyArray_DescrFromType(NPY_DOUBLE));
    if (PyErr_Occurred()){
        return NULL;
    }

    if (span > total) {
        PyErr_SetString(PyExc_RuntimeError, "Span must be smaller than the total number of points.");
        return NULL;
    }
    double w_ptr[total];
    double f_ptrs[y_dims[0]][y_dims[1]];

    try {
        int frame_end=span-1;
        std::cout << frame_end << '\n';
        for (int cur_p=0; cur_p<total; ++cur_p) {
                if ((cur_p % 1000) == 0) {
                    std::cout << cur_p << '\n';
                }
                if (cur_p>frame_end) { frame_end=cur_p; }
                const double& cur_point=x_ptr[cur_p];
                double back_dist=cur_point-x_ptr[frame_end-span+1], front_dist=x_ptr[frame_end]-cur_point,
                max_dist=(back_dist > front_dist ? back_dist : front_dist);

                while (frame_end < total-1 && cur_p+span-1>frame_end) {
                /* Every time we advance, we twiddle with the ends of the frame to see if we can't get
                 * a better fit. The frame will always advance in the forward direction. This is because the
                 * current frame is optimal with respect to the previous tag. If the previous maximal distance
                 * was at the back, shifting the frame backward will increase the back distance with respect to
                 * the current tag (and thus increase the maximal distance).
                 *
                 * If the previous maximal distance was at the front, shifting the frame backward may
                 * decrease the front distance with respect to the current tag. However, we note that
                 * because of optimality, having a previous maximal distance at the front must mean
                 * that a back-shifted frame will result in an even larger previous maximal distance at
                 * the back (otherwise the optimal frame would be located further back to start with). In
                 * short, shifting the frame backwards will flip the maximal distance to that of the back
                 * distance which is even larger than the non-shifted forward distance.
                 *
                 * Thus, the frame can only go forwards. Note that below, the frame is defined by
                 * the 'end' position which notes the end point of the current frame. The start
                 * point is inherently defined by revolving around the minimum point.
                 */
                back_dist=cur_point-x_ptr[frame_end-span+2];
                front_dist=x_ptr[frame_end+1]-cur_point;
                const double& next_max=(back_dist > front_dist ? back_dist : front_dist);
                /* This bit provides some protection against near-equal values, by forcing the frame
                 * forward provided that the difference between the lowest maximum distance and
                 * the maximum distance at any other frame is less than a low_value. This ensures
                 * that values following a stretch of identical x-coordinates are accessible
                 * to the algorithm (rather than being blocked off by inequalities introduced by
                 * double imprecision).
                 */
                const double diff=(next_max-max_dist)/max_dist;
                if (diff > low_value) {
                        break;
                } else if (diff < 0) {
                        max_dist=next_max;
                }
                ++frame_end;
                }
                /* Now that we've located our optimal window, we can calculate the weighted average
                 * across the points in the window (weighted according to distance from the current point).
                 * and we can calculate the leverages. Unfortunately, we have to loop over the points in the
                 * window because each weight must be recomputed according to its new distance and new maximal
                 * distance.
                 */
                double total_weight=0;
                double& out_leverage=(w_ptr[cur_p]=-1);
                for (int i=0; i<ncols; ++i) { f_ptrs[cur_p][i]=0; }
                /* For non-zero maximum distances, we can compute the relative distance; otherwise, we set it to zero.
                 * This means that all observations will have the same weight (user specifications aside). This makes
                 * sense as they all lie on the same x-coordinate. Note that funny calculations might happen with the
                 * leverage as there are multiple valid frames with the same minimum distance when many x-coordinates
                 * are equal.
                 *
                 * Note that we have to look for more than just the 'span' number of points. Consider the series
                 * A,B,C,C where each is a value and A < B < C and C - B > B - A. The algorithm above will move the
                 * frame to [1,3] when calculating the maximum distance for B. This is the same as [0, 2] in terms
                 * of distance, but only using the frame components to calculate the mean will miss out on element 0.
                 * So, the computation should work from [0, 3]. There's no need to worry about the extra 'C' as it
                 * will have weight zero.
                 */
                for (int m=frame_end; m>=0; --m) {
                    const double rel_dist=(max_dist > low_value ? std::abs(x_ptr[m]-cur_point)/max_dist : 0);
                    const double weight=std::pow(1-std::pow(rel_dist, 3.0), 3.0);
                    if (weight < 0) { continue; }
                    total_weight+=weight;

                    for (int i=0; i<ncols; ++i) {
                        f_ptrs[cur_p][i]+=weight*y_ptrs[m][i];
                    }
                    if (m==cur_p) {
                        out_leverage=weight;
                    }
                }
                // Normalizing by the total weight.
                out_leverage/=total_weight;
                for (int i=0; i<ncols; ++i) { f_ptrs[cur_p][i]/=total_weight; }
        }
        } catch (std::exception& e) {
                throw;
        }
    PyObject *f = PyArray_SimpleNew(2, y_dims, NPY_DOUBLE);
    double *f_data = (double *) PyArray_DATA(f);
    for (int i = 0; i<y_dims[0]; i++) {
        for (int j = 0; j<y_dims[1]; j++) {
            *f_data = f_ptrs[i][j];
            f_data++;
        }
    }
    PyObject *w = PyArray_SimpleNew(1, x_dims, NPY_DOUBLE);
    double *w_data = (double *) PyArray_DATA(w);
    for (int i = 0; i<x_dims[0]; i++) {
        *w_data = w_ptr[i];
        w_data++;
    }
    return Py_BuildValue("(OO)", f, w);
};

static PyObject *nb_deviance(PyObject *self, PyObject *args) {
    PyArrayObject *x, *loc, *scale;
    PyArg_ParseTuple(args, "OOO", &x, &loc, &scale);
    if (PyErr_Occurred()) {
        return NULL;
    }
    if (!(PyArray_Check(x) && PyArray_Check(loc) && PyArray_Check(scale))) {
        PyErr_SetString(PyExc_TypeError, "Arguments must be numpy arrays.");
        return NULL;
    }
    int64_t size = PyArray_SIZE(x);
    npy_intp *dims = PyArray_DIMS(x);
    double **x_data;
    double **loc_data;
    double **scale_data;
    PyArray_AsCArray((PyObject **)&x, &x_data, dims, 2, PyArray_DescrFromType(NPY_DOUBLE));
    if (PyErr_Occurred()) {
        return NULL;
    }
    PyArray_AsCArray((PyObject **)&loc, &loc_data, dims, 2, PyArray_DescrFromType(NPY_DOUBLE));
    if (PyErr_Occurred()) {
        return NULL;
    }
    PyArray_AsCArray((PyObject **)&scale, &scale_data, dims, 2, PyArray_DescrFromType(NPY_DOUBLE));
    if (PyErr_Occurred()) {
        return NULL;
    }

    double *x_ptr = &x_data[0][0];
    double *loc_ptr = &loc_data[0][0];
    double *scale_ptr = &scale_data[0][0];


    PyObject *result = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    double *result_data = (double*) PyArray_DATA((PyArrayObject *) result);

    for(int i=0; i<size; ++i) {
        result_data[i] = compute_unit_nb_deviance(*x_ptr, *loc_ptr, *scale_ptr);
        x_ptr++;
        loc_ptr++;
        scale_ptr++;
    }
    return result;
};


static PyMethodDef methods[] = {
    {"nb_deviance", nb_deviance, METH_VARARGS, "nb_dev"},
    {"loess_by_col", loess_by_col, METH_VARARGS, "loess_by_col"},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef c_utils {
    PyModuleDef_HEAD_INIT,
    "c_utils",
    "C accelerated functions needed for batchglm",
    -1,
    methods
};

PyMODINIT_FUNC PyInit_c_utils() {

    PyObject *module = PyModule_Create(&c_utils);
    import_array();
    return module;
};
