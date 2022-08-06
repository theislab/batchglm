import numpy as np
from scipy.linalg import qr

from .adjProfileLik import adjusted_profile_likelihood
from .aveLogCPM import calculate_avg_log_cpm
from .estimator import NBEstimator
from .external import InputDataGLM, NBModel
from .maximizeInterpolant import maximize_interpolant
from .prior_df import calculate_prior_df
from .residDF import combo_groups
from .wleb import wleb


def estimate_disp(
    model: NBModel,
    group=None,  #
    prior_df=None,  # TODO
    trend_method="loess",
    tagwise: bool = True,  # TODO
    span=None,  # TODO
    min_rowsum: int = 5,  # TODO
    grid_length: int = 21,  # TODO
    grid_range: tuple = (-10, 10),  # TODO
    robust: bool = False,  # TODO
    winsor_tail_p: tuple = (0.05, 0.1),  # TODO
    tol: float = 1e-6,  # TODO
    weights=None,  # TODO
    adjust: bool = True,
    *args,
    **kwargs,
):

    """
    Implements edgeR's estimateDisp function.

    :param y: np.ndarray of counts
    :param design: design_loc
    :param prior_df: prior degrees of freedom. It is used in calculating prior.n.?????
    :param trend_method: method for estimating dispersion trend. Possible values are
        "none" and "loess" (default).
    :param mixed_df: logical, only used when trend.method="locfit".
        If FALSE, locfit uses a polynomial of degree 0.
        If TRUE, locfit uses a polynomial of degree 1 for lowly expressed genes.
        Care is taken to smooth the curve. This argument is ignored since locfit isn't implemented.
    :param tagwise: logical, should the tagwise dispersions be estimated?
    :param span: width of the smoothing window, as a proportion of the data set.
    :param min_rowsum: numeric scalar giving a value for the filtering out of low abundance tags.
        Only tags with total sum of counts above this value are used.
        Low abundance tags can adversely affect the dispersion estimation,
        so this argument allows the user to select an appropriate filter threshold for the tag abundance.
    :param grid_length: the number of points on which the interpolation is applied for each tag.
    :param grid_range: the range of the grid points around the trend on a log2 scale.
    :param robust: logical, should the estimation of prior.df be robustified against outliers?
    :param winsor_tail_p: numeric vector of length 1 or 2, giving left and right tail proportions
        of the deviances to Winsorize when estimating prior.df.
    :param tol: the desired accuracy, passed to optimize
    :param group: vector or factor giving the experimental group/condition for each library.
    :param libsize: numeric vector giving the total count (sequence depth) for each library.
    :param offset: offset matrix for the log-linear model, as for glmFit.
        Defaults to the log-effective library sizes.
    :param weights: optional numeric matrix giving observation weights
    """

    # Spline points
    spline_pts = np.linspace(start=grid_range[0], stop=grid_range[1], num=grid_length)
    spline_disp = 0.1 * 2 ** spline_pts
    l0 = np.zeros((model.num_features, grid_length))

    # Identify which observations have means of zero (weights aren't needed here).
    print("Performing initial fit...", end="")
    estimator = NBEstimator(model, dispersion=0.05)
    estimator.train(maxit=250, tolerance=tol)

    zerofit = ((model.x < 1e-4) & (np.nan_to_num(model.location) < 1e-4)).compute()  # shape (obs, features)

    groups = combo_groups(zerofit)
    coefs = list()
    print("DONE.")
    print("Calculating adjusted profile likelihoods in subgroups...", end="")
    for subgroup in groups:
        not_zero_obs_in_group = ~zerofit[:, subgroup[0]]
        if not np.any(not_zero_obs_in_group):
            continue
        if np.all(not_zero_obs_in_group):
            design_new = model.design_loc
            new_dloc_names = model.design_loc_names
        else:
            design_new = model.design_loc[not_zero_obs_in_group]
            _, _, pivot = qr(design_new, mode="raw", pivoting=True)
            coefs_new = np.array(
                pivot[: np.linalg.matrix_rank(design_new.compute())]
            )  # explicitly make this array to keep dimension info
            if len(coefs_new) == design_new.shape[0]:
                continue
            design_new = design_new[:, coefs_new]
            new_dloc_names = model.design_loc_names[coefs_new]
        input_data = InputDataGLM(
            data=model.x.compute()[np.ix_(not_zero_obs_in_group, subgroup)],
            design_loc=design_new,
            design_loc_names=new_dloc_names,
            size_factors=model.size_factors[:, not_zero_obs_in_group] if model.size_factors is not None else None,
            design_scale=model.design_scale[not_zero_obs_in_group],
            design_scale_names=np.array(["Intercept"]),
            as_dask=True,
            chunk_size_cells=1000000,
            chunk_size_genes=1000000,
        )
        group_model = NBModel(input_data)
        estimator = NBEstimator(group_model, dispersion=0.05)
        for i in range(len(spline_disp)):
            estimator.reset_theta_scale(np.log(1 / spline_disp[i]))
            l0[subgroup, i] = adjusted_profile_likelihood(estimator, adjust=adjust)

        coefs.append(estimator._model_container.theta_location.compute())

    print("DONE.")

    # Calculate common dispersion
    overall = maximize_interpolant(spline_pts, l0.sum(axis=0, keepdims=True))  # (1, spline_pts)
    common_dispersion = 0.1 * 2 ** overall

    print(f"Common dispersion is {common_dispersion}.")

    # Allow dispersion trend?
    if trend_method is not None:
        print("Calculating trended dispersion...", flush=True)
        avg_log_cpm = calculate_avg_log_cpm(
            model.x.copy(), model_class=model.__class__, dispersion=common_dispersion[0], weights=weights
        )
        span, _, m0, trend, _ = wleb(
            theta=spline_pts,
            loglik=l0,
            covariate=avg_log_cpm[0],
            trend_method=trend_method,
            span=span,
            overall=False,
            individual=False,
        )
        disp_trend = 0.1 * 2 ** trend
        trended_dispersion = np.full(model.num_features, disp_trend[np.argmin(avg_log_cpm)])
        trended_dispersion = disp_trend
        print("DONE.")
    else:
        avg_log_cpm = None
        m0 = np.broadcast_to(l0.mean(axis=0), shape=(model.x.shape[1], len(spline_pts)))
        disp_trend = common_dispersion
        trended_dispersion = None

    # Are tagwise dispersions required?
    if not tagwise:
        return common_dispersion, trended_dispersion

    # Calculate prior.df
    print("Calculating featurewise dispersion...")
    if prior_df is None:  #
        prior_df = calculate_prior_df(
            model, avg_log_cpm[0].compute(), robust=robust, winsor_tail_p=winsor_tail_p, dispersion=disp_trend
        )
    print("DONE.")
    n_loc_params = model.design_loc.shape[1]
    prior_n = prior_df / (model.num_observations - n_loc_params)

    print("Calculating featurewise dispersion...")
    # Initiate featurewise dispersions
    if trend_method is not None:
        featurewise_dispersion = trended_dispersion.copy()
    else:
        featurewise_dispersion = np.full(model.num_features, common_dispersion)

    # Checking if the shrinkage is near-infinite.
    too_large = prior_n > 1e6
    if not np.all(too_large):
        temp_n = prior_n
        if np.any(too_large):
            temp_n[too_large] = 1e6

        # Estimating tagwise dispersions
        _, _, _, _, out_individual = wleb(
            theta=spline_pts,
            loglik=l0,
            prior_n=temp_n,
            covariate=avg_log_cpm,
            trend_method=trend_method,
            span=span,
            overall=False,
            trend=False,
            m0=m0,
        )
        if not robust or len(too_large) == 1:
            featurewise_dispersion = 0.1 * 2 ** out_individual
        else:
            featurewise_dispersion[~too_large] = 0.1 * 2 ** out_individual[~too_large]
    print("DONE.")
    if robust:
        temp_df = prior_df
        temp_n = prior_n
        prior_df = np.full(model.num_features, np.inf)
        prior_n = np.full(model.num_features, np.inf)
        prior_df = temp_df
        prior_n = temp_n

    return common_dispersion, trended_dispersion, featurewise_dispersion, span, prior_df, prior_n
