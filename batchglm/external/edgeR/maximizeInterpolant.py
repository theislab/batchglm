import numpy as np


def maximize_interpolant(x, y):
    # This function takes an ordered set of spline points and a likelihood matrix where each row
    # corresponds to a tag and each column corresponds to a spline point. It then calculates the
    # position at which the maximum interpolated likelihood occurs for each by solving the derivative
    # of the spline function.

    interpolator = Interpolator(n=len(x))
    output = np.zeros(y.shape[0], dtype=float)
    for i in range(y.shape[0]):
        output[i] = interpolator.find_max(x, y[i])
    return output


class Interpolator:
    def __init__(self, n: int):

        self.npts = n
        if self.npts < 2:
            raise ValueError("Must have at lest two points for interpolation")

        self.b = np.zeros(n, dtype=float)
        self.c = np.zeros(n, dtype=float)
        self.d = np.zeros(n, dtype=float)

    def find_max(self, x, y):
        maxed_at = np.argmax(y)
        maxed = y[maxed_at]
        """
        maxed = -1
        maxed_at = -1
        for i in range(self.npts):
            # Getting a good initial guess for the MLE.
            if maxed_at < 0 or y[i] > maxed:
                maxed = y[i]
                maxed_at = i
        """
        x_max = x[maxed_at]
        x, y, self.b, self.c, self.d = fmm_spline(self.npts, x, y, self.b, self.c, self.d)

        # First we have a look at the segment on the left and see if it contains the maximum.

        if maxed_at > 0:
            ld = self.d[maxed_at - 1]
            lc = self.c[maxed_at - 1]
            lb = self.b[maxed_at - 1]
            sol1_left, sol2_left, solvable_left = quad_solver(3 * ld, 2 * lc, lb)
            if solvable_left:
                """
                Using the solution with the maximum (not minimum). If the curve is mostly increasing, the
                maximal point is located at the smaller solution (i.e. sol1 for a>0). If the curve is mostly
                decreasing, the maximal is located at the larger solution (i.e., sol1 for a<0).
                """
                chosen_sol = sol1_left
                """
                The spline coefficients are designed such that 'x' in 'y + b*x + c*x^2 + d*x^3' is
                equal to 'x_t - x_l' where 'x_l' is the left limit of that spline segment and 'x_t'
                is where you want to get an interpolated value. This is necessary in 'splinefun' to
                ensure that you get 'y' (i.e. the original data point) when 'x=0'. For our purposes,
                the actual MLE corresponds to 'x_t' and is equal to 'solution + x_0'.
                """
                if (chosen_sol > 0) and (chosen_sol < (x[maxed_at] - x[maxed_at - 1])):
                    temp = ((ld * chosen_sol + lc) * chosen_sol + lb) * chosen_sol + y[maxed_at - 1]
                    if temp > maxed:
                        maxed = temp
                        x_max = chosen_sol + x[maxed_at - 1]

        # Repeating for the segment on the right.

        if maxed_at < self.npts - 1:
            rd = self.d[maxed_at]
            rc = self.c[maxed_at]
            rb = self.b[maxed_at]
            sol1_right, sol2_right, solvable_right = quad_solver(3 * rd, 2 * rc, rb)
            if solvable_right:
                chosen_sol = sol1_right
                print(sol1_right, sol2_right)
                if (chosen_sol > 0) and (chosen_sol < (x[maxed_at + 1] - x[maxed_at])):
                    temp = ((rd * chosen_sol + rc) * chosen_sol + rb) * chosen_sol + y[maxed_at]
                    if temp > maxed:
                        maxed = temp
                        x_max = chosen_sol + x[maxed_at]

        return x_max


def fmm_spline(n: int, x: np.ndarray, y: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray):
    """
    This code is a python derivative of fmm_spline in R core splines.c as implemented in edgeR.
    """
    """
        Spline Interpolation
        --------------------
        C code to perform spline fitting and interpolation.
        There is code here for:

        1. Splines with end-conditions determined by fitting
        cubics in the start and end intervals (Forsythe et al).


        Computational Techniques
        ------------------------
        A special LU decomposition for symmetric tridiagonal matrices
        is used for all computations, except for periodic splines where
        Choleski is more efficient.


        Splines a la Forsythe Malcolm and Moler
        ---------------------------------------
        In this case the end-conditions are determined by fitting
        cubic polynomials to the first and last 4 points and matching
        the third derivitives of the spline at the end-points to the
        third derivatives of these cubics at the end-points.
    """

    i = 0
    t = 0

    # Adjustment for 1-based arrays
    """
    x -= 1
    y -= 1
    b -= 1
    c -= 1
    d -= 1
    """

    if n < 2:
        return x, y, b, c, d

    if n < 3:
        t = y[1] - y[0]
        b[0] = t / (x[1] - x[0])
        b[1] = b[0]
        c[0] = 0.0
        c[1] = 0.0
        d[0] = 0.0
        d[1] = 0.0
        return x, y, b, c, d

    # Set up tridiagonal system
    # b = diagonal, d = offdiagonal, c = right hand side

    d[0] = x[1] - x[0]
    c[1] = (y[1] - y[0]) / d[0]  # ;/* = +/- Inf      for x[1]=x[2] -- problem? */
    for i in range(1, n - 1):
        d[i] = x[i + 1] - x[i]
        b[i] = 2.0 * (d[i - 1] + d[i])
        c[i + 1] = (y[i + 1] - y[i]) / d[i]
        c[i] = c[i + 1] - c[i]

    """
    End conditions.
    Third derivatives at x[0] and x[n-1] obtained
    from divided differences
    """

    b[0] = -d[0]

    b[n - 1] = -d[n - 2]
    c[0] = 0.0
    c[n - 1] = 0.0
    if n > 3:
        c[0] = c[2] / (x[3] - x[1]) - c[1] / (x[2] - x[0])
        c[n - 1] = c[n - 2] / (x[n - 1] - x[n - 3]) - c[n - 3] / (x[n - 2] - x[n - 4])
        c[0] = c[0] * d[0] * d[0] / (x[3] - x[0])
        c[n - 1] = -c[n - 1] * d[n - 2] * d[n - 2] / (x[n - 1] - x[n - 4])

    # Gaussian elimination
    for i in range(1, n):
        t = d[i - 1] / b[i - 1]
        b[i] = b[i] - t * d[i - 1]
        c[i] = c[i] - t * c[i - 1]

    # Backward substitution

    c[n - 1] = c[n - 1] / b[n - 1]
    for i in range(n - 2, -1, -1):
        c[i] = (c[i] - d[i] * c[i + 1]) / b[i]

    # c[i] is now the sigma[i-1] of the text
    # Compute polynomial coefficients

    b[n - 1] = (y[n - 1] - y[n - 2]) / d[n - 2] + d[n - 2] * (c[n - 2] + 2.0 * c[n - 1])
    for i in range(0, n - 1):
        b[i] = (y[i + 1] - y[i]) / d[i] - d[i] * (c[i + 1] + 2.0 * c[i])
        d[i] = (c[i + 1] - c[i]) / d[i]
        c[i] = 3.0 * c[i]

    c[n - 1] = 3.0 * c[n - 1]
    d[n - 1] = d[n - 2]
    return x, y, b, c, d


def quad_solver(a: float, b: float, c: float):
    """
    Find the two solutions for the formula x = (-b +- sqrt(b^2-4ac) / 2a
    :return tuple(sol1, sol2, solvable).
    """
    back = np.square(b) - 4 * a * c
    if back < 0:
        return None, None, False
    front = -b / (2 * a)
    back = np.sqrt(back) / (2 * a)
    return front - back, front + back, True
