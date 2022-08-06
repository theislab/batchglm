from scipy.linalg.lapack import dormqr


def calc_effects(qr, tau, y, trans: bool = True):
    """
    This function calculates the effects as returned by R's lm.fit(design, y)$effects.
    The input parameters are expected to originate from calculating a qr decomposition using
    (qr, tau), r = scipy.linalg.qr(design, mode='raw').
    This function is replicating R's qr.qty(qr.result, y) using the internal C-function qr_qy_real:
    https://github.com/SurajGupta/r-source/blob/a28e609e72ed7c47f6ddfbb86c85279a0750f0b7/src/modules/lapack/Lapack.c#L1206
    The function uses scipy's lapack interface to call the fortran soubroutine dormqr.
    """
    cq, work, info = dormqr(side="L", trans="T" if trans else "F", a=qr, tau=tau, c=y, lwork=-1)
    if info != 0:
        raise RuntimeError(f"dormqr in calc_effects returned error code {info}")

    cq, work, info = dormqr(side="L", trans="T" if trans else "F", a=qr, tau=tau, c=y, lwork=work[0])
    if info != 0:
        raise RuntimeError(f"dormqr in calc_effects returned error code {info}")
    return cq
