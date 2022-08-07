from typing import Optional

import numpy as np
from scipy.stats import rankdata


def calc_size_factors(x: np.ndarray, method: Optional[str] = None, *args, **kwargs):
    assert ~np.any(np.isnan(x)), "Counts matrix must not contain NaN!"
    x = x[:, np.sum(x, axis=0) > 0]

    if method is None:
        size_factors = np.ones((x.shape[1], 1), dtype=float)
    elif method.lower() == "tmm":
        size_factors = _calc_factor_tmm(data=x, *args, **kwargs)
    elif method.lower() == "tmmwsp":
        size_factors = _calc_factor_tmmwsp(data=x, *args, **kwargs)
    elif method.lower() == "rle":
        size_factors = _calc_factor_rle(data=x)
    elif method == "upperquartile":
        size_factors = _calc_factor_quantile(data=x, *args, **kwargs)
    else:
        raise ValueError(f"Method {method} not recognized.")

    # 	Factors should multiple to one
    size_factors = size_factors / np.exp(np.mean(np.log(size_factors)))
    return size_factors


def _calc_factor_rle(data: np.ndarray):
    # 	Scale factors as in Anders et al (2010)
    geometric_feature_means = np.exp(np.mean(np.log(data), axis=0))
    adjusted_data = data / geometric_feature_means
    return np.median(adjusted_data[:, geometric_feature_means > 0], axis=1, keepdims=True) / np.sum(
        data, axis=1, keepdims=True
    )


def _calc_factor_quantile(data, p=0.75):
    # 	Generalized version of upper-quartile normalization
    size_factors = np.quantile(data, q=p, axis=1, keepdims=True)
    if np.min(size_factors) == 0:
        print("Warning: One or more quantiles are zero.")
    size_factors = size_factors / np.sum(data, axis=1, keepdims=True)
    return size_factors


def _calc_factor_tmm(
    data: np.ndarray,
    ref_idx: Optional[int] = None,
    logratio_trim: float = 0.3,
    sum_trim: float = 0.05,
    do_weighting: bool = True,
    a_cutoff: float = -1e10,
):
    # 	TMM between two libraries

    if ref_idx is None:
        f75 = _calc_factor_quantile(data, p=0.75)
        if np.median(f75) < 1e-20:
            ref_idx = np.argmax(np.sum(np.sqrt(data), axis=1))
        else:
            ref_idx = np.argmin(np.abs(f75 - np.mean(f75)))

    sample_sums = np.sum(data, axis=1, keepdims=True)
    sum_normalized_data = data / sample_sums
    with np.errstate(divide="ignore", invalid="ignore"):
        opfer = sum_normalized_data / sum_normalized_data[ref_idx]
        log_ratios = np.log2(opfer)
        absolute_values = (np.log2(sum_normalized_data) + np.log2(sum_normalized_data[ref_idx])) / 2
        estimated_asymptotic_variance = (sample_sums - data) / sample_sums / data
        estimated_asymptotic_variance += (sample_sums[ref_idx] - data[ref_idx]) / sample_sums[ref_idx] / data[ref_idx]

    # 	remove infinite values, cutoff based on aCutOff
    finite_idx = np.isfinite(log_ratios) & np.isfinite(absolute_values) & (absolute_values > a_cutoff)

    size_factors = np.ones_like(sample_sums, dtype=float)
    for i in range(data.shape[0]):
        log_ratios_i = log_ratios[i, finite_idx[i]]
        absolute_values_i = absolute_values[i, finite_idx[i]]
        estimated_asymptotic_variance_i = estimated_asymptotic_variance[i, finite_idx[i]]

        if np.max(np.abs(log_ratios_i) < 1e-6):
            continue

        # 	taken from the original mean() function
        n = len(log_ratios_i)
        lo_l = np.floor(n * logratio_trim) + 1
        hi_l = n + 1 - lo_l
        lo_s = np.floor(n * sum_trim) + 1
        hi_s = n + 1 - lo_s

        keep = (rankdata(log_ratios_i) >= lo_l) & (rankdata(log_ratios_i) <= hi_l)
        keep &= (rankdata(absolute_values_i) >= lo_s) & (rankdata(absolute_values_i) <= hi_s)

        if do_weighting:
            size_factor_i = np.nansum(log_ratios_i[keep] / estimated_asymptotic_variance_i[keep])
            size_factor_i = size_factor_i / np.nansum(1 / estimated_asymptotic_variance_i[keep])
        else:
            size_factor_i = np.nanmean(log_ratios_i[keep])

        # 	Results will be missing if the two libraries share no features with positive counts
        # 	In this case, return unity
        if np.isnan(size_factor_i):
            size_factor_i = 0
        size_factors[i] = 2 ** size_factor_i
    return size_factors


def _calc_factor_tmmwsp(
    data: np.ndarray,
    ref_idx: Optional[int] = None,
    logratio_trim: float = 0.3,
    sum_trim: float = 0.05,
    do_weighting: bool = True,
    a_cutoff: float = -1e10,
):
    # 	TMM with pairing of singleton positive counts between the obs and ref libraries
    if ref_idx is None:
        ref_idx = np.argmax(np.sum(np.sqrt(data), axis=1))
    eps = 1e-14
    sample_sums = np.sum(data, axis=1, keepdims=True)

    # 	Identify zero counts
    n_pos = np.where(data > eps, 1, 0)
    n_pos = 2 * n_pos + n_pos[ref_idx]

    size_factors = np.ones_like(sample_sums, dtype=float)

    for i in range(data.shape[0]):
        # 	Remove double zeros and NAs
        keep = np.where(n_pos[i] > 0)
        data_i = data[i, keep]
        ref_i = data[ref_idx, keep]
        n_pos_i = n_pos[i, keep]

        # 	Pair up as many singleton positives as possible
        # 	The unpaired singleton positives are discarded so that no zeros remain
        zero_obs = n_pos_i == 1
        zero_ref = n_pos_i == 2
        k = zero_obs | zero_ref
        n_eligible_singles = np.min((np.sum(zero_obs), np.sum(zero_ref)))
        if n_eligible_singles > 0:
            ref_i_k = np.sort(ref_i[k])[::-1][:n_eligible_singles]
            data_i_k = np.sort(data_i[k])[::-1][:n_eligible_singles]
            data_i = np.concatenate((data_i[~k], data_i_k))
            ref_i = np.concatenate((ref_i[~k], ref_i_k))
        else:
            data_i = data_i[~k]
            ref_i = ref_i[~k]

        # 	Any left?
        n = len(data_i)
        if n == 0:
            continue

        # 	Compute M and A values
        data_i_p = data_i / sample_sums[i]
        ref_i_p = ref_i / sample_sums[ref_idx]
        m = np.log2(data_i_p / ref_i_p)
        a = 0.5 * np.log2(data_i_p * ref_i_p)

        # 	If M all zero, return 1
        if np.max(np.abs(m)) < 1e-6:
            continue

        # 	M order, breaking ties by shrunk M
        data_i_p_shrunk = (data_i + 0.5) / (sample_sums[i] + 0.5)
        ref_i_p_shrunk = (ref_i + 0.5) / (sample_sums[ref_idx] + 0.5)
        m_shrunk = np.log2(data_i_p_shrunk / ref_i_p_shrunk)
        m_ordered = np.argsort(
            np.array(list(zip(m, m_shrunk)), dtype={"names": ["m", "m_shrunk"], "formats": [m.dtype, m_shrunk.dtype]}),
            kind="stable",
        )

        # 	a order
        a_ordered = np.argsort(a, kind="stable")

        # 	Trim
        lo_m = int(n * logratio_trim)
        hi_m = n - lo_m
        keep_m = np.zeros(n, dtype=bool)
        keep_m[m_ordered[lo_m:hi_m]] = True
        lo_a = int(n * sum_trim)
        hi_a = n - lo_a
        keep_a = np.zeros(n, dtype=bool)
        keep_a[a_ordered[lo_a:hi_a]] = True
        keep = keep_a & keep_m
        m = m[keep]

        # 	Average the m values
        if do_weighting:
            data_i_p = data_i_p[keep]
            ref_i_p = ref_i_p[keep]
            v = (1 - data_i_p) / data_i_p / sample_sums[i] + (1 - ref_i_p) / ref_i_p / sample_sums[ref_idx]
            w = (1 + 1e-6) / (v + 1e-6)
            size_factor_i = np.sum(w * m) / np.sum(w)
        else:
            size_factor_i = np.mean(m)
        size_factors[i] = 2 ** size_factor_i
    return size_factors
