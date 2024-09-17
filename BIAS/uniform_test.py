import numpy as np
from numpy.polynomial.legendre import legval


def ddst_base_legendre(x, j):
    """
    Compute the j-th Legendre polynomial evaluated at x.

    Parameters:
    - x (array-like): The sample data.
    - j (int): The degree of the polynomial.

    Returns:
    - values (numpy.ndarray): The evaluated polynomial values at x.
    """
    # Map x from [0, 1] to [-1, 1]
    x_mapped = 2 * np.array(x) - 1
    # Coefficients for the j-th Legendre polynomial
    coefs = np.zeros(j + 1)
    coefs[j] = 1
    values = legval(x_mapped, coefs)
    return values


def ddst_phi(x, j, base):
    """
    Compute the coefficient for the j-th term using the base function.

    Parameters:
    - x (array-like): The sample data.
    - j (int): The degree of the polynomial.
    - base (function): The orthonormal base function.

    Returns:
    - coefficient (float): The computed coefficient.
    """
    # Evaluate the base function at x
    phi_values = base(x, j)
    # Compute the mean value
    coefficient = np.mean(phi_values)
    return coefficient


def ddst_uniform_Nk(x, base=None, Dmax=10):
    """
    Compute the cumulative sums for the data-driven smooth test of uniformity.

    Parameters:
    - x (array-like): The sample data.
    - base (function): The orthonormal base function to use (default is ddst_base_legendre).
    - Dmax (int): The maximum degree of the polynomial.

    Returns:
    - coord (numpy.ndarray): The cumulative sums of the transformed data.
    """
    if base is None:
        base = ddst_base_legendre

    n = len(x)
    maxN = max(min(Dmax, n - 2, 20), 1)
    coord = np.zeros(maxN)
    for j in range(1, maxN + 1):
        coord[j - 1] = ddst_phi(x, j, base)
    coord = np.cumsum(coord**2 * n)
    return coord


def ddst_IIC(coord, n, c=2.4):
    """
    Compute the model selection index l using the Information Criterion.

    Parameters:
    - coord (numpy.ndarray): The cumulative sums.
    - n (int): Sample size.
    - c (float): Calibrating parameter in the penalty in the model selection rule.

    Returns:
    - l (int): The selected index (starting from 1).
    """
    Dmax = len(coord)
    ic = coord - c * np.arange(1, Dmax + 1)
    l = np.argmin(ic) + 1  # Add 1 because numpy arrays are 0-indexed
    return l


def ddst_uniform_test(
    x,
    base=ddst_base_legendre,
    d_n=10,
    c=2.4,
    nr=100000,
    compute_p=True,
    alpha=0.05,
    compute_cv=True,
    **kwargs,
):
    """
    Data Driven Smooth Test for Uniformity.

    Parameters:
    - x (array-like): A (non-empty) numeric vector of data.
    - base (function): Function returning an orthonormal system (default is ddst_base_legendre).
    - d_n (int): Maximum dimension considered.
    - c (float): Calibrating parameter in the penalty in the model selection rule.
    - nr (int): Number of runs for p-value and critical value computation.
    - compute_p (bool): Whether to compute a p-value.
    - alpha (float): Significance level.
    - compute_cv (bool): Whether to compute a critical value corresponding to alpha.
    - kwargs: Further arguments.

    Returns:
    - result (dict): A dictionary containing test results.
    """
    # Only Legendre base is implemented yet
    base = ddst_base_legendre
    method_name = "ddst_base_legendre"

    x = np.asarray(x)
    n = len(x)
    if n < 5:
        raise ValueError("length(x) should be at least 5")

    # Compute coordinates
    coord = ddst_uniform_Nk(x, base=base, Dmax=d_n)
    # Compute model selection index l
    l = ddst_IIC(coord, n, c)
    # Test statistic t
    t = coord[l - 1]  # Adjust for zero-based indexing
    # Coordinates differences
    coord_diffs = coord - np.concatenate(([0], coord[:-1]))
    # Prepare result
    result = {
        "statistic": t,
        "parameter": l,
        "coordinates": coord_diffs,
        "method": "Data Driven Smooth Test for Uniformity",
    }

    # Compute p-value and critical value if required
    if compute_p or compute_cv:
        tmp = np.zeros(nr)
        for i in range(nr):
            y = np.random.uniform(0, 1, n)
            tmpC = ddst_uniform_Nk(y, base=base, Dmax=d_n)
            l_sim = ddst_IIC(tmpC, n, c)
            tmp[i] = tmpC[l_sim - 1]  # Adjust index for zero-based indexing
        if compute_p:
            result["p_value"] = np.mean(tmp > t)
        if compute_cv:
            result["cv"] = np.quantile(tmp, alpha)

    # Construct data name
    data_name = f"x, base: {method_name}  c: {c}  d_n: {d_n}" + (
        f"  cv({alpha}) : {result['cv']:.5f}" if compute_cv else ""
    )
    result["data_name"] = data_name

    return result
