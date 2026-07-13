"""Shared utilities for aDDM parameter-sweep experiments.

Provides functions for data generation (via efpt.aDDModel),
parameter recovery (MLE and TADA), and plotting.
"""

import glob
import os
import pickle
import time

import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds

from efpt import aDDModel
from efpt.cython.batch import compute_addm_nll, compute_tada_mean_nll

# Parameter ordering convention: [eta, kappa, a, x0]
PARAM_NAMES = ["eta", "kappa", "a", "x0"]


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def generate_sweep_data(
    swept_param_name,
    swept_values,
    fixed_params,
    n_trials=30000,
    gamma_shape=4.0,
    gamma_scale=0.1,
    r_range=(1, 5),
    T=20.0,
    dt=1e-4,
    n_threads=-1,
    rng=None,
):
    """Sweep one aDDM parameter and generate simulated data for each value.

    Parameters
    ----------
    swept_param_name : str
        One of "eta", "kappa", "sigma", "a", "b", "x0".
    swept_values : array-like
        Values to sweep over.
    fixed_params : dict
        Must contain keys: eta, kappa, sigma, a, b, x0.
        The swept_param_name key will be overridden per sweep value.
    n_trials : int
        Number of trials per sweep value.
    gamma_shape, gamma_scale : float
        Parameters for fixation-duration Gamma distribution.
    r_range : tuple of (int, int)
        Inclusive range for stimulus ratings.
    T : float
        Maximum trial duration.
    dt : float
        Euler-Maruyama time step.
    n_threads : int
        OpenMP threads for Cython simulator (-1 = all).
    rng : int, np.random.Generator, or None
        Seed or Generator for reproducibility.

    Returns
    -------
    str
        Path to the combined pkl file.
    """
    rng = np.random.default_rng(rng)
    all_data = {}

    for val in swept_values:
        params = dict(fixed_params)
        params[swept_param_name] = float(val)

        model = aDDModel(
            eta=params["eta"],
            kappa=params["kappa"],
            sigma=params["sigma"],
            a=params["a"],
            b=params["b"],
            x0=params["x0"],
        )

        print(f"Simulating {swept_param_name}={val:.4f} ...")
        start = time.time()
        experiment = model.generate_experiment(
            n_trials=n_trials,
            gamma_shape=gamma_shape,
            gamma_scale=gamma_scale,
            r_range=r_range,
            T=T,
            dt=dt,
            n_threads=n_threads,
            rng=rng,
        )
        elapsed = time.time() - start
        print(f"  {n_trials} trials in {elapsed:.1f}s")

        all_data[float(val)] = experiment

    fname = f"addm_data_all_{swept_param_name}_{time.strftime('%Y%m%d-%H%M%S')}.pkl"
    with open(fname, "wb") as f:
        pickle.dump(all_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {len(all_data)} configurations → {fname}")
    return fname


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_sweep_data(swept_param_name, pkl_path=None):
    """Load a combined sweep pkl file.

    Parameters
    ----------
    swept_param_name : str
        Name of the swept parameter (used to construct default glob pattern).
    pkl_path : str or None
        Explicit path. If None, uses the most recent matching pkl file.

    Returns
    -------
    all_data : dict
        Keyed by swept parameter value (float).
    data_keys : list
        Sorted list of swept parameter values.
    """
    if pkl_path is None:
        pattern = f"addm_data_all_{swept_param_name}_*.pkl"
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No files matching {pattern}")
        pkl_path = matches[-1]

    with open(pkl_path, "rb") as f:
        all_data = pickle.load(f)

    data_keys = sorted(all_data.keys())
    print(f"Loaded {len(data_keys)} {swept_param_name} values from {pkl_path}")
    return all_data, data_keys


# ---------------------------------------------------------------------------
# Parameter recovery
# ---------------------------------------------------------------------------


def _extract_trial_data(data, n_rows):
    """Extract and slice trial data from a canonical experiment dict.

    Returns a flat dict with all arrays sliced to n_rows and cast to proper dtypes.
    """
    rt_data = data["decision_data"]["rt_data"][:n_rows].astype(np.float64)
    choice_data = data["decision_data"]["choice_data"][:n_rows].astype(np.int32)
    r1_data = data["covariates"]["r1_data"][:n_rows].astype(np.float64)
    r2_data = data["covariates"]["r2_data"][:n_rows].astype(np.float64)
    flag_data = data["covariates"]["flag_data"][:n_rows].astype(np.int32)
    sacc_data = data["covariates"]["sacc_array_data"][:n_rows].astype(np.float64)
    length_data = data["covariates"]["d_data"][:n_rows].astype(np.int32)
    actual_n, max_d = sacc_data.shape

    return {
        "rt_data": rt_data,
        "choice_data": choice_data,
        "r1_data": r1_data,
        "r2_data": r2_data,
        "flag_data": flag_data,
        "sacc_data": sacc_data,
        "length_data": length_data,
        "max_d": max_d,
        "sigma": float(data["params"]["sigma"]),
        "b": float(data["params"]["b"]),
        "eta_true": float(data["params"]["eta"]),
        "kappa_true": float(data["params"]["kappa"]),
        "a_true": float(data["params"]["a"]),
        "x0_true": float(data["params"]["x0"]),
    }


def run_param_recovery(
    all_data,
    num_data_list,
    mode,
    method="trust-constr",
    initial_guess=None,
    n_threads=-1,
    verbose=True,
):
    """Run MLE or TADA parameter recovery across all swept values and data sizes.

    Parameters
    ----------
    all_data : dict
        Combined sweep data, keyed by swept parameter value.
    num_data_list : list of int
        Dataset sizes to test.
    mode : str
        "mle" for compute_addm_nll, "tada" for compute_tada_mean_nll.
    method : str
        scipy.optimize.minimize method.
    initial_guess : list of float or None
        Starting point [eta, kappa, a, x0]. Default: [0.5, 0.5, 1.0, 0.0].
    n_threads : int
        OpenMP threads for NLL computation.
    verbose : bool
        Print progress.

    Returns
    -------
    paras_true_arr : ndarray, shape (len(num_data_list), len(data_keys), 4)
    paras_est_arr : ndarray, shape (len(num_data_list), len(data_keys), 4)
    """
    if initial_guess is None:
        initial_guess = [0.5, 0.5, 1.0, 0.0]

    if mode == "mle":
        nll_fn = compute_addm_nll
        bounds = Bounds([0, 0, 0, -np.inf], [1, np.inf, np.inf, np.inf])
    elif mode == "tada":
        nll_fn = compute_tada_mean_nll
        bounds = Bounds([-np.inf, 0, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf])
    else:
        raise ValueError(f"mode must be 'mle' or 'tada', got {mode!r}")

    con = LinearConstraint(
        [[0, 0, 1, 1], [0, 0, 1, -1]],
        lb=[0, 0],
        ub=[np.inf, np.inf],
    )

    data_keys = sorted(all_data.keys())
    paras_true_arr = np.zeros((len(num_data_list), len(data_keys), 4))
    paras_est_arr = np.zeros((len(num_data_list), len(data_keys), 4))

    nll_kwargs = dict(n_threads=n_threads, invalid_policy="warn", warn=False)
    if mode == "mle":
        nll_kwargs["reduce"] = "mean"

    for i, data_name in enumerate(data_keys):
        for j, nd in enumerate(num_data_list):
            trial = _extract_trial_data(all_data[data_name], nd)

            if j == 0 and verbose:
                print(
                    f"true params: [{trial['eta_true']:.2f}, "
                    f"{trial['kappa_true']:.2f}, "
                    f"{trial['a_true']:.2f}, "
                    f"{trial['x0_true']:.2f}]"
                )

            func = lambda paras, t=trial: nll_fn(
                t["rt_data"],
                t["choice_data"],
                paras[0],
                paras[1],
                t["sigma"],
                paras[2],
                t["b"],
                paras[3],
                t["r1_data"],
                t["r2_data"],
                t["flag_data"],
                t["sacc_data"],
                t["length_data"],
                **nll_kwargs,
            )

            result = minimize(
                func,
                x0=initial_guess,
                bounds=bounds,
                constraints=con,
                method=method,
                options={"verbose": 0},
            )

            if not result.success and verbose:
                print(f"  WARNING: optimization failed for {data_name}")

            mle = result.x
            paras_true_arr[j, i] = [
                trial["eta_true"],
                trial["kappa_true"],
                trial["a_true"],
                trial["x0_true"],
            ]
            paras_est_arr[j, i] = mle

            if verbose:
                label = "mle" if mode == "mle" else "tada mle"
                print(
                    f"num_data: {nd:5d}, "
                    f"{label}: [{', '.join(f'{x:.5f}' for x in mle)}]"
                )

        if verbose:
            print()

    return paras_true_arr, paras_est_arr


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def setup_latex_matplotlib():
    """Configure matplotlib for LaTeX rendering.

    Attempts to enable LaTeX text rendering. Falls back to non-LaTeX
    rendering if a LaTeX installation is not found.
    """
    import matplotlib as mpl
    import shutil

    if shutil.which("latex") is None:
        import warnings
        warnings.warn(
            "LaTeX not found on PATH. Plots will use default matplotlib fonts.",
            stacklevel=2,
        )
        return

    mpl.rcParams["text.usetex"] = True
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["text.latex.preamble"] = (
        r"\usepackage{amsmath}\usepackage{amsfonts}"
    )


def plot_param_estimate(
    paras_true,
    paras_mle,
    paras_tada,
    param_idx,
    param_name,
    swept_idx,
    swept_name,
    num_data_list,
    num_data_idx=-2,
    identity_line=False,
    horizontal_line=None,
    xlim=None,
    ylim=None,
    figsize=(8, 8),
    dpi=300,
    save_path=None,
    ax=None,
):
    """Scatter plot comparing MLE and TADA estimates of one parameter.

    Parameters
    ----------
    paras_true, paras_mle, paras_tada : ndarray (n_sizes, n_swept, 4)
    param_idx : int
        Index into [eta, kappa, a, x0] for the y-axis (estimated parameter).
    param_name : str
        LaTeX name for the estimated parameter.
    swept_idx : int
        Index into [eta, kappa, a, x0] for the x-axis (swept parameter).
    swept_name : str
        LaTeX name for the swept parameter.
    num_data_list : list of int
    num_data_idx : int
        Which dataset size to plot.
    identity_line : bool
        Draw y=x line (when swept param == estimated param).
    horizontal_line : float or None
        Draw horizontal line at this value (true constant value).
    """
    import matplotlib.pyplot as plt

    n = num_data_list[num_data_idx]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.scatter(
        paras_true[num_data_idx, :, swept_idx],
        paras_mle[num_data_idx, :, param_idx],
        label=rf"$\widehat{{{param_name}}}_{{{n}}}^{{\text{{ML}}}}$",
        color="blue",
    )
    ax.scatter(
        paras_true[num_data_idx, :, swept_idx],
        paras_tada[num_data_idx, :, param_idx],
        label=rf"$\widehat{{{param_name}}}_{{{n}}}^{{\text{{TADA}}}}$",
        color="red",
    )

    x_range = paras_true[num_data_idx, :, swept_idx]
    if identity_line:
        ax.plot(
            np.linspace(x_range.min(), x_range.max(), 100),
            np.linspace(x_range.min(), x_range.max(), 100),
            c="k",
            label="identity function",
        )
    elif horizontal_line is not None:
        ax.axhline(
            horizontal_line, 0, 1, c="k",
            label=rf"${param_name}={horizontal_line}$",
        )

    ax.legend(fontsize=15)
    ax.set_xlabel(rf"${swept_name}$", fontsize=20)
    ax.set_ylabel(rf"estimate of ${param_name}$", fontsize=20)
    ax.tick_params(axis="both", labelsize=15)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


# ---------------------------------------------------------------------------
# Hypothesis testing helpers
# ---------------------------------------------------------------------------


def extract_data(experiment):
    """Extract arrays from a generate_experiment() output dict.

    Returns a flat dict with proper dtypes for use with compute_addm_nll
    and compute_tada_mean_nll.
    """
    return {
        "rt": experiment["decision_data"]["rt_data"].astype(np.float64),
        "choice": experiment["decision_data"]["choice_data"].astype(np.int32),
        "r1": experiment["covariates"]["r1_data"].astype(np.float64),
        "r2": experiment["covariates"]["r2_data"].astype(np.float64),
        "flag": experiment["covariates"]["flag_data"].astype(np.int32),
        "sacc": experiment["covariates"]["sacc_array_data"].astype(np.float64),
        "d": experiment["covariates"]["d_data"].astype(np.int32),
        "sigma": float(experiment["params"]["sigma"]),
        "b": float(experiment["params"]["b"]),
        "n": len(experiment["decision_data"]["rt_data"]),
    }


def subject_mean_nll(p, data, mode, n_threads=-1):
    """Mean NLL for one subject at parameters p = [eta, kappa, a, x0]."""
    if mode == "mle":
        return compute_addm_nll(
            data["rt"], data["choice"],
            p[0], p[1], data["sigma"], p[2], data["b"], p[3],
            data["r1"], data["r2"], data["flag"],
            data["sacc"], data["d"],
            n_threads=n_threads, reduce="mean",
            invalid_policy="warn", warn=False,
        )
    else:
        return compute_tada_mean_nll(
            data["rt"], data["choice"],
            p[0], p[1], data["sigma"], p[2], data["b"], p[3],
            data["r1"], data["r2"], data["flag"],
            data["sacc"], data["d"],
            n_threads=n_threads,
            invalid_policy="warn", warn=False,
        )


def subject_sum_nll(p, data, mode, n_threads=-1):
    """Summed NLL for one subject. LRT statistics require sums, not means."""
    return data["n"] * subject_mean_nll(p, data, mode, n_threads)


def fit_subject(data, mode, initial_guess=None, n_threads=-1):
    """Fit (eta, kappa, a, x0) to one subject.

    Returns the full scipy OptimizeResult (access .x for parameters, .fun for NLL).
    """
    if initial_guess is None:
        initial_guess = [0.5, 0.5, 1.0, 0.0]

    if mode == "mle":
        bounds = Bounds([0, 0, 0, -np.inf], [1, np.inf, np.inf, np.inf])
    else:
        bounds = Bounds([-np.inf, 0, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf])

    con = LinearConstraint([[0, 0, 1, 1], [0, 0, 1, -1]], lb=[0, 0])
    func = lambda p: subject_mean_nll(p, data, mode, n_threads)
    return minimize(func, x0=initial_guess, bounds=bounds, constraints=con,
                    method="trust-constr", options={"verbose": 0})


def fit_equal_eta(data_I, data_II, mode, initial_guess=None, n_threads=-1):
    """Fit shared eta with free nuisance params per population.

    Parameter vector: x = [eta, kappa_I, a_I, x0_I, kappa_II, a_II, x0_II].
    Objective is the joint summed NLL.

    Returns the full scipy OptimizeResult.
    """
    if initial_guess is None:
        initial_guess = [0.5, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0]

    if mode == "mle":
        bounds = Bounds(
            [0, 0, 0, -np.inf, 0, 0, -np.inf],
            [1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        )
    else:
        bounds = Bounds(
            [-np.inf, 0, 0, -np.inf, 0, 0, -np.inf],
            [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        )

    con = LinearConstraint(
        [[0, 0, 1, 1, 0, 0, 0], [0, 0, 1, -1, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 1, -1]],
        lb=[0, 0, 0, 0],
    )

    def func(x):
        pI = [x[0], x[1], x[2], x[3]]
        pII = [x[0], x[4], x[5], x[6]]
        return subject_sum_nll(pI, data_I, mode, n_threads) + \
               subject_sum_nll(pII, data_II, mode, n_threads)

    return minimize(func, x0=initial_guess, bounds=bounds, constraints=con,
                    method="trust-constr", options={"verbose": 0})


def greater_lrt(data_I, data_II, mode, critical_value, n_threads=-1):
    """One-sided LRT for H0: eta_I <= eta_II vs H1: eta_I > eta_II.

    Returns (Lambda_plus, reject, eta_I_hat, eta_II_hat).

    Lambda_+ = 0 if eta_I_hat <= eta_II_hat (full estimate already in H0).
    Otherwise Lambda_+ = 2*(NLL_boundary - NLL_full) where the boundary
    is eta_I = eta_II.
    """
    res_I = fit_subject(data_I, mode, n_threads=n_threads)
    res_II = fit_subject(data_II, mode, n_threads=n_threads)
    full_nll = subject_sum_nll(res_I.x, data_I, mode, n_threads) + \
               subject_sum_nll(res_II.x, data_II, mode, n_threads)

    if res_I.x[0] <= res_II.x[0]:
        return 0.0, False, res_I.x[0], res_II.x[0]

    init_eq = [0.5 * (res_I.x[0] + res_II.x[0]),
               res_I.x[1], res_I.x[2], res_I.x[3],
               res_II.x[1], res_II.x[2], res_II.x[3]]
    res_eq = fit_equal_eta(data_I, data_II, mode, initial_guess=init_eq,
                           n_threads=n_threads)
    Lambda_plus = max(2.0 * (res_eq.fun - full_nll), 0.0)
    return Lambda_plus, Lambda_plus > critical_value, res_I.x[0], res_II.x[0]


def less_lrt(data_I, data_II, mode, critical_value, n_threads=-1):
    """One-sided LRT for H0: eta_I >= eta_II vs H1: eta_I < eta_II.

    Returns (Lambda_minus, reject, eta_I_hat, eta_II_hat).

    Lambda_- = 0 if eta_I_hat >= eta_II_hat (full estimate already in H0).
    Otherwise Lambda_- = 2*(NLL_boundary - NLL_full).
    """
    res_I = fit_subject(data_I, mode, n_threads=n_threads)
    res_II = fit_subject(data_II, mode, n_threads=n_threads)
    full_nll = subject_sum_nll(res_I.x, data_I, mode, n_threads) + \
               subject_sum_nll(res_II.x, data_II, mode, n_threads)

    if res_I.x[0] >= res_II.x[0]:
        return 0.0, False, res_I.x[0], res_II.x[0]

    init_eq = [0.5 * (res_I.x[0] + res_II.x[0]),
               res_I.x[1], res_I.x[2], res_I.x[3],
               res_II.x[1], res_II.x[2], res_II.x[3]]
    res_eq = fit_equal_eta(data_I, data_II, mode, initial_guess=init_eq,
                           n_threads=n_threads)
    Lambda_minus = max(2.0 * (res_eq.fun - full_nll), 0.0)
    return Lambda_minus, Lambda_minus > critical_value, res_I.x[0], res_II.x[0]
