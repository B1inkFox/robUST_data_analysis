from __future__ import annotations

import numpy as np
import pandas as pd

from scipy.stats import f, wilcoxon, friedmanchisquare
import statsmodels.formula.api as smf


# ============================================================
# Basic extraction helpers
# ============================================================

def extract_metric_matrix(all_subject_summary, metric_name):
    """
    Convert nested summary structure into a subject x condition matrix.

    Parameters
    ----------
    all_subject_summary : list
        Shape conceptually: [subject][condition] -> summary dict
    metric_name : str
        Metric key, e.g. 'peak_error'

    Returns
    -------
    Y : ndarray, shape (n_subjects, n_conditions)
        Subject-level mean metric values.
    """
    n_subjects = len(all_subject_summary)
    n_conditions = len(all_subject_summary[0])

    Y = np.zeros((n_subjects, n_conditions), dtype=float)

    for s in range(n_subjects):
        for c in range(n_conditions):
            Y[s, c] = all_subject_summary[s][c][metric_name]["mean"]

    return Y


def print_condition_stats(Y, labels):
    """
    Print mean ± std across subjects for each condition.
    """
    print("\nCondition stats (mean ± std across subjects):")
    for i, label in enumerate(labels):
        m = np.mean(Y[:, i])
        s = np.std(Y[:, i], ddof=1)
        print(f"{label}: {m:.4f} ± {s:.4f}")


# ============================================================
# Repeated-measures ANOVA
# ============================================================

def repeated_measures_anova(Y):
    """
    One-way repeated-measures ANOVA for balanced subject x condition data.

    Parameters
    ----------
    Y : ndarray, shape (subjects, conditions)

    Returns
    -------
    dict with F, p, eta_sq, and decomposition terms.
    """
    Y = np.asarray(Y, dtype=float)
    S, C = Y.shape

    grand_mean = np.mean(Y)
    cond_means = np.mean(Y, axis=0)
    subj_means = np.mean(Y, axis=1)

    ss_total = np.sum((Y - grand_mean) ** 2)
    ss_cond = S * np.sum((cond_means - grand_mean) ** 2)
    ss_subj = C * np.sum((subj_means - grand_mean) ** 2)
    ss_error = ss_total - ss_cond - ss_subj

    df_cond = C - 1
    df_error = (S - 1) * (C - 1)

    ms_cond = ss_cond / df_cond
    ms_error = ss_error / df_error

    F_stat = ms_cond / ms_error
    p_value = 1.0 - f.cdf(F_stat, df_cond, df_error)

    eta_sq = ss_cond / (ss_cond + ss_error)

    return {
        "F": F_stat,
        "p": p_value,
        "eta_sq": eta_sq,
        "ss_cond": ss_cond,
        "ss_subj": ss_subj,
        "ss_error": ss_error,
        "df_cond": df_cond,
        "df_error": df_error,
    }


# ============================================================
# Friedman test
# ============================================================

def friedman_test(Y, labels=None):
    """
    Friedman test across repeated conditions.

    Parameters
    ----------
    Y : ndarray, shape (subjects, conditions)

    Returns
    -------
    dict with statistic and p-value
    """
    Y = np.asarray(Y, dtype=float)

    # scipy expects one array per condition
    cols = [Y[:, j] for j in range(Y.shape[1])]
    stat, p = friedmanchisquare(*cols)

    out = {
        "statistic": stat,
        "p": p,
    }

    if labels is not None:
        out["labels"] = labels

    return out


# ============================================================
# Pairwise Wilcoxon
# ============================================================

def pairwise_wilcoxon(Y, labels, zero_method="wilcox", correction=False):
    """
    Pairwise Wilcoxon signed-rank tests over conditions.

    Parameters
    ----------
    Y : ndarray, shape (subjects, conditions)
    labels : list[str]
    zero_method : str
        Passed to scipy.stats.wilcoxon
    correction : bool
        Passed to scipy.stats.wilcoxon

    Returns
    -------
    list[dict]
    """
    Y = np.asarray(Y, dtype=float)

    results = []
    pairs = [(0, 1), (0, 2), (1, 2)]

    for i, j in pairs:
        a = Y[:, i]
        b = Y[:, j]

        stat, p = wilcoxon(a, b, zero_method=zero_method, correction=correction)

        diff = a - b
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1) if len(diff) > 1 else 0.0

        # paired Cohen-like d on differences
        if std_diff > 0:
            effect_size_d = mean_diff / std_diff
        else:
            effect_size_d = np.nan

        results.append({
            "pair": f"{labels[i]} vs {labels[j]}",
            "i": i,
            "j": j,
            "statistic": stat,
            "p": p,
            "mean_diff": mean_diff,
            "std_diff": std_diff,
            "effect_size_d": effect_size_d,
        })

    return results


# ============================================================
# Event-level dataframe for mixed-effects model
# ============================================================

def build_event_level_dataframe(all_subject_results, metric_name, condition_labels):
    """
    Build long-format dataframe from event-level results.

    Parameters
    ----------
    all_subject_results : list
        Shape conceptually: [subject][condition] -> list[dict]
        Each innermost dict is one event result.
    metric_name : str
        Metric key to extract from each event dict.
    condition_labels : list[str]
        e.g. ["tran", "imp", "mpc"]

    Returns
    -------
    df : pandas.DataFrame
        Columns: subject, condition, event_idx, value
    """
    rows = []

    for subject_idx, subject_conditions in enumerate(all_subject_results):
        subject_label = f"S{subject_idx + 1}"

        for cond_idx, event_list in enumerate(subject_conditions):
            condition_label = condition_labels[cond_idx]

            for event_idx, event_dict in enumerate(event_list):
                if metric_name not in event_dict:
                    continue

                rows.append({
                    "subject": subject_label,
                    "condition": condition_label,
                    "event_idx": event_idx,
                    "value": float(event_dict[metric_name]),
                })

    df = pd.DataFrame(rows)
    return df


# ============================================================
# Linear mixed-effects model
# ============================================================

def fit_mixed_effects_model(df, baseline_label="tran"):
    """
    Fit linear mixed-effects model:
        value ~ C(condition, Treatment(reference=baseline_label))
    with random intercept by subject.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns: subject, condition, value
    baseline_label : str
        Baseline condition for treatment coding.

    Returns
    -------
    statsmodels fitted result
    """
    formula = f'value ~ C(condition, Treatment(reference="{baseline_label}"))'
    model = smf.mixedlm(formula, data=df, groups=df["subject"])
    result = model.fit()
    return result


def print_mixed_effects_summary(result):
    """
    Print a concise mixed-model summary.
    """
    print("\nLinear Mixed-Effects Model:")
    print(result.summary())


# ============================================================
# Main wrapper
# ============================================================

def analyze_metric(
    all_subject_summary,
    metric_name,
    trial_labels,
    all_subject_results=None,
    run_friedman=True,
    run_anova=True,
    run_wilcoxon=True,
    run_mixedlm=False,
    baseline_label="tran",
):
    """
    Run the requested statistics for one metric.

    Parameters
    ----------
    all_subject_summary : list
        [subject][condition] -> summary dict
    metric_name : str
    trial_labels : list[str]
    all_subject_results : list or None
        [subject][condition] -> list[event result dict]
        Needed for mixed-effects model.
    run_friedman, run_anova, run_wilcoxon, run_mixedlm : bool
    baseline_label : str
        Baseline level for mixed-effects treatment coding.

    Returns
    -------
    dict with all computed outputs
    """
    Y = extract_metric_matrix(all_subject_summary, metric_name)

    out = {
        "metric_name": metric_name,
        "Y": Y,
    }

    print(f"\n=== {metric_name} ===")
    print_condition_stats(Y, trial_labels)

    if run_anova:
        anova_res = repeated_measures_anova(Y)
        out["anova"] = anova_res
        print(
            f"\nANOVA: F = {anova_res['F']:.4f}, "
            f"p = {anova_res['p']:.4f}, "
            f"eta^2 = {anova_res['eta_sq']:.4f}"
        )

    if run_friedman:
        friedman_res = friedman_test(Y, labels=trial_labels)
        out["friedman"] = friedman_res
        print(
            f"Friedman: chi2 = {friedman_res['statistic']:.4f}, "
            f"p = {friedman_res['p']:.4f}"
        )

    if run_wilcoxon:
        wilcoxon_res = pairwise_wilcoxon(Y, trial_labels)
        out["wilcoxon"] = wilcoxon_res
        print("\nPairwise Wilcoxon:")
        for r in wilcoxon_res:
            print(
                f"{r['pair']}: "
                f"p = {r['p']:.4f}, "
                f"Δ = {r['mean_diff']:.4f}, "
                f"d = {r['effect_size_d']:.4f}"
            )

    if run_mixedlm:
        if all_subject_results is None:
            raise ValueError("all_subject_results must be provided when run_mixedlm=True")

        df = build_event_level_dataframe(all_subject_results, metric_name, trial_labels)
        out["mixedlm_df"] = df

        mixedlm_res = fit_mixed_effects_model(df, baseline_label=baseline_label)
        out["mixedlm"] = mixedlm_res

        print_mixed_effects_summary(mixedlm_res)

    return out