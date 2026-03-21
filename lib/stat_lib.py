from scipy.stats import f, wilcoxon
import numpy as np

def extract_metric_matrix(all_subject_summary, metric_name):
    n_subjects = len(all_subject_summary)
    n_conditions = len(all_subject_summary[0])

    M = np.zeros((n_subjects, n_conditions))

    for s in range(n_subjects):
        for c in range(n_conditions):
            M[s, c] = all_subject_summary[s][c][metric_name]["mean"]

    return M

def repeated_measures_anova(Y):
    """
    Y shape: (subjects, conditions)
    """

    S, C = Y.shape

    grand_mean = np.mean(Y)
    cond_means = np.mean(Y, axis=0)
    subj_means = np.mean(Y, axis=1)

    SS_total = np.sum((Y - grand_mean)**2)
    SS_cond = S * np.sum((cond_means - grand_mean)**2)
    SS_subj = C * np.sum((subj_means - grand_mean)**2)
    SS_error = SS_total - SS_cond - SS_subj

    df_cond = C - 1
    df_error = (S - 1) * (C - 1)

    MS_cond = SS_cond / df_cond
    MS_error = SS_error / df_error

    F_stat = MS_cond / MS_error
    p_value = 1 - f.cdf(F_stat, df_cond, df_error)

    eta_sq = SS_cond / (SS_cond + SS_error)

    return F_stat, p_value, eta_sq

def pairwise_tests(Y, labels):
    """
    Y: (subjects, conditions)
    """

    results = []
    pairs = [(0,1), (0,2), (1,2)]

    for i, j in pairs:
        a = Y[:, i]
        b = Y[:, j]

        stat, p = wilcoxon(a, b)

        diff = a - b
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)

        results.append({
            "pair": f"{labels[i]} vs {labels[j]}",
            "p": p,
            "mean_diff": mean_diff,
            "std_diff": std_diff
        })

    return results

def print_condition_stats(Y, labels):
    print("\nCondition stats (mean ± std across subjects):")
    for i, label in enumerate(labels):
        m = np.mean(Y[:, i])
        s = np.std(Y[:, i], ddof=1)
        print(f"{label}: {m:.4f} ± {s:.4f}")

def analyze_metric(all_subject_summary, metric_name, trial_labels):

    Y = extract_metric_matrix(all_subject_summary, metric_name)

    print(f"\n=== {metric_name} ===")

    # mean ± std
    print_condition_stats(Y, trial_labels)

    # ANOVA
    F_stat, p_value, eta_sq = repeated_measures_anova(Y)
    print(f"\nANOVA: F = {F_stat:.4f}, p = {p_value:.4f}, eta^2 = {eta_sq:.4f}")

    # pairwise
    pairwise = pairwise_tests(Y, trial_labels)
    print("\nPairwise Wilcoxon:")
    for r in pairwise:
        print(f"{r['pair']}: p = {r['p']:.4f}, Δ = {r['mean_diff']:.4f}")