import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import shapiro, bartlett, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def calculate_mean_std_ci(f2_list):
    mean_f2 = np.mean(f2_list)
    std_f2 = np.std(f2_list, ddof=1)
    n = len(f2_list)
    se_f2 = std_f2 / np.sqrt(n)
    ci_lower, ci_upper = stats.t.interval(0.95, df=n-1, loc=mean_f2, scale=se_f2)
    return mean_f2, std_f2, ci_lower, ci_upper

def format_results_to_table(KM_metrics_list, KM_f2_list, SVM_metrics_list, SVM_f2_list, RF_metrics_list, RF_f2_list):
    results = []
    f2_means_for_anova = []
    for metrics_list, f2_list, model_name in zip(
        [KM_metrics_list, SVM_metrics_list, RF_metrics_list],
        [KM_f2_list, SVM_f2_list, RF_f2_list], 
        ['K-Means', 'SVM', 'Random Forest']):
        
        mean_f2, std_f2, ci_lower, ci_upper = calculate_mean_std_ci(f2_list)
        f2_means_for_anova.append(f2_list)
        
        mean_metrics = np.mean(metrics_list, axis=0)  
        std_metrics = np.std(metrics_list, axis=0, ddof=1)

        row = {
            'Model': model_name,
            'F2 mean': mean_f2,
            'F2 std': std_f2,
            'F2 95% CI': f"{ci_lower:.2f} - {ci_upper:.2f}",
            'Accu mean': mean_metrics[0],
            'Accu std': std_metrics[0],
            'Precision mean': mean_metrics[1],
            'Precision std': std_metrics[1],
            'Recall mean': mean_metrics[2],
            'Recall std': std_metrics[2]
        }
        results.append(row)

    formatted_results = pd.DataFrame(results)

    F_statistic, p_value = stats.f_oneway(*f2_means_for_anova)
    print(f"ANOVA F-statistic: {F_statistic:.4f}, p-value: {p_value:.4g}")

    # Additional statistical tests and post-hoc analysis
    data = np.concatenate(f2_means_for_anova)
    groups = ['K-Means']*len(KM_f2_list) + ['SVM']*len(SVM_f2_list) + ['Random Forest']*len(RF_f2_list)

    # Normality Test
    print(f"Shapiro-Wilk Test for K-Means F2: {shapiro(KM_f2_list)[1]}")
    print(f"Shapiro-Wilk Test for SVM F2: {shapiro(SVM_f2_list)[1]}")
    print(f"Shapiro-Wilk Test for Random Forest F2: {shapiro(RF_f2_list)[1]}")

    # Homogeneity of Variances
    print(f"Bartlett's test: {bartlett(KM_f2_list, SVM_f2_list, RF_f2_list)[1]}")

    # Post-hoc Tukey HSD
    tukey_result = pairwise_tukeyhsd(endog=data, groups=groups, alpha=0.05)
    print(tukey_result)

    return formatted_results







