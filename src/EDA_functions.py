import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, fisher_exact, skew, mode

def cross_dependencies(df, cols, target, method="many_to_one"):
    """
    Compares one or more variables to a target variable using contingency tables.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.

    cols : str or list of str
        One or more column names to compare against the target(s).

    target : str or list of str
        One or more target column names to be compared.

    method : {'many_to_one', 'one_to_one', 'many_to_many'}, default='many_to_one'
        Defines the comparison strategy:
        - 'many_to_one': Each column in `cols` is compared against a single target column.
        - 'one_to_one': A single column is compared against a single target column.
        - 'many_to_many': Each column in `cols` is compared with each column in `target`.

    Returns
    -------
    statistic : float
        The test statistic from the chosen statistical test (e.g., chi-squared).

    p_value : float
        The p-value of the test.

    cramer_v : float
        The Cramér’s V statistic, measuring the strength of association.

    contingency_tables : DataFrame
        Contingency tables for each comparison.
    """
    methods=["many_to_one", "one_to_one", "many_to_many"]
    assert method in methods, "unknown method, please select one of the following: many_to_one, one_to_one, many_to_many"
    results=[]
    if method=="many_to_one":
        for i in cols:
            t_contingency=pd.crosstab(df[i], target, margins=True, margins_name="Totals")
            if len(t_contingency)-1==2 and t_contingency.shape[1]-1==2:
                t_con_reduced=pd.crosstab(df[i], target)
                results.append(["Fisher p-value: ", fisher_exact(t_con_reduced)[0], fisher_exact(t_con_reduced)[1], t_contingency])
            else:
                results.append(["Chi2 p-value: ", chi2_contingency(t_contingency)[0], chi2_contingency(t_contingency)[1], t_contingency])
    elif method=="one_to_one":
        t_contingency=pd.crosstab(cols, target, margins=True, margins_name="Totals")
        if len(t_contingency)-1==2 and t_contingency.shape[1]-1==2:
            t_con_reduced=pd.crosstab(cols, target)
            results.append(["Fisher p-value: ", fisher_exact(t_con_reduced)[0], fisher_exact(t_con_reduced)[1], t_contingency])
        else:
            results.append(["Chi2 p-value: ", chi2_contingency(t_contingency)[0], chi2_contingency(t_contingency)[1], t_contingency])
    else:
        for a in cols:
            for b in target:
                t_contingency=pd.crosstab(df[a], df[b], margins=True, margins_name="Totals")
                if len(t_contingency)-1==2 and t_contingency.shape[1]-1==2:
                    t_con_reduced=pd.crosstab(df[a], df[b])
                    results.append(["Fisher p-value: ", fisher_exact(t_con_reduced)[0], fisher_exact(t_con_reduced)[1], t_contingency])
                else:
                    results.append(["Chi2 p-value: ", chi2_contingency(t_contingency)[0], chi2_contingency(t_contingency)[1], t_contingency])
    for j in results:
            k=min(t_contingency.shape[1]-2, len(t_contingency)-2)
            V_Cramer=np.sqrt(j[1]/(len(df)*k))
            if j[2]>=0.05:
                continue
            else:
                print(f"{j[3]}\n\nStatistic: {j[1]}\n{j[0]}{j[2]}\nV-Cramer={V_Cramer})\n-----")
"-------------------------------"

def unique_values(df):
    """
    Returns the unique values for each variable in the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.

    Returns
    -------
    dict
        A dictionary where keys are column names and values are the list of unique values for each variable.
    """
    values={}
    for serie in df:
        if serie in values.keys():
            continue
        else:
            values[serie]= df[serie].unique()
    return values
"-------------------------------"

def matrix(df, method):
    """
    Plots a heatmap representing the correlation matrix of the input DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.

    method : {'pearson', 'spearman', 'kendall'}
        The method to compute pairwise correlation.
        - 'pearson': Standard correlation coefficient.
        - 'spearman': Rank correlation.
        - 'kendall': Kendall’s Tau coefficient.

    Returns 
    -------
    ax
        Axes object with the heatmap
    """
    sns.heatmap(df.corr(method=method),
                annot=True,
                fmt=".2f")
"-------------------------------" 

def bootstrap(distrib, obs, boot_rep):
    """
    Computes a list of observables from bootstrapped resamples of the input distribution.

    Parameters
    ----------
    distrib : array-like
        The original data distribution to resample from.

    obs : callable
        A function or statistic to compute for each resample (e.g., np.mean, np.median).

    boot_rep : int
        Number of resampling iterations.

    Returns
    -------
    list
        A list of computed observables from each bootstrap resample.
    """
    results=[]
    for _ in range(boot_rep):
        new_distrib=np.random.choice(distrib, size=len(distrib), replace=True)
        obs_value=obs(new_distrib)
        results.append(obs_value)
    return results
"-------------------------------"

def plot_distributions(df, cols=None, n=2, plot_type="histplot"):
    """
    Plots the distribution of one or more variables using the selected plot type.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.

    cols : list of str
        The columns to visualize (ignored if `plot_type='pairplot'`).

    n : int, default=2
        Number of decimal places to round numerical values.

    plot_type : str, default='histplot'
        Type of plot to generate. Options include:
        - 'histplot'
        - 'boxplot'
        - 'violinplot'
        - 'pairplot'

    Returns
    -------
    matplotlib.figure.Figure or seaborn.axisgrid.PairGrid
        The plot object.
    """
    plot_types=["histplot", "pairplot", "boxplot"]
    assert plot_type in plot_types, "unknown plot type; please select one of the following: histplot, pairplot or boxplot"
    if plot_type=="histplot":
        for serie in cols:
            fig, ax1=plt.subplots(1, 1, figsize=(10, 5))
            ax1=sns.histplot(df[serie].dropna(axis=0), ax=ax1)
            ax1.set_title(serie)
            print(f"\
                  Skewness {serie}: {np.round(skew(df[serie].dropna(axis=0)), n)}\n\
                  Mean {serie}: {np.round(np.mean(df[serie].dropna(axis=0)), n)}\n\
                  Median {serie}: {np.round(np.median(df[serie].dropna(axis=0)), n)}\n\
                  Mode {serie}: {np.round(mode(df[serie])[0], n)}")
            plt.show()
    elif plot_type=="pairplot":
        sns.pairplot(df)
    else:
        for serie in cols:
            fig, ax1=plt.subplots(1, 1, figsize=(10, 5))
            ax1=sns.boxplot(df[serie].dropna(axis=0), ax=ax1)
            ax1.set_title(serie)
            print(f"\
            Min: {np.round(np.min(df[serie].dropna(axis=0)), n)}\n\
            First Quantile: {np.round(np.quantile(df[serie].dropna(axis=0), 0.25), n)}\n\
            Median: {np.round(np.quantile(df[serie].dropna(axis=0), 0.5), n)}\n\
            Third Quantile: {np.round(np.quantile(df[serie].dropna(axis=0), 0.75), n)}\n\
            Max: {np.round(np.max(df[serie].dropna(axis=0)), n)}")
            plt.show()
"-------------------------------"


