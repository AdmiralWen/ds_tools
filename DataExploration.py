"""
Data Exploration Module
Author: Brandon Wen

Contains useful functions for data manipulation and exploration. Requires Pandas, Numpy, Matplotlib, and Seaborn.

dataInfo: Returns a dataframe of dataset properties (Data Types, # Observations, Unique Observations, by column).
extremeObs: Returns the top n highest and lowest observations of a variable; optional boxplot.
checkUniqueBy: Checks if a dataframe is unique by a given list of fields.
freqTab: Returns a dataframe containing the frequency tabulation of a categorical variable.
summaryTab: Returns a dataframe containing the summary tabulation of a categorical variable (by a summation variable).
describeBy: Adds "Non-NaN Count" and "Sum" to df.groupby().describe().
boolCounts: Returns a True/False tabulation for a given list of boolean variables (var_list).
naPerColumn: Returns a dataframe of of the NA's and percent-NA's for the variables in a dataframe.
visualizeNA: Returns a barchart of the dataframe generated by naPerColumn().
splitColumn: Splits the input variable into multiple columns according to a delimiter.
wtdQuantile: Returns a series for the quantile or weighted-quantile for a variable; can also be unweighted if desired.
corrHeatmap: Returns a correlation matrix heatmap for the variables in a dataframe.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def dataInfo(dataframe):
    '''
        Returns a list of the variables in the input dataframe, along with their types, observation counts,
        and NaN counts/percentages.
    '''
    dtypes = pd.Series(dataframe.dtypes)
    obs = dataframe.count()
    uniques = [len(dataframe[i].unique()) for i in dataframe.columns]

    na_count = dataframe.isnull().sum()
    na_pct = na_count/len(dataframe)

    info_df = pd.DataFrame({'Type':dtypes, 'Observations':obs, 'Unique Observations':uniques,
                            "NaN Count":na_count, "NaN Percent":na_pct})
    info_df['NaN Percent'] = info_df['NaN Percent'].map('{:,.2%}'.format)
    return info_df[['Type', 'Observations', 'Unique Observations', 'NaN Count', 'NaN Percent']]

def extremeObs(dataframe, variable, n = 10, boxplot = True, whis = 1.5):
    '''
        Displays the n (default n = 10) largest and smallest observations for a variable in a dataframe. If the
        boxplot argument is True, will also plot a boxplot for this variable. Use the whis argument to control the
        length of the whiskers (default is 1.5 IQR).
    '''
    largest = dataframe[variable].nlargest(n).reset_index()
    smallest = dataframe[variable].nsmallest(n).reset_index()

    outlier_df = pd.DataFrame({str(n) + '_Largest':largest[largest.columns[-1]],
                               'Index_Largest':largest['index'],
                               str(n) + '_Smallest':smallest[smallest.columns[-1]],
                               'Index_Smallest':smallest['index']
                              })
    print(outlier_df[[str(n) + '_Largest', 'Index_Largest', str(n) + '_Smallest', 'Index_Smallest']])

    if boxplot:
        sns.boxplot(dataframe[variable], whis = whis)
        plt.show()

def checkUniqueBy(dataframe, variables):
    '''
        Checks if a dataframe is unique by a given list of fields. The variables argument can be either a single column
        name (string) or a list. Keep in mind that "unique" doesn't mean the smallest possible number of fields that the
        dataframe will be unique by.
    '''
    l = len(dataframe[variables].drop_duplicates())
    return l == len(dataframe)

def freqTab(dataframe, variable, drop_na = False, sort_by_count = True):
    '''
        Returns the frequency tabulation of the input variable as a Pandas dataframe. Specify drop_na = True to drop
        NaNs from the tabulation (default is False), and specify sort_by_count = False to sort the result alphabetically
        instead of by the frequency counts (default is True).
    '''
    cnt = dataframe[variable].value_counts(sort = sort_by_count, dropna = drop_na)
    if not sort_by_count:
        cnt.sort_index(inplace = True)
    cnt_cumul = cnt.cumsum()
    pct = cnt/cnt.sum()
    pct_cumul = cnt_cumul/cnt_cumul.max()

    freq_df = pd.DataFrame({"Count":cnt,
                            "Percent":pct,
                            "Cumul. Count":cnt_cumul,
                            "Cumul. Percent":pct_cumul
                           })
    freq_df['Percent'] = freq_df['Percent'].map('{:,.2%}'.format)
    freq_df['Cumul. Percent'] = freq_df['Cumul. Percent'].map('{:,.2%}'.format)

    return freq_df[['Count', 'Percent', 'Cumul. Count', 'Cumul. Percent']]

def summaryTab(dataframe, groupby_var, sum_var, sort_by_sum = True):
    '''
        Returns the summary tabulation of the input variable as a Pandas dataframe. Be sure to enter the groupby_var
        and sum_var as strings; function can only support one group_by and sum variable. To sort by the grouping
        variable instead of the summary variable, specify sort_by_sum = False.
    '''
    sums_temp = dataframe.groupby(groupby_var, as_index = False)[sum_var].sum()
    if sort_by_sum:
        sums_temp.sort_values(sum_var, ascending = False, inplace = True)
    sums = sums_temp[sums_temp.columns[-1]]
    sums_cumul = sums.cumsum()
    pct = sums/sums.sum()
    pct_cumul = sums_cumul/sums_cumul.max()

    summary_df = pd.DataFrame({"Sum":sums,
                               "Percent":pct,
                               "Cumul. Sum":sums_cumul,
                               "Cumul. Percent":pct_cumul
                              })
    sums_temp[sums_temp.columns[0]].name = None
    summary_df.index = sums_temp[sums_temp.columns[0]]

    summary_df['Percent'] = summary_df['Percent'].map('{:,.2%}'.format)
    summary_df['Cumul. Percent'] = summary_df['Cumul. Percent'].map('{:,.2%}'.format)
    summary_df['Sum'] = summary_df['Sum'].map('{:,}'.format)
    summary_df['Cumul. Sum'] = summary_df['Cumul. Sum'].map('{:,}'.format)

    return summary_df[['Sum', 'Percent', 'Cumul. Sum', 'Cumul. Percent']]

def describeBy(dataframe, groupby_var, numeric_var):
    ''' Adds "Non-NaN Count" and "Sum" to df.groupby().describe(). '''
    by_cnt = dataframe.groupby(groupby_var).agg({groupby_var:'count'})
    by_cnt_non_null = dataframe.groupby(groupby_var).agg({numeric_var:'count'})
    by_sum = dataframe.groupby(groupby_var).agg({numeric_var:'sum'})
    by_desc = dataframe.groupby(groupby_var)[numeric_var].describe().drop('count', axis = 1)

    desc_stats = pd.concat([by_cnt, by_cnt_non_null, by_sum, by_desc], axis = 1)
    del desc_stats.index.name
    desc_stats.columns = ['Total Count', 'Non-NaN Count', 'Sum', 'Mean', 'Std. Dev.', 'Min', '25th Pctl',
                          'Median', '75th Pctl', 'Max']

    return desc_stats

def boolCounts(dataframe, var_list):
    ''' Returns a True/False tabulation for a given list of boolean variables (var_list). '''
    temp_counts = list()
    for var in var_list:
        assert dataframe[var].dtype == 'bool', 'All variables must be boolean.'
        temp_counts.append(dataframe[var].value_counts())

    temp_all = pd.DataFrame(temp_counts)
    temp_all.fillna(0, inplace = True)
    temp_all['Total'] = temp_all[True] + temp_all[False]
    temp_all['%-True'] = (temp_all[True]/temp_all['Total']).map('{:,.2%}'.format)
    temp_all['%-False'] = (temp_all[False]/temp_all['Total']).map('{:,.2%}'.format)

    return temp_all[[True, False, 'Total', '%-True', '%-False']]

def naPerColumn(dataframe):
    ''' Returns a tabulation of the NaNs for each column in the input dataframe. '''
    na_count = dataframe.isnull().sum()
    na_pct = na_count/len(dataframe)

    na_df = pd.DataFrame({"Count":na_count, "Percent":na_pct})
    na_df['Percent'] = na_df['Percent'].map('{:,.2%}'.format)

    return na_df

def visualizeNA(dataframe, sns_font_scale = 2, fig_size = (24, 16)):
    ''' Bargraph of the NaNs in the input dataframe. Use sns_font_scale and fig_size to set the font and graph sizes. '''
    sns.set(font_scale = sns_font_scale)
    df_na = (dataframe.isnull().sum()/len(dataframe))*100
    df_na = df_na.drop(df_na[df_na == 0].index).sort_values(ascending = False)

    f, ax = plt.subplots(figsize = fig_size)
    plt.xticks(rotation = '90')
    sns.barplot(x = df_na.index, y = df_na)
    ax.set(title = 'Percent NaN by Feature', ylabel = 'Percent Missing')
    plt.subplots_adjust(top = 0.95, bottom = 0.3)
    plt.show()

def splitColumn(dataframe, variable, delimiter, exp_cols_prefix, merge_orig = True, drop_orig = False):
    '''
        Splits a variable of a dataframe into multiple columns. You can specify the delimiter (which must be a string) using
        the delimiter argument, and the exp_cols_prefix argument (also a string) is used to prefix the split column names.
        The merge_orig and drop_orig arguments are used to control whether to merge back to the original dataframe and whether
        to drop the original column in the output.
    '''
    df_exp_temp = dataframe[variable].str.split(delimiter, expand = True)
    max_cols = len(df_exp_temp.columns)
    df_exp_temp.columns = [exp_cols_prefix + str(i) for i in range(max_cols)]
    df_exp_temp.fillna(value = np.nan, inplace = True)

    if merge_orig:
        df_mrg_temp = pd.concat((dataframe, df_exp_temp), axis = 1)
        if drop_orig:
            df_mrg_temp.drop(variable, axis = 1, inplace = True)
        return df_mrg_temp
    else:
        return df_exp_temp

def wtdQuantile(dataframe, var, weight = None, n = 10):
    '''
        Returns a Pandas Series for the quantile or weighted-quantile for a variable. The var argument is your variable
        of interest, weight is your weight variable, and n is the number of quantiles you desire.
    '''
    if weight == None:
        return pd.qcut(dataframe[var], n, labels = False)
    else:
        dataframe.sort_values(var, ascending = True, inplace = True)
        cum_sum = dataframe[weight].cumsum()
        cutoff = float(cum_sum[-1:])/n
        quantile = cum_sum/cutoff
        quantile[-1:] = n-1
        return quantile.map(int)

def corrHeatmap(dataframe, vars_list, sns_font_scale = 2, fig_size = (16, 16)):
    '''
        Creates a heatmap of the correlation matrix using the input dataframe and the variables list. The var_list
        argument should be a list of variable names (strings) that you wish to compute correlations for. Use
        sns_font_scale and fig_size to set the font and graph sizes.
    '''
    sns.set(font_scale = sns_font_scale)
    df_corrmat = dataframe[vars_list].corr()

    f, ax = plt.subplots(figsize = fig_size)
    sns.heatmap(df_corrmat, square=True, linewidths=0.5, annot=True)
    plt.xticks(rotation='90')
    plt.yticks(rotation='0')
    ax.set(title = 'Correlation Matrix of Selected Variables')
    plt.show()