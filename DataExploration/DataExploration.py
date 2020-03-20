"""
Data Exploration Tools
Author: Brandon Wen

Contains useful functions for data manipulation and exploration. Requires Pandas, Numpy, Matplotlib, and Seaborn.

data_info: Returns a dataframe of dataset properties (Data Types, # Observations, Unique Observations, by column).
extreme_obs: Returns the top n highest and lowest observations of a variable; optional boxplot.
check_unique_by: Checks if a dataframe is unique by a given list of columns.
non_unique_items: Returns the non-unique items and their non-unique count for the values in the given list of columns.
freq_tab: Returns a dataframe containing the frequency tabulation of a categorical variable.
summary_tab: Returns a dataframe containing the summary tabulation of a categorical variable (by a summation variable).
describe_by: Adds "Non-NaN Count" and "Sum" to df.groupby().describe().
boolean_counts: Returns a True/False tabulation for a given list of boolean variables (var_list).
na_per_column: Returns a dataframe of of the NA's and percent-NA's for the variables in a dataframe.
visualize_na: Returns a barchart of the dataframe generated by naPerColumn().
split_column: Splits the input variable into multiple columns according to a delimiter.
weighted_quantile: Returns a series for the quantile or weighted-quantile for a variable; can also be unweighted if desired.
correlation_heatmap: Returns a correlation matrix heatmap for the variables in a dataframe.
custom_boxplot: Creates a custom boxplot based on a dataframe of pre-calculated precentile metrics.
map_multi_level_dict: Converts a series to a multicolumn dummies dataframe based on a mapping dict where the values are a list.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

def data_info(dataframe):
    '''
    Returns a list of the variables in the input dataframe, along with their types, observation counts,
    and NaN counts/percentages.

    Parameters:
    -----------------
    dataframe: a pandas dataframe

    Returns:
    -----------------
    pandas.core.frame.DataFrame
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

def extreme_obs(dataframe, variable, n = 10, boxplot = True, whis = 1.5):
    '''
    Displays the n (default n = 10) largest and smallest observations for a variable in a dataframe. If the
    boxplot argument is True, will also plot a boxplot for this variable. Use the whis argument to control the
    length of the whiskers (default is 1.5 IQR).

    Parameters:
    -----------------
    dataframe: a pandas dataframe
        Your starting dataframe.
    variable: string
        The variable you wish to display extremem observations for.
    n: integer
        Display the top n largest and smallest values, default 10.
    boxplot: boolean
        Specify whether to return a boxplot alongside the top n values list.

    Returns:
    -----------------
    None; function directly prints and/or plots output.
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

def check_unique_by(dataframe, variables):
    '''
    Checks if a dataframe is unique by a given list of columns. The variables argument can be either a single column
    name (string) or a list. Keep in mind that "unique" doesn't mean the smallest possible number of fields that the
    dataframe will be unique by.

    Parameters:
    -----------------
    dataframe: a pandas dataframe
        Your starting dataframe.
    variables: list
        A list of variable names (strings) you wish to check uniqueness for.

    Returns:
    -----------------
    bool
    '''

    l = len(dataframe[variables].drop_duplicates())
    return l == len(dataframe)

def non_Unique_items(dataframe, variables):
    '''
    Groups the dataframe by the input variables and returns only the non-unique values of those variables, in decending order.

    Parameters:
    -----------------
    dataframe: a pandas dataframe
        Your starting dataframe.
    variables: list
        A list of variable names (strings) you wish to check non-uniqueness for.

    Returns:
    -----------------
    pandas.core.frame.DataFrame
    '''

    tmp = dataframe.groupby(variables).size().reset_index()
    tmp.rename(columns = {0:'nonUniqueItems_Count'}, inplace = True)
    tmp = tmp[tmp['nonUniqueItems_Count'] > 1].sort_values('nonUniqueItems_Count', ascending = False).reset_index()
    return tmp.drop('index', axis = 1)


def freq_tab(dataframe, variable, drop_na = False, sort_by_count = True, plot = None, fig_size = (16, 8)):
    '''
    Returns the frequency tabulation of the input variable as a Pandas dataframe. Specify drop_na = True to drop
    NaNs from the tabulation (default is False), and specify sort_by_count = False to sort the result alphabetically
    instead of by the frequency counts (default is True). Use the plot argument to specify the output type: frequency
    table, graph by count, or graph by percent (None, 'count', and 'percent' respectively).

    Parameters:
    -----------------
    dataframe: a pandas dataframe
        Your starting dataframe.
    variable: string
        The variable name you wish to compute the frequency tabulation by.
    drop_na: boolean
        Specify whether to drop NaN values from output, default False.
    sort_by_count: boolean
        Specify whether to sort by the frequency counts (True) or the frequency tabulation index (False), default True.
    plot: None or string
        Specify whether to return the frequency tabulation in barplot format rather than table format. If not plotting,
        set this to None. If plotting, specify 'count' or 'percent'. Default None.

    Returns:
    -----------------
    plot = None
        pandas.core.frame.DataFrame
    plot = 'count' or plot = 'percent'
        matplotlib.axes._subplots.AxesSubplot
    '''

    assert plot is None or plot.lower() in ('count', 'percent'), "Plot must be None, 'count', or 'percent'."

    cnt = dataframe[variable].value_counts(sort = sort_by_count, dropna = drop_na)
    if not sort_by_count:
        cnt.sort_index(inplace = True)
    cnt_cumul = cnt.cumsum()
    pct = cnt/cnt.sum()
    pct_cumul = cnt_cumul/cnt_cumul.max()

    freq_df = pd.DataFrame({"Count":cnt,
                            "Percent":pct,
                            "Cumul_Count":cnt_cumul,
                            "Cumul_Percent":pct_cumul
                           })
    freq_df['Percent'] = freq_df['Percent'].map('{:,.2%}'.format)
    freq_df['Cumul_Percent'] = freq_df['Cumul_Percent'].map('{:,.2%}'.format)

    # Returns table (plot = None):
    if not plot:
        return freq_df[['Count', 'Percent', 'Cumul_Count', 'Cumul_Percent']]
    # Returns barplot (plot = 'count' or 'percent'):
    else:
        if plot.lower() == 'count':
            plot2 = 'Count'
        elif plot.lower() == 'percent':
            plot2 = 'Percent'
            freq_df['Percent'] = freq_df['Percent'].map(lambda x: float(x[:-1])/100)
        plt.figure(figsize = fig_size)
        ax = sns.barplot(x = freq_df.index, y = plot2, data = freq_df)
        plt.xticks(rotation = 'vertical', fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.ylabel(plot2)
        if plot.lower() == 'percent':
            yvals = ax.get_yticks()
            ax.set_yticklabels(['{:,.2%}'.format(x) for x in yvals])
        plt.show()

def summary_tab(dataframe, groupby_var, sum_var, sort_by_sum = True):
    '''
    Similar to freq_tab(); returns the summary tabulation of the input variable as a Pandas dataframe. Be sure to
    enter the groupby_var and sum_var as strings; function can only support one group_by and sum variable. To sort
    by the grouping variable instead of the summary variable, specify sort_by_sum = False.

    Parameters:
    -----------------
    dataframe: a pandas dataframe
        Your starting dataframe.
    groupby_var: string
        The name of the categorical variable you wish to group-by.
    sum_var: string
        The name of the numeric variable you wish to aggregate.
    sort_by_sum: boolean
        Specify whether to sort by the summary values (True) or the group-by index (False), default True.

    Returns:
    -----------------
    pandas.core.frame.DataFrame
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

def describe_by(dataframe, groupby_var, numeric_var):
    '''
    Adds "Non-NaN Count" and "Sum" to df.groupby().describe().

    Parameters:
    -----------------
    dataframe: a pandas dataframe
        Your starting dataframe.
    groupby_var: string
        The name of the categorical variable you wish to group-by.
    sum_var: string
        The name of the numeric variable you wish to aggregate.

    Returns:
    -----------------
    pandas.core.frame.DataFrame
    '''

    by_cnt = dataframe.groupby(groupby_var).agg({groupby_var:'count'})
    by_cnt_non_null = dataframe.groupby(groupby_var).agg({numeric_var:'count'})
    by_sum = dataframe.groupby(groupby_var).agg({numeric_var:'sum'})
    by_desc = dataframe.groupby(groupby_var)[numeric_var].describe().drop('count', axis = 1)

    desc_stats = pd.concat([by_cnt, by_cnt_non_null, by_sum, by_desc], axis = 1)
    del desc_stats.index.name
    desc_stats.columns = ['Total Count', 'Non-NaN Count', 'Sum', 'Mean', 'Std. Dev.', 'Min', '25th Pctl',
                          'Median', '75th Pctl', 'Max']

    return desc_stats

def boolean_counts(dataframe, var_list):
    '''
    Returns a True/False tabulation for a given list of boolean variables (var_list).

    Parameters:
    -----------------
    dataframe: a pandas dataframe
        Your starting dataframe.
    var_list: list
        A list of boolean variable names (strings) you wish to tabulate.

    Returns:
    -----------------
    pandas.core.frame.DataFrame
    '''

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

def na_per_column(dataframe):
    '''
    Returns a tabulation of the NaNs for each column in the input dataframe.

    Parameters:
    -----------------
    dataframe: a pandas dataframe

    Returns:
    -----------------
    pandas.core.frame.DataFrame
    '''

    na_count = dataframe.isnull().sum()
    na_pct = na_count/len(dataframe)

    na_df = pd.DataFrame({"Count":na_count, "Percent":na_pct})
    na_df['Percent'] = na_df['Percent'].map('{:,.2%}'.format)

    return na_df

def visualize_na(dataframe, sns_font_scale = 2, fig_size = (24, 16)):
    '''
    Bargraph of the NaNs in the input dataframe. Use sns_font_scale and fig_size to set the font and graph sizes.

    Parameters:
    -----------------
    dataframe: a pandas dataframe
    sns_font_scale: integer, used to control Seaborn font size. Default 2.
    fig_size: tuple (x, y), used to control the figure size.

    Returns:
    -----------------
    None; function plots output directly.
    '''

    sns.set(font_scale = sns_font_scale)
    df_na = (dataframe.isnull().sum()/len(dataframe))*100
    df_na = df_na.drop(df_na[df_na == 0].index).sort_values(ascending = False)

    f, ax = plt.subplots(figsize = fig_size)
    plt.xticks(rotation = '90')
    sns.barplot(x = df_na.index, y = df_na)
    ax.set(title = 'Percent NaN by Feature', ylabel = 'Percent Missing')
    plt.subplots_adjust(top = 0.95, bottom = 0.3)
    plt.show()

def split_column(dataframe, variable, delimiter, exp_cols_prefix, merge_orig = True, drop_orig = False):
    '''
    Splits a variable of a dataframe into multiple columns. You can specify the delimiter (which must be a string) using
    the delimiter argument, and the exp_cols_prefix argument (also a string) is used to prefix the split column names.
    The merge_orig and drop_orig arguments are used to control whether to merge back to the original dataframe and whether
    to drop the original column in the output.

    Parameters:
    -----------------
    dataframe: a pandas dataframe
        Your starting dataframe.
    variable: string
        The name of the variable (string) you wish to split.
    delimiter: string
        The delimiter for the variable you wish to split.
    exp_cols_prefix: string
        The prefix for the expanded column names.
    merge_orig: boolean
        Specify whether to merge the expanded columns back to the original dataframe, default True.
    drop_orig: boolean
        Specify whether to drop the original column after mergin the expanded columns, default False.

    Returns:
    -----------------
    pandas.core.frame.DataFrame
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

def weighted_quantile(dataframe, var, weight = None, n = 10):
    '''
    Returns a Pandas Series for the quantile or weighted-quantile for a variable. The var argument is your variable
    of interest, weight is your weight variable, and n is the number of quantiles you desire.

    Parameters:
    -----------------
    dataframe: a pandas dataframe
        Your starting dataframe.
    var: string
        The name of the numeric variable (string) you're looking to compute quantiles for.
    weight: None or string
        The name of the weight variable, if applicable. Specify weight = None for unweighted quantiles. Default None.
    n: integer
        The number of quantiles to compute, default 10.

    Returns:
    -----------------
    pandas.core.series.Series
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

def correlation_heatmap(dataframe, vars_list, method = 'pearson', sns_font_scale = 2, fig_size = (16, 16)):
    '''
    Creates a heatmap of the correlation matrix using the input dataframe and the variables list. The var_list
    argument should be a list of variable names (strings) that you wish to compute correlations for. Use
    sns_font_scale and fig_size to set the font and graph sizes.

    Parameters:
    -----------------
    dataframe: a pandas dataframe
        Your starting dataframe.
    vars_list: list
        A list of column names (strings) to use as the input for your correlation matrix. All columns here should be numeric.
    method: string
        Specify the type of correlation - pearson, spearman, or kendall. Default 'pearson'.
    sns_font_scale: integer
        Used to control Seaborn font size. Default 2.
    fig_size: tuple (x, y)
        Used to control the figure size.

    Returns:
    -----------------
    None; function plots output directly.
    '''

    sns.set(font_scale = sns_font_scale)
    df_corrmat = dataframe[vars_list].corr(method = method)

    f, ax = plt.subplots(figsize = fig_size)
    sns.heatmap(df_corrmat, square=True, linewidths=0.5, annot=True)
    plt.xticks(rotation='90')
    plt.yticks(rotation='0')
    ax.set(title = 'Correlation Matrix of Selected Variables')
    plt.show()

def custom_boxplot(dataframe, category_field, plot_fields, n_obs_field = None, title = None, figsize = (12, 8),
                   box_color = 'lightsteelblue', bg_color = 'whitesmoke', n_obs_size = 12, n_obs_display_offset = 1,
                   x_label_size = 12, y_label_size = 12, x_axis_title = '', x_title_size = 12):
    '''
    Creates a custom boxplot based on pre-calculated precentile metrics. Function requires the percentile metrics to be in a dataframe. Each
    row of the dataframe should contain the name of the category we have metrics for, as well as 5 additional columns of percentile metrics in
    increasing order. For example:
        Category  p10  p25  p50  p75  p90
        Monday    1.0  4.0  7.1  8.4  10.0
        Tuesday  -0.2  0.9  2.4  5.0  7.5
        ... ... ...
    The function will create a horizontal boxplot by category (defined by the category_field argument), where the values of the boxplot - left
    whisker, left box, median, right box, right whisker - are given by the subsequent 5 fields (defined by the plot_fields argument). Use the
    n_obs_field to optionally add a display for the number of observations per box, which should also be a field in the input dataframe. You can
    also optionally adjust the title and color arguments.

    Required Parameters:
    --------------------
    dataframe: a pandas dataframe
        Your starting dataframe.
    category_field: string
        The name of the categorical variable (string) you're looking to plot distributions for.
    plot_fields: list
        A list of strings which should be the column names of the percentile metrics computed.
    n_obs_field: None or string
        The name of the column representing the number of observations per level in the category_field variable. If not applicable,
        set this to None. Default None.

    Returns:
    --------------------
    None; function plots output directly.
    '''

    assert len(plot_fields) == 5, "plot_fields must be a list of length 5."
    fig, ax = plt.subplots(figsize = figsize)

    # Turn input dataframe into a list-of-lists representation:
    data_list = dataframe[plot_fields].values.tolist()

    # Create boxplot object:
    box_plot = ax.boxplot(data_list, whis = 'range', vert = False, patch_artist = True, medianprops = {'linewidth':2, 'color':'darkred'})
    for patch in box_plot['boxes']:
        patch.set_facecolor(box_color)

    # Y-Axis adjustments:
    ytickNames = plt.setp(ax, yticklabels = list(dataframe[category_field]))
    plt.setp(ytickNames, fontsize = y_label_size)
    plt.gca().invert_yaxis()

    # X-Axis adjustments:
    plt.xticks(fontsize = x_label_size)
    plt.xlabel(x_axis_title, size = x_title_size)

    # Chart background grid and color:
    ax.xaxis.grid(True)
    ax.set_yticks([y+1 for y in range(len(data_list))], )
    ax.set_facecolor(bg_color)

    # Chart title:
    if title:
        ax.set_title(title, fontsize = 16)

    # Display n_obs for each box:
    if n_obs_field:
        n_obs = ['n = ' + str(int(i)) for i in dataframe[n_obs_field].values]
        medians = dataframe[plot_fields[-1]].values
        for box in range(len(n_obs)):
            ax.text(medians[box] + n_obs_display_offset, box + 1, n_obs[box], verticalalignment = 'center', size = n_obs_size, color = 'black', weight = 'semibold')

        plt.xlim(ax.get_xlim()[0], ax.get_xlim()[1]*1.1)

    plt.show()
