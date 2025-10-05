"""
Modeling and Evaluation Visualizations
Author: Brandon Wen
Copyright (c) 2018 Brandon Wen

Contains useful functions for plotting and visualizing modeling results.

plot_confusion_matrix: Plots the confusion matrix heatmap given the true and predicted labels.
gini: Computes the raw or normalized Gini coefficient given the true and predicted labels.
gini_plot: Plots the Gini Lorenz curve given the true and predicted labels.
single_lift_plot: Plots a single-lift chart for a given set of actual, predicted, and weight values.
rf_feature_importance: Plots the feature importance graph for random forest models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from sklearn.metrics import confusion_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .exploration import weighted_quantile

###########
# General #

def plot_confusion_matrix(ax, true_labels, pred_labels, normalize = False, cmap = plt.cm.Blues, show_colorbar = False,
                          title = None, title_size = 18, axis_label_size = 16, data_label_size = 15, data_label_decimals = 2,
                          x_tick_size = 14, x_tick_rotation = 0, x_tick_horiz_align = 'center', y_tick_size = 14):
    '''
    Plots the confusion matrix heatmap given the true and predicted labels, which can be either Pandas series or Numpy arrays.
    Use the normalize argument to display a normalized confusion matrix. Requires a Matplotlib axis object to be passed to the
    ax argument. The remaining arguments are used to control aesthetics.

    Required Parameters:
    --------------------
    ax: a matplotlib axes object
        Usually defined by plt.subplots().
    true_labels: numpy array or pandas series
        A series representing the true class lables.
    pred_labels: numpy array or pandas series
        A series representing the predicted class labels.
    normalize: boolean
        Used to specify whether or not to display the normalized confusion matrix (percentages) rather than the regular
        confusion matrix (counts). Default False.

    Returns:
    --------------------
    None; wrap function into a plt.subplot() block to create plots. See Example.

    Example:
    --------------------
    >> import random
    >> t = [random.randrange(0, 2, 1) for i in range(50)]
    >> p = [random.randrange(0, 2, 1) for i in range(50)]

    >> fig, ax = plt.subplots(1, 2, figsize = (10, 8))
    >> plot_confusion_matrix(ax[0], t, p, normalize = False, title = 'Test Plot 1', cmap = plt.cm.Blues)
    >> plot_confusion_matrix(ax[1], t, p, normalize = True, title = 'Test Plot 2', cmap = plt.cm.Blues)
    >> plt.show()
    '''

    cm = confusion_matrix(true_labels, pred_labels)
    classes = sorted(list(set(true_labels).union(set(pred_labels))))

    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]

    if not title:
        title = 'Normalized Confusion Matrix' if normalize else 'Confusion Matrix'

    # Plot color-coded matrix; use the cmap argument to control the color map:
    im = ax.imshow(cm, interpolation = 'nearest', cmap = cmap)
    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size = '5%', pad = 0.1)
        ax.colorbar(im, cax = cax, orientation = 'vertical')

    # Title and X/Y labels:
    ax.set_title(title, size = title_size)
    ax.set_ylabel('True Labels', size = axis_label_size)
    ax.set_xlabel('Predicted Labels', size = axis_label_size)

    # X and Y ticks:
    tick_marks = np.arange(len(classes))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(classes, rotation = x_tick_rotation, ha = x_tick_horiz_align, size = x_tick_size)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(classes, rotation = 0, va = 'center', size = y_tick_size)

    # Add numerical overlay for data values:
    fmt = '.{0}f'.format(data_label_decimals) if normalize else 'd'
    thresh = (cm.max() + cm.min())/2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment = "center",
                 color = "white" if cm[i, j] > thresh else "black", size = data_label_size)

    plt.tight_layout()

def _gini_raw(actual, predicted, weight = None):
    ''' Computes the raw Gini coefficient given the input actual and predicted response variables. Helper function used by gini() and gini_plot(). '''

    # Dummy weights if none are given:
    if weight is None:
        weight = np.ones(len(actual))

    # Basic check for array lengths:
    if len(actual) != len(predicted) or len(actual) != len(weight):
        raise Exception('Actual, Predicted, and Weight arrays must be of equal length.')

    # Compute sort order and cumsum calculations:
    srt_order = predicted.argsort()
    pred_srt, actual_srt, weight_srt = np.array(predicted)[srt_order], np.array(actual)[srt_order], np.array(weight)[srt_order]
    cumul_actual, cumul_weight = np.cumsum(actual_srt), np.cumsum(weight_srt)
    cumul_actual_pct, cumul_weight_pct = cumul_actual/cumul_actual[-1], cumul_weight/cumul_weight[-1]
    cumul_actual_pct_lag, cumul_weight_pct_lag = np.append(0, cumul_actual_pct[:-1]), np.append(0, cumul_weight_pct[:-1])

    # Gini coefficient using trapezoidal method:
    trpz_area = (cumul_weight_pct - cumul_weight_pct_lag)*(cumul_actual_pct + cumul_actual_pct_lag)/2
    return 2*(0.5 - sum(trpz_area))

def gini(actual, predicted, weight = None, normalize = True):
    '''
    Computes the Gini coefficient given the actual, predicted, and weight (optional) variables. Specify normalize = True for normalized Gini.
    Input can be either Pandas series or Numpy arrays.

    Parameters:
    --------------------
    actual: numpy array or pandas series
        A series/array of the true target variable.
    predicted: numpy array or pandas series
        A series/array of the predicted target labels.
    weight: numpy array or pandas series
        Used this to specify weighted Gini coefficient, default None.
    normalize: boolean
        Used to specify normalized vs raw Gini, default True.

    Returns:
    --------------------
    float

    Example:
    --------------------
    >> t = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 5, 6])
    >> p = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.31, 0.4, 0.73, 0.31, 0.4, 0.6, 0.2, 0.32, 0.53, 0.74, 0.1, 0.34,
                     0.9, 0.2, 0.11, 0.71, 0.3, 0.51, 0.61, 0.72, 0.52, 0.29, 0.8])
    >> w = np.array([8, 3, 4, 9, 6, 2, 13, 8, 11, 8, 7, 9, 8, 5, 13, 2, 7, 10, 16, 6, 8, 10, 1, 11, 15, 14, 7, 10, 12, 11])
    >> gini(t, p, weight = w, normalize = True)
    '''

    if normalize:
        return _gini_raw(actual, predicted, weight)/_gini_raw(actual, actual, weight)
    else:
        return _gini_raw(actual, predicted, weight)

def gini_plot(ax, actual, predicted, weight = None, normalize = True,
              title = 'Gini Lorenz Plot', x_axis_label = 'Cumulative %-Exposures', y_axis_label = 'Cumulative %-Losses',
              title_size = 16, axis_label_size = 14, annotation_size = 14, axis_tick_size = 12):
    '''
    Plots the Gini Lorenz curve for a given set of actual and predicted values. Inputs can be either Pandas series or Numpy
    arrays. Specify normalize = True to display the normalized Gini rather than the raw Gini in the plot. Requires a Matplotlib
    axis object to be passed to the ax argument The remaining arguments are used to control aesthetics.

    Required Parameters:
    --------------------
    ax: a matplotlib axis object
        Usually defined by plt.subplots().
    actual: numpy array or pandas series
        A series/array of the true target variable.
    predicted: numpy array or pandas series
        A series/array of the predicted target labels.
    weight: numpy array or pandas series
        Used this to specify weighted Gini coefficient, default None.
    normalize: boolean
        Used to specify normalized vs raw Gini, default True.

    Returns:
    --------------------
    None; wrap function into a plt.subplot() block to create plots. See Example.

    Example:
    --------------------
    >> import random
    >> t = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 5, 6])
    >> p = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.31, 0.4, 0.73, 0.31, 0.4, 0.6, 0.2, 0.32, 0.53, 0.74, 0.1, 0.34,
                     0.9, 0.2, 0.11, 0.71, 0.3, 0.51, 0.61, 0.72, 0.52, 0.29, 0.8])
    >> w = np.array([8, 3, 4, 9, 6, 2, 13, 8, 11, 8, 7, 9, 8, 5, 13, 2, 7, 10, 16, 6, 8, 10, 1, 11, 15, 14, 7, 10, 12, 11])
    >> r = np.array([round(random.random(), 1) for i in range(30)])

    >> fig, ax = plt.subplots(1, 2, figsize = (12, 6))
    >> gini_plot(ax[0], actual = t, predicted = r, weight = w, normalize = False, title = 'Example Plot - Random Predictions')
    >> gini_plot(ax[1], actual = t, predicted = p, weight = w, normalize = True, title = 'Example Plot - Simulated Predictions')
    >> plt.show()
    '''

    # Gini result:
    gini_rslt = round(gini(actual, predicted, weight, normalize), 5)

    # Dummy weights if none are given:
    if weight is None:
        weight = np.ones(len(actual))

    # Create plot vectors:
    srt_order = predicted.argsort()
    pred_srt, actual_srt, weight_srt = np.array(predicted)[srt_order], np.array(actual)[srt_order], np.array(weight)[srt_order]
    cumul_actual, cumul_weight = np.cumsum(actual_srt), np.cumsum(weight_srt)
    cumul_actual_pct, cumul_weight_pct = cumul_actual/cumul_actual[-1], cumul_weight/cumul_weight[-1]

    # Plot:
    sns.lineplot(x = cumul_weight_pct, y = cumul_actual_pct, ax = ax)
    ax.plot(np.linspace(0, 1), np.linspace(0, 1))

    # Annotation text:
    pref_txt = 'Normalized Gini: {}' if normalize is True else 'Gini: {}'
    ax.text(0, 1, pref_txt.format(gini_rslt), horizontalalignment = 'left', verticalalignment = 'center', size = annotation_size)

    # Graph and axes titles:
    ax.set_title(title, size = title_size)
    ax.set_xlabel(x_axis_label, size = axis_label_size)
    ax.set_ylabel(y_axis_label, size = axis_label_size)
    ax.tick_params(axis = 'both', which = 'major', labelsize = axis_tick_size)

    plt.tight_layout()

def single_lift_plot(ax, actual, predicted, weight = None, quantiles = 10,
                     title = 'Single Lift Plot', x_axis_label = 'Equal-Weight Quantile', y_axis_label = 'Outcome', y_axis2_label = 'Weight',
                     title_size = 16, axis_label_size = 14, axis_tick_size = 12, y_axis2_scale = 3):
    '''
    Plots a single-lift chart for a given set of actual, predicted, and weight values. Inputs can be either Pandas series or Numpy
    arrays. Default number of quantiles is 10. If no weight field is passed, then all observations are assumed to have equal weight.
    Requires a Matplotlib axis object to be passed to the ax argument The remaining arguments are used to control aesthetics.

    Required Parameters:
    --------------------
    ax: a matplotlib axis object
        Usually defined by plt.subplots().
    actual: numpy array or pandas series
        A series/array of the true target variable.
    predicted: numpy array or pandas series
        A series/array of the predicted target labels.
    weight: numpy array or pandas series
        Used this to specify weighted Gini coefficient, default None.
    quantiles: integer
        Specify the number of quantile groups, default is 10 (deciles).

    Returns:
    --------------------
    None; wrap function into a plt.subplot() block to create plots. See Example.

    Example:
    --------------------
    >> import random
    >> t = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 5, 6])
    >> p = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.31, 0.4, 0.73, 0.31, 0.4, 0.6, 0.2, 0.32, 0.53, 0.74, 0.1, 0.34,
                     0.9, 0.2, 0.11, 0.71, 0.3, 0.51, 0.61, 0.72, 0.52, 0.29, 0.8])
    >> w = np.array([8, 3, 4, 9, 6, 2, 13, 8, 11, 8, 7, 9, 8, 5, 13, 2, 7, 10, 16, 6, 8, 10, 1, 11, 15, 14, 7, 10, 12, 11])
    >> r = np.array([round(random.random(), 1) for i in range(30)])

    >> fig, ax = plt.subplots(1, 2, figsize = (12, 6))
    >> single_lift_plot(ax[0], actual = t, predicted = r, weight = w, title = 'Example Plot - Random Predictions')
    >> single_lift_plot(ax[1], actual = t, predicted = p, weight = w, title = 'Example Plot - Simulated Predictions')
    >> plt.show()
    '''

    # Create uniform weights if weight = None:
    if weight is None:
        weight = np.ones(len(actual))

    # Aggregate by weighted quantile:
    lift_df = pd.DataFrame(np.stack([weight, actual, predicted], axis = 1), columns = ['weight', 'actual_raw', 'predicted_raw'])
    lift_df['quantile'] = weighted_quantile(lift_df, 'predicted_raw', weight = 'weight', n = quantiles)
    lift_df_sum = lift_df.groupby(by = 'quantile', as_index = False).agg({'weight':'sum', 'actual_raw':'sum', 'predicted_raw':'sum'})

    # Create weighted averages and melt:
    lift_df_sum['Actual'] = lift_df_sum['actual_raw']/lift_df_sum['weight']
    lift_df_sum['Predicted'] = lift_df_sum['predicted_raw']/lift_df_sum['weight']

    melt_vars = ['quantile', 'weight', 'Actual', 'Predicted']
    lift_df_sum2 = lift_df_sum[melt_vars].melt(id_vars = ['quantile', 'weight'], var_name = ['actual_pred'], value_name = 'ratio')

    # Lift lines plot:
    sns.lineplot(ax = ax, data = lift_df_sum2, x = 'quantile', y = 'ratio', hue = 'actual_pred', palette = sns.color_palette(['darkred', 'darkgreen']))

    # Weight plot on secondary axis:
    ax2 = ax.twinx()
    sns.barplot(ax = ax2, data = lift_df_sum, x = 'quantile', y = 'weight', color = 'slategrey', alpha = 0.3)
    ax2.set_ylim(0, max(lift_df_sum['weight'])*y_axis2_scale)

    # Graph options:
    ax.set_xticklabels([i+1 for i in range(quantiles)])
    ax.legend(title = None)

    # Graph and axes titles:
    ax.set_title(title, size = title_size)
    ax.set_xlabel(x_axis_label, size = axis_label_size)
    ax.set_ylabel(y_axis_label, size = axis_label_size)
    ax.tick_params(axis = 'both', which = 'major', labelsize = axis_tick_size)
    ax2.set_ylabel(y_axis2_label, size = axis_label_size)
    ax2.tick_params(axis = 'both', which = 'major', labelsize = axis_tick_size)

    plt.tight_layout()

def double_lift_plot(ax, actual, predicted1, predicted2, weight = None, quantiles = 10,
                     title = 'Double Lift Plot', x_axis_label = 'Equal-Weight Quantile', y_axis_label = 'Outcome', y_axis2_label = 'Weight',
                     title_size = 16, axis_label_size = 14, axis_tick_size = 12, y_axis2_scale = 3):
    '''
    Plots a double-lift chart for a given set of actual, predicted (predicted1 and predicted2), and weight values. Inputs can be either Pandas
    series or Numpy arrays. Default number of quantiles is 10. If no weight field is passed, then all observations are assumed to have equal weight.
    Requires a Matplotlib axis object to be passed to the ax argument The remaining arguments are used to control aesthetics.

    Required Parameters:
    --------------------
    ax: a matplotlib axis object
        Usually defined by plt.subplots().
    actual: numpy array or pandas series
        A series/array of the true target variable.
    predicted1: numpy array or pandas series
        A series/array of the predicted target labels for the first model to compare.
    predicted2: numpy array or pandas series
        A series/array of the predicted target labels for the second model to compare.
    weight: numpy array or pandas series
        Used this to specify weighted Gini coefficient, default None.
    quantiles: integer
        Specify the number of quantile groups, default is 10 (deciles).

    Returns:
    --------------------
    None; wrap function into a plt.subplot() block to create plots. See Example.

    Example:
    --------------------
    >> import random
    >> t = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 5, 6])
    >> p = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.31, 0.4, 0.73, 0.31, 0.4, 0.6, 0.2, 0.32, 0.53, 0.74, 0.1, 0.34,
                     0.9, 0.2, 0.11, 0.71, 0.3, 0.51, 0.61, 0.72, 0.52, 0.29, 0.8])
    >> w = np.array([8, 3, 4, 9, 6, 2, 13, 8, 11, 8, 7, 9, 8, 5, 13, 2, 7, 10, 16, 6, 8, 10, 1, 11, 15, 14, 7, 10, 12, 11])
    >> r = np.array([round(random.random(), 1) for i in range(30)])

    >> fig, ax = plt.subplots(figsize = (6, 6))
    >> single_lift_plot(ax, actual = t, predicted1 = r, predicted2 = p, weight = w, title = 'Example Plot - Simulated against Random')
    >> plt.show()
    '''

    # Create uniform weights if weight = None:
    if weight is None:
        weight = np.ones(len(actual))

    # Create ratio of predicted values, aggregate by weighted quantile:
    lift_df = pd.DataFrame(np.stack([weight, actual, predicted1, predicted2], axis = 1), columns = ['weight', 'actual_raw', 'predicted1_raw', 'predicted2_raw'])
    lift_df['pred_ratio'] = lift_df['predicted1_raw']/lift_df['predicted2_raw']
    lift_df['quantile'] = weighted_quantile(lift_df, 'pred_ratio', weight = 'weight', n = quantiles)
    lift_df_sum = lift_df.groupby(by = 'quantile', as_index = False).agg({'weight':'sum', 'actual_raw':'sum', 'predicted1_raw':'sum', 'predicted2_raw':'sum'})

    # Create weighted averages and melt:
    lift_df_sum['Actual'] = lift_df_sum['actual_raw']/lift_df_sum['weight']
    lift_df_sum['Predicted1'] = lift_df_sum['predicted1_raw']/lift_df_sum['weight']
    lift_df_sum['Predicted2'] = lift_df_sum['predicted2_raw']/lift_df_sum['weight']

    melt_vars = ['quantile', 'weight', 'Actual', 'Predicted1', 'Predicted2']
    lift_df_sum2 = lift_df_sum[melt_vars].melt(id_vars = ['quantile', 'weight'], var_name = ['actual_pred'], value_name = 'ratio')

    # Lift lines plot:
    sns.lineplot(ax = ax, data = lift_df_sum2, x = 'quantile', y = 'ratio', hue = 'actual_pred', palette = sns.color_palette(['darkred', 'limegreen', 'teal']))

    # Weight plot on secondary axis:
    ax2 = ax.twinx()
    sns.barplot(ax = ax2, data = lift_df_sum, x = 'quantile', y = 'weight', color = 'slategrey', alpha = 0.3)
    ax2.set_ylim(0, max(lift_df_sum['weight'])*y_axis2_scale)

    # Graph options:
    ax.set_xticklabels([i+1 for i in range(quantiles)])
    ax.legend(title = None)

    # Graph and axes titles:
    ax.set_title(title, size = title_size)
    ax.set_xlabel(x_axis_label, size = axis_label_size)
    ax.set_ylabel(y_axis_label, size = axis_label_size)
    ax.tick_params(axis = 'both', which = 'major', labelsize = axis_tick_size)
    ax2.set_ylabel(y_axis2_label, size = axis_label_size)
    ax2.tick_params(axis = 'both', which = 'major', labelsize = axis_tick_size)

    plt.tight_layout()


#################
# Random Forest #

def rf_feature_importance(data_X, rf_model, sns_font_scale = 2):
	'''
    Feature importance graph for random forest models. Use the sns_font_scale argument to scale the font size.

    Parameters:
    --------------------
    data_X: numpy 2d-array
        The matrix of explanatory variables used as the input of the random forest model (the "X" matrix).
    rf_model: scikit-learn RandomForestClassifier object
        The trained random forest model object from scikit-learn.

    Returns:
    --------------------
    None; function plots output directly.
	'''

	sns.set(font_scale = sns_font_scale)
	features_list = data_X.columns.values
	feature_importance = rf_model.feature_importances_
	feature_importance = 100 * (feature_importance/feature_importance.max())

	sorted_idx = np.argsort(feature_importance)

	pos = np.arange(sorted_idx.shape[0])
	plt.barh(pos, feature_importance[sorted_idx], align='center')
	plt.yticks(pos, features_list[sorted_idx])
	plt.xlabel('Relative Importance')
	plt.title('Random Forest Variable Importance')
	plt.show()
