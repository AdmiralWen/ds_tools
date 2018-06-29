"""
Modeling Visualizations
Author: Brandon Wen

Contains useful functions for plotting and visualizing modeling results.

plotConfusionMatrix: Plots the confusion matrix heatmap given the true and predicted labels.
rfFeatureImportance: Plots the feature importance graph for random forest models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from sklearn.metrics import confusion_matrix

###########
# General #

def plotConfusionMatrix(true_labels, pred_labels, normalize = False, title = None, cmap = plt.cm.Blues, decimals = 2, x_tick_rotation = 0):
    '''
        Plots the confusion matrix heatmap given the true and predicted labels, which can be either Pandas series or Numpy arrays.
        Use the normalize argument to display a normalized confusion matrix. The remaining arguments are used to control aesthetics.
    '''
    cm = confusion_matrix(true_labels, pred_labels)
    classes = sorted(list(set(true_labels).union(set(pred_labels))))

    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]

    if not title:
        title = 'Normalized Confusion Matrix' if normalize else 'Confusion Matrix'

    # Plot color-coded matrix; use the cmap argument to control the color map:
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = x_tick_rotation)
    plt.yticks(tick_marks, classes)

    # Add numerical overlay:
    fmt = '.{0}f'.format(decimals) if normalize else 'd'
    thresh = cm.max()/2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment = "center",
                 color = "white" if cm[i, j] > thresh else "black")

    # Add X & Y labels:
    plt.tight_layout()
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')

    plt.show()

#################
# Random Forest #

def rfFeatureImportance(data_X, rf_model, sns_font_scale = 2):
	'''
		Feature importance graph for random forest models. Use the sns_font_scale argument to scale the font size.
		
		data_X: the matrix of explanatory variables used as the input of the random forest model (the "X" matrix)
		rf_model: the random forest model object
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