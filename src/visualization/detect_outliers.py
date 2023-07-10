import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(r'C:\Users\user\FITNESS_TRACKER')
from utils import mark_outliers_chauvenet,mark_outliers_iqr,plot_binary_outliers,mark_outliers_lof
import time


# load the data

df = pd.read_pickle('../../artifacts/transformed_data.pkl')
outlier_column = list(df.columns[:6])

plt.style.available
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = [20,5]
plt.rcParams['figure.dpi'] = 100

# check to see for outliers
df[['acc_x','label']].boxplot(by='label',figsize=[20,10]);
df[['acc_y','label']].boxplot(by='label',figsize=[20,10]);


# seperate box plot each label for the accelerometer data
df[outlier_column[:3] + ['label']].boxplot(by='label',figsize=[20,10],layout=[1,3])

# seperate box plot each label for the gyroscope data
df[outlier_column[3:6] + ['label']].boxplot(by='label',figsize=[20,10],layout=[1,3])




#  plotting all the columns using interquartile range
for col in outlier_column:
    dataset =  mark_outliers_iqr(df,col)
    plot_binary_outliers(dataset=dataset, col=col, outlier_col=col+'_outlier', reset_index=True,method='IQR')

# check for normal distribution
df[outlier_column[:3] + ['label']].plot.hist(by='label',figsize=[20,20],layout=[3,3])

df[outlier_column[3:6] + ['label']].plot.hist(by='label',figsize=[20,20],layout=[3,3])


# Chauvenet’s Criterion {assumes normal distribution using probilities band from the mean to get outliers}

#  plotting all the columns using chauvenet criterion
for col in outlier_column:
    dataset =  mark_outliers_chauvenet(df,col)
    plot_binary_outliers(dataset=dataset,col=col,outlier_col=col+'_outlier',reset_index=True)



#  plotting all the columns using local outlier factore
dataset, outliers, X_scores =  mark_outliers_lof(df,outlier_column)
for col in outlier_column:
    plot_binary_outliers(dataset=dataset,col=col,outlier_col='outlier_lof',reset_index=True)


# checking for outliers by labels for each column

# using   IQR
label = 'squat'
for col in outlier_column:
    dataset =  mark_outliers_iqr(df[df['label'] == label],col)
    plot_binary_outliers(dataset=dataset,col=col,outlier_col=col+'_outlier',reset_index=True)

# using chauvenet critirion
for col in outlier_column:
    dataset =  mark_outliers_chauvenet(df[df['label'] == label],col)
    plot_binary_outliers(dataset=dataset,col=col,outlier_col=col+'_outlier',reset_index=True)

# using lof
dataset,outliers,X_scores =  mark_outliers_lof(df[df['label'] == label],outlier_column)
for col in outlier_column:
    plot_binary_outliers(dataset=dataset,col=col,outlier_col='outlier_lof',reset_index=True)


# create a for loop to deal with outliers
outliers_remove_df = df.copy()
for col in outlier_column:
    for label in df['label'].unique():
        dataset = mark_outliers_chauvenet(df[df['label'] == label],col)

        # replace the columns marked as outliers as NaN
        dataset.loc[dataset[col+'_outlier'],col] = np.nan

        #update the column to the original dataframe

        outliers_remove_df.loc[(outliers_remove_df['label']== label),col] = dataset[col]
        n_outliers = len(dataset) - len(dataset[col].dropna())
        print(f'removed {n_outliers} from {col} for {label}')
