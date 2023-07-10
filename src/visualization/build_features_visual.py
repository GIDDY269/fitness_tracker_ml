import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(r'C:\Users\user\FITNESS_TRACKER')
from utils import LowPassFilter,PrincipalComponentAnalysis


# load the data

df = pd.read_pickle('../../artifacts/outlier_removed_chauvenet.pkl')

df.info()

predicitor_columns = list(df.columns[:6])

plt.style.available
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = [20,5]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['lines.linewidth'] = 2


# handling missing values using interpolate

for col in predicitor_columns:
    df[col] = df[col].interpolate()

df.info()

# calculating the duration of the set

df[df['set'] == 16]['acc_x'].plot()
df[df['set'] == 23]['acc_x'].plot()

for s in df['set'].unique():

    start = df[df['set'] == s].index[0]
    stop = df[df['set'] == s].index[-1]
    duration = stop - start
    df.loc[df['set'] == s,'Duration'] = duration.seconds

         
df.groupby('category')['Duration'].mean()


# applying low pass filter
lowpass_df = df.copy()
sf = 1000/200
cutoff = 1.3

lowpass = LowPassFilter()
lowpass_df = lowpass.low_pass_filter(lowpass_df,sampling_frequency=sf,cutoff_frequency=cutoff,col='acc_y')


subset = lowpass_df[df['set'] == 12]
print(subset['label'][0])

fig ,ax = plt.subplots(sharex=True,nrows=2,figsize = [20,10])
ax[0].plot(subset['acc_y'].reset_index(drop=True),label='raw data')
ax[1].plot(subset['acc_y_lowpass'].reset_index(drop=True),label='butterworth filter')
ax[0].legend(loc='upper center',fancybox=True,shadow=True,bbox_to_anchor=[0.5,1.15])
ax[1].legend(loc='upper center',fancybox=True,shadow=True,bbox_to_anchor=[0.5,1.15])
os.makedirs('../../reports/butterworth filter',exist_ok=True)
plt.savefig('../../reports/butterworth filter/ ohp [set 12].png')


for col in predicitor_columns:
    lowpass_df = lowpass.low_pass_filter(lowpass_df,sampling_frequency=sf,cutoff_frequency=cutoff,col=col)
    lowpass_df[col] = lowpass_df[col + '_lowpass']
    del lowpass_df[col + '_lowpass']


# principal component analysis

PCA = PrincipalComponentAnalysis()
pca_df = lowpass_df.copy()

#get pca values
pca_values = PCA.determine_pc_explained_variance(pca_df,predicitor_columns)

plt.figure(figsize=[10,10])
plt.plot(range(1,len(predicitor_columns) + 1) , pca_values)
plt.xlabel('prinicipal componets')
plt.ylabel('explained variance')
os.makedirs('../../reports/PCA',exist_ok=True)
plt.savefig('../../reports/PCA/elbow_method.png')


# appllying pca

pca_df = PCA.apply_pca(pca_df,predicitor_columns,3)

# sum of squares

df_squared = pca_df.copy()

acc_r = df_squared['acc_x']**2 + df_squared['acc_y']**2 + df_squared['acc_z']**2
gyr_r = df_squared['gyr_x']**2 + df_squared['gyr_y']**2 + df_squared['gyr_z']**2

df_squared['acc_r'] = np.sqrt(acc_r)
df_squared['gyr_r'] = np.sqrt(gyr_r)


subset = df_squared[df_squared['set'] == 12]
plt.figure()
subset[['acc_r','gyr_r']].plot(subplots=True)
os.makedirs('../../reports/sum_of_squares',exist_ok=True)
plt.savefig('../../reports/sum_of_squares/ acc_r and gyr_r plot.png')


