import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(r'C:\Users\user\FITNESS_TRACKER')
from utils import LowPassFilter,PrincipalComponentAnalysis,NumericalAbstraction,FourierTransformation
from sklearn.cluster import KMeans

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
del df['Duration']


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


 ## ADDING ROLLING AVERAGE 

df_temporal = df_squared.copy()
NumAbs = NumericalAbstraction()

predicitor_columns = predicitor_columns + ['acc_r','gyr_r']


ws = int(1000/200)
df_temporal_list = []
for s in df_temporal['set'].unique():
    subset = df_temporal[df_temporal['set']==s].copy()

    for col in predicitor_columns:
        subset = NumAbs.abstract_numerical(subset,[col],ws,'mean')
        subset = NumAbs.abstract_numerical(subset,[col],ws,'std')
    df_temporal_list.append(subset)

df_temporal =  pd.concat(df_temporal_list)


# frequency features using fast fourier transformation

freq_df = df_temporal.copy().reset_index() # changing it to discrete index not time series
fft = FourierTransformation()

sr = int(1000/200)
ws = int(2800/200) # average length of a repetition

df_freq_list = []

for s in freq_df['set'].unique():
    print(f'Applying fourier transformation in set {s}')
    subset = freq_df[freq_df['set'] == s].reset_index(drop=True).copy()
    subset = fft.abstract_frequency(subset,predicitor_columns,ws,sr)
    df_freq_list.append(subset)

freq_df = pd.concat(df_freq_list).set_index('epoch (ms)',drop=True)

# dealing with overllaping windows (columns are very correlated and could cause overfu=itting)

freq_df.dropna(inplace=True) #drop missing values
freq_df = freq_df[::2] # allowing 50% correlation between data



# adding clusters \
df_cluster = freq_df.copy()

cluster_columns = ['acc_x','acc_y','acc_z']
k_values = range(2,10)
inertia = []

for k in k_values:
    subset = df_cluster[cluster_columns]
    kmeans = KMeans(n_clusters=k,n_init=20,random_state=0)
    cluster_labels = kmeans.fit_predict(subset)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=[10,20])
plt.plot(k_values,inertia)
plt.xlabel('k')
plt.ylabel('sum of squared distance');



subset = df_cluster[cluster_columns]    
kmeans = KMeans(n_clusters=5,n_init=20,random_state=0)
df_cluster['cluster'] = kmeans.fit_predict(subset)

# cluster plot
fig = plt.figure(figsize=[35,35])
ax = fig.add_subplot(projection='3d')
for c in df_cluster['cluster'].unique():
    subset = df_cluster[df_cluster['cluster'] == c]
    ax.scatter(subset['acc_x'],subset['acc_y'],subset['acc_z'],label=c)
ax.set_xlabel('x_axis')
ax.set_ylabel('y_label')
ax.set_zlabel('z_label')
plt.legend()
os.makedirs('../../reports/kmeans_cluster',exist_ok=True)
plt.savefig('../../reports/kmeans_cluster/cluster_plot.png');


# cluster plot
fig = plt.figure(figsize=[35,35])
ax = fig.add_subplot(projection='3d')
for l in df_cluster['label'].unique():
    subset = df_cluster[df_cluster['label'] == l]
    ax.scatter(subset['acc_x'],subset['acc_y'],subset['acc_z'],label=l)
ax.set_xlabel('x_axis')
ax.set_ylabel('y_label')
ax.set_zlabel('z_label')
plt.legend()
os.makedirs('../../reports/kmeans_cluster',exist_ok=True)
plt.savefig('../../reports/kmeans_cluster/cluster_labelplot.png');  
