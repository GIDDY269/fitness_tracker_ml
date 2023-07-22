import os
import sys
sys.path.append('../../')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#from LearningAlgorithms import ClassificationAlgorithms
#import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix
from utils import ClassificationAlgorithms


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

df = pd.read_pickle('../../artifacts/feature_engineered_data.pkl')

# creating training and tet set

df_train = df.drop(['participants','category','set'],axis = 1)

x = df_train.drop('label',axis=1)
y = df_train['label']

x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size=0.25,random_state=0,stratify=y
)


# select feature subset

basic_feature  = ['acc_x','acc_y','acc_z','gyr_x','gyr_y','gyr_z']
square_features = ['acc_r','gyr_r']
pca_features = ['pca_1','pca_2','pca_3']
tim_features = [f  for f in df_train.columns if '_temp_' in f]
freq_features = [f  for f in df_train.columns if ('_freq' in f) or ('_pse' in f)]
cluster_features = ['cluster']

print(f'basic features : {len(basic_feature)}')
print(f'square features : {len(square_features)}')
print(f'pca features : {len(pca_features)}')
print(f'time features : {len(tim_features)}')
print(f'frequency features : {len(freq_features)}')
print(f'cluster features : {len(cluster_features)}')


feature_set_1 =  basic_feature
feature_set_2 = list(set(basic_feature + square_features + pca_features))
feature_set_3 = list(set(feature_set_2 + tim_features))
feature_set_4 = list(set(feature_set_3 + freq_features + cluster_features))

# performing forward feature selection using dicision tree

classifier = ClassificationAlgorithms()
max_features = 10

selected_features, ordered_features, ordered_score = classifier.forward_selection(
    max_features,x_train,y_train
)


selected_features = [
    'acc_z_freq_0.0_Hz_ws_14',
    'acc_x_freq_0.0_Hz_ws_14',
    'gyr_r_freq_0.0_Hz_ws_14',
    'acc_y_temp_std_ws_5',
    'gyr_r_freq_2.143_Hz_ws_14',
    'gyr_x',
    'acc_r_max_freq',
    'gyr_y_max_freq',
    'cluster',
    'gyr_y_temp_mean_ws_5'
]

plt.figure(figsize=[10,5])
plt.plot(np.arange(1,max_features + 1,1) , ordered_score)
plt.xlabel('number of features')
plt.ylabel('Accuracy')
plt.xticks(np.arange(1,max_features+1,1))
plt.tight_layout()
os.makedirs('../../reports/model building',exist_ok=True)
plt.savefig('../../reports/model building/selected feature accuracy.png');

# Grid search for hyperparemeter tuning and model selection

possible_feature_set = [
    feature_set_1,
    feature_set_2,
    feature_set_3,
    feature_set_4,
    selected_features
]

feature_name  =  [
    'feature_set_1',
    'feature_set_2',
    'feature_set_3',
    'feature_set_4',
    'selected_features'
]


iterations = 1
score_df = pd.DataFrame()
learner = ClassificationAlgorithms()

for i,f in zip(range(len(possible_feature_set)),feature_name):
    print(f'features set : {i}')
    selected_x_train = x_train[possible_feature_set[i]]
    selected_x_test = x_test[possible_feature_set[i]]

    #    First run non deterministic classifiers to average their score.  

    performance_test_nn = 0
    performance_test_rf = 0


    for it in range(0,iterations):
        print('\t training neural network',it)

        (
            class_train_y,
            class_test_y,
            class_train_proba_y,
            class_test_proba_y
        ) = learner.feedforward_neural_network(
            selected_x_train,y_train,selected_x_test,gridsearch=False,print_model_details=True
        )

        performance_test_nn += accuracy_score(y_test,class_test_y)

        print('\t training random forest',it)

        (
            class_train_y,
            class_test_y,
            class_train_proba_y,
            class_test_proba_y
        ) = learner.random_forest(
            selected_x_train,y_train,selected_x_test,print_model_details=True
        )

        performance_test_rf += accuracy_score(y_test,class_test_y)

    # calculating average of all model performance


    performance_test_nn = performance_test_nn / iterations
    performance_test_rf = performance_test_rf / iterations


    print('\t training knn')

    (
        class_train_y,
        class_test_y,
        class_train_proba_y,
        class_test_proba_y
    ) = learner.k_nearest_neighbor(
        selected_x_train,y_train,selected_x_test,gridsearch=True,print_model_details=True
    )

    performance_knn = accuracy_score(y_test,class_test_y)


    print("\tTraining decision tree")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.decision_tree(
        selected_x_train, y_train, selected_x_test, gridsearch=True,print_model_details=True
    )
    performance_test_dt = accuracy_score(y_test, class_test_y)


    print("\tTraining naive bayes")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.naive_bayes(selected_x_train, y_train, selected_x_test)

    performance_test_nb = accuracy_score(y_test, class_test_y)


    # Save results to dataframe
    models = ["NN", "RF", "KNN", "DT", "NB"]
    new_scores = pd.DataFrame(
        {
            "model": models,
            "feature_set": f,
            "accuracy": [
                performance_test_nn,
                performance_test_rf,
                performance_knn,
                performance_test_dt,
                performance_test_nb,
            ],
        }
    )
    score_df = pd.concat([score_df, new_scores]).sort_values(by='accuracy',ascending=False).reset_index(drop=True)










   