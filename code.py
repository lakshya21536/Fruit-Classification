import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
category_map = {'Apple_Raw': 0, 'Apple_Ripe': 1, 'Banana_Raw': 2, 'Banana_Ripe': 3, 
                'Coconut_Raw': 4, 'Coconut_Ripe': 5, 'Guava_Raw': 6, 'Guava_Ripe': 7, 
                'Leeche_Raw': 8, 'Leeche_Ripe': 9, 'Mango_Raw': 10, 'Mango_Ripe': 11, 
                'Orange_Raw': 12, 'Orange_Ripe': 13, 'Papaya_Raw': 14, 'Papaya_Ripe': 15, 
                'Pomengranate_Raw': 16, 'Pomengranate_Ripe': 17, 'Strawberry_Raw': 18, 'Strawberry_Ripe': 19}

# Loading the data
train_data = pd.read_csv('../input/sml-project/train.csv')
test_data = pd.read_csv('../input/sml-project/test.csv')

print("TRAIN DATA before encoding\n",train_data)
train_data['category'] = train_data['category'].map(category_map)
print("TRAIN DATA after encoding\n",train_data)

train_cat = train_data['category']
test_id = test_data['ID']

print("TRAIN DATA before dropping\n",train_data.shape)
print("TEST DATA before dropping\n",test_data.shape)

test_data=test_data.drop(columns=['ID'])
train_data=train_data.drop(columns=['ID', 'category'])

print("TRAIN DATA after dropping\n",train_data.shape)
print("TEST DATA after dropping\n",test_data.shape)

# Outlier detection using Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred_lof = lof.fit_predict(train_data)
train_data_lof = train_data[y_pred_lof == 1]
train_cat_lof = train_cat[y_pred_lof == 1]

pca = PCA(n_components=400)
train_data = pca.fit_transform(train_data)
test_data = pca.transform(test_data)

print("TRAIN DATA AFTER PCA\n",train_data.shape)
print("TEST DATA AFTER PCA\n",test_data.shape)

# Feature transformation using LDA followed by PCA
lda = LDA(n_components=19)
train_data = lda.fit_transform(train_data, train_cat)
test_data = lda.transform(test_data)
                             

print("TRAIN DATA AFTER LDA\n",train_data.shape)
print("TEST DATA AFTER LDA\n",test_data.shape)


nn1 = MLPClassifier(hidden_layer_sizes=(400), 
                   activation='logistic', 
                   solver='adam', 
                   alpha=0.0001, 
                   max_iter=5000, 
                   batch_size=128,
                   learning_rate='constant', 
                   learning_rate_init=0.001, 
                   verbose=True)
nn2 = MLPClassifier(hidden_layer_sizes=(400), 
                   activation='tanh', 
                   solver='adam', 
                   alpha=0.0001,  
                   max_iter=5000, 
                   batch_size=128,
                   learning_rate='constant', 
                   learning_rate_init=0.001,
                   verbose=True)

kmeans = KMeans(n_clusters=20, random_state=0)
kmeans.fit(train_data)
train_data_kmeans = np.column_stack((train_data, kmeans.labels_))
test_data_kmeans = np.column_stack((test_data, kmeans.predict(test_data)))

lr1 = LogisticRegression(C = 0.1, max_iter= 5000, penalty= 'l2', solver ='newton-cg')
lr2 = LogisticRegression(C = 0.1, max_iter= 5000, penalty= 'l2', solver ='lbfgs')
# lr3 = LogisticRegression(C = 0.1, max_iter= 5000, penalty= 'l2', solver ='saga')

voting_clf = VotingClassifier(estimators=[('nn1',nn1),('nn2',nn2),('lr2',lr2),('lr1',lr1)], voting='hard')
voting_clf_val = voting_clf

# Training and validating classifiers
name = 'Logistic Regression+MLP'
print(f'Training and validating {name} classifier...')
X_train, X_test, y_train, y_test = train_test_split(train_data, train_cat, test_size=0.2, random_state=0)
voting_clf_val.fit(X_train, y_train)
y_pred = voting_clf_val.predict(X_test)
print()
kf = KFold(n_splits=3, shuffle=True, random_state=0)
scores = cross_val_score(voting_clf_val, train_data, train_cat, cv=kf, scoring='accuracy')
accuracy = accuracy_score(y_test, y_pred)
print()
print(f'{name} classifier - Validation Accuracy:', accuracy)
print(f'{name} classifier - Cross Validation Scores:', scores)
print()
print()
print()

voting_clf.fit(train_data, train_cat)
# Predicting on test data
test_pred = voting_clf.predict(test_data)
print("TEST DATA PREDICTION\n",test_pred)
# Saving the predictions
test_pred_category = pd.Series(test_pred).map({v: k for k, v in category_map.items()})
output = pd.DataFrame({'ID': test_id, 'category': test_pred_category})
output.to_csv('SubmissionTrial.csv', index=False)