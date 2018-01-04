# Titanic Kaggle using SVM Kernel

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
#Sel_ind = [1, 3, 4, 5, 6, 8, 10]
Sel_ind = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Cabin']
Sel_ind_num = [2, 3, 4, 5]
Sel_ind_cat = [0, 6, 7]

train = pd.read_csv('train.csv')
#list(train)
#X_train = np.delete(train.values,obj = 1, axis = 1) # Remove survived column 
X_train = train[Sel_ind].values
y_train = train.Survived.values 

test = pd.read_csv('test.csv')
X_test = test[Sel_ind].values

# Test Indices
# 0 passenger id 
# 1 - Pclass
# 2 - Name 
# 3 - Sex binary
# 4 - Age
# 5 - SibSp Categorical
# 6 - Parch Categorical
# 7 - Ticket
# 8 - Fare Numbers
# 9 Cabin Numbers and letters Alot of nan
# 10 - Embarked  Categorical

# Find NaN
#import math
#np.any(np.isnan(X_train))
#np.any(np.isnan(X_test))
#np.where(np.isnan(X_test))



####### Missing Data ###########
# Taking care of missing data: Choose mean for missing values
from sklearn.preprocessing import Imputer
# Missing data for Training dataset
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
#imputer = imputer.fit(X_train[:, 2:7])
imputer = imputer.fit(X_train[:, Sel_ind_num])
X_train[:, Sel_ind_num] = imputer.transform(X_train[:, Sel_ind_num])

# Missing data for Testing dataset
imputer_test = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer_test = imputer_test.fit(X_test[:, Sel_ind_num])
X_test[:, Sel_ind_num] = imputer.transform(X_test[:, Sel_ind_num])

# Missing data for Cabin feature
# Set NaN in Cabin column as a string. Choose the first letter in cabin
X_train[:,7][pd.isnull(X_train[:,7])]  = 'NaN'
X_train_str = X_train[:,7].astype(str)
for i in range(0, len(X_train)):
    #temp_first.append(word[0])
    if X_train_str[i] != 'NaN':
        X_train[i, 7] = X_train_str[i][0]
        #print(temp[i][0])

X_test[:,7][pd.isnull(X_test[:,7])]  = 'NaN'
X_test_str = X_test[:,7].astype(str)
for i in range(0, len(X_test)):
    #temp_first.append(word[0])
    if X_test_str[i] != 'NaN':
        X_test[i, 7] = X_test_str[i][0]
        #print(temp[i][0])
        
        
#imputer_Embarked = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
#imputer_Embarked = imputer_Embarked.fit(X_train[:, 6])
#X_train[:, Sel_ind_num] = imputer.transform(X_train[:, Sel_ind_num])

# Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
#imputer_Embarked.fit_transform(X_train[:, 6])


#ind = pd.isnull(X_train[:,6]).nonzero()[0]
#X_train[61,6] = 'Q'

############ Label categorical data ########################
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Label Sex Categorical data
labelencoder_Sex = LabelEncoder()
X_train[:, 1] = labelencoder_Sex.fit_transform(X_train[:, 1])
X_test[:, 1] = labelencoder_Sex.fit_transform(X_test[:, 1])

# Label Embarked column data
labelencoder_Embarked = LabelEncoder()
labelencoder_Embarked.fit(['S','C','Q','NaN'])
#list(labelencoder_Embarked.classes_)
#labelencoder_Embarked.transform(["S", "Q"])
X_train[:,6][pd.isnull(X_train[:,6])]  = 'NaN'
X_test[:,6][pd.isnull(X_test[:,6])]  = 'NaN'
X_train[:, 6] = labelencoder_Embarked.transform(X_train[:,6])
X_test[:, 6] = labelencoder_Embarked.transform(X_test[:, 6])

# Label Cabin
labelencoder_Cabin = LabelEncoder()
#dataset_Cabin = pd.concat(( X_train[:,6],X_test[:,6] ))
dataset_Cabin = pd.concat([pd.DataFrame(X_train[:,7]), pd.DataFrame(X_test[:,7])], axis = 0)
dataset_Cabin = labelencoder_Cabin.fit_transform(dataset_Cabin)
X_train[:,7] = dataset_Cabin[0:len(X_train)]
X_test[:,7] = dataset_Cabin[len(X_train):len(dataset_Cabin)]


################## Encode the Categorical Class ###################
onehotencoder = OneHotEncoder(categorical_features = Sel_ind_cat)
dataset = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_train)], axis = 0)
dataset = dataset.values
onehotencoder.fit(dataset)
#dataset_Encoder = onehotencoder.transform(dataset).toarray()
X_train = onehotencoder.transform(X_train).toarray()
X_test = onehotencoder.transform(X_test).toarray()
#pd.isnull(X_train).any()
#pd.isnull(X_test).any()

#X_train = temp = onehotencoder.fit_transform(X_train).toarray()
#X_test = temp2 = onehotencoder.fit_transform(X_test).toarray()


##################### Feature Scaling ####################
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

################ Fitting classifier to the Training set #########
# Create your classifier here
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf',
                 C = 2.5, 
                 gamma = 0.05)
classifier.fit(X_train, y_train)

#from xgboost import XGBClassifier
#classifier = XGBClassifier()
## n_estimators = 100 number of trees
#classifier.fit(X_train, y_train)


################### Apply kfold validation ###################
from sklearn.cross_validation import cross_val_score
accuracies = cross_val_score(estimator = classifier,
                             X = X_train,
                             y = y_train,
                             cv = 10) # 10 fold cross validation
# if you have lot of dataset, n_jobs = -1 to use all CPU's 
accuracies.mean()
accuracies.std()

################## Apply Grid Search ###################
from sklearn.grid_search import GridSearchCV
parameters = [{'C':[1, 0.5, 1.5, 2, 2.5, 3, 2.4, 2.6], 'kernel': ['rbf'], 'gamma': [0.3, 0.2, 0.1, 0.05, 0.01, 0.001]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_



##################### Predict Survival of Test set ############
y_pred = classifier.predict(X_test)

################### Create CSV File  #######################
PassengerID = test.iloc[:, 0].values
Final = np.matrix([PassengerID, y_pred]).transpose()
np.savetxt(fname = "Submission5.csv", 
           X = Final, 
           fmt = "%d",
           delimiter=",",
           header = "PassengerId,Survived",
           comments = "")
temp = pd.read_csv('Submission5.csv')

