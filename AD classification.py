#%% checking/changing working directory
import os
# allows checking/changing of working directory
os.chdir("C:/Users/gullm/OneDrive/Documents/MATH516 Coursework/python scripts/data/MRI Dataset - CW")
# sets working directory as file with MRI training data in

#%% MRI training data
import pandas as pd
# imports pandas for working with dataframes
mridata = pd.read_csv("Data_training_and_val_set_CNvsAD.csv")
'''
- running mridata.info() shows there are no missing values
- running mridata.duplicated().sum() shows there are no duplicated
rows
'''
import matplotlib.pyplot as plt
import seaborn as sns
# imports matplotlib.pyplot and seaborn modules for plotting

'''
f = plt.figure(figsize = (10, 8))
# sets the figure size so it is square and larger than normal
sns.heatmap(mridata.corr(method = "spearman"), annot = True)
# plots a heatmap to show how strongly variables are correlated
plt.tight_layout()
# makes sure everything fits onto the figure size
'''
'''
some of the variables seem to be quite strongly correlated
- this could imply not every variable will be necessary when
producing a machine learning model for predicting DXCURREN
'''
'''
sns.pairplot(data = mridata, hue = "DXCURREN")
plt.show()
# produces a plot to show relationship between each pair of variables
'''
#%% plotting and visual EDA
'''
DXCURREN contains a binary variable which takes the value 0
or 1. By running mridata["DXCURREN"].sum() the total sum of
this column is 110 which implies there are 110 1s and therefore
also 110 0s. This means the data is balanced and weighting
won't be a problem when fitting the classifier
'''
f = plt.figure()
sns.boxplot(data = mridata, y = "BRAIN")
plt.show()
# distribution of BRAIN variable seems pretty symmetrical
'''
BRAIN and EICV are the 2 most strongly correlated variables,
to visualise this here is a scatterplot of the 2 variables
'''
sns.scatterplot(data = mridata, x = "BRAIN", y = "EICV", hue = "DXCURREN")
plt.title("Scatterplot of BRAIN and EICV")
plt.show()
# creates a scatterplot to show correlation between BRAIN and EICV
# points are coloured according to DXCURREN, there seems to be a 
# pattern between the points
'''
also creates a scatterplot for LHIPPOC and RHIPPOC as they
also are strongly correlated
'''

''' LHIPPOC and RHIPPOC'''
sns.scatterplot(data = mridata, x = "LHIPPOC", y = "RHIPPOC", hue = "DXCURREN")
plt.title("Scatterplot of RHIPPOC and LHIPPOC")
plt.show()
# creates a scatterplot of LHIPPOC and RHIPPOC with points coloured
# according to DXCURREN - also seems to be a pattern
sns.boxenplot(data = mridata[["RHIPPOC", "LHIPPOC"]])
plt.title("Letter value plot of LHIPPOC and RHIPPOC")
plt.show()
# creates a letter value plot to show and compare distribution
# of LHIPPOC and RHIPPOC. LHIPPOC generally seems to take slightly
# lower values than RHIPPOC but both are relatively symmertrically
# distributed

''' LINFLATVEN and RINFLATVEN '''
sns.scatterplot(data = mridata, x = "LINFLATVEN", y = "RINFLATVEN", hue = "DXCURREN")
plt.title("Scatterplot of LINFLATVEN and RINFLATVEN")
plt.show()
# creates a scatterplot of LINFLATVEN and RINFLATVEN with points coloured
# according to DXCURREN - also seems to be a pattern
sns.boxenplot(data = mridata[["RINFLATVEN", "LINFLATVEN"]])
plt.title("Letter value plot of LINFLATVEN and RINFLATVEN")
plt.show()
# creates a letter value plot to show and compare distribution
# of LINFLATVEN and RINFLATVEN

''' LMIDTEMP and RMIDTEMP '''
sns.scatterplot(data = mridata, x = "LMIDTEMP", y = "RMIDTEMP", hue = "DXCURREN")
plt.title("Scatterplot of LMIDTEMP and RMIDTEMP")
plt.show()
# creates a scatterplot of LMIDTEMP and RMIDTEMP with points coloured
# according to DXCURREN - also seems to be a pattern
sns.boxenplot(data = mridata[["RMIDTEMP", "LMIDTEMP"]])
plt.title("Letter value plot of LMIDTEMP and RMIDTEMP")
plt.show()
# creates a letter value plot to show and compare distribution
# of LMIDTEMP and RMIDTEMP

''' LINFTEMP and RINFTEMP '''
sns.scatterplot(data = mridata, x = "LINFTEMP", y = "RINFTEMP", hue = "DXCURREN")
plt.title("Scatterplot of LINFTEMP and RINFTEMP")
plt.show()
# creates a scatterplot of LINFTEMP and RINFTEMP with points coloured
# according to DXCURREN - also seems to be a pattern
sns.boxenplot(data = mridata[["RINFTEMP", "LINFTEMP"]])
plt.title("Letter value plot of LINFTEMP and RINFTEMP")
plt.show()
# creates a letter value plot to show and compare distribution
# of LINFTEMP and RINFTEMP

''' LFUSIFORM and RFUSIFORM '''
sns.scatterplot(data = mridata, x = "LFUSIFORM", y = "RFUSIFORM", hue = "DXCURREN")
plt.title("Scatterplot of LFUSIFORM and RFUSIFORM")
plt.show()
# creates a scatterplot of LFUSIFORM and RFUSIFORM with points coloured
# according to DXCURREN - also seems to be a pattern
sns.boxenplot(data = mridata[["RFUSIFORM", "LFUSIFORM"]])
plt.title("Letter value plot of LFUSIFORM and RFUSIFORM")
plt.show()
# creates a letter value plot to show and compare distribution
# of LFUSIFORM and RFUSIFORM

''' LENTHORIN and RENTHORIN '''
sns.scatterplot(data = mridata, x = "LENTORHIN", y = "RENTORHIN", hue = "DXCURREN")
plt.title("Scatterplot of LENTORHIN and RENTORHIN")
plt.show()
# creates a scatterplot of LENTORHIN and RENTORHIN with points coloured
# according to DXCURREN - also seems to be a pattern
sns.boxenplot(data = mridata[["RENTORHIN", "LENTORHIN"]])
plt.title("Letter value plot of LENTORHIN and RENTORHIN")
plt.show()
# creates a letter value plot to show and compare distribution
# of LENTORHIN and RENTORHIN

#%% Decision trees and random forest classifyers

''' Decision Trees '''

'''
decision trees are effective machine learning algorithms because they
are easily interpretable

they are limited in that they don't work well with missing data or 
non-numeric/ordinal data but neither of these appear in the mridata
'''
from sklearn.tree import DecisionTreeClassifier
# imports the DecisionTreeClassifier function for making decision 
# trees from the sklearn.tree module
from sklearn.tree import plot_tree
# imports the plot_tree function for visualising decision trees
# from the sklearn.tree module
from sklearn.metrics import accuracy_score
# imports the accuracy_score function from sklearn.metrics module
# for checking the training error of the decision tree
from sklearn.model_selection import train_test_split
# imports the train test split function from sklearn.model_selection

mrivars = mridata.copy()
# creates a copy of the mridata called mrivars
mriclass = mrivars.pop("DXCURREN")
# removes the DXCURREN columns from mrivars and calls it mriclass

trainvars, testvars, trainclass, testclass = train_test_split(mrivars, mriclass, 
                                                              test_size = 0.15,
                                                              random_state = 0)
# splits the data into a training set and a test set

tree = DecisionTreeClassifier(random_state = 1)
# renames the DecisionTreeClassifier() function to just tree
tree.fit(trainvars, trainclass)
# fits a decision tree classifier to training data
f2 = plt.figure(figsize = (20, 16))
plot_tree(tree)
plt.show()
# creates a plot of the generated decision tree

predicted_classes = tree.predict(testvars)
# creates predicted class values for test data
testerror = accuracy_score(testclass, predicted_classes)
# this retruns a value of 0.7879, meaning the decision tree is
# about 82% accurate for test data

'''
decisions tress tend to overfit to the training data. This is
likely the cause of the 0% training error here, generally this
makes random forest classifiers more useful as they are less
likely to overfit to the training data
'''

''' Random Forest Classifyers '''

from sklearn.ensemble import RandomForestClassifier
# imports the random forest classifier from sklearn.ensemble
# module for fitting a random forest classifier
from sklearn.model_selection import cross_val_score
# imports the cross_val_score function from sklearn.ensemble
# module for checking how well the classifier fits via
# cross validation
from sklearn.feature_selection import SelectFromModel
# imports the SelectFromModel function from sklearn.feature_selection
# to identify important features in random forests

slctfeatures = SelectFromModel(RandomForestClassifier(n_estimators = 100))
# creates a moment for feature selection function to identify most
# important features using Random forest classifier
slctfeatures.fit(trainvars, trainclass)
# fits training data to random forest in slctfeatures
importances = slctfeatures.estimator_.feature_importances_
# this gives use the importances of each feature when training the
# Random Forest classifier, the final model will be trained using
# features where the importance is greater than 0.04 to be as
# inclusive as possible while eliminating unimportant features

trainvars2 = pd.DataFrame()
# creates an empty dataframe

for i in range(0, len(importances)):
    if importances[i] >= 0.05:
        new_col = trainvars.columns[i]
        trainvars2[new_col] = trainvars[new_col]   
# creates new training data only containing important columns

testvars2 = pd.DataFrame()
# creates an empty dataframe

for i in range(0, len(importances)):
    if importances[i] >= 0.05:
        new_col = testvars.columns[i]
        testvars2[new_col] = testvars[new_col]   
# creates new test data only containing important columns

forest = RandomForestClassifier(random_state = 1)
# creates a moment of the RandomForestClassifier function with 100 
# decision trees for ease of use
scores = cross_val_score(forest, trainvars2, trainclass, cv = 5)
# calculates errors using cross validation to gauge how
# effective the random forest classifier using default 
# hyperparameters is at classifying the mridata 
scores.mean()
scores.std()
# calculates the mean and standard devaitation score
# mean = 0.8455
# std = 0.033
'''
by fitting a random forest with default hyperparameters
to the data a mean cross valdation score of 84.5% can 
be acheived with a low standard deviation as well
'''
'''

from sklearn.model_selection import GridSearchCV
# this functions make hypertuning a classifier easier

pramtrgrd = {
    "n_estimators": [25, 50, 75, 100],
    "max_features": ["sqrt", "log2", None],
    "max_depth": [3, 4, 5, 6, 7, 8, 9, 10],
    "max_leaf_nodes": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}
# creates a grid of possible hyperparameter values to check

gridsearch = GridSearchCV(RandomForestClassifier(), 
                          param_grid = pramtrgrd,
                          cv = 5)
# creates a moment for GridSearchCV called gridsearch which
# will use the hyperparameters defined in parametergrid to find
# the best parameters for a RandomForestClassifier using 
# crossValidation with 5 splits to evaluate the classifier
gridsearch.fit(trainvars2, trainclass)
# fits the training data with important features to the gridsearch 
# algorithm to find the best
# hyperparameters for a random forest classifier

# running above code helps tune the random forest classifier 
# to find best hyperparameters for classifying mridata

'''
'''
by running gridsearch.best_estimator_ it appears that the best
estimator is RandomForestClassifier(max_depth = 9,
                                    max_features = "sqrt",
                                    max_leaf_nodes = 9,
                                    n_estimators = 25)
This classifier will be trained using the full training data
'''
bestforest = RandomForestClassifier(max_depth = 9, max_features = "sqrt",
                                    max_leaf_nodes = 9, n_estimators = 25, 
                                    random_state = 2)
# creates a moment for the tuned random forest classifier
tndscores = cross_val_score(bestforest, trainvars2, trainclass, cv = 5)
# calculates cross validation scores for the tuned forest
tndscores.mean()
tndscores.std()
# calculates mean and standard devaition of the tuned scores to
# compare to untuned scores
'''
the tuned random forest classifier returns a mean cross
validation score of 94%. This is the same as the untuned
forest classifier but it has a lower standard devation and I'm
not sure what other metrics GridSearchCV uses to determine the
best model
'''
bestforest.fit(trainvars2, trainclass)
# fits the bestforest model to the training data
prdctclass = bestforest.predict(testvars2)
# generates predicted classes for test data using best random forest
# algorithm
from sklearn.metrics import confusion_matrix
# imports the confusion_matrix function from sklearn.metrics module
# for assessing performance of classifier
cmrf = confusion_matrix(testclass, prdctclass)
# creates a confusion matrix for random forest classifier performance
# with test data
accuracy2 = accuracy_score(testclass, prdctclass)
# calculates the test error for optimised random forest
# gets a final accuracy score of 94% for test data
sensitivityrf = cmrf[0, 0]/(cmrf[0, 0] + cmrf[0, 1])
specificityrf = cmrf[1, 1]/(cmrf[1, 0] + cmrf[1, 1])
# calculates the sensitivty and specificty for the Random Forest 
# classifier using the test data
# sensitivity = 88% for test data
# specificty = 100% for test data


#%% SVM classifier
from sklearn.model_selection import train_test_split
# imports the train_test_split function from sklearn.model_selection
from sklearn.svm import SVC
# imports the SVC class from sklearn.svm for SVM classification
from sklearn.metrics import accuracy_score
# imports the accuracy_score function from sklearn.metrics
from sklearn.preprocessing import StandardScaler
# imports the StandardScaler function from sklearn.preprocessing

mrivars = mridata.copy()
# creates a copy of the mridata called mrivars
mriclass = mrivars.pop("DXCURREN")
# removes the DXCURREN columns from mrivars and calls it mriclass

trainvars, testvars, trainclass, testclass = train_test_split(mrivars, mriclass, 
                                                              test_size = 0.15,
                                                              random_state = 0)
# splits the data into a training set and a test set

scaler = StandardScaler()
# creates a moment for the  StandardScaler function
scldtrainvars = scaler.fit_transform(trainvars)
# scales the training data variables

svmlinear = SVC(kernel = "linear")
# creates a moment for the SVC function with a linear kernel

lin_scores = cross_val_score(svmlinear, scldtrainvars, trainclass, cv = 5)
# uses cross validation and the scaled training data to gauge
# how effective SVM with a linear kernel is at classifying the data
lin_scores_mean = lin_scores.mean()
lin_scores_std = lin_scores.std()
# calculates the mean cross validation errors for linear kerlled SVM
# linear mean = 85%, linear std = 0.027

svmpoly = SVC(kernel = "poly")
# creates a moment for the SVC function with a polynomial kernel

poly_scores = cross_val_score(svmpoly, scldtrainvars, trainclass, cv = 5)
# uses cross validation and the scaled training data to gauge
# how effective SVM with a polynomial kernel is at classifying the data
poly_scores_mean = poly_scores.mean()
poly_scores_std = poly_scores.std()
# calculates the mean cross validation errors for polynomial kerlled SVM
# polynomial mean = 81%, polynomial std = 0.027


scldtestvars = scaler.transform(testvars)
# scales the test data variables

svmrad = SVC(kernel = "rbf")
# creates a moment for the SVC function with a radial kernel

rad_scores = cross_val_score(svmrad, scldtrainvars, trainclass, cv = 5)
# uses cross validation and the scaled training data to gauge
# how effective SVM with a radial kernel is at classifying the data
rad_scores_mean = rad_scores.mean()
rad_scores_std = rad_scores.std()
# calculates the mean cross validation errors for radial kerlled SVM
# radial mean = 87%, radial std = 0.054

svmsig = SVC(kernel = "sigmoid")
# creates a moment for the SVC function with a sigmoidal kernel

sig_scores = cross_val_score(svmsig, scldtrainvars, trainclass, cv = 5)
# uses cross validation and the scaled training data to gauge
# how effective SVM with a sigmoidal kernel is at classifying the data
sig_scores_mean = sig_scores.mean()
sig_scores_std = sig_scores.std()
# calculates the mean cross validation errors for sigmoidal kerlled SVM
# sigmoidal mean = 84.5%, sigmoidal std = 0.049

'''
with cross validation it seems SVM with all types of kernels perform
well on this data. Models will be produces using SVMs with linear and
radial kernels as these seemed to be the best performing models with
cross validation
'''
'''
from sklearn.model_selection import GridSearchCV
# imports the GridSearchCV function from sklearn.model_selection
# module
parametergrid = {
    "C": [2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 10],
    "kernel": ["linear", "rbf", "sigmoid"],
    "gamma": ["scale", "auto", 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]}
# creates a grid of parameters that will be checked to find the best
# model for the SVM classifier
grdsrch = GridSearchCV(SVC(), param_grid = parametergrid,
                       cv = 5)
# creates a moment for the GridSearchCV algorithm with SVM classifier,
# and parammeter defined in parametergrid
grdsrch.fit(scldtrainvars, trainclass)
# fits the training data to GridSearchCV algorithm
'''
'''
after running the above code run grdsrch.best_estimator_ to 
find best SVM classifier for data. It gives the best model to be:
    SVC(C = 5, kernel = "rbf", gamma = 0.01)
'''
svmbest = SVC(C = 3, kernel = "rbf", gamma = 0.02, random_state = 3)
# fits the best SVM model that could be found for the MRI data
svmbest.fit(scldtrainvars, trainclass)
# fits the training data to best SVM model

scldtestvars = scaler.transform(testvars)
# scales the test data variables

predictions = svmbest.predict(scldtestvars)
# creates predicted classes from the test data
from sklearn.metrics import confusion_matrix
# imports the confusion_matrix function from sklearn.metrics module
# for assessing performance of classifier
cmsvm = confusion_matrix(testclass, predictions)
# creates a confusion matrix for the SVM classifier using
# the test data
accuracy3 = accuracy_score(testclass, predictions)
# calculates the accuracy score for the best SVM model
# using test data
# SVM classifier gets an accuracy score of 97% for test data
sensitivitysvm = cmsvm[0, 0]/(cmsvm[0, 0] + cmsvm[0, 1])
specificitysvm = cmsvm[1, 1]/(cmsvm[1, 0] + cmsvm[1, 1])
# calculates the sensitivty and specificty for the SVM classifier
# using the test data
# sensitivity = 94% for test data
# specificty = 100% for test data
'''
the best SVM model obtains an accuracy score of 97% for the test data
'''

