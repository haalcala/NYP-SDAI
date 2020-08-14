# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 19:47:23 2020

@author: Harold
"""

#
# load and prepare data
# 

import ds_util as util

ds_util = util.DSUtil()
ds_util.load_csv("files/bs140513_032310.csv")

ds_util.blow_my_mind()

#drop columns not needed for calculations and get dummy variables

ds_util.drop_columns(["customer", "zipMerchant", "zipcodeOri"])

df = ds_util.get_dummies()


###################################################################
# kNN Stuffs                                                      #
###################################################################

#
# Training
#

#We need to import the k-NN Classifier from skleart.neighbors
from sklearn.neighbors import KNeighborsClassifier

target_column = "fraud"

#Set the target column as our label (target value to be predicted)
YTrain = df[target_column]

#Remove the target column from our input variables
XTrain = df.drop(target_column, axis=1)

#Create a k-NN Classifier
classifier = KNeighborsClassifier(n_neighbors=1, algorithm='brute')

def train_fn():
    #Train the classifier using our training data
    classifier.fit(XTrain, YTrain)

ds_util.activity_wrapper("Training", train_fn)

#
# Testing
#

XTest_fraud = df.loc[df.fraud==1].drop(target_column, axis=1).iloc[0].values
XTest_not_fraud = df.loc[df.fraud==0].drop(target_column, axis=1).iloc[0].values

# XTest = pd.DataFrame([
#     # this should give value 1 result
#     XTest_fraud,
#     # this should give value 0 result
#     XTest_not_fraud
# ])

# for index, row in XTest.iterrows():
#     print(classifier.predict(row.values.reshape(1, -1)))

total_test_count = 2
tests_count_done = 0
tests_passed = 0
tests_failed = 0

def test_fn():
    global tests_count_done, tests_passed, tests_failed
    
    tests_count_done = tests_count_done+1
    if classifier.predict(XTest_fraud.reshape(1,-1))[0] == 1:
        tests_passed = tests_passed+1
        print("Tested fraud entry to be fraud")
    else:
        tests_failed = tests_failed+1
        print("Failed testing fraud to be fraud")
    
    tests_count_done = tests_count_done+1
    if classifier.predict(XTest_not_fraud.reshape(1,-1))[0] == 0:
        tests_passed = tests_passed+1
        print("Tested not fraud entry to be not fraud")
    else:
        tests_failed = tests_failed+1
        print("Failed testing not fraud to be not fraud")
        
ds_util.activity_wrapper("Testing", test_fn)
     
print("Test result: {}/{} done. passed: {}, failed: {}.".format(tests_count_done, total_test_count, tests_passed, tests_failed))

