# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 20:41:24 2020

@author: Harold
"""

#
# load and prepare data
# 
import ds_util as util

ds_util = util.DSUtil()
ds_util.load_csv("bs140513_032310.csv")

ds_util.blow_my_mind()

#drop columns not needed for calculations and get dummy variables
ds_util.drop_columns(["customer", "zipMerchant", "zipcodeOri"])

df = ds_util.df

#
# Data prep
#

from sklearn.model_selection import train_test_split

target_column = "fraud"

y = df[target_column]

X = df.drop(target_column, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=42)


###################################################################
# Logistic Regression Stuffs                                                      #
###################################################################

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

#
# Training
#

lr_clf = LogisticRegression(solver='liblinear', random_state=42)
y_pred = None

def train_fn():
    global lr_clf
    
    lr_clf.fit(X_train, y_train)

ds_util.activity_wrapper("Training", train_fn)


#
# Testing
#
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.legend(loc="center right", fontsize=16) 
    plt.xlabel("Threshold", fontsize=16)        
    plt.grid(True)                                           



def test_fn():
    global y_pred
    
    y_pred = lr_clf.predict(X_test) 
    
    print(accuracy_score(y_test, y_pred))
    
    print(cross_val_score(lr_clf, X_train, y_train, cv=3, scoring="accuracy"))
    
    y_train_pred = cross_val_predict(lr_clf, X_train, y_train, cv=3)
    
    print("confusion_matrix", confusion_matrix(y_train, y_train_pred))
    
    y_train_perfect_predictions = y_train
    
    print("confusion_matrix", confusion_matrix(y_train, y_train_perfect_predictions))
    
    print("recall_score:", recall_score(y_train, y_train_pred, pos_label=0))
    print("precision_score:", precision_score(y_train, y_train_pred, pos_label=0))
    print(("f1_score:",f1_score(y_train, y_train_pred, pos_label=0)))
    
    print(classification_report(y_train, y_train_pred))
    
    y_scores = cross_val_predict(lr_clf, X_train, y_train, cv=3, method="decision_function")
    
    print(y_scores)
    
    precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)
    
    plt.figure(figsize=(8, 4))                      
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
    
    plt.show()
    
    
    

    
ds_util.activity_wrapper("Testing", test_fn)



