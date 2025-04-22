import argparse
import sys
import re
import os
# the ML libraries take forever to load, so checking for the json file first saves time
parser = argparse.ArgumentParser(description='Pass a file name.')
parser.add_argument('--colsample_bytree', default = 1, required = False) # range (0,1]
parser.add_argument('--file',        required = True)
parser.add_argument('--output_stem', required = True)
parser.add_argument('--target',      required = True)
# https://xgboost.readthedocs.io/en/latest/parameter.html
parser.add_argument('--max_depth',   default = 6, required = False)
# max_depth range: [0,∞] (0 is only accepted in lossguided growing policy when tree_method is set as hist or gpu_hist)
parser.add_argument('--gamma',     default = 0,  required = False)
# Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be. range: [0,∞]

parser.add_argument('--learning_rate', default = 0.3, required = False) # range [0,1]
parser.add_argument('--n_jobs', required = False, default = 1, type = int)
parser.add_argument('--categorical', nargs='+', required = False)
parser.add_argument('--drop', nargs='+', required = False)
args = parser.parse_args()

if not os.path.isfile(args.file):
	sys.exit(args.file + ' is not a file')

import mlflow
import json
import mlflow.sklearn
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate
import xgboost
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, RocCurveDisplay

def ref_to_json_file(data, filename):
	json1=json.dumps(data)
	f = open(filename,"w+")
	print(json1,file=f)

xgb_dir = args.output_stem + '.xgb.reg.output/'
if not os.path.isdir(xgb_dir):
	os.mkdir(xgb_dir)
xgb_x_test_dir = xgb_dir + 'X_test/'
xgb_json_dir   = xgb_dir + 'json/'
xgb_result_obj_dir = xgb_dir + 'xg_clf/'
image_dir      = xgb_dir + 'Images/'
for directory in xgb_x_test_dir, xgb_json_dir, xgb_result_obj_dir, image_dir:
	if not os.path.isdir(directory):
		os.mkdir(directory)

def xgbclassifier_wrapper( input_file, category_cols, dependent_var, output_stem, colsample_bytree, gamma, learning_rate, max_depth):
  #https://xgboost.readthedocs.io/en/latest/parameter.html
  if re.search(r'\.json$', input_file):
    pandasDF = pd.read_json(input_file)
  elif re.search(r'\.csv$', input_file):
    pandasDF = pd.read_csv(input_file)
  elif re.search(r'\.xlsx$', input_file):
    pandasDF = pd.read_excel(input_file)
  # https://stackoverflow.com/questions/58101126/using-scikit-learn-onehotencoder-with-a-pandas-dataframe
  if args.drop:
  	for col in args.drop:
  		pandasDF = pandasDF.drop(columns = [col])
  print(pandasDF.columns)
  if args.categorical != None:
	  for category in args.categorical:
  		pandasDF = pd.get_dummies(pandasDF, prefix = [category], columns = [category], drop_first = True)
  print(pandasDF)
  Y = pandasDF[dependent_var]
  X = pandasDF.drop([dependent_var], axis=1)
  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
  mlflow.sklearn.autolog()

  # With autolog() enabled, all model parameters, a model score, and the fitted model are automatically logged.  
  with mlflow.start_run():
    # Set the model parameters. 
    n_estimators = 200
    #colsample_bytree = 0.3 # colsample_bytree is the subsample ratio of columns when constructing each tree. Subsampling occurs once for every tree constructed.
    #learning_rate = 0.05
    #max_depth = 6# default 6; max. depth of a tree. Increasing this value will make the model more complex and more likely to overfit. 0 is only accepted in lossguided growing policy when tree_method is set as hist or gpu_hist and it indicates no limit on depth. Beware that XGBoost aggressively consumes memory when training a deep tree.
    #min_child_rate = 0
    #gamma = 0 # default = 0; Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be.

    # Create and train model.
    xg_clf = xgboost.XGBClassifier( n_estimators=n_estimators, colsample_bytree=colsample_bytree, learning_rate=learning_rate, max_depth=max_depth, gamma = gamma, use_label_encoder=False, eval_metric = 'logloss')
    xg_clf.fit(X_train, y_train)
    # Use the model to make predictions on the test dataset.
    predictions = xg_clf.predict(X_test)
  accuracy = accuracy_score(y_test, predictions)
  pre_score  = precision_score(y_test, predictions)
  return_dict = {}
  return_dict['Feature Importance'] = {}
  for importance_type in ('weight', 'gain', 'cover', 'total_gain', 'total_cover'):
    return_dict['Feature Importance'][importance_type] = xg_clf.get_booster().get_score( importance_type = importance_type)
  kfold = KFold(n_splits=10)
  results = cross_val_score(xg_clf, X, Y, cv=kfold)
  accuracy = results.mean() * 100
  y_score = xg_clf.predict_proba(X_test)[:, 1]
  roc = RocCurveDisplay.from_predictions(y_test, y_score)#, name = dependent_var)
  roc_svg = image_dir + output_stem + '_ROC.svg'
  plt.savefig(roc_svg)
  return_dict['ROC_SVG'] = roc_svg
  return_dict['precision_score'] = pre_score
  return_dict['accuracy']        = accuracy
  return_dict['roc_auc']         = roc.roc_auc
  scores = cross_validate(estimator=xg_clf, X=X_train, y=y_train, cv=kfold, n_jobs=args.n_jobs, scoring=['accuracy', 'roc_auc', 'precision', 'recall', 'f1'])
  return_dict['cross_validate'] = {}
  return_dict['cross_validate']['AUC mean'] = scores['test_roc_auc'].mean()
  return_dict['cross_validate']['Accuracy mean'] = scores['test_accuracy'].mean()
  return_dict['cross_validate']['Precision mean'] = scores['test_precision'].mean()
  return_dict['cross_validate']['Recall mean'] = scores['test_recall'].mean()
  return_dict['cross_validate']['F1 mean'] = scores['test_f1'].mean()
  ref_to_json_file(return_dict, xgb_json_dir + output_stem + '.xgboost.json')
  xgb_obj = open(xgb_result_obj_dir + output_stem + '.xgboost.obj', 'wb')
  pickle.dump(xg_clf, xgb_obj)
  X_test_obj = open(xgb_x_test_dir + output_stem + '.xgboost.X_test.obj', 'wb')
  pickle.dump(X_test, X_test_obj)
  return 0

tmp = xgbclassifier_wrapper(args.file, args.categorical, args.target, args.output_stem, args.colsample_bytree, args.gamma, args.learning_rate, args.max_depth)
