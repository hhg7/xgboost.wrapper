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

parser.add_argument('--learning_rate', default = 0.6, required = False) # range [0,1]
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
from sklearn.metrics import mean_absolute_error, mean_squared_error
#from sklearn.metrics import accuracy_score, mean_squared_error

def ref_to_json_file(data, filename):
	json1=json.dumps(data)
	f = open(filename,"w+")
	print(json1,file=f)

xgb_dir = args.output_stem + '.xgb.reg.output/'
if not os.path.isdir(xgb_dir):
	os.mkdir(xgb_dir)
xgb_x_test_dir = xgb_dir + 'X_test/'
xgb_json_dir   = xgb_dir + 'json/'
xgb_result_obj_dir = xgb_dir + 'xg_rgs/'
image_dir      = xgb_dir + 'Images/'
for directory in xgb_x_test_dir, xgb_json_dir, xgb_result_obj_dir, image_dir:
	if not os.path.isdir(directory):
		os.mkdir(directory)

def xgb_regressor_wrapper( input_file, category_cols, dependent_var, output_stem, colsample_bytree, gamma, learning_rate, max_depth):
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
  if args.categorical != None:
	  for category in args.categorical:
  		pandasDF = pd.get_dummies(pandasDF, prefix = [category], columns = [category], drop_first = True)
  Y = pandasDF[dependent_var]
  print(len(Y))
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
    xg_rgs = xgboost.XGBRegressor( n_estimators=n_estimators, colsample_bytree=colsample_bytree, learning_rate=learning_rate, max_depth=max_depth, gamma = gamma, eval_metric = 'logloss')
    xg_rgs.fit(X_train, y_train)
    # Use the model to make predictions on the test dataset.
    predictions = xg_rgs.predict(X_test)
  #accuracy = accuracy_score(y_test, predictions)
  #pre_score  = precision_score(y_test, predictions)
  return_dict = {}
  return_dict['MAE'] = mean_absolute_error( y_test, predictions )
  return_dict['Feature Importance'] = {} # python3 requires declaring
  for importance_type in ('weight', 'gain', 'cover', 'total_gain', 'total_cover'):
    return_dict['Feature Importance'][importance_type] = xg_rgs.get_booster().get_score( importance_type = importance_type)
  kfold = KFold(n_splits=10)
  #return_dict['mean absolute error'] = results.mean()
  ref_to_json_file(return_dict, xgb_json_dir + output_stem + '.xgboost.json')
  xgb_obj = open(xgb_result_obj_dir + output_stem + '.xgboost.obj', 'wb')
  pickle.dump(xg_rgs, xgb_obj)
  X_test_obj = open(xgb_x_test_dir + output_stem + '.xgboost.X_test.obj', 'wb')
  pickle.dump(X_test, X_test_obj)
  return 0

tmp = xgb_regressor_wrapper(args.file, args.categorical, args.target, args.output_stem, args.colsample_bytree, args.gamma, args.learning_rate, args.max_depth)
