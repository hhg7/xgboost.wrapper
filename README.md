These two files are wrappers to run XGBoost's regressor and classifier.  They take command line options to python 

`python xgb.regressor.py --file file.csv --target 'target.column.in.csv' --categorical 'categorical.column' --output_stem stem`

`python 0.xgboost.py --file hcvdat0.clean.csv --output_stem hcv --target Category_1=Hepatitis --categorical Category Sex`

the scripts should be self-explanatory.  They just make using XGBoost easier for me, there's not much to them.
