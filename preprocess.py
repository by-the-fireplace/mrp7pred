"""
Data preprocessing

Input: A csv file with three columns: 
	name, SMILES, label

1. Drop nulls
2. Remove duplicates
3. Feature engineering
4. Randomly split train/test

Output: pandas dataframe as pickle files:
	X_train.pkl, X_test.pkl, y_train.pkl, y_test.pkl
Columns:
	X -> name, [features]
	y -> name, label
"""

"__author__" = "Jingquan Wang"
"__email__" = "jq.wang1214@gmail.com"

