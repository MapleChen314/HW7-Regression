"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
from regression import LogisticRegressor
from regression import utils
from regression import logreg
from sklearn.preprocessing import StandardScaler
import numpy as np
# (you will probably need to import more things here)


def test_prediction():

	X_train, X_val, y_train, y_val = utils.loadDataset(
		features=[
			'Penicillin V Potassium 500 MG',
			'Computed tomography of chest and abdomen',
			'Plain chest X-ray (procedure)',
			'Low Density Lipoprotein Cholesterol',
			'Creatinine',
			'AGE_DIAGNOSIS'
		],
		split_percent=0.8,
		split_seed=1
	)

	# Scale the data, since values vary across feature. Note that we
	# fit on the training data and use the same scaler for X_val.
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_val = sc.transform(X_val)
	# For testing purposes, once you've added your code.
	# CAUTION: hyperparameters have not been optimized.
	lr = logreg.LogisticRegressor(num_feats=6, learning_rate=0.001, tol=0.01, max_iter=100, batch_size=10)
	lr.train_model(X_train, y_train, X_val, y_val)
	X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
	X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
	y_hat=lr.make_prediction(X_val)
	assert y_hat[5]<1
	assert y_hat[5]>0


def test_loss_function():
	X_train, X_val, y_train, y_val = utils.loadDataset(
		features=[
			'Penicillin V Potassium 500 MG',
			'Computed tomography of chest and abdomen',
			'Plain chest X-ray (procedure)',
			'Low Density Lipoprotein Cholesterol',
			'Creatinine',
			'AGE_DIAGNOSIS'
		],
		split_percent=0.8,
		split_seed=1
	)

	# Scale the data, since values vary across feature. Note that we
	# fit on the training data and use the same scaler for X_val.
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_val = sc.transform(X_val)

	lr = logreg.LogisticRegressor(num_feats=6, learning_rate=0.001, tol=0.01, max_iter=100, batch_size=10)
	lr.train_model(X_train, y_train, X_val, y_val)
	X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
	X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
	y_hat=lr.make_prediction(X_val)
	test_loss=lr.loss_function(y_val,y_hat)
	assert test_loss > 0


def test_gradient():
	X_train, X_val, y_train, y_val = utils.loadDataset(
		features=[
			'Penicillin V Potassium 500 MG',
			'Computed tomography of chest and abdomen',
			'Plain chest X-ray (procedure)',
			'Low Density Lipoprotein Cholesterol',
			'Creatinine',
			'AGE_DIAGNOSIS'
		],
		split_percent=0.8,
		split_seed=1
	)

	# Scale the data, since values vary across feature. Note that we
	# fit on the training data and use the same scaler for X_val.
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_val = sc.transform(X_val)
	
	lr = logreg.LogisticRegressor(num_feats=6, learning_rate=0.001, tol=0.01, max_iter=100, batch_size=10)
	lr.train_model(X_train, y_train, X_val, y_val)
	X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
	X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
	test_grad=lr.calculate_gradient( y_val, X_val)
	assert abs(test_grad[0])<1

def test_training():
	X_train, X_val, y_train, y_val = utils.loadDataset(
		features=[
			'Penicillin V Potassium 500 MG',
			'Computed tomography of chest and abdomen',
			'Plain chest X-ray (procedure)',
			'Low Density Lipoprotein Cholesterol',
			'Creatinine',
			'AGE_DIAGNOSIS'
		],
		split_percent=0.8,
		split_seed=1
	)
	# Scale the data, since values vary across feature. Note that we
	# fit on the training data and use the same scaler for X_val.
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_val = sc.transform(X_val)
	# For testing purposes, once you've added your code.
	# CAUTION: hyperparameters have not been optimized.
	lr = logreg.LogisticRegressor(num_feats=6, learning_rate=0.001, tol=0.01, max_iter=100, batch_size=10)
	W_before=lr.W
	lr.train_model(X_train, y_train, X_val, y_val)
	W_after=lr.W
	assert (W_before != W_after).any