"""
file: prob2_fit.py
language: python3
author: Anurag Kallurwar, ak6491@rit.edu
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
CSCI 635: Introduction to Machine Learning
Problem 2: Polynomial Regression &

@author/lecturer - Alexander G. Ororbia II

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

# NOTE: you will need to tinker with the meta-parameters below yourself
#       (do not think of them as defaults by any means)
# meta-parameters for program
trial_name = 'p1_fit' # will add a unique sub-string to output of this program
degree = 15 # p, order of model
beta = 0.001 # regularization coefficient
alpha = 0.9 # step size coefficient
eps = 0.000001 # controls convergence criterion
n_epoch = 500000 # number of epochs (full passes through the dataset)
output_dir = "out/"

# begin simulation

def map_features(X):
	"""
	This function creates a feature map for the attribute in X
	:param X: Input dataset
	:return: Data with the corresponding feature map
	"""
	X_new = []
	for point in X:
		point_new = []
		for p in range(1, degree + 1):
			point_new.append(point[0] ** p)
		X_new.append(point_new)
	return np.array(X_new)

def regress(X, theta):
    ############################################################################
	# WRITEME: write your code here to complete the routine
	return X @ theta[1].T + theta[0]
    ############################################################################

def gaussian_log_likelihood(mu, y):
    ############################################################################
	# WRITEME: write your code here to complete the routine
	return (np.sum((mu - y) ** 2)) / (2 * y.shape[0])
    ############################################################################

def computeCost(X, y, theta, beta): ## loss is now Bernoulli cross-entropy/log likelihood
    ############################################################################
	# WRITEME: write your code here to complete the routine
	return gaussian_log_likelihood(regress(X, theta), y) + (beta * np.sum(
		theta[1] ** 2) / (2 * y.shape[0]))
    ############################################################################

def regularize_loss(theta):
	"""
	Calculates the regularized loss values
	:param theta:
	:return:
	"""
	return (beta * theta / y.shape[0])

def computeGrad(X, y, theta, beta):
    ############################################################################
	# WRITEME: write your code here to complete the routine (
	# NOTE: you do not have to use the partial derivative symbols below, they are there to guide your thinking)
	dL_dfy = None # derivative w.r.t. to model output units (fy)
	y_pred = regress(X, theta)
	dL_db = (1 / y.shape[0]) * (np.sum(y_pred - y))  # derivative w.r.t. model bias b
	dL_dw = (1 / y.shape[0]) * ((y_pred - y).T @ X) + regularize_loss(theta[1]) # derivative w.r.t model weights w
	nabla = (dL_db, dL_dw) # nabla represents the full gradient
	return nabla
    ############################################################################

path = os.getcwd() + '/data/prob2.dat'
data = pd.read_csv(path, header=None, names=['X', 'Y'])

# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

# convert from data frames to numpy matrices
X = np.array(X.values)
y = np.array(y.values)

############################################################################
# apply feature map to input features x1
# WRITEME: write code to turn X_feat into a polynomial feature map (hint: you
#          could use a loop and array concatenation)
X = map_features(X)
############################################################################

# convert to numpy arrays and initalize the parameter array theta
w = np.zeros((1,X.shape[1]))
b = np.array([0])
theta = (b, w)

print("=================================================================")
L = computeCost(X, y, theta, beta)
L_prev = L
halt = 0 # halting variable (you can use these to terminate the loop if you have converged)
print("-1 L = {0}".format(L))
i = 0
while(i < n_epoch and halt == 0):
	dL_db, dL_dw = computeGrad(X, y, theta, beta)
	b = theta[0]
	w = theta[1]
    ############################################################################
	# update rules go here...
	# WRITEME: write your code here to perform a step of gradient descent & record anything else desired for later
	b = b - (alpha * dL_db)
	w = w - (alpha * dL_dw)
	theta = (b, w)
    ############################################################################

	L = computeCost(X, y, theta, beta)

    ############################################################################
	# WRITEME: write code to perform a check for convergence (or simply to halt early)
	if abs(L - L_prev) < eps:
		halt = 1
	L_prev = L
    ############################################################################

	print(" {0} L = {1}".format(i,L))
	i += 1
# print parameter values found after the search
print("w = ",w)
print("b = ",b)

kludge = 0.25
# visualize the fit against the data
X_test = np.linspace(data.X.min(), data.X.max(), 100)
X_feat = np.expand_dims(X_test, axis=1) # we need this otherwise, the dimension is missing (turns shape(value,) to shape(value,value))

############################################################################
# apply feature map to input features x1
# WRITEME: write code to turn X_feat into a polynomial feature map (hint: you
#          could use a loop and array concatenation)
X_feat = map_features(X_feat)
############################################################################

print("\n=================================================================")
print("Plotting scatter plot! Close it to continue...")
plt.plot(X_test, regress(X_feat, theta), label="Model")
plt.scatter(X[:,0], y, edgecolor='g', s=20, label="Samples")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim((np.amin(X_test) - kludge, np.amax(X_test) + kludge))
plt.ylim((np.amin(y) - kludge, np.amax(y) + kludge))
plt.legend(loc="best")
plt.suptitle("Scatter Plot")
plt.title("Polynomial Regression, Order = " + str(degree) + ", beta = " +
		  str(beta))
############################################################################
# WRITEME: write your code here to save plot to disk (look up documentation or
#          the inter-webs for matplotlib)
plt.savefig(output_dir + "prob2_model_degree_" + str(degree))
############################################################################

plt.show()
