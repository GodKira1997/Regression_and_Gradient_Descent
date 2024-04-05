"""
file: prob1_fit.py
language: python3
author: Anurag Kallurwar, ak6491@rit.edu
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
CSCI 635: Introduction to Machine Learning
Problem 1: Univariate Regression

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

# NOTE: you will need to tinker with the meta-parameters below yourself (do not think of them as defaults by any means)
# meta-parameters for program
alpha = 0.02 # step size coefficient
eps = 0.00000 # controls convergence criterion
n_epoch = 2000 # number of epochs (full passes through the dataset)
output_dir = "out/"

# begin simulation

def regress(X, theta):
    ############################################################################
	# WRITEME: write your code here to complete the routine
	return X @ theta[1].T + theta[0]
    ############################################################################

def gaussian_log_likelihood(mu, y):
    ############################################################################
	# WRITEME: write your code here to complete the sub-routine
	return (np.sum((mu - y) ** 2)) / (2 * y.shape[0])
    ############################################################################

def computeCost(X, y, theta): # loss is now Bernoulli cross-entropy/log likelihood
    ############################################################################
	# WRITEME: write your code here to complete the routine
	return gaussian_log_likelihood(regress(X, theta), y)
    ############################################################################

def computeGrad(X, y, theta):
    ############################################################################
	# WRITEME: write your code here to complete the routine
	# NOTE: you do not have to use the partial derivative symbols below, they are there to guide your thinking)
	dL_dfy = None # derivative w.r.t. to model output units (fy)
	y_pred = regress(X, theta)
	dL_db = (1 / y.shape[0]) * (np.sum(y_pred - y)) # derivative w.r.t. model bias b
	dL_dw = (1 / y.shape[0]) * ((y_pred - y).T @ X) # derivative w.r.t model weights w
	nabla = (dL_db, dL_dw) # nabla represents the full gradient
	return nabla
    ############################################################################

path = os.getcwd() + '/data/prob1.dat'
data = pd.read_csv(path, header=None, names=['X', 'Y'])

# display some information about the dataset itself here
############################################################################
# WRITEME: write your code here to print out information/statistics about the
#          data-set "data" using Pandas (consult the Pandas documentation to learn how)
print("=================================================================")
print("Statistical information of the dataset:")
print(data.info())
print("===================================")
print(data.describe())

# WRITEME: write your code here to create a simple scatterplot of the dataset
#          itself and print/save to disk the result
print("\n=================================================================")
print("Plotting Scatter plot! Close it to continue...")
kludge = 0.25
data.plot.scatter(x='X', y='Y', edgecolor='b', s=20, label="Samples")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim((np.amin(data['X']) - kludge, np.amax(data['X']) + kludge))
plt.ylim((np.amin(data['Y']) - kludge, np.amax(data['Y']) + kludge))
plt.title("Scatter Plot - Food Truck Dataset")
plt.legend(loc="best")
plt.savefig(output_dir + "prob1_scatterplot")
plt.show()
############################################################################

# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

# convert from data frames to numpy matrices
X = np.array(X.values)
y = np.array(y.values)

# convert to numpy arrays and initalize the parameter array theta
w = np.zeros((1,X.shape[1]))
b = np.array([0])
theta = (b, w)

print("\n=================================================================")
L = computeCost(X, y, theta)
print("-1 L = {0}".format(L))
L_best = L
halt = 0 # halting variable (you can use these to terminate the loop if you have converged)
i = 0
cost = [] # you can use this list variable to help you create the loss versus epoch plot at the end (if you want)
while(i < n_epoch and halt == 0):
	dL_db, dL_dw = computeGrad(X, y, theta)
	b = theta[0]
	w = theta[1]
    ############################################################################
	# update rules go here...
	# WRITEME: write your code here to perform a step of gradient descent &
    #          record anything else desired for later
	b = b - (alpha * dL_db)
	w = w - (alpha * dL_dw)
	theta = (b, w)
    ############################################################################

	# (note: don't forget to override the theta variable...)
	L = computeCost(X, y, theta) # track our loss after performing a single step
	cost.append(L)

	print(" {0} L = {1}".format(i,L))
	i += 1
# print parameter values found after the search
print("w = ",w)
print("b = ",b)

kludge = 0.25 # helps with printing the plots (you can tweak this value if you like)
# visualize the fit against the data
X_test = np.linspace(data.X.min(), data.X.max(), 100)
X_test = np.expand_dims(X_test, axis=1)
print("\n=================================================================")
print("Plotting scatter plot! Close it to continue...")
plt.plot(X_test, regress(X_test, theta), label="Model")
plt.scatter(X[:,0], y, edgecolor='g', s=20, label="Samples")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim((np.amin(X_test) - kludge, np.amax(X_test) + kludge))
plt.ylim((np.amin(y) - kludge, np.amax(y) + kludge))
plt.legend(loc="best")
plt.suptitle("Scatter Plot")
plt.title("Univariate Linear Regression")
############################################################################
# WRITEME: write your code here to save plot to disk (look up documentation or
#          the inter-webs for matplotlib)
plt.savefig(output_dir + "prob1_model")
############################################################################
plt.show()

############################################################################
# visualize the loss as a function of passes through the dataset
# WRITEME: write your code here create and save a plot of loss versus epoch
print("\n=================================================================")
print("Plotting Loss function plot! Close it to continue...")
plt.plot(cost, label="Loss as a function", color = 'r')
plt.xlabel("Cost of model")
plt.ylabel("epochs")
plt.legend(loc="best")
plt.title("Plot of loss as a function")
plt.savefig(output_dir + "prob1_loss")
############################################################################

plt.show() # convenience command to force plots to pop up on desktop
