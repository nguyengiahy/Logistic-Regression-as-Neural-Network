import numpy as np
import h5py
import copy

def load_dataset():
	train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
	train_x = np.array(train_dataset["train_set_x"][:])
	train_y = np.array(train_dataset["train_set_y"][:])

	test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
	test_x = np.array(test_dataset["test_set_x"][:])
	test_y = np.array(test_dataset["test_set_y"][:])

	classes = np.array(test_dataset["list_classes"][:])

	train_y = train_y.reshape((1, train_y.shape[0]))
	test_y = test_y.reshape((1, test_y.shape[0]))

	return train_x, train_y, test_x, test_y, classes

def sigmoid(z):
	'''
	Compute the sigmoid of z

	Arguments: 
	z -- A scalar or numpy array of any size

	Returns:
	s -- Sigmoid(z)
	'''
	s = 1/(1+np.exp(-z))
	return s

def initialize_with_zeros(dim):
	'''
	Initialize 'w' and 'b' to be zeros

	Arguments:
	dim -- size of w 

	Returns:
	w -- initialized vector of shape (dim, 1)
	b -- initialized scalar of type float
	'''
	w = np.zeros((dim, 1))
	b = 0.0

	return w, b

def propagate(w, b, X, Y):
	'''
	Implement cost function and its gradient for the propagation

	Arguments:
	w -- weights, a numpy array of size (num_px * num_px * 3, 1)
	b -- bias, a scalar
	X -- input data, a numpy array of size (num_px * num_px * 3, number of examples)
	Y -- true "label" vector of size (1, number of examples)

	Returns:
	cost -- negative log-likelihood cost for logistic regression
	dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
	'''
	m = X.shape[1]
	# Forward propagation
	z = np.dot(w.T, X) + b
	A = sigmoid(z)
	cost = -1/m * np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))
	# Backward propagation
	dw = 1/m * np.dot(X, (A-Y).T)
	db = 1/m * np.sum(A-Y)

	cost = np.squeeze(np.array(cost))

	grads = {"dw": dw, "db": db}

	return grads, cost

def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
	'''
	This function optimizes 'w' and 'b' with gradient descent

	Arguments:
	w -- weights, a numpy array of size (num_px * num_px * 3, 1)
	b -- bias, a scalar
	X -- input data, a numpy array of size (num_px * num_px * 3, number of examples)
	Y -- true "label" vector of size (1, number of examples)
	num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

	Returns:
	params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
	'''
	w = copy.deepcopy(w)
	b = copy.deepcopy(b)
	costs = []

	for i in range(num_iterations):
		grads, cost = propagate(w, b, X, Y)
		dw = grads["dw"]
		db = grads["db"]
		# Gradient descent
		w = w - learning_rate * dw
		b = b - learning_rate * db

		# Record the costs
		if i % 100 == 0:
			costs.append(cost)
			if print_cost:
				print("Cost after iteration {}: {}".format(i, cost))

	params = {"w": w, "b": b}
	grads = {"dw": dw, "db": db}

	return params, grads, costs

def predict(w, b, X):
	'''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    A -- a numpy array (vector) of size (1, number of examples) containing all predictions (0/1)
    '''
	z = np.dot(w.T, X) + b
	A = sigmoid(z)

	for i in range(A.shape[1]):
		if A[0][i] > 0.5:
			A[0][i] = 1
		else:
			A[0][i] = 0

	return A	