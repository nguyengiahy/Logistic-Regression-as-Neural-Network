from utils import load_dataset, sigmoid, initialize_with_zeros, propagate, optimize, predict
import matplotlib.pyplot as plt
import numpy as np

# Load data
train_x, train_y, test_x, test_y, classes = load_dataset()
'''
All data is stored as numpy array. Data shape:
- train_x = (209, 64, 64, 3) - - Train set has 209 images, each image is 64x64 pixels with each pixel is a 3 RGB value
- train_y = (1, 209)
- test_x = (50, 64, 64, 3) 
- test_y = (1, 50)
'''

# Show an example input image
# index = 25
# plt.imshow(train_x[index])
# print("y = " + str(train_y[0][index]) + ", it's a '" + classes[train_y[0][index]].decode("utf-8") + "' picture.")

# Extract dimension values
m_train = train_x.shape[0]		# 209
m_test = test_x.shape[0]		# 50
num_px = train_x.shape[1]		# 64

# Reshape data
train_x = train_x.reshape(train_x.shape[0], -1).T   # shape = (64*64*3, 209)
test_x = test_x.reshape(test_x.shape[0], -1).T 		# shape = (64*64*3, 50)

# Standardize the dataset
train_x = train_x / 255.
test_x = test_x / 255.

# Hyperparameters
num_iterations = 2000
learning_rate = 0.005
print_cost = True

'''
Build the logistic regression model
'''

# Initialize parameters with zeros 
w, b = initialize_with_zeros(train_x.shape[0])
# Run gradient descent
params, grads, costs = optimize(w, b, train_x, train_y, num_iterations, learning_rate, print_cost)
# Retrieve the trained parameters w and b
w, b = params["w"], params["b"]
# Predict
Y_prediction_train = predict(w, b, train_x)
Y_prediction_test = predict(w, b, test_x)

# Print train/test errors
if print_cost:
	print("train accuracy: {}%".format(100 - np.mean(np.abs(Y_prediction_train - train_y)) * 100))
	print("test accuracy: {}%".format(100 - np.mean(np.abs(Y_prediction_test - test_y)) * 100))

# Predict an image
index = 1
plt.imshow(test_x[:, index].reshape((num_px, num_px, 3)))
print("y = " + str(test_y[0][index]) + ", the model predicted this image has a label of y = " + str(int(Y_prediction_test[0][index])))
plt.show()