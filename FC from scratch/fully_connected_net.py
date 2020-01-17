import numpy as np 
import matplotlib.pyplot as plt 

in_features = 2
out_features = 3 
hidden_size = 10

in_to_hid = (np.random.randn(hidden_size, in_features))/1000
hid_to_out = (np.random.randn(out_features, hidden_size))/1000
b1 = (np.random.randn(hidden_size, 1))/1000
b2 = (np.random.randn(out_features, 1))/1000

def sigmoid(x):
	return 1/(1+np.exp(-x))

def softmax(x):
	maxx = np.max(h)
	x = x - maxx
	exponent = np.exp(x)
	return exponent/np.sum(exponent)	

def d_sigmoid(x):
	return sigmoid(x) * sigmoid(-x)

def forward(x):
	h = in_to_hid.T @ x
	h += b1
	h = sigmoid(h)

	output = hid_to_out.T @ h
	output += b2 
	output = softmax(output)
	return output

def get_loss(input, target):

	p = target
	q = input 
	CE = -np.dot(p, np.log(q))
	return CE 

def backward(loss):
	# working the gradients by hand  

	pass 


x_1 = np.random.multivariate_normal(mean = [1,1], cov = np.eye(2), size = 200).T
y_1 = [1]*200

x_2 = np.random.multivariate_normal(mean = [-1,1], cov = np.eye(2), size = 200).T
y_2 = [-1]*200

# plt.scatter(x_1[0,:], x_1[1,:], color = 'r')
# plt.scatter(x_2[0,:], x_2[1,:], color = 'c')
# plt.show()