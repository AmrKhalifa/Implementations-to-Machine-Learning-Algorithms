import numpy as np 
import matplotlib.pyplot as plt 

in_features = 2
out_features = 2 
hidden_size = 10

in_to_hid = (np.random.randn(hidden_size, in_features))/1000
hid_to_out = (np.random.randn(out_features, hidden_size))/1000
b1 = (np.random.randn(hidden_size, 1))/1000
b2 = (np.random.randn(out_features, 1))/1000

def sigmoid(x):
	return 1/(1+np.exp(-x))

def softmax(x):
	x = x.T
	maxx = np.max(x, axis =0)
	x = x - maxx
	exponent = np.exp(x)
	return exponent/np.sum(exponent)	

def d_sigmoid(x):
	return sigmoid(x) * sigmoid(-x)

def forward(x):
	h = in_to_hid @ x
	h += b1
	h = sigmoid(h)

	logits = hid_to_out @ h
	logits += b2 
	output = softmax(logits)
	return x, h, output

def get_loss(input, target):

	p = target
	print(input)
	q = np.argmax(input, axis  =1)
	print(p == q)
	CE = -np.dot(q, np.log(p))
	return CE 

def backward(x, h, output, target):
	# working the gradients by hand

	d1 = (output - target)
	d2 = d1.T @ hid_to_out @ d_sigmoid(h)

	dw2 = h @ d1.T
	db2 = d1.T
	dw1 = d2.T @ x.T
	db1 = d2. T 

	pass 


x_1 = np.random.multivariate_normal(mean = [1,1], cov = np.eye(2), size = 200).T
y_1 = [1]*200

x_2 = np.random.multivariate_normal(mean = [-1,1], cov = np.eye(2), size = 200).T
y_2 = [0]*200

# plt.scatter(x_1[0,:], x_1[1,:], color = 'r')
# plt.scatter(x_2[0,:], x_2[1,:], color = 'c')
# plt.show()


x, h, output = forward(x_1)

print(get_loss(output, y_1))
#backward(x, h, output, y_1)