import numpy as np 

in_features = 2
out_features = 3 
hidden_size = 10

in_to_hid = np.randn((hidden_size, in_features))/1000
hid_to_out = np.randn((out_features, hidden_size))/1000

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
	h = sigmoid(h)
	output = hid_to_out.T @ h
	output = softmax(output)
	return output