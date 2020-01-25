import numpy as np 
import matplotlib.pyplot as plt 

class NN():

	def __init__(self): 
		self.in_features = 2
		self.out_features = 2 
		self.hidden_size = 10

		self.in_to_hid = (np.random.randn(self.hidden_size, self.in_features))/1000
		self.hid_to_out = (np.random.randn(self.out_features, self.hidden_size))/1000
		self.b1 = (np.random.randn(self.hidden_size, 1))/1000
		self.b2 = (np.random.randn(self.out_features, 1))/1000

		self.lr = .01

	def sigmoid(self, x):
		return 1/(1+np.exp(-x))

	def softmax(self, x):
		x = x
		maxx = np.max(x)
		x = x - maxx
		exponent = np.exp(x)
		output = exponent/np.sum(exponent)
		return 	output

	def d_sigmoid(self, x):
		return self.sigmoid(x) * self.sigmoid(-x)

	def forward(self, x):
		h = self.in_to_hid @ x
		h += self.b1
		h = self.sigmoid(h)

		logits = self.hid_to_out @ h
		logits += self.b2 
		output = self.softmax(logits)
		output = output
		return x, h, output

	def get_loss(self, input, target):

		p = target
		q = input
		CE = -np.sum(p*np.log(q))

		return CE 

	def backward(self, x, h, output, target):
		# working the gradients by hand
		
		lr = self.lr 
		d_1 = (output - target).T	
		d_2 = (d_1 @ self.hid_to_out)*self.d_sigmoid(h).reshape(1, -1)	
		dw2 = d_1.T @ h.T
		db2 = d_1.T
		dw1 = d_2.T@ x.T
		db1 = d_2.T

		self.in_to_hid -= lr*dw1
		self.hid_to_out -= lr*dw2
		self.b1 -= lr*db1
		self.b2 -= lr*db2 
		pass 


x_1 = np.random.multivariate_normal(mean = [1,1], cov = np.eye(2), size = 200)
x_1 = x_1.reshape(200, 2, 1)
y_1 = np.array([0, 1]*200).reshape(200,2, 1)

x_2 = np.random.multivariate_normal(mean = [-1,1], cov = np.eye(2), size = 200)
y_2 = np.array([1, 0]*200).reshape(-1, 2)

net = NN()

for x, y in zip(x_1, y_1):
	x, h, output = net.forward(x)
	print(net.get_loss(output, y)) 
	net.backward(x, h, output, y)