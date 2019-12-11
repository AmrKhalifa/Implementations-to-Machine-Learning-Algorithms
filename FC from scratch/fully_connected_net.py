import numpy as np 


def sigmoid(x):
	return 1/(1+np.exp(-x))

def d_sigmoid(x):
	return sigmoid(x) * sigmoid(-x)
