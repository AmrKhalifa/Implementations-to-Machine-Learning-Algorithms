import numpy as np 
from rnn_data import train_data, test_data


class RNN():

	def __init__(self, train_data, hidden_size, num_classes):
		
		self.vocab = list(set([word for example in train_data.keys() for word in example.split(" ")]))
		self.train_data = list([(example, label) for example, label in train_data.items()])
		self.labels = [item[1] for item in train_data.items()]
		self.vocab_size = len(self.vocab)

		self.word2ind = {word:index for word, index in zip(self.vocab, range(self.vocab_size))}
		self.ind2word = {index:word for word, index in self.word2ind.items()}

		self.label2ind = {label : 0 if label == False else 1 for label in self.labels}

		self.vocab_size = self.vocab_size 
		self.input_size = self.vocab_size
		self.hidden_size = hidden_size
		self.num_classes = num_classes


		self.Whx = np.random.normal(0, .01, size = (self.hidden_size, self.input_size))
		self.Whh = np.random.normal(0, .01, size = (self.hidden_size, self.hidden_size))
		self.Why = np.random.normal(0, .01, size = (self.num_classes, self.hidden_size))

		self.bh = np.random.normal(0, .01, size = (self.hidden_size)).reshape(-1, 1)
		self.by = np.random.normal(0, .01, size = (self.num_classes)).reshape(-1, 1)
		self.ih = np.random.normal(0, .01, size = (self.hidden_size)).reshape(-1, 1)
		

	def vectorize(self, text, vocab_size):
	## returns a matrix whose columns are a 1-hot vector representation of the words in the text 
		mat = np.matrix(np.zeros((vocab_size, len(text))))
		for i, word in enumerate(text):
			vec = np.zeros(vocab_size).reshape(-1, 1)
			vec[self.word2ind[word]] = 1
			mat[:,i] = vec

		return mat 


	def softmax(self, x):
		## returns a column vector 
		x = x.reshape(-1,1)
		exponent = np.exp(x - np.max(x))
		return exponent/np.sum(exponent)


	def forward(self, example):
		mat = self.vectorize(example.split(" "), self.vocab_size)
		record = []
		h = self.ih
		record.append((h, mat[:,0].reshape(1, -1)))
		for x in mat.T:
			input_info = self.Whx @ x.reshape(-1, 1)
			recur_info = self.Whh @ h.reshape(-1,1)
			h = np.tanh(input_info + recur_info + self.bh)
			record.append((h, x))

		y = self.Why @ h.reshape(-1, 1) + self.by
		props = self.softmax(y)
		return props, record  

	def compute_loss_grad(self, props, label):

		target_idx = self.label2ind[label]
		loss = float(-np.log(props[target_idx]))
		target_vec = np.zeros((self.num_classes,1))
		target_vec[target_idx] = 1
		gradLoss_y = props - target_vec

		return loss, gradLoss_y 

	def backward(self, dL_dy, props, record, clip = False):

		dL_dy = dL_dy.reshape(-1, 1)
		rev_record = list(reversed(record))
		
		h_n = rev_record[0][0].reshape(1, -1)
		dL_dWhy = dL_dy @ h_n
		dL_dby = dL_dy

		dL_dWhx = np.zeros_like(self.Whx)
		dL_dWhh = np.zeros_like(self.Whh)
		dL_dbh = np.zeros_like(self.bh)
		
		dL_h = self.Why.T @ dL_dy

		for time_step in rev_record[1:]: 
			h_t, x_t = time_step
			h_t = np.array(h_t)
			comulative = np.multiply((1 - h_t ** 2) , dL_h)

			dL_dbh += comulative
			dL_dWhh += comulative @ h_t.T
			dL_dWhx += comulative @ x_t
			dL_h = self.Whh @ comulative


		if clip == True: 
			for grad in [dL_dWhy, dL_dby, dL_dWhx, dL_dWhh, dL_dbh]:
				np.clip(grad, -1, 1, out=grad)
			
		return dL_dWhy, dL_dby, dL_dWhx, dL_dWhh, dL_dbh
	

	def train(self, train_set = None, n_epochs  = 10, lr = 1e-3): 
		
		for epoch in range(n_epochs):

			for example, label in self.train_data: 
				props, record = self.forward(example)
				
				Loss, dL_dy = self.compute_loss_grad(props, label)
				print("epoch: ",epoch, "The loss is: ", Loss)
				dL_dWhy, dL_dby, dL_dWhx, dL_dWhh, dL_dbh = self.backward(dL_dy, props, record, clip = True)

				self.Why -= lr * dL_dWhy
				self.Whx -= lr * dL_dWhx
				self.Whh -= lr * dL_dWhh 

				self.bh -= lr * dL_dbh
				self.by -= lr * dL_dby 





rnn = RNN(train_data, 10, 2)
rnn.train(n_epochs = 1000)



