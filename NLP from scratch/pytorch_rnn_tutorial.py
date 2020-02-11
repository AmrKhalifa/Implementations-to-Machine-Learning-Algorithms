from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import torch
import torch.nn as nn 
from torch.optim import Adam, SGD 
import random
import matplotlib.pyplot as plt 
import torch.nn.functional as F 
import numpy as np 

def findFiles(path): return glob.glob(path)

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

category_lines = {}
all_categories = []

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

cat2ind = {cat: ind for ind, cat in enumerate(all_categories)}

############################################

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    category_index = cat2ind[category]
    line = randomChoice(category_lines[category])
    return category, category_index, line

#####################################################################
################################ Computation graph

class NN(nn.Module):
	def __init__(self):
		super(NN, self).__init__() 
		self.in2hid = nn.Linear(in_features = 57, out_features = 120) 
		self.cell = nn.GRU(input_size = 120, hidden_size = 150)
		self.hid2out = nn.Linear(in_features = 150, out_features = len(all_categories))

	def forward(self, input, hidden):

		input = self.in2hid(input)
		input = F.relu(input)
		output, hidden = self.cell(input, hidden)
		output = F.relu(output)
		hidden = F.relu(hidden)

		output = self.hid2out(output)

		return output, hidden

model = NN()
optimizer = Adam(model.parameters(), lr =.001)
criterion = nn.CrossEntropyLoss()
losses = []

def train_step(): 
	cat, category_index, line = randomTrainingExample()
	hidden = torch.zeros((1, 1, 150))
	
	for char in line:
		charTensor = letterToTensor(char).unsqueeze(0)
		output, hidden = model(charTensor, hidden)

	output = output.reshape(1, len(all_categories))
	loss = criterion(output, torch.tensor(category_index).flatten())
	
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	print("loss after processing this example is: ", loss.item())
	losses.append(loss.item())

for i in range(20000):
	train_step()

def runningMeanFast(x, N):
    return np.convolve(x, np.ones((N,))/N)[(N-1):]

mean = runningMeanFast(losses, 100)
plt.plot(mean)
plt.show()