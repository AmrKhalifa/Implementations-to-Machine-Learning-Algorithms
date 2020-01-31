import torch
import torch.nn as nn 

info = "hello"
vocab = list(set([char for char in info]))
word2ind = {w:i for i, w in enumerate(vocab)}
words = torch.eye(len(word2ind))

embedding = nn.Embedding.from_pretrained(torch.tensor(words))

x = embedding(torch.tensor([[1, 2, 3], [1, 2, 3]]))
#x = x.unsqueeze(dim =0)
print(x.shape)
cell = nn.RNN(input_size = 4, hidden_size = 2, num_layers = 1)
output, hidden = cell(x)

print(output, output.shape)
print(hidden, hidden.shape)