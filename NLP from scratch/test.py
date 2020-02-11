import torch
import torch.nn as nn 

loss = nn.CrossEntropyLoss()
input = torch.randn(1, 5, requires_grad=True)
target = torch.empty(1, dtype=torch.long).random_(5)

print(input, input.shape)
print(target, target.shape)
z = torch.tensor(1).flatten()

output = loss(input, target)
output.backward()

