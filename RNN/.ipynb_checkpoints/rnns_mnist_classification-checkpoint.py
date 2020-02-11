from data_processor import train_set, test_set
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt  
# torch.manual_seed(69)

batch_size = 5
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size) 

first_batch = next(iter(train_loader))

images, labels = first_batch


