import torch
from torchvision import datasets, transforms

train_set = datasets.MNIST(root='../data/', train=True, download=False, 
	transform=transforms.Compose([
            transforms.ToTensor()
            ]))

test_set = datasets.MNIST(root='../data/', train=False, download=False, 
	transform=transforms.Compose([
            transforms.ToTensor()
            ]))

def main():
	pass 
	
if __name__ == "__main__":

	main()