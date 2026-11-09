
import torch
import torch.nn as nn
import torch.nn.functional as F

class  MLP(nn.Module):
	"""
	Network / Multilayer perseptron

	"""
	def __init__(self, n_units, n_in, n_out):
		"""
		Construct multilayer perceptron

		Parameters
		----------
		n_units	: int
			dimension of hidden layer
		n_in	: int
			dimension of input
		n_out	: int
			dimension of output
		"""
		super(MLP, self).__init__()

		self.fc1 = nn.Linear(n_in, n_units)	# n_in -> n_units
		self.fc2 = nn.Linear(n_units, n_units)	# n_units -> n_units
		self.fc3 = nn.Linear(n_units, n_out)	# n_units -> n_out

	def forward(self, x):
		"""
		Calculate forward propagation

		Parameters
		----------
		x	: torch.Tensor
			input to the network
			batchsize * input_dimension

		Returns
		-------
		y	: torch.Tensor
			output from the network
			batchsize * outout_dimension
		"""
		h = F.relu(self.fc1(x))
		h = F.relu(self.fc2(h))
		y = self.fc3(h)
		return y

class  MnistCNN(nn.Module):
	"""
	Network / Convolutional newral network for Mnist task
	"""
	def __init__(self, n_out):
		"""
		Construct CNN

		Parameters
		----------
		n_out	: int
			dimension of output
		"""
		super(MnistCNN, self).__init__()
		# Search or See https://pytorch.org/docs/stable/nn.html for information
		self.conv1 = nn.Conv2d(1,5,5,stride = 2,padding = 2)	# 1 * 28 * 28 -> 5 * 14 * 14
		self.conv2 = nn.Conv2d(5,10,5,stride = 2,padding = 2)	# 5 * 14 * 14 -> 10 * 7 * 7
		self.fc = nn.Linear(10 * 7 * 7, n_out)	# 490(=10*7*7) -> n_out

	def forward(self, x):
		"""
		Calculate forward propagation

		Parameters
		----------
		x	: torch.Tensor
			input to the network
			batchsize * channel * height * width

		Returns
		-------
		y	: torch.Tensor
			output from the network
			batchsize * outout_dimension
		"""
		h = F.relu(self.conv1(x))
		h = F.relu(self.conv2(h))
		h = h.view(-1, 10 * 7 * 7)
		y = self.fc(h)
		return y

class   CifarCNN(nn.Module):
	"""
	Network / Convolutional newral network for Cifar10 task
	"""
	def __init__(self, n_out, ch1=6, ch2=16):
		"""
		Construct CNN

		Parameters
		----------
		n_out	: int
			dimension of output
		ch1	: int
			number of output channels of conv1
		ch2	: int
			number of output channels of conv2
		"""
		super(CifarCNN, self).__init__()

		self.ch2 = ch2
		self.conv1 = nn.Conv2d(3, ch1, 5, stride=1)	# 3 * 32 * 32 -> ch1 * 28 * 28
		self.pool = nn.MaxPool2d(2, 2)	# ch1 * 28 * 28 -> ch1 * 14 * 14
		self.conv2 = nn.Conv2d(ch1, ch2, 5, stride=1)	# ch1 * 14 * 14 -> ch2 * 10 * 10
		#self.pool = nn.MaxPool2d(2, 2)	# ch2 * 10 * 10 -> ch2 * 5 * 5
		self.fc = nn.Linear(ch2 * 5 * 5, n_out)	# ch2 * 5 * 5 -> n_out

	def forward(self, x):
		"""
		Calculate forward propagation

		Parameters
		----------
		x	: torch.Tensor
			input to the network
			batchsize * channel * height * width

		Returns
		-------
		y	: torch.Tensor
			output from the network
			batchsize * outout_dimension
		"""
		h = self.pool(F.relu(self.conv1(x)))
		h = self.pool(F.relu(self.conv2(h)))
		h = h.view(-1, self.ch2 * 5 * 5)
		y = self.fc(h)
		return y
