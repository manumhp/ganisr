import torch
import argparse,os
import random,math
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import GenNet

from dataset import MNISTLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# need to learn this dataset thing

from torchvision import models
import torch.utils.model_zoo as model_zoo


parser = argparse.ArgumentParser(description="PyTorch SRGAN")
parser.add_argument("--batchSize", type=int, default=16, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=500, help="")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. 1e-4")
parser.add_argument("--step", type=int, default=200, help="Sets the learning rate to initial")
parser.add_argument("--cuda", action="store_true", help="Use Cuda ?")
#parser.add_argument("--resume")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number")
#parser.add_argument("--threads", type="int", default=1, help="The number of threads for data loader to use ")
def main():

	global opt, model
	opt = parser.parse_args()
	print (opt)

	loss_list = []

	cuda = opt.cuda
	if cuda and not torch.cuda.is_available():
		raise Exception("No GPU found, please run without cuda")

	cudnn.benchmark = True	

	print("--> Loading datasets")

	# include the path to the datasets
	train_set = MNISTLoader()
	training_data_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize, shuffle=True)

	print("--> Building the model")
	model = GenNet()
	criterion = nn.MSELoss(size_average=False)

	print("--> Setting GPU")
	if cuda:
		model = model.cuda()
		criterion = criterion.cuda()

	print("--> Setting Optimizer")	
	optimizer = optim.Adam(model.parameters(), lr=opt.lr)

	print("--> Training")
	for epoch in range(opt.start_epoch, opt.nEpochs + 1):
		current_loss = train(training_data_loader, optimizer, model, criterion, epoch)
		#give save checkpoint command 
		loss_list.append(current_loss)
		save_plot(loss_list)


def adjust_learning_rate(optimizer, epoch):
	lr = opt.lr * (0.1 ** (epoch/opt.step))
	return lr


def train(training_data_loader, optimizer, model, criterion, epoch):

	lr = adjust_learning_rate(optimizer, epoch-1)

	for param_group in optimizer.param_groups:
		param_group["lr"] = lr

	print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
	
	model.train()

	for iteration, batch in enumerate(training_data_loader, 1):

		input, target = Variable(batch[0]), Variable(batch[1], requires_grad = False)

		if opt.cuda:
			input = input.cuda()
			target = target.cuda()

		output = model(input)
		loss = criterion(output, target)

		optimizer.zero_grad()

		loss.backward()

		optimizer.step()

		if iteration%10 == 0:
			print("--> Epoch[{}]({}/{}): Loss: {:.5} Content_loss {:.5}".format(epoch, iteration, len(training_data_loader), loss.data[0], content_loss.data[0]))

		return loss.data[0]


def save_plot(loss_list):
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.plot(loss_list)	
	plt.savefig('/output/plot.png')

if __name__ == "__main__":
	main()