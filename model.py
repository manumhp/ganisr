import torch 
import torch.nn as nn
import math

class ResidualBlock(nn.Module):
    def __init__(self):

       super(ResidualBlock,self).__init__()

       self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
       self.in1 = nn.InstanceNorm2d(64, affine=True)
       self.relu = nn.LeakyReLU(0.2, inplace=True)
       self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)  
       self.in2 = nn.InstanceNorm2d(64, affine=True)

    def forward(self,x):
    	input_data = x
    	output1 = self.relu(self.in1(self.conv1(x)))
        output2 = self.in2(self.conv2(output1))
        output = torch.add(output2,input_data)
        return output

class GenNet(nn.Module):

    def __init__(self):

        super(GenNet,self).__init__()

        self.conv_input = nn.Conv2d(in_channels=1 ,out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)        
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.residual = self.make_layer(ResidualBlock,16)	

        self.conv_mid = nn.Conv2d(in_channels=64,out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(64, affine=True)

        self.upscale4x = nn.Sequential(
            
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        	)

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=9, stride=1,padding=4, bias=False)

    
    	# for m in self.modules():

    	# did not understand why the above block of code is required


    def make_layer(self,block,num_of_layers):
    	layers=[]
    	for a in range(num_of_layers):
    		layers.append(block())
    		return nn.Sequential(*layers)


    def forward(self,x):
        out = self.relu(self.conv_input(x))
        residual = out
        out = self.residual(out)
        out = self.bn_mid(self.conv_mid(out))
        out = torch.add(out,residual)
        out = self.upscale4x(out)
        out = self.conv_output(out)
        return out