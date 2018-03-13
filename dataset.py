import torch
from torchvision import transforms
import torchvision.datasets as thedataset
from torch.utils.data import Dataset
from PIL import Image

class MNISTLoader(Dataset):

    def __init__(self,root='./data',download=True):
        trans = transforms.Compose([transforms.ToTensor()])
        self.hr_images = thedataset.MNIST(root = root,train=True,transform=trans,download=True)

    def __getitem__(self,idx):
        hr_image = self.hr_images[idx][0]

        ToPILFunction = transforms.ToPILImage()
        lr_image = ToPILFunction(hr_image)
        lr_image = lr_image.resize((7,7), Image.BICUBIC)
        lr_image = lr_image

        ToTensorFunction = transforms.ToTensor()
        lr_image = ToTensorFunction(lr_image)

        return lr_image,hr_image

    def __len__(self):
        return len(self.hr_images)	

