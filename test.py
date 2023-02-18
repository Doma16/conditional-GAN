import torch
import torch.nn as nn
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

from model import Generator

PATH = './trained_models/gen'

NUM_CLASSES = 10
VECT_DIM = 64
EMB_DIM = 64
CHANNELS_IMG = 1
FEATURES_G = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gen = Generator(VECT_DIM,FEATURES_G,CHANNELS_IMG,NUM_CLASSES,EMB_DIM)
gen.load_state_dict(torch.load(PATH))
gen = gen.to(device)
gen.eval()

for i in range(20):

    with torch.no_grad():
        noise = torch.randn((1,VECT_DIM,1,1)).to(device)
        label = torch.randint(0,10,(1,)).to(device)

        img = gen(noise,label)
        img = img.cpu().detach().numpy().squeeze()

        print(f'{label.cpu().detach().numpy()}')
        plt.imshow(img)
        plt.show()

