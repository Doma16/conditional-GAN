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

img_dict = {}

for i in range(20):

    with torch.no_grad():
        noise = torch.randn((1,VECT_DIM,1,1)).to(device)
        label = torch.randint(0,10,(1,)).to(device)

        img = gen(noise,label)
        # img = img / torch.norm(img)
        # img = torch.sigmoid(img)

        maxs = torch.max(img)
        mins = torch.min(img)
        img = (img - mins) / (maxs-mins)
        img = img * 255
        img = img.cpu().detach().numpy().squeeze()
        img = np.uint8(img)

        label = label.cpu().detach().numpy()

        img_dict[label[0]] = img
        '''
        im = Image.fromarray(img)
        im.show(title=f'{label}')
        print(f'{label[0]}')
        input(f"Press enter to continue: ")
        '''


breakpoint()
pass