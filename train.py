import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from time import perf_counter

from model import Generator, Discriminator
from utils import my_penalty

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

#hyperparameters
LR = 1e-4
BATCH_SIZE = 64
IMG_SIZE = 32
CHANNELS_IMG = 1
VECT_DIM = 64
EMB_DIM = 64
EPOCHS = 5
FEATURES_DISC = 32
FEATURES_GEN = 64
DISC_ITERATIONS = 5
NUM_CLASSES = 10
LAMBDA = 5

transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    #transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)])
])

dataset = datasets.MNIST(root='../dataset/', train=True, transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

gen = Generator(VECT_DIM,FEATURES_GEN,CHANNELS_IMG,NUM_CLASSES,EMB_DIM)
disc = Discriminator(CHANNELS_IMG,FEATURES_DISC,IMG_SIZE,NUM_CLASSES)

gen = gen.to(device)
disc = disc.to(device)

# COULD INITIALIZE WEIGHTS

opt_gen = optim.Adam(gen.parameters(), lr=LR, betas=(0.0, 0.95))
opt_disc = optim.Adam(disc.parameters(), lr=LR, betas=(0.0,0.95))

writer_real = SummaryWriter(f'./logs/real')
writer_fake = SummaryWriter(f'./logs/fake')
step = 0

gen.train()
disc.train()

for epoch in range(EPOCHS):
    start = perf_counter()
    for batchid, (real,label) in enumerate(loader):
        
        real = real.to(device)
        label = label.view(-1,1)
        label = label.to(device)

        for _ in range(DISC_ITERATIONS):
            noise = torch.randn((BATCH_SIZE,VECT_DIM,1,1)).to(device)

            fake = gen(noise,label)

            disc_real = disc(real,label).reshape(-1)
            disc_fake = disc(fake,label).reshape(-1)

            # grad. penalty not implemented
            gp = my_penalty(disc,label,real,fake,device)

            # E(D(x)) - E(D(G(z)))
            loss_disc = (
                -(torch.mean(disc_real) - torch.mean(disc_fake)) + LAMBDA*gp
            )

            disc.zero_grad()
            loss_disc.backward(retain_graph=True)
            opt_disc.step()

        y = disc(fake,label).reshape(-1)
        # E(D(G(z)))
        # y_ = disc(real,label).reshape(-1)
        loss_gen = -torch.mean(y) # + torch.mean(y_)

        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batchid % 10 == 0:
            print(
                f'Epoch: {epoch}, Batch: {batchid}/{len(loader)} \ '
                f'Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f} \ '
                f'Grad. pen.: {gp.cpu().detach().numpy():.4f}'
            )

            with torch.no_grad():
                fake = gen(noise,label)

                img_grid_real = torchvision.utils.make_grid(
                    real[:32], normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake[:32], normalize=True
                )

                writer_real.add_image('Real', img_grid_real, global_step=step)
                writer_fake.add_image('Fake', img_grid_fake, global_step=step)

            step += 1
    end = perf_counter()
    print(f'Epoch TIME: {end-start:.3f}s')

torch.save(gen.state_dict(), f'./trained_models/gen')
torch.save(disc.state_dict(), f'./trained_models/disc')