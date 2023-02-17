import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_channels, features, img_size, num_classes) -> None:
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.disc = nn.Sequential(
            # 32 x 32
            nn.Conv2d(in_channels+1, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.02),
            self.block(features, 2*features,4,2,1),
            self.block(2*features, 4*features,4,2,1),
            self.block(4*features, 8*features,4,2,1),
            nn.Conv2d(8*features,1, kernel_size=4, stride=2, padding=1),
        )
        self.embed = nn.Embedding(num_classes, img_size*img_size)

    def block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.02)
        )

    def forward(self, x, label):
        embedded = self.embed(label).view(label.shape[0], 1, self.img_size, self.img_size)
        x = torch.cat([x,embedded], dim=1)
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, vect_dim, features, out_channels, num_classes, emb_dim) -> None:
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self.block(vect_dim + emb_dim, 8*features, 4, 2, 1),
            self.block(8*features, 4*features, 4, 2, 1),
            self.block(4*features, 2*features, 4, 2, 1),
            self.block(2*features, features, 4, 2, 1),
            nn.ConvTranspose2d(features, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        self.embed = nn.Embedding(num_classes, emb_dim)

    def block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride,padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x, label):
        embedded = self.embed(label).view(label.shape[0],-1,1,1)
        x = torch.cat([x,embedded], dim=1)
        return self.gen(x)


'''
def test():
    # test disc
    
    disc = Discriminator(1,4,32, 10)
    test_img = torch.randn((50,1,32,32))
    labels = torch.randint(0,10,size=(50,1))
    disc(test_img,labels) 
    
    # test gen
    
    gen = Generator(64,4,1,10,64)
    noise_vector = torch.randn((50,64,1,1))
    gen(noise_vector, labels)
    
    print('Test went well')
test()
'''