import torch
import torch.nn as nn

def my_penalty(discriminator,label,real,fake,device='cpu'):
    batch_size, c, h, w = real.shape
    eps = torch.rand((batch_size,1,1,1)).repeat(1,c,h,w).to(device)

    interpolated_img = eps * real + (1 - eps) * fake

    scores = discriminator(interpolated_img,label)

    gradient = torch.autograd.grad(
        inputs=interpolated_img,
        outputs=scores,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True
    )[0]

    gradient = gradient.view(gradient.shape[0],-1)
    gradient = torch.abs(gradient)
    gradient = gradient - 1
    gradient = torch.relu(gradient)
    gradient_pen = torch.mean(gradient)
    return gradient_pen