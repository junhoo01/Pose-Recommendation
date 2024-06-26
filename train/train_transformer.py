from __future__ import print_function
import os

import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pose_resnet import *
from config import Config
from dataloader import MaskDataset
from loss import CustomLoss
from loss import MyLoss
from transformer import *
from resnet import MultiHeadResNet34

from VAE import *


def train_epoch(train_loader, criterion, optimizer):

    total_loss = []
    for image, norm_pose_keypoints, class_label in train_loader:

        # Forward pass
        outputs = model(image.to(device))

        targets = torch.cat((norm_pose_keypoints, class_label.unsqueeze(-1)), dim=-1)
        # Compute the loss
        loss = criterion(outputs.to(device), targets.to(device))

        total_loss.append(loss.item())

        optimizer.zero_grad()
        # Backward pass
        loss.backward()
        # Update the parameters
        optimizer.step()
    return np.mean(total_loss)


cfg = Config()

device = cfg.device

if cfg.model_type == 'pose_resnet':
    model = PoseResNet(Bottleneck, [3, 4, 6, 3])
    model.load_state_dict(torch.load('checkpoints/experiment1/saved_model_poseres_300_5.489701014487341.pth', map_location=device),strict=False)
    #model.load_state_dict(torch.load("./pretrained/pose_resnet_50_256x192.pth.tar", map_location=device),strict=False)
    #custom_layer = model.custom_layer  
    #for param in custom_layer.parameters():
    #    param.data.fill_(0.05)
if cfg.model_type == 'resnet':
    model = MultiHeadResNet34()

if cfg.model_type == 'transformer':
    model = KeypointAttention()

if cfg.model_type == 'vae':
    model = VAE('ResNet34')    

model.train()
model.to(device)

data_path = './data'
data_list = []                  
with open(os.path.join(data_path, 'mask_data.csv'), 'r') as f:
      data_list = list(f.readlines())

train_dataset = MaskDataset('./data', data_list, cfg)
train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)


##########################################################

#criterion = vae_loss(CustomLoss)

criterion = CustomLoss()
#criterion = MyLoss()
optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

total_time=0
for epoch in range(cfg.num_epochs):

    epoch_start_time = time.time()


    loss = train_epoch(train_dataloader, criterion, optimizer)

    # Optionally, print the loss after each epoch
    epoch_end_time = time.time()
    total_time = total_time + (epoch_end_time-epoch_start_time)
    print(f"Epoch {epoch+1}, Loss: {loss.item()},Time: {total_time}")
    if epoch % 10 == 0:
        torch.save(model.state_dict(), os.path.join(cfg.checkpoint_dir, f'saved_model_transformer_{epoch+1}_{loss}.pth'))
