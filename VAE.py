import torch
import torch.nn as nn
from torchvision import models

import torch.optim as optim
from loss import CustomLoss

class VAE(nn.Module):
    def __init__(self, backbone):
        super(VAE, self).__init__()
        if backbone == 'VGG':
            new_model = models.vgg16(pretrained=True)
            self.backbone = nn.Sequential(
            *list(new_model.features.children()),
            new_model.avgpool,
            nn.Flatten(),
            *list(new_model.classifier.children())[:-1],
            nn.ReLU(),
            nn.Linear(4096, 512)
        )   
        elif backbone == 'ResNet18':
            new_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.backbone = nn.Sequential(
            *list(new_model.children())[:-2],  
            new_model.avgpool,
            nn.Linear(512, 512)
        )
        elif backbone == 'ResNet34':
            new_model = models.resnet34(pretrained=True)
            self.backbone = nn.Sequential(
            *list(new_model.children())[:-2],  
            new_model.avgpool,
            nn.Linear(512,512)
        )
        else:
            new_model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
            self.backbone = nn.Sequential(
            *list(new_model.features.children()),
            new_model.avgpool,
            nn.Flatten(),
            *list(new_model.classifier.children())[:-1],
            nn.ReLU(),
            nn.Linear(4096, 512)
        )

        self.fc1 = nn.Linear(51, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5_1 = nn.Linear(1024, 30)
        self.fc5_2 = nn.Linear(1024, 30)
        self.fc6 = nn.Linear(30, 512)
        self.fc7 = nn.Linear(512, 512)
        self.fc8 = nn.Linear(512, 512)
        self.fc9 = nn.Linear(1024, 512)
        self.fc10 = nn.Linear(512, 512)
        self.fc11 = nn.Linear(512, 34)
        self.fc12 = nn.Linear(512, 17)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()
    
    def encode(self, target, image): #target.size() = [B, 17, 3]
        x1 = self.relu(self.fc1(self.flatten(target)))
        x2 = self.relu(self.backbone(image))

        x1 = self.relu(self.fc3(x1))
        x2 = self.relu(self.fc4(x2))
        
        mu = self.fc5_1(torch.cat([x1,x2], dim=1))
        logvar = self.fc5_2(torch.cat([x1,x2], dim=1))
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z, image):
        x2 = self.relu(self.backbone(image))

        z = self.relu(self.fc6(z))
        x = self.relu(self.fc4(x2))
        
        z = self.relu(self.fc7(z))
        x = self.relu(self.fc8(x))
        
        x = self.relu(self.fc9(torch.cat([z,x], dim=1)))
        x = self.relu(self.fc10(x))

        reg = self.fc11(x).view(-1, 17, 2)
        cls = self.sigmoid(self.fc12(x)).unsqueeze(-1)
        x = torch.cat((reg, cls), dim=-1)

        return x
    
    def forward(self, target, origin_image, masked_image):
        mu, logvar = self.encode(target, origin_image)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, masked_image), mu, logvar
    
class vae_loss(nn.Module):
    def __init__(self, LossFunction):
        super(vae_loss, self).__init__()
        self.criterion = LossFunction()
    def forward(self, output, target, mu, logvar):
        L2 = self.criterion(output, target)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return L2 + KLD


'''
model = VAE('VGG')
criterion = vae_loss(CustomLoss)


origin_image = torch.randn(1,3,224,224)
masked_image = torch.randn(1,3,224,224)
random_label = torch.randn(1, 17, 3)
random_label[:, :, 2] = 1
print(random_scale)
optimizer = optim.Adam(model.parameters(), lr=0.01)
optimizer.zero_grad()

output, mu, logvar = model(random_scale, origin_image, masked_image)
loss = criterion(output, random_label, mu, logvar)

print("loss!!")
print(loss)
loss.backward()
optimizer.step()
'''