import torch
import torch.nn as nn
from torchvision import models

class MultiHeadResNet34(nn.Module):
    def __init__(self):
        super(MultiHeadResNet34, self).__init__()
        self.pretrained_model = models.resnet34(pretrained = True)
        num_ftrs = self.pretrained_model.fc.in_features
        self.pretrained_model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128), #layer 디자인 어떻게...??
            nn.ReLU()
        )

        coordinate_head = nn.Linear(128, 34) #(x1, y1, ... , x17, y17)

        classification_head = nn.Sequential(
            nn.Linear(128, 17), #(c1, ... , c17)
            nn.Sigmoid()
        )
        self.head1 = coordinate_head
        self.head2 = classification_head

    def forward(self, x):
        x = self.pretrained_model(x)
        output1 = self.head1(x)
        output2 = self.head2(x)
        output1 = output1.view(-1, 17, 2)
        output2 = output2.unsqueeze(-1)
        output = torch.cat((output1, output2), dim=-1)
        return output

if __name__ == '__main__':
    x = torch.randn((1, 3, 224, 224))
    model = MultiHeadResNet34()
    print(model(x).shape)