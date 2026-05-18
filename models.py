import torch
import torch.nn as nn
import torchvision.models as models

class DeepFakeModel(nn.Module):
    def __init__(self):
        super(DeepFakeModel, self).__init__()

        self.network = models.resnet18(weights="IMAGENET1K_V1")

        for param in self.network.parameters():
            param.requires_grad = False

        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        return self.network(x)

print("Models.py successfully loaded DeepFakeModel!")