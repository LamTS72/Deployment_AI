from torchvision.models import resnet18
import torch
import torch.nn as nn

class CatDogModel(nn.Module):
        def __init__(self, n_classes):
                super(CatDogModel, self).__init__()
                resnet_model = resnet18(weights="IMAGENET1K_V1")
                self.backbone = nn.Sequential(*list(resnet_model.children())[:-1])
                for param in self.backbone.parameters():
                        param.requires_grad = False

                in_features = resnet_model.fc.in_features
                self.fc = nn.Linear(in_features, n_classes)

        def forward(self, x):
                x = self.backbone(x)
                x = torch.flatten(x)
                x = self.fc(x)
                return x
        