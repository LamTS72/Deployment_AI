from torchvision.models import resnet18
import torch
import torch.nn as nn

class CatDogModel(nn.Module):
        def __init__(self, n_classes, hidden_output):
                super(CatDogModel, self).__init__()
                resnet_model = resnet18(weights="IMAGENET1K_V1")
                self.backbone = nn.Sequential(*list(resnet_model.children())[:-1])
                # for param in list(self.backbone.parameters())[:-11]:
                #         param.requires_grad = False

                in_features = resnet_model.fc.in_features
                self.fc1 = nn.Linear(in_features, hidden_output)
                self.dropout = nn.Dropout(p=0.2)
                self.fc2 = nn.Linear(hidden_output, n_classes)


        def forward(self, x):
                x = self.backbone(x)
                x = torch.flatten(x, 1)
                x = self.fc1(x)
                x = self.dropout(x)
                x = self.fc2(x)
                return x
        