# %%
!pip install datasets
!pip install torchinfo

# %%
import torch
import torch.nn as nn

from PIL import Image
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from torchvision import transforms
from torchinfo import summary
import numpy as np

# %%
TEST_SIZE=0.2
IMG_SIZE = 64
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
DATASET_NAME = "microsoft/cats_vs_dogs"
DROPOUT = 0.2
HIDDEN_OUTPUT = 256

# %%
datasets = load_dataset(DATASET_NAME)
datasets

# %%
datasets["train"][0]['image'], np.asarray(datasets["train"][0]['image']).shape

# %%
datasets = datasets["train"].train_test_split(test_size=TEST_SIZE)
img_transforms = transforms.Compose(
        [
                transforms.Resize(size=(IMG_SIZE,IMG_SIZE)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(
                        NORMALIZE_MEAN,
                        NORMALIZE_STD
                )
        ]
)
img_transforms

# %%
class CatDogDataset(Dataset):
        def __init__(self, data, transform=None):
                self.data = data
                self.transform = transform

        def __len__(self):
                return len(self.data)
        
        def __getitem__(self, idx):
                images = self.data[idx]["image"]
                labels = self.data[idx]["labels"]

                if self.transform:
                        images = self.transform(images)
                
                labels =torch.tensor(labels, dtype=torch.long)
                return images, labels

# %%
TRAIN_BATCH_SIZE = 512
VAL_BATCH_SIZE = 256

train_dataset = CatDogDataset(
        datasets["train"], transform=img_transforms
)

test_dataset = CatDogDataset(
        datasets["test"], transform=img_transforms
)

train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True
)

test_loader = DataLoader(
        test_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False
)

# %%
img, label = next(iter(train_loader))
img.shape, label.shape

# %%
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

# %%
device  = "cuda" if torch.cuda.is_available() else "cpu"

# %%
device

# %%
N_CLASSES = 2
model = CatDogModel(N_CLASSES, HIDDEN_OUTPUT).to(device)
test_input = torch.rand(1,3, 224, 224).to(device)
with torch.no_grad():
        output = model(test_input)
        print(output.shape)

# %%
model

# %%
summary(model, input_size=[1, 3, 224 ,224]) 

# %%
EPOCHS = 10
LR = 1e-3
WEIGHT_DECAY = 1e-5

# %%
optimizer = torch.optim.Adam(model.parameters(), lr=LR)#, weight_decay=WEIGHT_DECAY)
losses= torch.nn.CrossEntropyLoss()

results = {"train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
          }

for  epoch in range(EPOCHS):
        train_losses, train_acc = [], []
        model.train()
        for batch, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                optimizer.zero_grad()
                loss = losses(outputs, labels)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

                
                y_pred_class = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                train_acc.append((y_pred_class == labels).sum().item()/len(outputs))
        
        train_loss = sum(train_losses) / len(train_losses)
        train_acc = sum(train_acc) / len(train_acc)

        val_losses, val_acc = [], []
        model.eval()
        with torch.no_grad():
                for batch, (images, labels) in enumerate(test_loader):
                        images = images.to(device)
                        labels = labels.to(device)

                        outputs = model(images)
                        loss = losses(outputs, labels)
                        val_losses.append(loss.item())
                        test_pred_labels = outputs.argmax(dim=1)
                        val_acc.append(((test_pred_labels == labels).sum().item()/len(test_pred_labels)))

        val_loss = sum(val_losses) / len(val_losses)
        val_acc = sum(val_acc) / len(val_acc)
        print(f"EPOCH  {epoch+1}: \tTrain Loss: {train_loss:.3f}\tTrain Acc: {train_acc:.3f}\tValidation Loss: {val_loss:.3f}\tVal Acc: {val_acc:.3f}")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

# %%
import matplotlib.pyplot as plt

def plot_loss_curves(results):
  
    results = dict(list(results.items()))

    # Get the loss values of the results dictionary (training and val)
    loss = results['train_loss']
    val_loss = results['val_loss']

    # Get the accuracy values of the results dictionary (training and val)
    accuracy = results['train_acc']
    val_accuracy = results['val_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot 
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();

plot_loss_curves(results)


# %%
SAVE_PATH = "catdog_weights.pt"
torch.save(model.state_dict(), SAVE_PATH)

# %%



