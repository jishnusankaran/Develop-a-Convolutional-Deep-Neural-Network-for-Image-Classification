# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
Design and implement a Convolutional Neural Network (CNN) capable of classifying images into their respective categories. The model should automatically extract image features, learn patterns during training, and accurately classify unseen images

## Neural Network Model

<img width="990" height="692" alt="image" src="https://github.com/user-attachments/assets/977fc15a-d5d3-410a-b93c-f70e2f2a9784" />

## DESIGN STEPS
### STEP 1: 

Import the required libraries such as PyTorch, Torchvision, NumPy, and Matplotlib.

### STEP 2: 

Load and preprocess the CIFAR-10 dataset by applying transformations like normalization and converting images into tensors.

### STEP 3: 

Design the CNN architecture using convolutional layers, ReLU activation functions, max-pooling layers, and fully connected layers.

### STEP 4: 

Initialize the loss function (CrossEntropyLoss) and optimizer (Adam optimizer) for model training.

### STEP 5: 

Train the CNN model for multiple epochs using the training dataset while monitoring the training loss.

### STEP 6: 

Evaluate the trained model using the test dataset, generate the confusion matrix and classification report, and verify predictions using new sample images.


## PROGRAM

### Name: JISHNUPRIYAN S

### Register Number: 212223240061

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

transform = transforms.Compose([
    transforms.ToTensor(),         
    transforms.Normalize((0.5,), (0.5,)) 
])

train_dataset = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=transform, download=True)

image, label = train_dataset[0]
print(image.shape)
print(len(train_dataset))

image, label = test_dataset[0]
print(image.shape)
print(len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1 = nn.Linear(128*3*3,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0),-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


from torchsummary import summary

model=CNNclassifier1()
criterion=nn.CrossEntropyLoss()
op=optim.Adam(model.parameters(),lr=0.001)
if torch.cuda.is_available():
    print(torch.cuda.is_available())
    device=torch.device('cuda')
    model.to(device)


summary(model,input_size=(1,28,28))

epochs=3
running_loss=0.0

for i in range(epochs):
    model.train()
    for a,b in trl:
        op.zero_grad()
        pred=model(a)
        loss=criterion(pred,b)
        loss.backward()
        op.step()
        running_loss+=loss.item()
    print(f"Loss:{i}",running_loss/len(trl))

t=0
c=0
act=[]
pre=[]
model.eval()
with torch.no_grad():
    for img,labels in tstl:
        output=model(img)
        _,predicted=torch.max(output,1)
        t=t+labels.size(0)
        c+=(predicted==labels).sum().item()
        pre.extend(predicted.cpu().numpy())
        act.extend(labels.cpu().numpy())
accuracy=c/t*100
print("Accuracy Score:",accuracy)
conf_matrix=confusion_matrix(act,pre)
class_report=classification_report(act,pre,target_names=test_set.classes)
print("Classification Report:",class_report)
sns.heatmap(conf_matrix,annot=True,fmt='d',cmap='Blues',xticklabels=test_set.classes,yticklabels=test_set.classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()



```

### OUTPUT

## Training Loss per Epoch

<img width="450" height="120" alt="image" src="https://github.com/user-attachments/assets/bc1d382a-08d4-4000-b14a-ea6468b1b6e9" />


## Confusion Matrix
<img width="928" height="780" alt="image" src="https://github.com/user-attachments/assets/d2decaca-717f-4fcc-87a1-ea849d0fbf79" />


## Classification Report
<img width="1031" height="522" alt="image" src="https://github.com/user-attachments/assets/fb6055fa-46c9-4723-89d9-bdaa1b3ad424" />



## RESULT
A Convolutional Neural Network (CNN) was successfully developed and trained for image classification using the CIFAR-10 dataset. The model effectively extracted image features through convolution and pooling layers, achieved good classification accuracy on the test dataset, and correctly predicted the class of new input images.
