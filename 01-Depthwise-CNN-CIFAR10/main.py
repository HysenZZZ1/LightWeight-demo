import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import torch.optim as optim
import matplotlib.pyplot as plt

# 深度可分离卷积块：深度卷积(3x3) + 逐点卷积(1x1)，无池化，用stride下采样
def ConBlock(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=in_ch
        ),
        nn.BatchNorm2d(in_ch),
        nn.ReLU(),

        nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            padding=0
        ),
        nn.BatchNorm2d(out_ch),
        nn.ReLU()    
    )

# CIFAR-10分类网络
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConBlock(3, 16)
        self.conv2 = ConBlock(16, 32)

        self.fc1 = nn.Linear(32*8*8,256)
        self.fc2 = nn.Linear(256,10)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.flatten(1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    

batch_size = 64
epochs = 10
lr = 1e-3

#dataloader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010])
])
transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010])
])
train_set = datasets.CIFAR10('./data', train=True, transform=transform_train, download=True)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_set = datasets.CIFAR10('./data', train=False, transform=transform, download=True)
val_loader = DataLoader(val_set, batch_size=batch_size,shuffle=False)

model = Network()
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(),lr=lr)
train_losses = []
test_accs = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss=total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    correct = 0
    with torch.no_grad():
        for images,labels in val_loader:
            outputs = model(images)
            pred = outputs.argmax(dim=1)
            correct += (pred==labels).sum().item()
    acc = correct / len(val_set)
    test_accs.append(acc)
    print(f"Val Acc: {acc:.4f}")

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(range(1,epochs+1),train_losses, marker='o', label='train loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Train_loss')
plt.grid(True)
plt.legend()

#准确率
plt.subplot(1,2,2)
plt.plot(range(1,epochs+1),test_accs,marker='s', label = 'accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('validation accuracy curve')
plt.grid(True)
plt.legend()

plt.show()    
