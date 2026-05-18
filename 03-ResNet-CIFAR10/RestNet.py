#ResNet-CIFAR10
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import torch.optim as optim
import torchvision.models as model
import matplotlib.pyplot as plt

def ConvBlock(in_ch,out_ch,stride):
    return nn.Sequential(
        nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=3,
            stride=stride,
            padding=1
        ),
        nn.BatchNorm2d(out_ch),
        nn.ReLU()
    )

class ResidualBlock(nn.Module):
    def __init__(self,in_ch,out_ch,downsample = None,stride=1):
        super().__init__()
        #第一个卷积，允许向下采样
        self.conv1 = ConvBlock(in_ch,out_ch,stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
#问题一：为什么残差相加之前的一层不先激活，而是残差加上之后再激活

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)
#问题2：downsample是干嘛的？
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = ConvBlock(3, 64, stride=1)

        #residual block
        self.layer1 = self.make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self.make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self.make_layer(128, 256, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, num_classes)
    
    def make_layer(self,in_ch, out_ch, blocks, stride):
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        layers=[]
        layers.append(ResidualBlock(in_ch, out_ch, downsample, stride))
        for _ in range(1,blocks):
            layers.append(ResidualBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x

batch_size = 32
lr = 1e-3
epochs = 10

transform_train = transforms.Compose([
    transforms.RandomCrop(32,padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010])
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010])
])
train_set = datasets.CIFAR10(root='./data',train=True,transform=transform_train,download=True)
train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True)
test_set = datasets.CIFAR10('./data',train=False,transform=transform_test,download=True)
test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = ResNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model1.parameters(), lr=lr)

train_losses = []
test_accs =[]

for epoch in range (epochs):
    model1.train()
    total_loss = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model1(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss=total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

    model1.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            output = model1(images)
            pred = output.argmax(dim=1)
            correct += (pred==labels).sum().item()
    acc = correct / len(test_set)
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
