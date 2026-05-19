#DW-ResNet-CIFAR10
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import torch.optim as optim
import matplotlib.pyplot as plt


def DwConvBlock(in_ch, out_ch, stride, add_relu=True):
    Block = [
        nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=3,
            stride=stride,  # 步幅作用于深度卷积层
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
    ]
    if add_relu:
        Block.append(nn.ReLU())
    return nn.Sequential(*Block)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=None, stride=1):
        super().__init__()
        self.conv1 = DwConvBlock(in_ch, out_ch, stride, add_relu=True)
        self.conv2 = DwConvBlock(out_ch, out_ch, 1, add_relu=False)
        self.downsample = downsample
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)
        
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = DwConvBlock(3, 32, stride=1)

        self.layer1 = self.make_layer(32, 64, 3, 1)
        self.layer2 = self.make_layer(64, 128, 4, 2)
        self.layer3 = self.make_layer(128, 256, 6, 2)
        self.layer4 = self.make_layer(256, 512, 3, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)

    
    def make_layer(self, in_ch, out_ch, blocks, stride):
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        layer = []
        # 第一个残差块应用stride
        layer.append(ResidualBlock(in_ch, out_ch, downsample=downsample, stride=stride))
        for _ in range(1,blocks):
            layer.append(ResidualBlock(out_ch, out_ch, stride=1))

        return nn.Sequential(*layer)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x


batch_size = 32
lr = 1e-3
epochs = 10

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010])
])

train_set = datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_set = datasets.CIFAR10(root='./data', train=False, transform=transform_test, download=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = ResNet().to(device)
optimizer = optim.AdamW(model1.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

train_losses = []
test_accs = []

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
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    
    # 每个epoch都测试准确率（原代码只在最后测试一次，修正后更合理）
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
        print(f"Epoch {epoch+1}, Val Acc: {acc:.4f}")

# 绘图
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
