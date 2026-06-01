#mobileV1test
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import torch.optim as optim
import matplotlib.pyplot as plt

def dwdpBlock(in_ch, out_ch, stride):
    Block = [
        nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=3,
            stride=stride,
            padding = 1,
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
    ]
    return nn.Sequential(*Block)


class MobileV1_net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        #图片小，减少一次下采样
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer1 = dwdpBlock(32, 64, stride=2)
        self.layer2 = self.make_layer(64, 128, blocks=2)
        self.layer3 = self.make_layer(128, 256, blocks=2)
        self.layer4 = self.make_layer(256, 512, blocks=6)
        self.layer5 = self.make_layer(512, 1024, blocks=2)
        self.fc = nn.Linear(1024, num_classes)
        self.maxpool = nn.AdaptiveMaxPool2d(1)

    def make_layer(self, in_ch, out_ch, blocks):
        layer = [dwdpBlock(in_ch, out_ch, stride=2)]          # 第一个块：下采样 + 通道翻倍
        for _ in range(1, blocks):
            layer.append(dwdpBlock(out_ch, out_ch, stride=1)) # 后续块：保持通道不变
            return nn.Sequential(*layer)
        
    #逻辑正确，运行正确，但思路与原文正好相反。原文是先下采样。
    #这样写计算量稍高
    #def make_layer(self, in_ch, out_ch, blocks):
    #    layer = []
    #    for _ in range(1, blocks):
    #        layer.append(dwdpBlock(in_ch, in_ch, stride=1))
    #    layer.append(dwdpBlock(in_ch, out_ch, stride=2))
    #    return nn.Sequential(*layer)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
batch_size = 64
lr = 1e-3
epochs = 10

transforms_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010])

])

transforms_val = transforms.Compose([
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010]),
    transforms.ToTensor()
])

train_set = datasets.CIFAR10(root='./data', train=True, transform=transforms_train, download=True)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

val_set = datasets.CIFAR10('./data', train=False, transform=transforms_val, download=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = MobileV1_net(num_classes=10).to(device)
optimizer = optim.AdamW(model1.parameters(), lr = lr)
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
    
    # 每个epoch都测试准确率
    model1.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            output = model1(images)
            pred = output.argmax(dim=1)
            correct += (pred==labels).sum().item()
        acc = correct / len(val_set)
        test_accs.append(acc)
        print(f"Epoch {epoch+1}, Val Acc: {acc:.4f}")
