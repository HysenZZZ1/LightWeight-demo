import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import torch.optim as optim
import torchvision.models as model

def ConvBlock(in_ch, out_ch, pool):
    layer = [
        nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=3,
            padding=1
        ),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=3,
            padding=1
        ),
        nn.BatchNorm2d(out_ch),
        nn.ReLU()
    ]
    if pool:
        #易遗忘混淆点：特征图的大小是根据卷积核和图片原始尺寸算的。然后pool缩小的是单张特征图的尺寸，而不是特征图的数量，更无关通道数。
        layer.append(nn.MaxPool2d(2))
    return nn.Sequential(*layer)

#具体解释
# 1.特征提纯：第一个卷积（3→64）只是把原始的 RGB 图像映射到 64 个基础特征图（比如边缘、纹理）。
# 第二个卷积（64→64）在这些特征图的基础上再进行一次非线性变换，组合出更抽象的模式（比如角点、简单形状）。这种“同通道数的卷积”相当于对已有特征做进一步加工。
# 2.增加非线性：两个卷积层中间夹着一个 ReLU，让网络具备更强的表达能力。如果没有第二个卷积，模型就只是一个单层线性映射，无法学习复杂函数。
# 3.扩大感受野：两个 3×3 卷积堆叠，等效感受野 = 5×5（相比单个 3×3 卷积）。这样，第二个卷积的每个输出像素能看到输入图像中 5×5 的区域，有利于捕捉局部上下文。
# 4.遵循 VGG 设计：VGG 的基本单元就是“连续两个或三个 3×3 卷积 + 一个池化”。在你的 Block 1 中就是“两个 3×3 卷积 + 一个 2×2 池化”，这符合 VGG 的经典结构。
class VGG_CIFAR10(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()
        self.feature = nn.Sequential(
            ConvBlock(3,64,pool=True),
            #在第一个卷积将输入从 3 通道升到 64 通道之后，继续用 64 通道做一次 3×3 卷积，
            #从而在保持通道数不变的前提下，进一步提取更高阶的特征，同时扩大感受野。
            ConvBlock(64,128,pool=True),
            ConvBlock(128,256,pool=True),
            ConvBlock(256,512,pool=False)

        )

        self.Affine_pool = nn.Sequential(nn.AdaptiveMaxPool2d(1))

        self.Affine = nn.Sequential(
             nn.Linear(512,256), 
             nn.ReLU(), 
             nn.Dropout(0.2),
             nn.Linear(256, num_classes)
        )
    
    def forward(self,x):
        x = self.feature(x)
        x = self.Affine_pool(x)
        x = x.flatten(1)
        x = self.Affine(x)
        
        return x

batch_size = 32
epochs = 10
lr = 1e-3

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010])
])

train_set = datasets.CIFAR10(root='./data',train=True,transform=transform,download=True)
train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True)
test_set = datasets.CIFAR10(root='./data',train=False, download=True, transform=transform)
test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model1 = VGG_CIFAR10().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model1.parameters(),lr=lr)

for epoch in range(epochs):
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
    print(f"Val Acc: {acc:.4f}")
