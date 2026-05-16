import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import torch.optim as optim
import matplotlib.pyplot as plt

# 深度可分离卷积块：深度卷积(3x3) + 逐点卷积(1x1)，无池化，用stride下采样
# 纯特征提取，不下采样，多叠几层
def DWConv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch),
        nn.BatchNorm2d(in_ch),
        nn.ReLU(),
        nn.Conv2d(in_ch, out_ch, 1, 1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU()
    )

# 下采样块，用来缩小特征图
def DWConvDown(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, 3, 2, 1, groups=in_ch),
        nn.BatchNorm2d(in_ch),
        nn.ReLU(),
        nn.Conv2d(in_ch, out_ch, 1, 1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU()
    )

# CIFAR-10分类网络
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # 先同尺度连卷2层挖特征
        self.layer1 = nn.Sequential(
            DWConv(3,16),
            DWConv(16,16)   # 多卷一层，吃透32*32原图特征
        )
        self.down1 = DWConvDown(16,32) # 再下采样到16*16

        # 16*16尺度继续多卷
        self.layer2 = nn.Sequential(
            DWConv(32,32),
            DWConv(32,32)
        )
        self.down2 = DWConvDown(32,64) # 下采样到8*8

        self.fc = nn.Linear(64*8*8, 10)

    def forward(self,x):
        x = self.layer1(x)
        x = self.down1(x)
        x = self.layer2(x)
        x = self.down2(x)
        x = x.flatten(1)
        return self.fc(x)
    

batch_size = 32
epochs = 20
lr = 1e-3

#dataloader
#RandomHorizontalFlip 水平翻转：语义完全不变，猫还是猫、车还是车，属于温和无伤害增强，几乎所有模型都能吃。
#RandomCrop 随机裁剪 + padding：会切掉边缘、偏移主体，图片局部结构变多、有效特征被打乱，属于高难度增强。
#模型小，训练次数少，不适合用crop,反而会掉精度。
transform_train = transforms.Compose([
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
val_set = datasets.CIFAR10('./data', train=False, transform=transform_val, download=True)
val_loader = DataLoader(val_set, batch_size=batch_size,shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Network().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
train_losses = []
test_accs = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
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
            images = images.to(device)
            labels = labels.to(device)
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

