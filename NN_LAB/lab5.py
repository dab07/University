import torch
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch import nn, optim

transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True)


trainiter=iter(trainloader)
images,labels=trainiter.next()
print(images.shape)

plt.imshow((make_grid(images).numpy().transpose((2, 1, 0))))


def conv_layer(in_channels, out_channels, kernel_size):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
        nn.MaxPool2d(kernel_size=(3, 3)),
        nn.ReLU()
    )

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = conv_layer(1, 32, 5)
        self.layer2 = conv_layer(32, 64, 5)
        # self.layer3 = conv_layer(64, 128, 5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

model = Classifier()
print(model)

criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 5
running_loss = 0

for i in range(epochs):
    for images, labels in trainloader:
        log_ps = model(images)
        loss = criterion(log_ps, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    else:
        print(f"[EPOCH {i}] - Training loss: {running_loss / len(trainloader)}")


