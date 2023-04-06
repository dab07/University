import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import time
import utils

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=6000, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor()),
    batch_size=1000)

train_data = enumerate(train_loader)
batch_idx, (train_set, train_targets) = next(train_data)
test_data = enumerate(test_loader)
batch_idx, (test_set, test_targets) = next(test_data)


class two_layers(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super(two_layers, self).__init__()

        self.layer1 = nn.Linear(inputSize, hiddenSize, bias=True)
        self.layer2 = nn.Linear(hiddenSize, outputSize, bias=True)

    def forward(self, x):
        yHidden = self.layer1(x)
        y = self.layer2(F.relu(yHidden))

        return y


net = two_layers(inputSize=784, hiddenSize=512, outputSize=10)
print(net)
utils.display_num_param(net)

cross_entropy = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
bs = 16


def error_on_test_set():
    current_error = 0
    num_batches = 0

    for batch in range(0, 1000, bs):
        test_mini_batch = test_set[batch:batch + bs]
        input_test = test_mini_batch.view(bs, 784)
        label_minibatch = test_targets[batch:batch + bs]
        y_hat = net(input_test)
        error = utils.get_error(y_hat, label_minibatch)

        num_batches += 1
        current_error += error.item()

    avg_error = current_error / num_batches
    print('The error on test set = ' + str(avg_error * 100) + '%')


start = time.time()
for epoch in range(100):

    current_loss = 0
    current_error = 0
    num_batches = 0

    shuffled_indices = torch.randperm(6000)
    for batch in range(0, 6000, bs):
        shuffled_batch = shuffled_indices[batch:batch + bs]
        train_minibatch = train_set[shuffled_batch]
        label_minibatch = train_targets[shuffled_batch]
        optimizer.zero_grad()
        inputs = train_minibatch.view(bs, 784)
        inputs.requires_grad_()
        y_hat = net(inputs)
        loss = cross_entropy(y_hat, label_minibatch)
        loss.backward()
        optimizer.step()
        error = utils.get_error(y_hat.detach(), label_minibatch)

        num_batches += 1
        current_loss += loss.detach().item()
        current_error += error.item()

    avg_loss = current_loss / num_batches
    avg_error = current_error / num_batches
    elapsed_time = time.time() - start

    if epoch % 10 == 0:
        print('The loss for epoch number ' + str(epoch) + ' = ' + str(avg_loss))
        print('The error for epoch number ' + str(epoch) + ' = ' + str(avg_error))

        error_on_test_set()
