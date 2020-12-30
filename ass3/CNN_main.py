import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc1 = nn.Linear(50 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(model, train_loader, optimizer, epoch):
    running_loss = 0.0
    for i, (data, label) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 1000 == 999:    # print every 1000 mini-batches
            print('Train epoch:%2d    [%5d/%5d]    loss: %.5f' %
                  (epoch, i * len(data),  len(train_loader.dataset), running_loss / 1000))
            running_loss = 0.0

def test(model, test_loader):
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, label in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, label, reduction='sum').item()
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == label).sum().item()
    size = len(test_loader.dataset)
    test_loss /= size
    print('Test: Validation loss: %.5f, Accuracy: %5d/%5d %.2f%%\n' %
        (test_loss, correct, size, 100.0 * correct / size))

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data/', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True)

testset = torchvision.datasets.MNIST(root='./data/', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000,
                                         shuffle=False)

net = Net()
trainepoch = 10
opt_algorithm = 0
if opt_algorithm == 0:
    optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.5)
elif opt_algorithm == 1:
    optimizer = optim.Adam(net.parameters())

for epoch in range(1, trainepoch + 1):
    train(net, trainloader, optimizer, epoch)
    test(net, testloader)