import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from net import Net

CUDA=torch.cuda.is_available()
device = torch.device("cuda:0" if CUDA else "cpu")

trainset = None
testset = None
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
classes = (None)

net = Net(len(classes))
accuracy_values=[]
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
epoch_number=[]

for epoch in range(1):  # loop over the dataset multiple times. Here 10 means 10 epochs
    for i, (inputs,labels) in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        if CUDA:
            inputs = inputs.cuda()
            labels = labels.cuda()
        else:
            inputs = inputs.cpu()
            labels = labels.cpu()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            if CUDA:
                images = images.cuda()
                labels = labels.cuda()
            else:
                images = images.cpu()
                labels =labels.cpu()

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            if CUDA:
                correct += (predicted.cpu()==labels.cpu()).sum().item()
            else:
                correct += (predicted==labels).sum().item()

        TestAccuracy = 100 * correct / total
        epoch_number += [epoch+1]
        accuracy_values += [TestAccuracy]
        print('Epoch=%d Test Accuracy=%.3f\n' %
                (epoch + 1, TestAccuracy))