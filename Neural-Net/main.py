import torch
import torch.optim as optim
import torch.nn as nn
import os
from PIL import Image
import numpy as np
from net import AlexNet, LeNet
import pandas as pd
import matplotlib.pyplot as plt

def preProcessData(directory_path, data_type="train"):
    trainset_X = []
    trainset_Y = []

    for dir_path, dir_names, file_names in os.walk(directory_path): #Loop directories
        for filename in file_names: #Loop files
            if not filename.endswith('.ppm'):
                continue

            file_path = os.path.join(dir_path, filename)

            try:
                with Image.open(file_path) as img:
                    image = img.resize((IMG_HEIGHT,IMG_WIDTH))
                    image = np.array(image)
                    trainset_X.append(image)

                if(data_type == "train"):
                    class_Id = int(dir_path[-5:])
                    trainset_Y.append(class_Id)

            except Exception as e:
                print(f'Error reading file: {filename}')
                print(e)

    if(data_type == "test"):
        trainset_Y = pd.read_csv(f"{directory_path}/GT-final_test.csv", sep=";", usecols=[7]).to_numpy()

    trainset_X = np.array(trainset_X) / 255.
    trainset_Y = np.array(trainset_Y).reshape(-1,1)

    tensor_X = torch.Tensor(trainset_X)
    tensor_X = tensor_X.permute(0, 3, 1, 2)
    tensor_y = torch.LongTensor(trainset_Y)

    dataset = torch.utils.data.TensorDataset(tensor_X, tensor_y)

    return dataset

# Training loop
def train(model, optimizer, train_dataloader):
    criterion = nn.CrossEntropyLoss()
    model.train()  # entering training mode (dropout behaves differently)
    for batch_idx, (data, target) in enumerate(train_dataloader):
        data, target = data.to(device), target.to(device)  # move data to the same device
        optimizer.zero_grad()  # clear existing gradients
        output = model(data)  # forward pass
        target = target.view(-1)
        loss = criterion(output, target)  # compute the loss
        loss.backward()  # backward pass: calculate the gradients
        optimizer.step()  # take a gradient step

    return model

def test(net, test_dataloader):
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            if CUDA:
                correct += (predicted.cpu()==labels.cpu()).sum().item()
            else:
                correct += torch.eq(predicted.unsqueeze(1), labels).sum().item()

    test_accuracy = 100 * (correct / total)
    return test_accuracy

def learn(train_dataloader, test_dataloader, learning_rate=0.001, momentum=0.9):
    net = LeNet(NUM_CLASSES)
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    accuracy_list = []

    for epoch in range(NUM_EPOCHS):
        net = train(net, optimizer,train_dataloader)
        acc = test(net, test_dataloader)
        accuracy_list.append(acc)
        print('Epoch=%d Test Accuracy=%.3f' % (epoch + 1, acc))

    return accuracy_list

def plotAccuracies(accuracies, labels):
    accuracies = np.array(accuracies).reshape(len(labels), -1)

    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0,100)

    for i in range(len(labels)):
        plt.plot(np.arange(1,NUM_EPOCHS+1), accuracies[i], label=labels[i])
    
    plt.legend()
    #plt.savefig(f'plots/GTSRB/{plt.gca().get_title()}')
    plt.clf()

def runExperiments():
    #Hyper params
    batch_sizes = [1, 4, 16, 64]
    learning_rates = [0.0001, 0.001, 0.01, 0.1]
    momentums = [0.5, 0.75, 0.9, 0.99]

    # Test batch sizes
    print("\nTesting batch sizes...")
    accuracies = []
    labels = []
    for _, value in enumerate(batch_sizes):
        print(f'Batch-size: {value}')
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=value, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
        accuracies += learn(train_dataloader, test_dataloader)
        labels.append(f'Batch size: {value}')

    plt.title('GTSRB Batch Sizes')
    plotAccuracies(accuracies, labels)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

    # Test learning rates
    print("\nTesting learning rates...")
    accuracies = []
    labels = []
    for _, value in enumerate(learning_rates):
        print(f'LR: {value}')
        accuracies += learn(train_dataloader, test_dataloader, learning_rate=value)
        labels.append(f'LR: {value}')

    plt.title('GTSRB Learning Rates')
    plotAccuracies(accuracies, labels)

    # Test momentums
    print("\nTesting momentum...")
    accuracies = []
    labels = []
    for _, value in enumerate(momentums):
        print(f'Momentum: {value}')
        accuracies += learn(train_dataloader, test_dataloader, momentum=value)
        labels.append(f'Momentum: {value}')

    plt.title('GTSRB Momentum')
    plotAccuracies(accuracies, labels)


CUDA=torch.cuda.is_available()
device = torch.device("cuda:0" if CUDA else "cpu")
print(device)

NUM_CLASSES = 43
IMG_HEIGHT = 32
IMG_WIDTH = 32
NUM_EPOCHS = 10

print("Loading data...")
train_dataset = preProcessData('data/GTSRB/train-data/')
test_dataset = preProcessData('data/GTSRB/test-data', "test")

runExperiments()
