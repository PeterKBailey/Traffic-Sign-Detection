import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
from PIL import Image
import numpy as np
from net import LeNet, VGNet
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.mobile_optimizer import optimize_for_mobile

NUM_CLASSES = 43
IMG_HEIGHT = 32
IMG_WIDTH = 32
NUM_EPOCHS = 20

def preProcessData2(directory_path, data_type="train"):
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
    tensor_X = torch.Tensor(trainset_X)
    tensor_X = tensor_X.permute(0, 3, 1, 2)
    tensor_y = torch.LongTensor(trainset_Y)

    dataset = torch.utils.data.TensorDataset(tensor_X, tensor_y)

    return dataset

def preProcessData():
    def processDataset(dataset):
        X = []
        Y = []
        for image, label in dataset:
            X.append(image)
            Y.append(label)

        X = torch.stack(X)
        Y = torch.LongTensor(Y)
        return torch.utils.data.TensorDataset(X, Y)
    
    transform = transforms.Compose([
        transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),  
        transforms.ToTensor(),
    ])
    
    train_data = datasets.GTSRB(root="data2", split="train", download=True, transform=transform)
    test_data = datasets.GTSRB(root="data2", split="test", download=True, transform=transform)
    
    train_dataset = processDataset(train_data)
    test_dataset = processDataset(test_data)
    return train_dataset, test_dataset

# Training loop
def train(model, optimizer, train_dataloader):
    criterion = nn.CrossEntropyLoss()
    model.train()  # entering training mode (dropout behaves differently)
    for batch_idx, (data, target) in enumerate(train_dataloader):
        data, target = data.to(device), target.to(device)  # move data to the same device
        optimizer.zero_grad()  # clear existing gradients
        output = model(data)  # forward pass
        #target = target.view(-1)
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
                correct += torch.eq(predicted.cpu(), labels.cpu()).sum().item()
            else:
                correct += torch.eq(predicted, labels.cpu()).sum().item()

    test_accuracy = 100 * (correct / total)
    return test_accuracy

def learn(train_dataloader, test_dataloader, learning_rate=0.001, momentum=0.9, dropout=0.75):
    net = VGNet(NUM_CLASSES, dropout)
    net = net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    accuracy_list = []

    for epoch in range(NUM_EPOCHS):
        net = train(net, optimizer, train_dataloader)
        test_acc = test(net, test_dataloader)
        train_acc = test(net, train_dataloader)
        accuracy_list.append(test_acc)
        print('Epoch=%d Train Accuracy=%.3f Test Accuracy=%.3f' % (epoch + 1, train_acc, test_acc))

    return net, accuracy_list

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
    batch_sizes = [16, 32, 64]
    learning_rates = [0.0001, 0.001, 0.01, 0.1]
    momentums = [0.5, 0.75, 0.9, 1]
    dropouts = [0, 0.1, 0.25, 0.5, 0.75]
    
    # Test batch sizes
    print("\nTesting batch sizes...")
    accuracies = []
    labels = []
    for _, value in enumerate(batch_sizes):
        print(f'Batch-size: {value}')
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=value, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
        _, acc = learn(train_dataloader, test_dataloader)
        accuracies+= acc
        labels.append(f'Batch size: {value}')

    plt.title('GTSRB Batch Sizes')
    plotAccuracies(accuracies, labels)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

    # Test learning rates
    print("\nTesting learning rates...")
    accuracies = []
    labels = []
    for _, value in enumerate(learning_rates):
        print(f'LR: {value}')
        _, acc = learn(train_dataloader, test_dataloader, learning_rate=value)
        accuracies+= acc
        labels.append(f'LR: {value}')

    plt.title('GTSRB Learning Rates')
    plotAccuracies(accuracies, labels)

    # Test momentums
    print("\nTesting momentum...")
    accuracies = []
    labels = []
    for _, value in enumerate(momentums):
        print(f'Momentum: {value}')
        _, acc = learn(train_dataloader, test_dataloader, momentum=value)
        accuracies+= acc
        labels.append(f'Momentum: {value}')

    plt.title('GTSRB Momentum')
    plotAccuracies(accuracies, labels)

    # Test dropouts
    print("\nTesting dropouts...")
    accuracies = []
    labels = []
    for _, value in enumerate(dropouts):
        print(f'Dropout: {value}')
        _, acc = learn(train_dataloader, test_dataloader, dropout=value)
        accuracies+= acc
        labels.append(f'Dropout: {value}')

    plt.title('GTSRB Dropout')
    plotAccuracies(accuracies, labels)

def runExperiments2():
    #Hyper params
    learning_rates = [0.0001, 0.001]
    momentums = [0.5, 0.75, 0.9]
    dropouts = [0.1, 0.25, 0.5, 0.75] 

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

    for _, lr in enumerate(learning_rates):
        for _, m in enumerate(momentums):
            for _, d in enumerate(dropouts):
                print(f'\nLR: {lr}, M: {m}, D: {d}')
                learn(train_dataloader, test_dataloader, learning_rate=lr, momentum=m, dropout=d)

def createModel(batch_size, learning_rate, momentum, dropout):
    print("Creating model...")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
    model, _ = learn(train_dataloader, test_dataloader, learning_rate=learning_rate, momentum=momentum, dropout=dropout)
    model = model.cpu()
    #torch.save(model.state_dict(), "models/model_scripted.pt")

def optimizeModel():
    model = VGNet(NUM_CLASSES)
    model.load_state_dict(torch.load('models/model_scripted.pt',map_location="cpu"))
    model.eval()
    example = torch.rand(1, 3, IMG_WIDTH, IMG_HEIGHT)
    traced_script_module = torch.jit.trace(model, example)
    optimized_traced_model = optimize_for_mobile(traced_script_module)
    #optimized_traced_model._save_for_lite_interpreter("model.pt")

def loadData(train_path='data/gtsrb/train.pth', test_path='data/gtsrb/test.pth'):
    print("Loading data...")
    try:
        train_dataset = torch.load(train_path)
        test_dataset = torch.load(test_path)
    except FileNotFoundError:
        train_dataset, test_dataset = preProcessData()
        torch.save(train_dataset, train_path)
        torch.save(test_dataset, test_path)
    return train_dataset, test_dataset

CUDA=torch.cuda.is_available()
device = torch.device("cuda:0" if CUDA else "cpu")
print(device)

train_dataset, test_dataset = loadData()

#print("Running experiments...")
#runExperiments()
#createModel(batch_size=4, learning_rate=0.001, momentum=0.9, dropout=0.75)
#optimizeModel()