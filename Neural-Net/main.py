import torch
import torch.optim as optim
import torch.nn as nn
import os
from PIL import Image
import numpy as np
from net import AlexNet, LeNet
import pandas as pd

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
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    return dataloader

# Training loop
def train(model, optimizer, train_dataloader):
    criterion = nn.CrossEntropyLoss()
    model.train()  # entering training mode (dropout behaves differently)
    for batch_idx, (data, target) in enumerate(train_dataloader):
        data, target = data.to(device), target.to(device)  # move data to the same device
        optimizer.zero_grad()  # clear existing gradients
        output = model(data)  # forward pass
        loss = criterion(output, target.squeeze())  # compute the loss
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

CUDA=torch.cuda.is_available()
device = torch.device("cuda:0" if CUDA else "cpu")
print(device)

NUM_CLASSES = 43
IMG_HEIGHT = 32
IMG_WIDTH = 32
net = LeNet(NUM_CLASSES)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print("Loading Images...")
train_dataloader = preProcessData('data/GTSRB/train-data/')
test_dataloader = preProcessData('data/GTSRB/test-data', "test")

print("Training model...")
for epoch in range(10):
    net = train(net, optimizer,train_dataloader)
    acc = test(net, test_dataloader)
    print('Epoch=%d Test Accuracy=%.3f\n' % (epoch + 1, acc))