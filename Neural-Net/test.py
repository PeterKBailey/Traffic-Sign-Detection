import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from main import IMG_HEIGHT, IMG_WIDTH
from net import VGNet

def imshow(img, labels):
    npimg = img.numpy()

    plt.figure(figsize=(15, 5))
    plt.text(0, 50, "Predicted: " + labels[0], fontsize=12)
    plt.text(0, 60, "Actual: " + labels[1], fontsize=12)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def generateSubset():
    try:
        test_dataset = torch.load("data/demo/test_subset.pth")
    except FileNotFoundError:
        transform = transforms.Compose([
            transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),  
            transforms.ToTensor(),
        ])
        
        test_data = datasets.GTSRB(root="data", split="test", download=True, transform=transform)
        test_dataset = [test_data[i] for i in range(100)]
        torch.save(test_dataset, "data/demo/test_subset.pth")

    return test_dataset

def main():
    classes = []
    with open('data/demo/labels_gtsrb.txt') as f:
        for line in f.readlines():
            classes.append(line.strip())

    test_subset = generateSubset()
    samples = 8
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=samples, shuffle=True, num_workers=2)
    dataiter = iter(test_loader)
    
    net = VGNet()
    net.load_state_dict(torch.load('models/model_scripted.pt',map_location="cpu"))
    net.eval()

    images, labels = next(dataiter)
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    predicted_labels = ',  '.join('%5s' % classes[predicted[j]] for j in range(samples))
    labels = ',  '.join('%5s' % classes[labels[j]] for j in range(samples))
    imshow(torchvision.utils.make_grid(images), (predicted_labels, labels))

if __name__ == "__main__":
    main()
