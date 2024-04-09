from main import NUM_EPOCHS, loadData, learn

import matplotlib.pyplot as plt
import numpy as np
import torch

def plotAccuracies(accuracies, labels):
    accuracies = np.array(accuracies).reshape(len(labels), -1)

    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0,100)

    for i in range(len(labels)):
        plt.plot(np.arange(1,NUM_EPOCHS+1), accuracies[i], label=labels[i])
    
    plt.legend()
    plt.savefig(f'plots/GTSRB/{plt.gca().get_title()}')
    plt.clf()

def runExperiments(train_dataset, test_dataset):
    #Hyper params
    batch_sizes = [16, 32, 64]
    learning_rates = [0.0001, 0.001, 0.01, 0.1]
    momentums = [0.5, 0.75, 0.9, 1]
    dropouts = [0, 0.1, 0.25, 0.5, 0.75]
    
    # # Test batch sizes
    # print("\nTesting batch sizes...")
    # accuracies = []
    # labels = []
    # for _, value in enumerate(batch_sizes):
    #     print(f'Batch-size: {value}')
    #     train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=value, shuffle=True)
    #     test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
    #     _, acc = learn(train_dataloader, test_dataloader)
    #     accuracies+= acc
    #     labels.append(f'Batch size: {value}')

    # plt.title('GTSRB Batch Sizes')
    # plotAccuracies(accuracies, labels)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

    # # Test learning rates
    # print("\nTesting learning rates...")
    # accuracies = []
    # labels = []
    # for _, value in enumerate(learning_rates):
    #     print(f'LR: {value}')
    #     _, acc = learn(train_dataloader, test_dataloader, learning_rate=value)
    #     accuracies+= acc
    #     labels.append(f'LR: {value}')

    # plt.title('GTSRB Learning Rates')
    # plotAccuracies(accuracies, labels)

    # # Test momentums
    # print("\nTesting momentum...")
    # accuracies = []
    # labels = []
    # for _, value in enumerate(momentums):
    #     print(f'Momentum: {value}')
    #     _, acc = learn(train_dataloader, test_dataloader, momentum=value)
    #     accuracies+= acc
    #     labels.append(f'Momentum: {value}')

    # plt.title('GTSRB Momentum')
    # plotAccuracies(accuracies, labels)

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

def runExperiments2(train_dataset, test_dataset):
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

def main():
    train_dataset, test_dataset = loadData()
    runExperiments(train_dataset, test_dataset)
    #runExperiments2(train_dataset, test_dataset)

if __name__ == "__main__":
    main()