import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_data_loader(training = True):
    """
    TODO: implement this function.

    INPUT:
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    custom_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    train_set = datasets.MNIST('./data', train=True, download=True,
                       transform=custom_transform)

    test_set = datasets.MNIST('./data', train=False,
                       transform=custom_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size = 50, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = 50, shuffle=False)

    if (training == False):
        return test_loader
    return train_loader


def build_model():
    """
    TODO: implement this function.

    INPUT:
        None

    RETURNS:
        An untrained neural network model
    """
    m = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
    )
    return m


def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT:
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy
        T - number of epochs for training

    RETURNS:
        None
    """
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()

    for epoch in range(T):
        running_loss = 0.0
        total = 0
        accuracy = 0
        for images, labels in train_loader:
            opt.zero_grad() #zero the parameter gradients

            outputs = model(images) #get logits

            loss = criterion(outputs, labels)
            loss.backward()

            opt.step() #optimize

            running_loss += loss.item() #get loss
            total += labels.size(0)
            _, predicted = torch.max(outputs.data, 1) #get predicted labels
            accuracy += (predicted == labels).sum().item() #compare predicted to expected

        train_accuracy = (accuracy / total) * 100

        print("Train Epoch: %d   Accuracy: %d/%d" %(epoch, accuracy, total), end="");
        print("(%.2f%%)" %(train_accuracy), end="  ")
        print("Loss: %.3f" % (running_loss/len(train_loader)))


def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT:
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy

    RETURNS:
        None
    """
    model.eval()
    with torch.no_grad():
        accuracy = 0
        total = 0
        running_loss = 0.0
        for images, labels in test_loader:
            outputs = model(images)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0) #get total number of labels
            accuracy += (predicted == labels).sum().item() #compare predicted to expected

        if (show_loss == True):
            print("Average loss: %.4f" %(running_loss/total)) #should be the average loss I think
        print("Accuracy: %.2f%%" %((accuracy/total)*100))


def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT:
        model - the trained model
        test_images   -  test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    #test_images is tensor
    class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

    logits = model(test_images[index])

    prob = F.softmax(logits, dim=1) #get probabilities
    n_prob = (prob[0] * 100).tolist() #for ease, turn into list of percentages

    for i in range(3):
        maxProb = max(n_prob)
        print("%s: %.2f%%" %(class_names[n_prob.index(maxProb)], maxProb)) #I'm guessing the indices of prob correspond to the labels
        n_prob[n_prob.index(maxProb)] = -1 #dont factor in the current max again


if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions.
    Note that this part will not be graded.
    '''
    train_loader = get_data_loader()
    test_loader = get_data_loader(False)
    model = build_model()
    criterion = nn.CrossEntropyLoss()

    train_model(model, train_loader, criterion, T = 5)
    print()

    evaluate_model(model, test_loader, criterion)
    print()

    #tensors = []
    #for images, labels in test_loader:
        #tensors.append(images)
    #test_images = torch.cat(tensors, dim=0)
    test_images, _ = iter(test_loader).next()

    predict_label(model, test_images, 1)
