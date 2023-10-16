import numpy as np

import logging
import os

import torch
import torchvision
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from tqdm import tqdm

from make_image import makeImage

transform = torchvision.transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor()])

trainSet = torchvision.datasets.ImageFolder('../images', transform = transform)
testSet = torchvision.datasets.ImageFolder('../test_images', transform = transform)

trainLoader = torch.utils.data.DataLoader(trainSet, batch_size = 8, shuffle=True)

testLoader = torch.utils.data.DataLoader(testSet, batch_size = 8, shuffle=True)

loss = torch.nn.CrossEntropyLoss()

def train(model, optimizer, lossFunction, numEpochs = 50):
    
    model.train()
    for epoch in range(numEpochs):
        for images, labels in tqdm(trainLoader):
            images = images.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()

            scores, features = model(images)
            targetLoss = lossFunction(scores, labels)

            predLoss = torch.sum(targetLoss) / targetLoss.size(0)

            predLoss.backward()

            optimizer.step()

def test(model):
    model.eval()

    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in tqdm(testLoader):
            inputs = inputs.cuda()
            labels = labels.cuda()

            scores, _ = model(inputs)
            _, preds = torch.max(scores.data, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return correct / total

def getInitialAcc():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    testAcc = test(model)
    return testAcc

def doImage(phrase, dire):

    makeImage(phrase, dire)

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    optimizer = torch.optim.SGD(lr=1e-4, momentum=0.9)

    train(model, optimizer, loss)

    testAcc = test(model)

    return testAcc

if __name__ == "__main__":
    print(getInitialAcc())