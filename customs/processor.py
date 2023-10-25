import numpy as np

import logging
import os

import torch
import torchvision
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from tqdm import tqdm

from make_image import makeImage

currentDir = os.path.dirname(os.path.realpath(__file__))

transform = torchvision.transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor()])

device = torch.device("cuda")

trainPath = os.path.join(currentDir, "../images")
testPath = os.path.join(currentDir, "../test_images")

def train(model, optimizer, lossFunction, trainLoader, numEpochs = 20):
    
    model.train(True)
    for epoch in range(numEpochs):
        for images, labels in tqdm(trainLoader):
            images = images.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()

            scores = model(images)
            # targetLoss = lossFunction(scores, labels)

            # logging.info(f"{targetLoss}")

            # predLoss = torch.sum(targetLoss) / targetLoss.size(0)

            predLoss = lossFunction(scores, labels)

            predLoss.backward()

            optimizer.step()

def test(model, testLoader):
    model.eval()

    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in tqdm(testLoader):
            inputs = inputs.cuda()
            labels = labels.cuda()

            scores = model(inputs)
            _, preds = torch.max(scores.data, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return correct / total

def getInitialAcc():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    testSet = torchvision.datasets.ImageFolder(testPath, transform = transform) 
    testLoader = torch.utils.data.DataLoader(testSet, batch_size = 8, shuffle=True)
    model.to(device)
    testAcc = test(model, testLoader)
    return testAcc

def doImage(phrase, dire):
    makeImage(phrase, dire)

    trainSet = torchvision.datasets.ImageFolder(trainPath, transform = transform)
    testSet = torchvision.datasets.ImageFolder(testPath, transform = transform) 

    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size = 8, shuffle=True)
    testLoader = torch.utils.data.DataLoader(testSet, batch_size = 8, shuffle=True)

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    loss = torch.nn.CrossEntropyLoss()

    train(model, optimizer, loss, trainLoader)

    testAcc = test(model, testLoader)

    return testAcc

if __name__ == "__main__":
    logging.basicConfig(filename='my.log', format='%(asctime)s : %(levelname)s : %(message)s', encoding='utf-8', level=logging.DEBUG)
    doImage("a", "a")