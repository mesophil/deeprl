import numpy as np

import logging
import os

import torch
import torchvision
from torchvision import transforms

from tqdm import tqdm

from make_image import makeImage

from config import batch_size, learning_rate, momentum, numEpochs, test_batch_size, numClasses

from accelerate import Accelerator

currentDir = os.path.dirname(os.path.realpath(__file__))

#device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

logging.basicConfig(filename='my2.log', format='%(asctime)s : %(levelname)s : %(message)s', encoding='utf-8', level=logging.INFO)

normalize = transforms.Normalize(mean=[0.507, 0.4865, 0.4408], std=[0.2673, 0.2564, 0.2761])

trainTransform = torchvision.transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 normalize])

testTransform = torchvision.transforms.Compose([transforms.ToTensor(),
                                                normalize])


trainPath = os.path.join(currentDir, "../images")
testPath = os.path.join(currentDir, "../test_images")
trainPathFull = os.path.join(currentDir, "../train_images")

trainSetFull = torchvision.datasets.CIFAR10(root=trainPathFull, train=True, download=True, transform=trainTransform)
testSet = torchvision.datasets.CIFAR10(root=testPath, train=False, download=True, transform=testTransform)
testLoader = torch.utils.data.DataLoader(testSet, batch_size=test_batch_size, shuffle=False)

def train(model, optimizer, lossFunction, trainLoader, numEpochs = numEpochs):
    model.train()
    
    ## EXPERIMENTAL
    accelerator = Accelerator()
    model, optimizer, trainLoader = accelerator.prepare(model, optimizer, trainLoader)

    for epoch in range(numEpochs):
        for images, labels in tqdm(trainLoader):
            # images = images.cuda()
            # labels = labels.cuda()

            optimizer.zero_grad()

            scores = model(images)
            predLoss = lossFunction(scores, labels)
            # predLoss.backward()
            accelerator.backward(predLoss)
            optimizer.step()

def test(model, testLoader):
    model.eval()

    # EXPERIMENTAL
    accelerator = Accelerator()
    model, testLoader = accelerator.prepare(model, testLoader)

    correct, total = 0, 0
    classCorrect = [0 for c in range(numClasses)]
    classTotals = [0 for c in range(numClasses)]

    with torch.no_grad():
        for inputs, labels in tqdm(testLoader):
            # inputs = inputs.cuda()
            # labels = labels.cuda()

            scores = model(inputs)
            _, preds = torch.max(scores.data, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            for c in range(numClasses):
                classCorrect[c] += ((preds == labels) * (labels == c)).float().sum().item()
                classTotals[c] += (labels == c).sum().item()

    overallAcc = correct / total
    classAccuracies = [i/j for i, j in zip(classCorrect, classTotals)]
    
    return overallAcc, classAccuracies

def getInitialAcc():
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
    # model.to(device)

    testAcc, testClassAcc = test(model, testLoader)
    return testAcc, testClassAcc

def doImage(phrase, dire):
    makeImage(phrase, dire)

    trainSet = torchvision.datasets.ImageFolder(trainPath, transform = trainTransform)
    concatenatedTrainSet = torch.utils.data.ConcatDataset([trainSet, trainSetFull])
    trainLoader = torch.utils.data.DataLoader(concatenatedTrainSet, batch_size = batch_size, shuffle=True,
                                              pin_memory=True, num_workers=8)

    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
    # model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    loss = torch.nn.CrossEntropyLoss()

    train(model, optimizer, loss, trainLoader)

    testAcc, testClassAcc = test(model, testLoader)
    return testAcc, testClassAcc

if __name__ == "__main__":
    print(getInitialAcc())
    print('done')