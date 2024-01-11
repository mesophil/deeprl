import numpy as np

import logging
import os

import torch
import torchvision
from torchvision import transforms

from tqdm import tqdm
from pathlib import Path

from make_image import makeImage

from tin_config import batch_size, learning_rate, momentum, numEpochs, test_batch_size, numClasses
from tinyimagenet_utils.tin3 import TinyImageNet

from accelerate import Accelerator


logging.basicConfig(filename='tinyimagenet.log', format='%(asctime)s : %(levelname)s : %(message)s', encoding='utf-8', level=logging.INFO)

currentDir = os.path.dirname(os.path.realpath(__file__))
trainPath = os.path.join(currentDir, "../images")
tinPath = os.path.join(currentDir, "../tiny_image_net/tiny-imagenet-200/")

device = torch.device("cuda")

normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
trainTransform = torchvision.transforms.Compose([transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 normalize])
testTransform = torchvision.transforms.Compose([transforms.ToTensor(),
                                                normalize])

tinyTrain = TinyImageNet(tinPath, transform=trainTransform, split='train')
tinyValid = TinyImageNet(tinPath, transform=testTransform, split='val')

validationLoader = torch.utils.data.DataLoader(tinyValid, batch_size = test_batch_size, shuffle=False)

def train(model, optimizer, lossFunction, sched, trainLoader, numEpochs = numEpochs):
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
        
        sched.step()
        logging.info(f"Epoch {epoch} Loss: {predLoss}")

def valid(model, validationLoader):
    model.eval()

    correct, total = 0, 0
    classCorrect = [0 for c in range(numClasses)]
    classTotals = [0 for c in range(numClasses)]

    with torch.no_grad():
        for inputs, labels in tqdm(validationLoader):
            inputs, labels = inputs.to(device), labels.to(device)

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
    weights = torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model = torchvision.models.efficientnet_v2_s(weights=weights)

    model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=numClasses)

    model.to(device)

    testAcc, testClassAcc = valid(model, validationLoader)
    return testAcc, testClassAcc

def doImage(phrase, dire):
    makeImage(phrase, dire)

    trainSet = torchvision.datasets.ImageFolder(trainPath, transform = trainTransform)
    concatenatedTrainSet = torch.utils.data.ConcatDataset([trainSet, tinyTrain])
    trainLoader = torch.utils.data.DataLoader(concatenatedTrainSet, batch_size = batch_size, shuffle=True,
                                              pin_memory=True, num_workers=8)

    weights = torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model = torchvision.models.efficientnet_v2_s(weights=weights)

    # FREEZE HIDDEN LAYERS
    # for params in model.parameters():
    #     params.requires_grad = False

    model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=numClasses)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=numEpochs)
    loss = torch.nn.CrossEntropyLoss()

    train(model, optimizer, loss, sched, trainLoader)

    testAcc, testClassAcc = valid(model, validationLoader)
    return testAcc, testClassAcc


def dryRun():
    weights = torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model = torchvision.models.efficientnet_v2_s(weights=weights)

    model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=numClasses)

    model.to(device)

    trainLoader = torch.utils.data.DataLoader(tinyTrain, batch_size = batch_size, shuffle=True, pin_memory=True, num_workers=8)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=numEpochs)
    loss = torch.nn.CrossEntropyLoss()

    train(model, optimizer, loss, sched, trainLoader)

    testAcc, testClassAcc = valid(model, validationLoader)
    return testAcc, testClassAcc

if __name__ == "__main__":
    #print(getInitialAcc())
    #print(doImage(1, 2))
    print(dryRun())
    print('done')
