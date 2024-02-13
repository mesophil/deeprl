import numpy as np

import logging
import os
from datetime import datetime

import torch
import torchvision
from torchvision import transforms

from functools import partial

import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path

from make_image import makeImage

from tin_config import batch_size, learning_rate, momentum, numEpochs, test_batch_size, numClasses
from tinyimagenet_utils.tin3 import TinyImageNet

from accelerate import Accelerator

from moco import MoCo_ResNet


logging.basicConfig(filename='tinyimagenet.log', format='%(asctime)s : %(levelname)s : %(message)s', encoding='utf-8', level=logging.INFO)
date = datetime.now().strftime('%m%d_%H%M_')

currentDir = os.path.dirname(os.path.realpath(__file__))
modelPath = os.path.join(currentDir, "../models")
trainPath = os.path.join(currentDir, "../images")
tinPath = os.path.join(currentDir, "../tiny_image_net/tiny-imagenet-200/")

device = torch.device("cuda")

normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
trainTransform = torchvision.transforms.Compose([transforms.Resize(64),
                                                 transforms.RandomHorizontalFlip(),
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
        #logging.info(f"Epoch {epoch} Loss: {predLoss}")

def valid(model, validationLoader):
    model.eval()

    correct, total = 0, 0
    classCorrect = [0 for c in range(numClasses)]
    classTotals = [0 for c in range(numClasses)]

    accelerator = Accelerator()
    model, validationLoader = accelerator.prepare(model, validationLoader)

    with torch.no_grad():
        for inputs, labels in tqdm(validationLoader):
            # inputs, labels = inputs.to(device), labels.to(device)

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
    # efficientnet
    # weights = torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
    # model = torchvision.models.efficientnet_v2_s(weights=weights)
    
    # model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=numClasses)
    
    
    
    # model.fc =  torch.nn.Linear(in_features=1280, out_features=numClasses)

    # checkpoint = torch.load(os.path.join(modelPath, "0110_2059EfficientNetTIN.pt"))

    # model.load_state_dict(checkpoint['model_state_dict'])
    
    # moco
    model = torchvision.models.__dict__['resnet50']()
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()
    
    checkpoint = torch.load(os.path.join(modelPath, "r-50-1000ep.pth.tar"))
    state_dict = checkpoint['state_dict']
    
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % 'fc'):
            # remove prefix
            state_dict[k[len("module.base_encoder."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
        
    model.fc = torch.nn.Linear(in_features=2048, out_features=numClasses)
        
    

    testAcc, testClassAcc = valid(model, validationLoader)
    return testAcc, testClassAcc

def doImage(phrase, dire):
    makeImage(phrase, dire)

    trainSet = torchvision.datasets.ImageFolder(trainPath, transform = trainTransform)
    concatenatedTrainSet = torch.utils.data.ConcatDataset([trainSet, tinyTrain])
    trainLoader = torch.utils.data.DataLoader(concatenatedTrainSet, batch_size = batch_size, shuffle=True,
                                              pin_memory=True, num_workers=8)

    # weights = torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
    # model = torchvision.models.efficientnet_v2_s(weights=weights)
    
    # model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=numClasses)
    
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    model = torchvision.models.resnet50(weights=weights)
    model.fc = torch.nn.Linear(in_features=1280, out_features=numClasses)

    # FREEZE HIDDEN LAYERS
    # for params in model.parameters():
    #     params.requires_grad = False

    

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=numEpochs)
    loss = torch.nn.CrossEntropyLoss()

    # EFFICIENTNET TIN
    
    # checkpoint = torch.load(os.path.join(modelPath, "0110_2059EfficientNetTIN.pt"))

    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    
    # MOCO???
    checkpoint = torch.load(os.path.join(modelPath, "r-50-1000ep.pth.tar"))
    
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    train(model, optimizer, loss, sched, trainLoader)

    testAcc, testClassAcc = valid(model, validationLoader)
    return testAcc, testClassAcc


def pretrainModel():
    logging.info(f"Pretraining: LR {learning_rate} BS {batch_size} EPOCHS {numEpochs} MOM {momentum}")
    weights = torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model = torchvision.models.efficientnet_v2_s(weights=weights)

    model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=numClasses)

    # model.to(device)

    trainLoader = torch.utils.data.DataLoader(tinyTrain, batch_size = batch_size, shuffle=True, pin_memory=True, num_workers=8)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=numEpochs)
    loss = torch.nn.CrossEntropyLoss()

    train(model, optimizer, loss, sched, trainLoader)

    testAcc, testClassAcc = valid(model, validationLoader)

    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(modelPath, "".join([date, "EfficientNetTIN.pt"])))

    return testAcc, testClassAcc


def ckptTest():
    logging.info(f"Training from checkpoint: LR {learning_rate} BS {batch_size} EPOCHS {numEpochs} MOM {momentum}")

    weights = torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model = torchvision.models.efficientnet_v2_s(weights=weights)

    model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=numClasses)

    # model.to(device)

    trainLoader = torch.utils.data.DataLoader(tinyTrain, batch_size = batch_size, shuffle=True, pin_memory=True, num_workers=8)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=numEpochs)
    loss = torch.nn.CrossEntropyLoss()

    checkpoint = torch.load(os.path.join(modelPath, "0110_2059EfficientNetTIN.pt"))

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    train(model, optimizer, loss, sched, trainLoader)

    testAcc, testClassAcc = valid(model, validationLoader)

    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(modelPath, "".join([date, "EfficientNetTIN.pt"])))

    return testAcc, testClassAcc

def mocoTest():
    model = torchvision.models.__dict__['resnet50']()
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()
    
    checkpoint = torch.load(os.path.join(modelPath, "r-50-1000ep.pth.tar"))
    state_dict = checkpoint['state_dict']
    
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % 'fc'):
            # remove prefix
            state_dict[k[len("module.base_encoder."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
        
    model.fc = torch.nn.Linear(in_features=2048, out_features=numClasses)
    
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))

    optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=momentum)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=numEpochs)
    loss = torch.nn.CrossEntropyLoss()

    # model.load_state_dict(checkpoint['state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer'])
    
    trainLoader = torch.utils.data.DataLoader(tinyTrain, batch_size = batch_size, shuffle=True, pin_memory=True, num_workers=8)

    train(model, optimizer, loss, sched, trainLoader)

    testAcc, testClassAcc = valid(model, validationLoader)
    
    return testAcc, testClassAcc

if __name__ == "__main__":
    #print(getInitialAcc())
    #print(doImage(1, 2))
    #logging.info(f"Results: {pretrainModel()}")
    #logging.info(f"Results: {ckptTest()}")
    print(mocoTest())
    print('done')
