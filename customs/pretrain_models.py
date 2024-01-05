import numpy as np

import logging
import os

import torch
import torchvision
from torchvision import transforms

from tqdm import tqdm

from config import pretrain_bs, pretrain_lr, pretrain_momentum, pretrain_epochs, pretrain_weight_decay, test_batch_size, numClasses

from architectures.resnet import resnet20, resnet56

device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")

currentDir = os.path.dirname(os.path.realpath(__file__))

logging.basicConfig(filename='pretrain.log', format='%(asctime)s : %(levelname)s : %(message)s', encoding='utf-8', level=logging.INFO)


# ORIGINAL
#normalize = transforms.Normalize(mean=[0.507, 0.4865, 0.4408], std=[0.2673, 0.2564, 0.2761])

# MODIF
normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

trainTransform = torchvision.transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 normalize])

testTransform = torchvision.transforms.Compose([transforms.ToTensor(),
                                                normalize])

savePath = os.path.join(currentDir, "../models")

testPath = os.path.join(currentDir, "../test_images")
trainPathFull = os.path.join(currentDir, "../train_images")

trainSetFull = torchvision.datasets.CIFAR10(root=trainPathFull, train=True, download=True, transform=trainTransform)

testSet = torchvision.datasets.CIFAR10(root=testPath, train=False, download=True, transform=testTransform)

testLoader = torch.utils.data.DataLoader(testSet, batch_size=test_batch_size, shuffle=False)
trainLoader = torch.utils.data.DataLoader(trainSetFull, batch_size=pretrain_bs, shuffle=False)


def train(model, optimizer, lossFunction, sched, numEpochs=pretrain_epochs):
    model.train()

    for epoch in range(numEpochs):
        for images, labels in tqdm(trainLoader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            scores = model(images)
            predLoss = lossFunction(scores, labels)
            predLoss.backward()
            optimizer.step()

        logging.info(f"Epoch: {epoch}, Loss: {predLoss}")
        sched.step()

def test(model):
    model.eval()

    correct, total = 0, 0
    classCorrect = [0 for c in range(numClasses)]
    classTotals = [0 for c in range(numClasses)]

    with torch.no_grad():
        for inputs, labels in tqdm(testLoader):
            inputs = inputs.cuda()
            labels = labels.cuda()

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

def main():
    
    # CHANGE THIS
    # important note: torchvision hub resnet has an imagenet config; we gotta make our own resnet for this to work.

    model = resnet20()

    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=pretrain_lr, momentum=pretrain_momentum, weight_decay=pretrain_weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=pretrain_epochs)
    loss = torch.nn.CrossEntropyLoss()

    train(model, optimizer, loss, sched, numEpochs=pretrain_epochs)

    testAcc, classAcc = test(model)

    print(testAcc)
    logging.info(testAcc)

    torch.save(model.state_dict(), "".join([savePath, "/cifar_resnet20.pt"]))


if __name__ == "__main__":
    logging.info(f"START: LR = {pretrain_lr}, BS = {pretrain_bs}, EPOCHS = {pretrain_epochs}")
    main()