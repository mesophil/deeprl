import numpy as np

import logging
import os

import torch
import torchvision
from torchvision import transforms
from torchvision.models import resnet18, resnet50, ResNet18_Weights
import timm

from tqdm import tqdm

from make_image import makeImage

from transformers import AutoModelForImageClassification

from config import batch_size, learning_rate, momentum, numEpochs, test_batch_size

import evaluate

currentDir = os.path.dirname(os.path.realpath(__file__))

transform = torchvision.transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor()])

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

logging.basicConfig(filename='my2.log', format='%(asctime)s : %(levelname)s : %(message)s', encoding='utf-8', level=logging.INFO)

trainPath = os.path.join(currentDir, "../images")
testPath = os.path.join(currentDir, "../test_images")

testSet = torchvision.datasets.CIFAR10(root=testPath, train=False, download=True, transform=transform)
testLoader = torch.utils.data.DataLoader(testSet, batch_size = test_batch_size, shuffle=False)

def train(model, optimizer, lossFunction, trainLoader, numEpochs = numEpochs):
    
    model.train()
    for epoch in range(numEpochs):
        # for images, labels in tqdm(trainLoader):
        #     images = images.cuda()
        #     labels = labels.cuda()

            # scores = model(images)
            # predLoss = lossFunction(scores, labels)
            # predLoss.backward()
        for batch in tqdm(trainLoader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

def test_old(model, testLoader):
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

def test(model, testLoader):
    metric = evaluate.load("accuracy")
    model.eval()

    for batch in tqdm(testLoader):
        print(batch)
        batch = {k: v.to(device) for k, v in batch}

        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    
    return metric.compute()


def getInitialAcc():
    # model = resnet50(weights="../models/pytorch_model.bin")
    model = AutoModelForImageClassification.from_pretrained("yukiyan/resnet-50-finetuned-eurosat")
    model.to(device)
    testAcc = test(model, testLoader)
    return testAcc

def doImage(phrase, dire):
    makeImage(phrase, dire)

    trainSet = torchvision.datasets.ImageFolder(trainPath, transform = transform)
    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size = batch_size, shuffle=True)
    model = resnet50(weights="../models/pytorch_model.bin")
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    loss = torch.nn.CrossEntropyLoss()

    train(model, optimizer, loss, trainLoader)

    testAcc = test(model, testLoader)

    return testAcc

if __name__ == "__main__":
    print(getInitialAcc())
    print('done')