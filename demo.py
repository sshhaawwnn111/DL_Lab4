import torch
import argparse
import dataloader
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
from tqdm import tqdm
from torchvision import models
import matplotlib.pyplot as plt
from ResNet import ResNet50, ResNet18
from dataloader import RetinopathyLoader
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader


def inference(model, loader):
    avg_acc = 0.0
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            for i in range(len(labels)):
                if int(pred[i]) == int(labels[i]):
                    avg_acc += 1

        avg_acc = (avg_acc / len(loader.dataset)) * 100

    cm = confusion_matrix(labels, pred.cpu())
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return avg_acc, cmn


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 64
    root = './new_dataset'
    test_dataset = RetinopathyLoader(root, 'test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for j in range(2):
        for i in range(2):
            if i == 1:
                pretrained = True
            else:
                pretrained = False

            num_classes = 5

            if j == 0:
                exp_name = 'ResNet18'
                if pretrained:
                    model = models.resnet18(pretrained=pretrained)
                    fc_inputs = model.fc.in_features
                    model.fc = nn.Linear(fc_inputs, num_classes)
                else:
                    model = ResNet18()
                model_name = 'ResNet18'
            else:
                exp_name = 'ResNet50'
                if pretrained:
                    model = models.resnet50(pretrained=pretrained)
                    fc_inputs = model.fc.in_features
                    model.fc = nn.Linear(fc_inputs, num_classes)
                else:
                    model = ResNet50()
                model_name = 'ResNet50'
            
            model.to(device)
            model.load_state_dict(torch.load(f'./demo_weights/{exp_name}_{pretrained}.pt'))

            avg_acc, cmn = inference(model, test_loader)
            

            print(f'{model_name} pre-trained-{pretrained}: {avg_acc:.2f}%')

            # print(f'+=======================================================+')
            # print(f'                          Demo                           ')
            # print(f'                                                         ')
            # print(f'      - Model: {model_name}                              ')
            # print(f'      - Pre-trained: {pretrained}                        ')
            # print(f'      - Inference Acc.(%): {avg_acc:.2f}%                ')
            # print(f'                                                         ')
            # print(f'+=======================================================+')