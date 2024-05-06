import json
import os
import re

import pandas as pd
import torch
import torch.utils.data
from torch import nn
import numpy as np

device = torch.device('cuda')

label_size = ['small', 'big']
label_shape = ['cylinder', 'cube', 'sphere']
label_material = ['rubber', 'metal']
label_color = ['red', 'green', 'blue', 'yellow', 'gray', 'brown', 'purple', 'cyan']


def load_model_dict(pth='training/1th_cache/models/model_task-0_0_81.82%.pth'):
    model_state_dict = torch.load(pth, map_location=device)
    flattened_weights = []
    for param_tensor in model_state_dict:
        flattened_array = model_state_dict[param_tensor].flatten()
        flattened_weights.append(flattened_array)

    max_len = max([len(x) for x in flattened_weights])
    for i in range(len(flattened_weights)):
        if len(flattened_weights[i]) < max_len:
            flattened_weights[i] = nn.functional.pad(flattened_weights[i], (0, max_len - len(flattened_weights[i])))
    weight_matrix = torch.stack(flattened_weights)
    return weight_matrix


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # input shape: (1, 10, 262144)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((11, 11))
        # self.polymerization = nn.Sequential(
        #     nn.Linear(10, 512),
        #     nn.LayerNorm(512),
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 1),
        #     nn.ReLU(),
        # )

        self.mid_fc = nn.Sequential(
            nn.Linear(256 * 11 * 11, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        self.fc_size = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )
        self.fc_shape = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
        )
        self.fc_material = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )
        self.fc_color = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 8),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.avgpool(x)
        # x = x.squeeze(1)
        # x =x.permute(0, 2, 1)
        # x = self.polymerization(x)
        x = torch.flatten(x, 1)
        x = self.mid_fc(x)
        size = self.fc_size(x)
        shape = self.fc_shape(x)
        material = self.fc_material(x)
        color = self.fc_color(x)
        return size, shape, material, color


class WeightDataset(torch.utils.data.Dataset):
    def __init__(self, model_dir='training/1th_cache/models/', split_file='example/example_spilt.csv'):
        self.models = [os.path.join(model_dir, x) for x in os.listdir(model_dir) if x.endswith('.pth')]
        self.models = [{'index': int(re.search(r'(task-)(\d+)', x).group(2)), 'path': x} for x in self.models]
        self.split_data = pd.read_csv(split_file)

    def __len__(self):
        return len(self.models)

    def __getitem__(self, idx):
        x = self.models[idx]
        y_index = x['index']
        x = load_model_dict(x['path']).unsqueeze(0)
        y = self.split_data.iloc[y_index]
        rule = json.loads(y['positive'])
        size = list(map(lambda x: 1 if x in rule['size'] else 0, label_size))
        shape = list(map(lambda x: 1 if x in rule['shape'] else 0, label_shape))
        material = list(map(lambda x: 1 if x in rule['material'] else 0, label_material))
        color = list(map(lambda x: 1 if x in rule['color'] else 0, label_color))
        size, shape, material, color = map(lambda x: torch.tensor(x, dtype=torch.float32, device=device), [size, shape, material, color])
        return x, size, shape, material, color


def collate_fn(batch):
    x, size, shape, material, color = zip(*batch)
    x = torch.stack(x)
    size = torch.stack(size)
    shape = torch.stack(shape)
    material = torch.stack(material)
    color = torch.stack(color)
    return x, size, shape, material, color


def train():
    dataset = WeightDataset()
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.MSELoss()
    losses = []
    accuracies = []
    for x, size, shape, material, color in train_loader:
        optimizer.zero_grad()
        size_pred, shape_pred, material_pred, color_pred = model(x)
        size_loss = criterion(size_pred, size)
        shape_loss = criterion(shape_pred, shape)
        material_loss = criterion(material_pred, material)
        color_loss = criterion(color_pred, color)
        loss = size_loss + shape_loss + material_loss + color_loss / 4
        loss.backward()
        optimizer.step()
        accuracy_size = (size_pred.int() == size.int()).sum() / (size.shape[0] * 2)
        accuracy_shape = (shape_pred.int() == shape.int()).sum() / (shape.shape[0] * 3)
        accuracy_material = (material_pred.int() == material.int()).sum() / (shape.shape[0] * 2)
        accuracy_color = (color_pred.int() == color.int()).sum() / (shape.shape[0] * 8)
        accuracy = (accuracy_size + accuracy_shape + accuracy_material + accuracy_color) / 4
        losses.append(loss.item())
        accuracies.append(accuracy.item())
        print(f"loss: {np.mean(losses[-10:]):.3f}, accuracy: {np.mean(accuracies[-10:]) * 100:.2f}%")


def main():
    train()


if __name__ == '__main__':
    main()
