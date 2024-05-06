from typing import Optional

import numpy as np
import torch
import torch.utils.data
import pandas as pd
import argparse
from tqdm.rich import tqdm
from loguru import logger

import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# device = 'cpu'

class AttentionModule(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionModule, self).__init__()
        # 注意力得分层
        self.attention_layer = nn.Linear(feature_dim, 1)

    def forward(self, x):
        # 生成注意力权重
        attention_scores = self.attention_layer(x)
        attention_weights = torch.sigmoid(attention_scores)
        # 应用注意力权重
        attended_output = x * attention_weights
        return attended_output


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return x, y

    def __len__(self):
        return len(self.data)


class SimpleDataset2(torch.utils.data.Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __getitem__(self, index):
        x = self.data[index]
        x, relation = x[:20], x[20:]
        relation = relation.reshape(4, 5, 5)
        y = self.target[index]
        return x, relation, y

    def __len__(self):
        return len(self.data)


class Net(torch.nn.Module):
    def __init__(self, input_size=4 * 5, output_size=2, hidden_size=512, layers=4, activation_function: Optional[torch.nn.ReLU, torch.nn.GELU, torch.nn.LeakyReLU, torch.nn.ReLU6] = torch.nn.ReLU,
                 layer_norm: bool = True, dropout: float = 0.1):
        super(Net, self).__init__()
        layers_list = [torch.nn.Linear(input_size, hidden_size)]
        if layer_norm:
            layers_list.append(torch.nn.LayerNorm(hidden_size))
        layers_list.append(activation_function())
        for _ in range(layers):
            layers_list.append(torch.nn.Linear(hidden_size, hidden_size))
            layers_list.append(activation_function())
        if dropout > 0:
            layers_list.append(torch.nn.Dropout(dropout))
        layers_list.append(torch.nn.Linear(hidden_size, output_size))

        self.fc = torch.nn.Sequential(*layers_list)

    def forward(self, x):
        return self.fc(x)


class Net2(torch.nn.Module):
    def __init__(self, input_size=4 * 5, graph_size=(5, 5), output_size=2, hidden_size=512, gcn_hidden=16):
        super(Net2, self).__init__()
        self.gcn = torch.nn.Sequential(
            nn.Conv2d(4, gcn_hidden, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(gcn_hidden, gcn_hidden, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(gcn_hidden, gcn_hidden, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_size + gcn_hidden * graph_size[0] * graph_size[1], hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size),
        )

    def forward(self, x, relation):
        relation = self.gcn(relation)
        relation = relation.view(relation.size(0), -1)
        x = torch.cat([x, relation], dim=1)
        return self.fc(x)


def collate_fn(batch):
    x, y = zip(*batch)
    return torch.tensor(x, dtype=torch.float32), torch.nn.functional.one_hot(torch.tensor(y), 2)


def collate_fn2(batch):
    x, relation, y = zip(*batch)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(relation, dtype=torch.float32), torch.nn.functional.one_hot(torch.tensor(y), 2)


def load_data(file_path):
    data = pd.read_csv(file_path)
    target = data['label']
    data = data.drop(columns=['label'])
    return data.to_numpy(), target.to_numpy()


def main_attr(args):
    data, target = load_data(args.data)
    dataset = SimpleDataset(data, target)
    logger.info(f'Data shape {data.shape}')
    train_set, test_set = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.75), len(dataset) - int(len(dataset) * 0.75)])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    x_width = data.shape[1]

    model = Net(x_width, hidden_size=512).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    accs = []
    window = 100
    train_acc = []
    for epoch in range(100):
        for i, (x, y) in enumerate(progress := tqdm(train_loader)):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            # y_pred = torch.softmax(y_pred, dim=1)
            loss = criterion(y_pred, torch.argmax(y, dim=1)) if type(criterion) in [torch.nn.CrossEntropyLoss] else criterion(y_pred, y.float())
            loss.backward()
            optimizer.step()
            accuracy = (torch.argmax(y_pred, dim=1) == torch.argmax(y, dim=1)).sum().item() / args.batch_size
            accs.append(accuracy)
            progress.set_description(f'Epoch {epoch} Iter {i} Loss {loss.item():.4f} Accuracy {accuracy * 100:.4f}% Avg Acc {sum(accs) / len(accs) * 100:.4f}%')
            if len(accs) > window:
                accs.pop(0)

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            lss = []
            for i, (x, y) in enumerate(progress := tqdm(test_loader)):
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                loss = criterion(y_pred, torch.argmax(y, dim=1)) if type(criterion) in [torch.nn.CrossEntropyLoss] else criterion(y_pred, y.float())
                lss.append(loss.item())
                correct += (torch.argmax(y_pred, dim=1) == torch.argmax(y, dim=1)).sum().item()
                total += args.batch_size
                progress.set_description(f'Test Accuracy {correct / total * 100:.4f}%, {np.mean(lss)}')

    dataset_len = len(dataset)


def main_attr_save(args, save_dir):
    stop_acc_train = 0.95
    stop_acc_test = 0.75
    data, target = load_data(args.data)
    dataset = SimpleDataset(data, target)
    logger.info(f'Data shape {data.shape}')
    train_set, test_set = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.75), len(dataset) - int(len(dataset) * 0.75)])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    x_width = data.shape[1]

    model = Net(x_width, hidden_size=512).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    accs = []
    window = 100
    train_acc = []
    for epoch in range(100):
        for i, (x, y) in enumerate(progress := tqdm(train_loader)):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            # y_pred = torch.softmax(y_pred, dim=1)
            loss = criterion(y_pred, torch.argmax(y, dim=1)) if type(criterion) in [torch.nn.CrossEntropyLoss] else criterion(y_pred, y.float())
            loss.backward()
            optimizer.step()
            accuracy = (torch.argmax(y_pred, dim=1) == torch.argmax(y, dim=1)).sum().item() / args.batch_size
            accs.append(accuracy)
            progress.set_description(f'Epoch {epoch} Iter {i} Loss {loss.item():.4f} Accuracy {accuracy * 100:.4f}% Avg Acc {sum(accs) / len(accs) * 100:.4f}%')
            if len(accs) > window:
                accs.pop(0)

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            lss = []
            for i, (x, y) in enumerate(progress := tqdm(test_loader)):
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                loss = criterion(y_pred, torch.argmax(y, dim=1)) if type(criterion) in [torch.nn.CrossEntropyLoss] else criterion(y_pred, y.float())
                lss.append(loss.item())
                correct += (torch.argmax(y_pred, dim=1) == torch.argmax(y, dim=1)).sum().item()
                total += args.batch_size
                progress.set_description(f'Test Accuracy {correct / total * 100:.4f}%, {np.mean(lss)}')

        if correct / total > stop_acc_test:
            torch.save(model.state_dict(), f'{save_dir}/model.pth')
            break


def main_relation(args):
    data, target = load_data(args.data)
    dataset = SimpleDataset2(data, target)
    logger.info(f'Data shape {data.shape}')
    train_set, test_set = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.75), len(dataset) - int(len(dataset) * 0.75)])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn2)

    x_width = data.shape[1]

    model = Net2(20, hidden_size=512, gcn_hidden=32).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    accs = []
    window = 100
    for epoch in range(100):
        for i, (x, relation, y) in enumerate(progress := tqdm(train_loader)):
            x, relation, y = x.to(device), relation.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x, relation)
            # y_pred = torch.softmax(y_pred, dim=1)
            loss = criterion(y_pred, torch.argmax(y, dim=1)) if type(criterion) in [torch.nn.CrossEntropyLoss] else criterion(y_pred, y.float())
            loss.backward()
            optimizer.step()
            accuracy = (torch.argmax(y_pred, dim=1) == torch.argmax(y, dim=1)).sum().item() / args.batch_size
            accs.append(accuracy)
            progress.set_description(f'Epoch {epoch} Iter {i} Loss {loss.item():.4f} Accuracy {accuracy * 100:.4f}% Avg Acc {sum(accs) / len(accs) * 100:.4f}%')
            if len(accs) > window:
                accs.pop(0)

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            lss = []
            for i, (x, relation, y) in enumerate(progress := tqdm(test_loader)):
                x, relation, y = x.to(device), relation.to(device), y.to(device)
                y_pred = model(x, relation)
                loss = criterion(y_pred, torch.argmax(y, dim=1)) if type(criterion) in [torch.nn.CrossEntropyLoss] else criterion(y_pred, y.float())
                lss.append(loss.item())
                correct += (torch.argmax(y_pred, dim=1) == torch.argmax(y, dim=1)).sum().item()
                total += args.batch_size
                progress.set_description(f'Test Accuracy {correct / total * 100:.4f}% Loss: {loss.item()}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data.csv')
    parser.add_argument('--batch_size', type=int, default=50)
    return parser.parse_args()


if __name__ == '__main__':
    main_attr(parse_args())
