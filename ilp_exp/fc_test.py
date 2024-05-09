from typing import Optional

import numpy as np
import torch
import torch.utils.data
import pandas as pd
import argparse

from matplotlib import pyplot as plt
from tqdm.rich import tqdm
from loguru import logger
import torch.onnx

import torch.nn as nn
import torch.nn.functional as F

import configs

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
    def __init__(self, input_size=4 * 5, output_size=2, hidden_size=512, layers=4, activation_function=torch.nn.ReLU,
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
    def __init__(self, input_size=4 * 5, graph_size=(5, 5), output_size=2, hidden_size=512, gcn_hidden=16, layers=4, activation_function=torch.nn.ReLU,
                 layer_norm: bool = True, dropout: float = 0.1):
        super(Net2, self).__init__()
        self.gcn = torch.nn.Sequential(
            nn.Conv2d(4, gcn_hidden, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(gcn_hidden, gcn_hidden, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(gcn_hidden, gcn_hidden, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
        )

        layers_list = [torch.nn.Linear(input_size + gcn_hidden * graph_size[0] * graph_size[1], hidden_size)]
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

        # self.fc = torch.nn.Sequential(
        #     torch.nn.Linear(input_size + gcn_hidden * graph_size[0] * graph_size[1], hidden_size),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(hidden_size, hidden_size),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(hidden_size, hidden_size),
        #     torch.nn.Dropout(0.1),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(hidden_size, output_size),
        # )

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


def main_attr_data_len_test(args, output_dir='plot/attr_relation'):
    X, Y = [], []
    for i in [50, 75, 100, 250, 500, 1000, 1500, 2000, 3000]:
        name, x, y = list(main_attr(args, output_dir=output_dir, dataset_limit=i))[0]
        X.append(x)
        Y.append(y)

    fig = plt.figure()
    plt.plot(X, Y)
    plt.xlabel('Dataset Size')
    plt.ylabel('Final Accuracy(%)')
    plt.title('Final Accuracy vs Dataset Size')
    plt.savefig(f'{output_dir}/dataset size VS acc.png')
    plt.close(fig)


def main_attr_log(args, output_dir='plot/attr_relation'):
    # pass
    return list(main_attr(args, output_dir=output_dir))


def main_attr(args, cfgs=configs.exp_attr, output_dir='plot/test', dataset_limit=-1):
    logger.info(f'Running with {cfgs}')
    data, target = load_data(args.data)
    if dataset_limit > 0:
        data = data[:dataset_limit]
        target = target[:dataset_limit]
        cfgs = [configs.exp_attr_base]
    dataset = SimpleDataset(data, target)
    logger.info(f'Data shape {data.shape}')
    for cfg in cfgs:
        train_set, test_set = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.75), len(dataset) - int(len(dataset) * 0.75)])
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

        x_width = data.shape[1]

        model = Net(x_width, hidden_size=cfg['hidden_size'], layers=cfg['layers'], activation_function=cfg['activation_function'], layer_norm=cfg['layer_norm'], dropout=cfg['dropout']).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        train_epoch_accuracy = []
        test_epoch_accuracy = []
        window = 50
        for epoch in range(50):
            train_accuracy = []
            for i, (x, y) in enumerate(progress := tqdm(train_loader)):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                y_pred = model(x)
                # y_pred = torch.softmax(y_pred, dim=1)
                loss = criterion(y_pred, torch.argmax(y, dim=1)) if type(criterion) in [torch.nn.CrossEntropyLoss] else criterion(y_pred, y.float())
                loss.backward()
                optimizer.step()
                accuracy = (torch.argmax(y_pred, dim=1) == torch.argmax(y, dim=1)).sum().item() / args.batch_size
                train_accuracy.append(accuracy)
                progress.set_description(
                    f'Epoch {epoch} Iter {i} Loss {loss.item():.4f} Accuracy {accuracy * 100:.4f}% Avg Acc {sum(train_accuracy[-window:]) / len(train_accuracy[-window:]) * 100:.4f}%')
            train_epoch_accuracy.append(sum(train_accuracy) / len(train_accuracy))
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

                test_epoch_accuracy.append(correct / total)

        dataset_len = len(dataset)

        fig = plt.figure()
        plt.plot(range(len(train_epoch_accuracy)), train_epoch_accuracy, label='Train')
        plt.plot(range(len(test_epoch_accuracy)), test_epoch_accuracy, label='Test')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Epoch\n' + f'Final Test Accuracy: {np.mean(test_epoch_accuracy[-7:]) * 100:.4f}%')
        plt.savefig(f'{output_dir}/{("num_exp_" if dataset_limit != -1 else "") + cfg["name"]}-data_size_{dataset_len}-Acc_{np.mean(test_epoch_accuracy[-7:]) * 100:.2f}%.png')
        plt.close(fig)

        yield cfg['name'], dataset_len, np.mean(test_epoch_accuracy[-7:])


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


def main_relation_data_len_test(args, output_dir='plot/attr_relation_gnn'):
    X, Y = [], []
    for i in [50, 75, 100, 250, 500, 1000, 1500, 2000, 3000]:
        name, x, y = list(main_relation(args, output_dir=output_dir, dataset_limit=i))[0]
        X.append(x)
        Y.append(y)

    fig = plt.figure()
    plt.plot(X, Y)
    plt.xlabel('Dataset Size')
    plt.ylabel('Final Accuracy(%)')
    plt.title('Final Accuracy vs Dataset Size')
    plt.savefig(f'{output_dir}/dataset size VS acc.png')
    plt.close(fig)


def main_relation_log(args):
    list(main_relation(args))


def main_relation(args, cfgs=configs.exp_attr, output_dir='plot/attr_relation_gnn', dataset_limit=-1):
    logger.info(f'Running with {cfgs}')
    data, target = load_data(args.data)
    if dataset_limit > 0:
        data = data[:dataset_limit]
        target = target[:dataset_limit]
        cfgs = [configs.exp_attr_base]
    dataset = SimpleDataset2(data, target)
    logger.info(f'Data shape {data.shape}')
    for cfg in cfgs:
        train_set, test_set = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.75), len(dataset) - int(len(dataset) * 0.75)])
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn2)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn2)

        x_width = data.shape[1]

        model = Net2(20, gcn_hidden=32, hidden_size=cfg['hidden_size'], layers=cfg['layers'], activation_function=cfg['activation_function'], layer_norm=cfg['layer_norm'], dropout=cfg['dropout']).to(
            device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        train_epoch_accuracy = []
        test_epoch_accuracy = []
        window = 100
        for epoch in range(50):
            train_accuracy = []
            for i, (x, relation, y) in enumerate(progress := tqdm(train_loader)):
                x, relation, y = x.to(device), relation.to(device), y.to(device)
                optimizer.zero_grad()
                y_pred = model(x, relation)
                # y_pred = torch.softmax(y_pred, dim=1)
                loss = criterion(y_pred, torch.argmax(y, dim=1)) if type(criterion) in [torch.nn.CrossEntropyLoss] else criterion(y_pred, y.float())
                loss.backward()
                optimizer.step()
                accuracy = (torch.argmax(y_pred, dim=1) == torch.argmax(y, dim=1)).sum().item() / args.batch_size
                train_accuracy.append(accuracy)
                progress.set_description(
                    f'Epoch {epoch} Iter {i} Loss {loss.item():.4f} Accuracy {accuracy * 100:.4f}% Avg Acc {sum(train_accuracy[-window:]) / len(train_accuracy[-window:]) * 100:.4f}%')

            train_epoch_accuracy.append(sum(train_accuracy) / len(train_accuracy))

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

                test_epoch_accuracy.append(correct / total)

        dataset_len = len(dataset)

        fig = plt.figure()
        plt.plot(range(len(train_epoch_accuracy)), train_epoch_accuracy, label='Train')
        plt.plot(range(len(test_epoch_accuracy)), test_epoch_accuracy, label='Test')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Epoch\n' + f'Final Test Accuracy: {np.mean(test_epoch_accuracy[-7:]) * 100:.4f}%')
        plt.savefig(f'{output_dir}/{("num_exp_" if dataset_limit != -1 else "") + cfg["name"]}-data_size_{dataset_len}-Acc_{np.mean(test_epoch_accuracy[-7:]) * 100:.2f}%.png')
        plt.close(fig)

        yield cfg['name'], dataset_len, np.mean(test_epoch_accuracy[-7:])


def model_print(args):
    model = Net()
    model.eval()
    input_tensor = torch.randn(1, 20)
    onnx_path = "model-1.onnx"
    torch.onnx.export(model,  # 被导出的模型
                      input_tensor,  # 模型的输入数据
                      onnx_path,  # 导出的 ONNX 文件的路径
                      export_params=True,  # 是否导出权重
                      opset_version=10,  # ONNX 的操作集版本
                      do_constant_folding=True,  # 是否进行常数折叠优化
                      input_names=['input'],  # 输入数据的名字
                      output_names=['output'],  # 输出数据的名字
                      dynamic_axes={'input': {0: 'batch_size'},  # 可变维度
                                    'output': {0: 'batch_size'}})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data.csv')
    parser.add_argument('--batch_size', type=int, default=50)
    return parser.parse_args()


if __name__ == '__main__':
    model_print(parse_args())
