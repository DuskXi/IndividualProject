import json
import os.path

import numpy as np
import torch
import torch.utils.data
import pandas as pd
import argparse

from rich.logging import RichHandler
from rich.progress import Progress
from tqdm.rich import tqdm
from loguru import logger

import torch.nn as nn
logger.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FistDataset(torch.utils.data.Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return x, y

    def __len__(self):
        return len(self.data)


class Net(torch.nn.Module):
    def __init__(self, input_size=4 * 5, output_size=2, hidden_size=512):
        super(Net, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.fc(x)


def collate_fn(batch):
    x, y = zip(*batch)
    return torch.tensor(x, dtype=torch.float32), torch.nn.functional.one_hot(torch.tensor(y), 2)


def train_epoch(batch_size, criterion, model, optimizer, train_loader, progress: Progress):
    accuracies = []
    losses = []
    # task = progress.add_task('Training...', total=len(train_loader))
    model.train()
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, torch.argmax(y, dim=1)) if type(criterion) in [torch.nn.CrossEntropyLoss] else criterion(y_pred, y.float())
        loss.backward()
        optimizer.step()
        accuracy = (torch.argmax(y_pred, dim=1) == torch.argmax(y, dim=1)).sum().item() / batch_size
        accuracies.append(accuracy)
        losses.append(loss.item())
        # progress.update(task, advance=1, accuracy=accuracy)
        # TODO: Add logging
    return np.mean(accuracies), np.mean(losses)


def test_epoch(batch_size, criterion, model, test_loader, progress: Progress):
    accuracies = []
    losses = []
    # task = progress.add_task('Testing...', total=len(test_loader))
    model.eval()
    for i, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, torch.argmax(y, dim=1)) if type(criterion) in [torch.nn.CrossEntropyLoss] else criterion(y_pred, y.float())
        accuracy = (torch.argmax(y_pred, dim=1) == torch.argmax(y, dim=1)).sum().item() / batch_size
        accuracies.append(accuracy)
        losses.append(loss.item())
        # progress.update(task, advance=1, accuracy=accuracy)

    return np.mean(accuracies), np.mean(losses)


def train(model, train_loader, test_loader, optimizer, criterion, stop_acc_test, stop_delay, epochs, batch_size, progress):
    task = progress.add_task('Epoch...', total=epochs)
    for epoch in range(epochs):
        train_accuracy, train_loss = train_epoch(batch_size, criterion, model, optimizer, train_loader, progress)
        test_accuracy, test_loss = test_epoch(batch_size, criterion, model, test_loader, progress)
        progress.update(task, advance=1, description=f"Epoch {epoch + 1}/{epochs}, Train Accuracy: {train_accuracy * 100:.2f}%, Train Loss: {np.mean(train_loss):.4f}, Test Accuracy: {test_accuracy * 100:.2f}%, Test Loss: {np.mean(test_loss):.4f}")
        if test_accuracy >= stop_acc_test:
            if stop_delay > 0:
                yield epoch, train_accuracy, train_loss, test_accuracy, test_loss
            stop_delay -= 1
            if stop_delay == 0:
                break
        if epoch == epochs - 1:
            yield epoch, train_accuracy, train_loss, test_accuracy, test_loss


def train_rule(save_dir, name, model, train_loader, test_loader, optimizer, criterion, stop_acc_test, stop_delay, tolerance, epochs, batch_size, progress: Progress):
    best_epoch = 0
    best_test_accuracy = 0
    for epoch, train_accuracy, train_loss, test_accuracy, test_loss in train(model, train_loader, test_loader, optimizer, criterion, stop_acc_test, stop_delay, epochs, batch_size, progress):
        if test_accuracy > stop_acc_test - tolerance:
            model_name = f"model_{name}_{epoch}_{test_accuracy * 100:.2f}%.pth"
            torch.save(model.state_dict(), os.path.join(save_dir, model_name))
            logger.info(f"Model saved: {model_name}")
            if test_accuracy > best_test_accuracy:
                best_epoch = epoch
                best_test_accuracy = test_accuracy
    return best_epoch, best_test_accuracy


def run_task(dataset: FistDataset, save_dir, name, batch_size, split_ratio, epochs, stop_acc_test, stop_delay, tolerance, progress: Progress):
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    split_size = int(len(dataset) * split_ratio)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [split_size, len(dataset) - split_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    best_epoch, best_test_accuracy = train_rule(save_dir, name, model, train_loader, test_loader, optimizer, criterion, stop_acc_test, stop_delay, tolerance, epochs, batch_size, progress)
    return best_epoch, best_test_accuracy


def load_data(split_file, data_file, min_len=100):
    split = pd.read_csv(split_file)
    data = pd.read_csv(data_file)
    data.drop(columns=['label'], inplace=True)
    # split =split[['positive_index', 'negative_index']].to_numpy()
    for i, (pos_idx, neg_idx) in enumerate(split[['positive_index', 'negative_index']].to_numpy()):
        pos_idx = json.loads(pos_idx)
        neg_idx = json.loads(neg_idx)
        pos_idx = list(map(lambda x: int(x.split('_')[1]), pos_idx))
        neg_idx = list(map(lambda x: int(x.split('_')[1]), neg_idx))
        if len(pos_idx) + len(neg_idx) < min_len:
            continue
        full_idx = pos_idx + neg_idx
        sub_data = data.iloc[full_idx].copy()
        sub_data['label'] = 0
        sub_data.loc[pos_idx, 'label'] = 1
        X = sub_data.drop(columns=['label']).to_numpy()
        y = sub_data['label'].to_numpy()
        dataset = FistDataset(X, y)
        yield dataset, i


def main(args):
    dataset = load_data(args.split_file, args.data_file)
    split = pd.read_csv(args.split_file)
    with Progress() as progress:
        task = progress.add_task('Training...', total=len(split))
        for i, (data, index) in enumerate(dataset):
            save_dir = f'training/1th_cache/models'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            best_epoch, best_test_accuracy = run_task(data, save_dir, f'task-{i}', 32, 0.8, 100, 0.78, 5, 0.02, progress)
            logger.info(f"Task {i} best epoch: {best_epoch}, best test accuracy: {best_test_accuracy * 100:.2f}%")
            progress.update(task, advance=1, description=f"Task {i} best epoch: {best_epoch}, best test accuracy: {best_test_accuracy * 100:.2f}%")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_file', type=str, default='example/example_spilt.csv')
    parser.add_argument('--data_file', type=str, default='example/example_data.csv')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
