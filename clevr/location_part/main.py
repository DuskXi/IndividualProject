import os

import numpy as np
from loguru import logger

from clevr.location_part.dataset import ClevrBoxPositionDataset, ClevrBoxPositionDatasetV3
import torch
import torch.utils.data

from clevr.location_part.model import Relationship, RelationshipV2, RelationshipV3

from rich.logging import RichHandler
from tqdm.rich import tqdm

logger.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


def main_1():
    dataset = ClevrBoxPositionDataset(r"D:\projects\IndividualProject\clevr\Dataset Generation\output\scenes.json", 5)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    model = Relationship(5)
    model.train()
    model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    for epoch in (progress_epoch := tqdm(range(1000), desc="Epoch")):
        losses = []
        accuracies = []
        window = 20
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels['front'].cuda()
            optimizer.zero_grad()
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            result_binary = torch.where(outputs >= 0.5, torch.tensor(1.0), torch.tensor(0.0))
            acc = (labels == result_binary).flatten()
            accuracy = (acc.sum().item() / acc.numel())
            accuracies.append(accuracy)
            if len(losses) > window:
                losses.pop(0)
                accuracies.pop(0)
            progress_epoch.set_description(f"Epoch {epoch + 1} loss: {np.average(losses):.4f}, accuracy: {np.average(accuracies) * 100:.4f} %, lr: {lr_scheduler.get_last_lr()}")
        lr_scheduler.step()
        progress_epoch.set_postfix({"lr": lr_scheduler.get_last_lr()})


def test(model, criterion, test_dataloader):
    model.eval()
    losses = []
    accuracies = []
    window = 20
    for i, data in enumerate(test_dataloader, 0):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels['front'].cuda()
        outputs = model(inputs).squeeze(1)
        loss = criterion(outputs, labels)
        losses.append(loss.item())
        result_binary = torch.where(outputs >= 0.5, torch.tensor(1.0), torch.tensor(0.0))
        acc = (labels == result_binary).flatten()
        accuracy = (acc.sum().item() / acc.numel())
        accuracies.append(accuracy)
        if len(losses) > window:
            losses.pop(0)
            accuracies.pop(0)
    return np.average(losses), np.average(accuracies) * 100


def main_2():
    dataset = ClevrBoxPositionDataset(r"D:\projects\IndividualProject\clevr\Dataset Generation\output\scenes.json", 5)
    # did not need random split, just give 200 data for test set
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - 200, 200])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

    model = RelationshipV2(5)
    model.train()
    model.cuda()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    for epoch in (progress_epoch := tqdm(range(1000), desc="Epoch")):
        losses = []
        accuracies = []
        window = 20
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels['front'].cuda()
            optimizer.zero_grad()
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            result_binary = torch.where(outputs >= 0.5, torch.tensor(1.0), torch.tensor(0.0))
            acc = (labels == result_binary).flatten()
            accuracy = (acc.sum().item() / acc.numel())
            accuracies.append(accuracy)
            if len(losses) > window:
                losses.pop(0)
                accuracies.pop(0)
            progress_epoch.set_description(f"Epoch {epoch + 1} loss: {np.average(losses):.4f}, accuracy: {np.average(accuracies) * 100:.4f} %, lr: {lr_scheduler.get_last_lr()}")

        test_loss, test_accuracy = test(model, criterion, test_dataloader)
        logger.info(f"Test loss: {test_loss:.4f}, accuracy: {test_accuracy:.4f} %")
        lr_scheduler.step()
        progress_epoch.set_postfix({"lr": lr_scheduler.get_last_lr()})


def test_3(model, criterion, test_dataloader):
    model.eval()
    losses = []
    accuracies = []
    window = 20
    for i, data in enumerate(test_dataloader, 0):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs).squeeze(1)
        loss = criterion(outputs, labels)
        losses.append(loss.item())
        result_binary = torch.where(outputs >= 0.5, torch.tensor(1.0), torch.tensor(0.0))
        acc = (labels == result_binary).flatten()
        accuracy = (acc.sum().item() / acc.numel())
        accuracies.append(accuracy)
        if len(losses) > window:
            losses.pop(0)
            accuracies.pop(0)
    return np.average(losses), np.average(accuracies) * 100


def save_model(model, model_dir, epoch, accuracy):
    model_path = os.path.join(model_dir, f"model_{epoch}_acc_{int(accuracy * 100)}.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")


def main_3():
    dataset = ClevrBoxPositionDatasetV3(r"D:\projects\IndividualProject\clevr\Dataset Generation\output\scenes.json", 5)
    # did not need random split, just give 200 data for test set
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - 200, 200])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

    model = RelationshipV3(5)
    model.train()
    model.cuda()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    for epoch in (progress_epoch := tqdm(range(1000), desc="Epoch")):
        losses = []
        accuracies = []
        window = 20
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            result_binary = torch.where(outputs >= 0.5, torch.tensor(1.0), torch.tensor(0.0))
            acc = (labels == result_binary).flatten()
            accuracy = (acc.sum().item() / acc.numel())
            accuracies.append(accuracy)
            if len(losses) > window:
                losses.pop(0)
                accuracies.pop(0)
            progress_epoch.set_description(f"Epoch {epoch + 1} loss: {np.average(losses):.4f}, accuracy: {np.average(accuracies) * 100:.4f} %, lr: {lr_scheduler.get_last_lr()}")

        # save model
        acc = np.average(accuracies)
        if epoch % 10 == 0:
            save_model(model, "D:\projects\IndividualProject\clevr\location_part\model", epoch, acc)

        test_loss, test_accuracy = test_3(model, criterion, test_dataloader)
        logger.info(f"Test loss: {test_loss:.4f}, accuracy: {test_accuracy:.4f} %")
        lr_scheduler.step()
        progress_epoch.set_postfix({"lr": lr_scheduler.get_last_lr()})


if __name__ == '__main__':
    main_3()
