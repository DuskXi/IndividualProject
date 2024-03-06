import unittest

import numpy as np
import torch
import torch.utils.data
from loguru import logger
from tqdm.rich import tqdm
from rich.logging import RichHandler

from clevr.location_part.dataset import ClevrBoxPositionDataset, ClevrBoxPositionDatasetV3
from clevr.location_part.model import Relationship, RelationshipV2, RelationshipV3

logger.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


class TestModel(unittest.TestCase):
    def test_model_init(self):
        model = Relationship(5)
        x = torch.rand(1, 5, 4, dtype=torch.float32)
        pred = model(x)
        pred = pred.argmax(dim=1)
        result = pred.cpu().detach().numpy()
        self.assertEqual(True, True)

    def test_V3_model_input(self):
        model = RelationshipV3(5)
        x = torch.rand(1, 5, 4, dtype=torch.float32)
        pred = model(x)
        self.assertEqual(True, True)

    def test_V3_model_dataset(self):
        dataset = ClevrBoxPositionDatasetV3(r"D:\projects\IndividualProject\clevr\Dataset Generation\output\scenes.json", 5)
        x, y = dataset[0]
        self.assertEqual(True, True)

    def test_dataset(self):
        dataset = ClevrBoxPositionDataset(r"D:\projects\IndividualProject\clevr\Dataset Generation\output\scenes.json", 5)
        x, y = dataset[0]
        self.assertEqual(True, True)

    def test_train(self):
        dataset = ClevrBoxPositionDataset(r"D:\projects\IndividualProject\clevr\Dataset Generation\output\scenes.json", 5)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

        model = Relationship(5)
        model.train()
        model.cuda()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
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
                progress_epoch.set_postfix({"loss": np.average(losses), "accuracy": np.average(accuracies) * 100})
                # if i % 100 == 0 and i != 0:
                #     # print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100}, accuracy: {np.average(accuracies) * 100} %')
                #     running_loss = 0.0
                #     accuracies = []
            lr_scheduler.step()
            progress_epoch.set_postfix({"lr": lr_scheduler.get_last_lr()})

        self.assertEqual(True, True)

    def test_train(self):
        dataset = ClevrBoxPositionDataset(r"D:\projects\IndividualProject\clevr\Dataset Generation\output\scenes.json", 5)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

        model = RelationshipV2(5)
        model.train()
        model.cuda()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
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
                progress_epoch.set_postfix({"loss": np.average(losses), "accuracy": np.average(accuracies) * 100})
                # if i % 100 == 0 and i != 0:
                #     # print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100}, accuracy: {np.average(accuracies) * 100} %')
                #     running_loss = 0.0
                #     accuracies = []
            lr_scheduler.step()
            progress_epoch.set_postfix({"lr": lr_scheduler.get_last_lr()})

        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
