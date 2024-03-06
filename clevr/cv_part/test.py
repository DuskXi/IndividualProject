import json
import unittest

import numpy as np
import torch
import torch.utils.data
import torchvision.models as models
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm

from clevr.cv_part.dataset import ClevrObjectDetectionDataset
from clevr.cv_part.model import Mobius, ClassificationFC, ClassificationMobius, SimpleCountDataset
from clevr.cv_part.utils import draw_boxes, display_image
import cv2

from torchvision.utils import draw_bounding_boxes
def test_model(model, x, y):
    model.eval()
    prediction = model(x)
    loss = torch.nn.functional.cross_entropy(prediction, y)
    accuracy = (prediction.argmax(1) == y.argmax(1)).float().mean()
    plt.imshow(x[0].permute(1, 2, 0).cpu().numpy())
    plt.title(f"Loss: {loss.item()}, Acc: {accuracy.item()}, Pred: {prediction.argmax(1)[0].item()}, True: {y.argmax(1)[0].item()}")
    plt.show()
    return loss


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.base_loss = nn.MSELoss()

    def forward(self, outputs, targets, it, max_iter):
        # 计算基础损失
        loss = self.base_loss(outputs, targets)
        loss = loss + it / max_iter
        return loss


class TestNN(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)  # add assertion here

    def test_rcnn(self):
        model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
        model.eval()
        test_tensor = torch.rand(1, 3, 300, 400)
        prediction = model(test_tensor)
        self.assertEqual(True, True)

    def test_ssd(self):
        model = models.detection.ssd300_vgg16(pretrained=False)
        model.eval()
        test_tensor = torch.rand(1, 3, 300, 300)
        prediction = model(test_tensor)
        self.assertEqual(True, True)

    def test_simple_classify(self):
        dataset = SimpleCountDataset(r"D:\projects\IndividualProject\clevr\Dataset Generation\output\scenes.json")
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

        model = ClassificationFC(3, 6)
        model.cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(10):
            progress = tqdm(enumerate(dataloader))
            losses = []
            accs = []
            window = 10
            for i, (x, y) in progress:
                x = x.cuda()
                y = y.cuda()
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()

                acc = (pred.argmax(1) == y.argmax(1)).float().mean()
                losses.append(loss.item())
                accs.append(acc.item())
                if len(losses) > window:
                    losses.pop(0)
                    accs.pop(0)
                progress.set_description(f"Epoch {epoch}, Loss: {np.average(losses)}, Acc: {np.average(accs)}")

                if i % 40 == 0 and i != 0:
                    test_model(model, x, y)

        self.assertEqual(True, True)

    def test_logical(self):
        class ComparisonNetwork(nn.Module):
            def __init__(self):
                super(ComparisonNetwork, self).__init__()
                self.fc1 = nn.Linear(100, 128)  # 输入层到隐藏层
                self.fc2 = nn.Linear(128, 100)  # 隐藏层到输出层

            def forward(self, x):
                x = torch.relu(self.fc1(x))  # 使用ReLU激活函数
                x = self.fc2(x)
                return x

        # 初始化网络
        model = ComparisonNetwork()
        model.cuda()
        model.train()

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # 训练数据准备
        # 假设data是输入数据，labels是标签，是数据中在0.5-0.6区间数字的数量
        data = torch.rand(3000, 100)
        labels = torch.tensor([len(list(filter(lambda x: 0.5 < x < 0.6, data[i]))) for i in range(len(data))])
        data = data.cuda()
        labels = labels.cuda()

        # 训练网络
        for epoch in range(1000):  # 训练轮数
            progress = tqdm(range(0, len(data), 32))
            for i in progress:
                x = data[i:i + 32]
                y = labels[i:i + 32]
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                acc = (output.argmax(1) == y).float().mean()
                if i % 100 == 0:
                    # print(f'Epoch [{epoch}/1000], Loss: {loss.item()}, Acc: {acc.item()}')
                    progress.set_description(f'Epoch [{epoch}/1000], Loss: {loss.item()}, Acc: {acc.item()}')
        self.assertEqual(True, True)

    def test_1(self):
        path = r"D:\projects\IndividualProject\clevr\Dataset Generation\output\scenes.json"
        with open(path, "r") as f:
            data = json.load(f)
        for scene in data["scenes"]:
            for obj in scene["objects"]:
                tmp = obj["color_name"]
                obj["color_name"] = obj["size_name"]
                obj["size_name"] = tmp

        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        self.assertEqual(True, True)

    def test_dataset(self):
        dataset = ClevrObjectDetectionDataset(r"D:\projects\IndividualProject\clevr\Dataset Generation\output", r"D:\projects\IndividualProject\clevr\Dataset Generation\output\scenes.json")
        image, prediction = dataset[0]
        # display_image(image, y, dataset.labels_class)
        labels = [dataset.labels_class[i] for i in prediction["labels"]]
        box = draw_bounding_boxes(image, boxes=prediction["boxes"],
                                  labels=labels,
                                  colors="red",
                                  width=4, font_size=30)
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
