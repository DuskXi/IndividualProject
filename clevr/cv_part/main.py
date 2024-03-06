import faulthandler
import os

import numpy as np
import torch
import torchvision
from loguru import logger

from rich.logging import RichHandler
from tqdm.rich import tqdm
import torch.utils.data

from clevr.cv_part.dataset import ClevrObjectDetectionDataset
from clevr.cv_part.utils import display_image

logger.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


def train_epoch(model, optimizer, train_loader, test_loader, device, epoch, labels_class, test_per_step=100):
    model.train()
    accuracies = []
    window = 5
    for i, (x, y) in enumerate(progress := tqdm(train_loader, desc=f"Training, epoch:{epoch}")):
        x = x.to(device)
        y = [{"boxes": box.to(device), "labels": label.to(device)} for box, label in zip(y["boxes"], y["labels"])]
        optimizer.zero_grad()
        loss = model(x, y)
        total_loss = sum(loss for loss in loss.values())
        total_loss.backward()
        optimizer.step()
        progress.set_description(f"Training, epoch:{epoch}, loss:{total_loss.item()}, acc:{np.average(accuracies) * 100:.4f}%")
        if i * x.shape[0] % test_per_step == 0:
            acc = test_epoch(model, test_loader, device, epoch, labels_class, only_once=True)
            accuracies.append(acc)
            if len(accuracies) > window:
                accuracies.pop(0)
            model.train()


def test_epoch(model, test_loader, device, epoch, labels_class, only_once=False):
    model.eval()
    accuracies = []
    for x, y in test_loader:
        x = x.to(device)
        y = [{"boxes": box.to(device), "labels": label.to(device)} for box, label in zip(y["boxes"], y["labels"])]
        with torch.no_grad():
            result = model(x, y)
            accuracy = 0
            for i in range(len(result)):
                correct = 0
                for j in range(len(result[i]["labels"])):
                    if result[i]["labels"][j] in y[i]["labels"]:
                        correct += 1
                accuracy += correct / len(y[i]["labels"])
            accuracy /= len(result)
            accuracies.append(accuracy)
        try:
            display_image(x[0].cpu(), result[0], labels_class, title=f"Epoch:{epoch}, accuracy:{accuracy * 100:.4f}%")
        except Exception as e:
            logger.error(e)
            logger.error(result)
        if only_once:
            break
        # progress.set_description(f"Testing, epoch:{epoch}, loss:{total_loss.item()}")
    return np.average(accuracies)


def save_model(model, model_dir, epoch, accuracy):
    model_path = os.path.join(model_dir, f"model_{epoch}_acc_{int(accuracy*100)}.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")


def main():
    batch_size = 10
    test_per_step = 100
    dataset = ClevrObjectDetectionDataset(r"D:\projects\IndividualProject\clevr\Dataset Generation\output", r"D:\projects\IndividualProject\clevr\Dataset Generation\output\scenes.json")
    train, test = torch.utils.data.random_split(dataset, [int(0.95 * len(dataset)), len(dataset) - int(0.95 * len(dataset))])
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=len(dataset.labels_class), box_score_thresh=0.5, box_nms_thresh=0.5)
    model.to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        train_epoch(model, optimizer, train_loader, test_loader, "cuda", epoch, dataset.labels_class, test_per_step)
        accuracy = test_epoch(model, test_loader, "cuda", epoch, dataset.labels_class)
        save_model(model, r"D:\projects\IndividualProject\clevr\cv_part", epoch, accuracy)


if __name__ == '__main__':
    faulthandler.enable()
    main()
