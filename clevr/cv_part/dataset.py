import json
import os

import numpy as np
import torch
from PIL import Image
from loguru import logger
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class ClevrObjectDetectionDataset(Dataset):
    def __init__(self, image_base_dir, scene_path):
        with open(scene_path, 'r') as f:
            data = json.load(f)
        self.data = data["scenes"]
        self.image_base_dir = image_base_dir
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ])
        self.sizes = ["small", "large"]
        self.colors = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
        self.materials = ["rubber", "metal"]
        self.shapes = ["cube", "cylinder", "sphere"]
        self.labels_class = self.all_labels()

    @staticmethod
    def attributes_to_label(attributes):
        return f"{attributes['size']}_{attributes['color']}_{attributes['material']}_{attributes['shape']}"

    @staticmethod
    def get_total_classes():
        return 2 * 8 * 2 * 3

    def all_labels(self):
        labels = []
        for size in self.sizes:
            for color in self.colors:
                for material in self.materials:
                    for shape in self.shapes:
                        labels.append(self.attributes_to_label({
                            "size": size,
                            "color": color,
                            "material": material,
                            "shape": shape
                        }))
        return labels

    def attributes_to_label_index(self, attributes):
        return self.labels_class.index(self.attributes_to_label(attributes))

    def label_index_to_attributes(self, label_index):
        label = self.labels_class[label_index]
        size, color, material, shape = label.split("_")
        return {
            "size": size,
            "color": color,
            "material": material,
            "shape": shape
        }

    def get_image(self, idx):
        item = self.data[idx]
        return Image.open(os.path.join(self.image_base_dir, item["image_filename"])).convert('RGB')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(os.path.join(self.image_base_dir, item["image_filename"])).convert('RGB')
        image = self.transform(image)
        height, width = image.shape[1:]
        boxes = []
        labels = []
        for obj in item["objects"]:
            x1 = (obj['bounding_box']['xmin'] + 1) * width / 2
            y1 = (obj['bounding_box']['ymin'] + 1) * height / 2
            x2 = (obj['bounding_box']['xmax'] + 1) * width / 2
            y2 = (obj['bounding_box']['ymax'] + 1) * height / 2
            boxes.append([x1, y1, x2, y2])
            attributes = {
                "size": obj["size_name"],
                "color": obj["color_name"],
                "material": obj["material"],
                "shape": obj["shape"]
            }
            labels.append(self.attributes_to_label_index(attributes))
        return image, {
            "boxes": torch.tensor(np.array(boxes), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long)
        }
