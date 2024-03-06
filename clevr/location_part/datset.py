import json

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset


class ClevrBoxPositionDataset(Dataset):
    def __init__(self, scene_path, num_objects=5):
        with open(scene_path, 'r') as f:
            data = json.load(f)
        self.data = data["scenes"]
        self.num_objects = num_objects

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        boxes = []
        relationships = item.get("relationships", {}).copy()
        for obj in item["objects"]:
            box = obj["bounding_box"]
            boxes.append([box["xmin"], box["xmax"], box["ymin"], box["ymax"]])
        for key, value in relationships.items():
            adj_list = value
            adj_matrix = [[0] * self.num_objects for _ in range(self.num_objects)]
            for i, line in enumerate(adj_list):
                for v in line:
                    try:
                        adj_matrix[i][v] = 1
                    except:
                        print("")
            relationships[key] = torch.tensor(adj_matrix, dtype=torch.float32)
        return torch.tensor(np.array(boxes), dtype=torch.float32), relationships
