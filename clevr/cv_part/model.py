import json
import os

import torch
from PIL import Image
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class Mobius(nn.Module):
    def __init__(self, in_features, out_features, max_iter=100, attenuation=0.1, out_threshold=0.5):
        super(Mobius, self).__init__()
        self.max_iter = max_iter
        self.attenuation = attenuation
        self.out_threshold = out_threshold
        self.fc = nn.Linear(in_features, in_features)
        self.fc_out = nn.Linear(in_features, out_features)
        self.threshold = nn.Linear(in_features, 1)

    def merge(self, prev, x):
        if self.attenuation == 0:
            return x
        x = x.clone()
        mut = self.attenuation
        for i in reversed(range(len(prev))):
            x = x * (1.0 - mut) + mut * prev[i]
            mut *= self.attenuation
        return x

    def forward(self, x):
        # s = []
        it = -1
        if type(x) == tuple:
            x, it = x
        for i in range(self.max_iter):
            x = self.fc(x)
            x = F.relu(x)
            # s.append(x)
            threshold = self.threshold(x)
            if threshold[0] > self.out_threshold:
                break
            # x = self.merge(s, x)

        x = self.fc_out(x)
        return x, i if it == -1 else (it + i) / 2



class ClassificationFC(nn.Module):
    def __init__(self, in_channel, num_class):
        super(ClassificationFC, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_class)
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.softmax(x, dim=1)


class ClassificationMobius(nn.Module):
    def __init__(self, in_channel, num_class):
        super(ClassificationMobius, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            Mobius(256, 64),
            Mobius(64, 32),
            Mobius(32, num_class)
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x, it = self.fc(x)
        return F.softmax(x, dim=1), it


class SimpleCountDataset(Dataset):
    def __init__(self, json_path, target_color='red', target_color2='blue', num_class=5):
        self.target_color = target_color
        self.target_color2 = target_color2
        self.data = json.load(open(json_path))
        self.data = self.data['scenes']
        self.base_dir = os.path.dirname(json_path)
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()
        ])

    def get(self, idx):
        data = self.data[idx]
        image = Image.open(os.path.join(self.base_dir, data['image_filename'])).convert('RGB')
        count_t1 = len([obj for obj in data['objects'] if obj['color_name'] == self.target_color])
        count_t2 = len([obj for obj in data['objects'] if obj['color_name'] == self.target_color2])
        label = -1
        if count_t1 > count_t2:
            label = 0
        elif count_t1 < count_t2:
            label = 1
        else:
            label = 2
        return image, count_t1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, count = self.get(idx)
        # label to one-hot
        label = torch.zeros(6)
        label[count] = 1
        return self.transform(image), label
