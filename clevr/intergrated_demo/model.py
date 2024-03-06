import torch
import torchvision
from torch import nn


class Relationship(nn.Module):
    def __init__(self, num_objects, hidden_size=256, upper_sampling_hidden_size=32):
        # this model neet output a matrix of size num_objects x num_objects that is a adjacency matrix
        super(Relationship, self).__init__()
        anchor = 4
        self.fc = nn.Sequential(
            nn.Linear(anchor * num_objects, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_objects * num_objects),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # upper sampling, 3 layers, target size is (num_objects, num_objects), target channel is 2
        self.upper_samplings = [nn.Sequential(
            nn.Conv2d(1, upper_sampling_hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(upper_sampling_hidden_size),
            nn.Conv2d(upper_sampling_hidden_size, upper_sampling_hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(upper_sampling_hidden_size, 1, kernel_size=3, padding=1),
        ) for _ in ["left", "right", "front", "behind"]]
        for i in range(len(self.upper_samplings)):
            setattr(self, f"upper_sampling_{i}", self.upper_samplings[i])

    def forward(self, x):
        # x: (batch_size, num_objects, 4)
        batch_size, num_objects, _ = x.size()
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = x.view(batch_size, num_objects, num_objects)
        x = x.unsqueeze(1)
        results = []
        for upper_sampling in self.upper_samplings:
            results.append(upper_sampling(x))
        # merge the results
        results = torch.cat(results, dim=1)
        return results


class ModelLoader:
    @staticmethod
    def load_relationship(path, num_objects, hidden_size=256, upper_sampling_hidden_size=32):
        model = Relationship(num_objects, hidden_size, upper_sampling_hidden_size)
        model.load_state_dict(torch.load(path))
        return model

    @staticmethod
    def load_faster_rcnn(path, num_classes=96, box_score_thresh=0.6, box_nms_thresh=0.6):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes, box_score_thresh=box_score_thresh, box_nms_thresh=box_nms_thresh)
        model.load_state_dict(torch.load(path))
        return model
