import torch
from torch import nn
from torch.nn import functional as F


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
            nn.ReLU()
        )
        # upper sampling, 3 layers, target size is (num_objects, num_objects), target channel is 2
        self.upper_sampling = nn.Sequential(
            nn.Conv2d(1, upper_sampling_hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(upper_sampling_hidden_size, upper_sampling_hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(upper_sampling_hidden_size, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # x: (batch_size, num_objects, 4)
        batch_size, num_objects, _ = x.size()
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = x.view(batch_size, num_objects, num_objects)
        x = self.upper_sampling(x.unsqueeze(1))
        return x


class RelationshipV2(nn.Module):
    def __init__(self, num_objects, hidden_size=256, upper_sampling_hidden_size=32):
        # this model neet output a matrix of size num_objects x num_objects that is a adjacency matrix
        super(RelationshipV2, self).__init__()
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
        self.upper_sampling = nn.Sequential(
            nn.Conv2d(1, upper_sampling_hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(upper_sampling_hidden_size),
            nn.Conv2d(upper_sampling_hidden_size, upper_sampling_hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(upper_sampling_hidden_size, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # x: (batch_size, num_objects, 4)
        batch_size, num_objects, _ = x.size()
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = x.view(batch_size, num_objects, num_objects)
        x = self.upper_sampling(x.unsqueeze(1))
        return x


class RelationshipV3(nn.Module):
    def __init__(self, num_objects, hidden_size=256, upper_sampling_hidden_size=32):
        # this model neet output a matrix of size num_objects x num_objects that is a adjacency matrix
        super(RelationshipV3, self).__init__()
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


class RelationshipAED(nn.Module):
    def __init__(self, hidden_size=32):
        super(RelationshipAED, self).__init__()
        self.down_sampling = nn.Sequential(
            nn.Conv2d(1, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, 1, kernel_size=3, padding=1),
        )

        self.upper_sampling = nn.Sequential(
            nn.Conv2d(1, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # x: (batch_size, num_objects, num_objects)
        x = self.down_sampling(x.unsqueeze(1))
        x = self.upper_sampling(x)
        return x
