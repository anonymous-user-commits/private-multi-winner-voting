import torch.nn as nn
import torch.nn.functional as F
from time import time


class MnistNetPate(nn.Module):
    """Class used to initialize model of student/teacher"""

    def __init__(self, name, args):
        super(MnistNetPate, self).__init__()
        self.name = name
        self.args = args
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

        self.first_time = 0
        self.middle_time = 0
        self.last_time = 0

    def forward(self, x):
        start = time()
        x = F.relu(self.conv1(x))
        self.first_time += time() - start

        start = time()
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        self.middle_time += time() - start

        start = time()
        x = self.fc2(x)
        self.last_time += time() - start

        return x
