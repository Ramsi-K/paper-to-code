# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    """
    T-Net for learning global point cloud features.
    """

    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        iden = (
            torch.eye(self.k, requires_grad=True)
            .view(1, self.k * self.k)
            .repeat(batchsize, 1)
        )
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNet(nn.Module):
    """
    PointNet architecture for classification.
    """

    def __init__(self, num_classes=10):
        super(PointNet, self).__init__()
        self.input_transform = TNet(k=3)
        self.feature_transform = TNet(k=64)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.input_transform(x)
        x = torch.bmm(torch.transpose(x, 1, 2), trans).transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=-1)


class PointNetSegmentation(nn.Module):
    """
    PointNet architecture for segmentation.
    """

    def __init__(self, num_classes=10):
        super(PointNetSegmentation, self).__init__()
        self.input_transform = TNet(k=3)
        self.feature_transform = TNet(k=64)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.input_transform(x)
        x = torch.bmm(torch.transpose(x, 1, 2), trans).transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        trans_feat = torch.max(x, 2, keepdim=True)[0]
        trans_feat = trans_feat.view(-1, 1024)
        trans_feat = self.fc1(trans_feat)
        trans_feat = self.fc2(trans_feat)
        trans_feat = self.fc3(trans_feat)
        net_transform = trans_feat.view(-1, num_classes, 1).repeat(1, 1, n_pts)

        x = torch.cat([x, net_transform], 1)
        return x


# Test PointNet model
def test_PointNet():
    batch_size = 32
    num_points = 1024
    num_classes = 10
    model = PointNet(num_classes=num_classes)
    input_data = torch.randn(batch_size, 3, num_points)
    output = model(input_data)
    assert output.size(0) == batch_size
    assert output.size(1) == num_classes
    print("PointNet classification model passed the test.")


# Test PointNetSegmentation model
def test_PointNetSegmentation():
    batch_size = 32
    num_points = 1024
    num_classes = 10
    model = PointNetSegmentation(num_classes=num_classes)
    input_data = torch.randn(batch_size, 3, num_points)
    output = model(input_data)
    assert output.size(0) == batch_size
    assert (
        output.size(1) == num_classes + 3
    )  # Include extra dimensions for segmentation
    assert output.size(2) == num_points
    print("PointNet segmentation model passed the test.")


if __name__ == "__main__":
    test_PointNet()
    test_PointNetSegmentation()
