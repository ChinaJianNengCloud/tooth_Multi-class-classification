# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from mypath import Path
import torchvision.models as models

class ResNet50(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(ResNet50, self).__init__()

        # 使用预训练的ResNet50模型
        resnet50 = models.resnet50(weights=None)
        resnet50.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # 从ResNet50中提取卷积层
        self.features = nn.Sequential(*list(resnet50.children())[:-2])
        
        # 修改全连接层以适应你的问题
        self.fc6 = nn.Linear(2048, 8192)
        self.fc7 = nn.Linear(8192, 4096)
        self.fc8 = nn.Linear(4096, 4096)
        self.fc9 = nn.Linear(4096, num_classes)
        self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])  # 在宽度和高度上取平均，以获得2D特征图
        x = x.view(x.size(0), -1)  # 展平
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)
        x = self.relu(self.fc8(x))
        x = self.dropout(x)

        # x = self.relu(self.fc9(x))
        # logits = self.sigmoid(x)  # 使用Sigmoid激活函数
        # return logits
        logits = self.fc9(x)
        return logits


    def __load_pretrained_weights(self):
        """Initialize network with pretrained weights."""
        corresp_name = {
            # Conv1
            "features.0.weight": "conv1.weight",
            "features.0.bias": "conv1.bias",
            # Conv2
            "features.3.weight": "conv2.weight",
            "features.3.bias": "conv2.bias",
            # Conv3a
            "features.6.weight": "conv3a.weight",
            "features.6.bias": "conv3a.bias",
            # Conv3b
            "features.8.weight": "conv3b.weight",
            "features.8.bias": "conv3b.bias",
            # Conv4a
            "features.11.weight": "conv4a.weight",
            "features.11.bias": "conv4a.bias",
            # Conv4b
            "features.13.weight": "conv4b.weight",
            "features.13.bias": "conv4b.bias",
            # Conv5a
            "features.16.weight": "conv5a.weight",
            "features.16.bias": "conv5a.bias",
            # Conv5b
            "features.18.weight": "conv5b.weight",
            "features.18.bias": "conv5b.bias",
            # fc6
            "fc6.weight": "fc6.weight",  # 修改为适应2D任务的形状
            "fc6.bias": "fc6.bias",
            # fc7
            "fc7.weight": "fc7.weight",
            "fc7.bias": "fc7.bias",
        }

        p_dict = torch.load(Path.model_dir())
        s_dict = self.state_dict()
        for name in p_dict:
            if name not in corresp_name:
                continue
            s_dict[corresp_name[name]] = p_dict[name]
        self.load_state_dict(s_dict)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # 修改为Conv2d
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):  # 修改为BatchNorm2d
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def get_1x_lr_params(model):
    """
    This generator returns all the parameters for conv and two fc layers of the net.
    """
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    # 注意这里使用的是 model.features 替代了原来的 model.conv1, model.conv2, ...
    b = [model.features, model.fc6, model.fc7]  # 添加其他需要微调的层

    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last fc layer of the net.
    """
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    # 注意这里使用的是 model.fc8 替代了原来的 model.fc8
    b = [model.fc8]

    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k

if __name__ == "__main__":
    inputs = torch.rand(32, 3, 112, 112)  # 输入修改为2D图像
    net = ResNet50(num_classes=7)

    outputs = net.forward(inputs)
    print(outputs.size())