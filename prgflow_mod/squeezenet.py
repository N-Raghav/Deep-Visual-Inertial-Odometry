import torch
import torch.nn as nn
import torch.nn.functional as F


class Fire(nn.Module):
    def __init__(self, in_ch, squeeze_ch, expand_ch):
        super().__init__()
        self.squeeze = nn.Conv2d(in_ch, squeeze_ch, kernel_size=1)
        self.expand1 = nn.Conv2d(squeeze_ch, expand_ch, kernel_size=1)
        self.expand3 = nn.Conv2d(squeeze_ch, expand_ch, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.squeeze(x), inplace=True)
        a = F.relu(self.expand1(x), inplace=True)
        b = F.relu(self.expand3(x), inplace=True)
        return torch.cat([a, b], dim=1)


class SqueezeHead(nn.Module):
    def __init__(self, out_dim, base_channels=32, fire_blocks=4, dropout=0.7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, base_channels, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=5, stride=2, padding=2)

        blocks = []
        ch = base_channels * 2
        for i in range(fire_blocks):
            squeeze_ch = max(16, ch // 4)
            expand_ch = max(base_channels, ch // 2)
            blocks.append(Fire(ch, squeeze_ch, expand_ch))
            ch = expand_ch * 2
            if i % 2 == 1:
                blocks.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        self.blocks = nn.Sequential(*blocks)
        self.final_conv = nn.Conv2d(ch, 64, kernel_size=1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(64, out_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = self.blocks(x)
        x = F.relu(self.final_conv(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


class THead(SqueezeHead):
    def __init__(self, base_channels=32, fire_blocks=4, dropout=0.7):
        super().__init__(out_dim=2, base_channels=base_channels, fire_blocks=fire_blocks, dropout=dropout)


class SHead(SqueezeHead):
    def __init__(self, base_channels=32, fire_blocks=4, dropout=0.7):
        super().__init__(out_dim=1, base_channels=base_channels, fire_blocks=fire_blocks, dropout=dropout)


def model_size_mb(model):
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    return total / (1024.0 * 1024.0)
