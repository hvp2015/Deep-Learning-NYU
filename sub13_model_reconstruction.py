"""RECONSTRUCTION - not the original code of submission 13.

The notebook that produced submission 13 (private 92.9%, public 93.3%) was run
locally and was not kept. This file only rebuilds the architecture described in
the project report: a three-stage ResNet, widths 64/128/256, three basic blocks
per stage, global average pooling. It is the same model class used in
sub09-sub12 and has 4,327,754 trainable parameters.

What is NOT recovered: the number of epochs, the learning-rate schedule and the
exact brightness/contrast augmentation ranges used for submission 13. Use
sub12_rerun_80ep.ipynb for the closest fully documented training recipe.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.skip = None if (stride == 1 and cin == cout) else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)), inplace=True)
        y = self.b2(self.c2(y))
        return F.relu(y + (x if self.skip is None else self.skip(x)), inplace=True)


class Net(nn.Module):
    def __init__(self, widths=(64, 128, 256), blocks=(3, 3, 3), classes=10, drop=0.0):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, widths[0], 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True))
        layers, cin = [], widths[0]
        for k, (cout, n) in enumerate(zip(widths, blocks)):
            for b in range(n):
                layers.append(Block(cin, cout, 2 if (b == 0 and k > 0) else 1))
                cin = cout
        self.layers = nn.Sequential(*layers)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(cin, classes)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(self.layers(self.stem(x)), 1).flatten(1)
        return self.fc(self.drop(x))


if __name__ == "__main__":
    n = sum(p.numel() for p in Net().parameters() if p.requires_grad)
    print("trainable parameters:", n)
    assert n == 4_327_754 and n <= 5_000_000
