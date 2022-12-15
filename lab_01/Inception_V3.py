import torch.nn as nn
import torch


class GridReduction(nn.Module):
    def __init__(self, in_fts, out_fts):
        super(GridReduction, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_fts, out_channels=out_fts, kernel_size=(3, 3), stride=(2, 2))
        )

        self.branch2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        )

    def forward(self, input_img):
        o1 = self.branch1(input_img)
        o2 = self.branch2(input_img)
        x = torch.cat([o1, o2], dim=1)
        return x


class Inceptionx3(nn.Module):
    def __init__(self, in_fts, out_fts):
        super(Inceptionx3, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_fts, out_channels=out_fts[0], kernel_size=(1, 1), stride=(1, 1)),
            nn.Conv2d(in_channels=out_fts[0], out_channels=out_fts[0], kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.Conv2d(in_channels=out_fts[0], out_channels=out_fts[0], kernel_size=(3, 3), stride=(1, 1), padding=1)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=in_fts, out_channels=out_fts[1], kernel_size=(1, 1), stride=(1, 1)),
            nn.Conv2d(in_channels=out_fts[1], out_channels=out_fts[1], kernel_size=(3, 3), stride=(1, 1), padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.Conv2d(in_channels=in_fts, out_channels=out_fts[2], kernel_size=(1, 1), stride=(1, 1))
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels=in_fts, out_channels=out_fts[3], kernel_size=(1, 1), stride=(1, 1))
        )

    def forward(self, input_img):
        o1 = self.branch1(input_img)
        o2 = self.branch2(input_img)
        o3 = self.branch3(input_img)
        o4 = self.branch4(input_img)
        x = torch.cat([o1, o2, o3, o4], dim=1)
        return x


class Inceptionx5(nn.Module):
    def __init__(self, in_fts, out_fts, n=7):
        super(Inceptionx5, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_fts, out_channels=out_fts[0], kernel_size=(1, 1), stride=(1, 1)),
            nn.Conv2d(in_channels=out_fts[0], out_channels=out_fts[0], kernel_size=(1, n), stride=(1, 1),
                      padding=(0, n // 2)),
            nn.Conv2d(in_channels=out_fts[0], out_channels=out_fts[0], kernel_size=(n, 1), stride=(1, 1),
                      padding=(n // 2, 0)),
            nn.Conv2d(in_channels=out_fts[0], out_channels=out_fts[0], kernel_size=(1, n), stride=(1, 1),
                      padding=(0, n // 2)),
            nn.Conv2d(in_channels=out_fts[0], out_channels=out_fts[0], kernel_size=(n, 1), stride=(1, 1),
                      padding=(n // 2, 0)),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=in_fts, out_channels=out_fts[1], kernel_size=(1, 1), stride=(1, 1)),
            nn.Conv2d(in_channels=out_fts[1], out_channels=out_fts[1], kernel_size=(1, n), stride=(1, 1),
                      padding=(0, n // 2)),
            nn.Conv2d(in_channels=out_fts[1], out_channels=out_fts[1], kernel_size=(n, 1), stride=(1, 1),
                      padding=(n // 2, 0)),
        )
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.Conv2d(in_channels=in_fts, out_channels=out_fts[2], kernel_size=(1, 1), stride=(1, 1))
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels=in_fts, out_channels=out_fts[3], kernel_size=(1, 1), stride=(1, 1))
        )

    def forward(self, input_img):
        o1 = self.branch1(input_img)
        o2 = self.branch2(input_img)
        o3 = self.branch3(input_img)
        o4 = self.branch4(input_img)
        x = torch.cat([o1, o2, o3, o4], dim=1)
        return x


class Inceptionx2(nn.Module):
    def __init__(self, in_fts, out_fts):
        super(Inceptionx2, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_fts, out_channels=out_fts[0] // 4, kernel_size=(1, 1)),
            nn.Conv2d(in_channels=out_fts[0] // 4, out_channels=out_fts[0] // 4, kernel_size=(3, 3), stride=(1, 1),
                      padding=1)
        )
        self.subbranch1_1 = nn.Sequential(
            nn.Conv2d(in_channels=out_fts[0] // 4, out_channels=out_fts[0], kernel_size=(1, 3), stride=(1, 1),
                      padding=(0, 3 // 2))
        )
        self.subbranch1_2 = nn.Sequential(
            nn.Conv2d(in_channels=out_fts[0] // 4, out_channels=out_fts[1], kernel_size=(3, 1), stride=(1, 1),
                      padding=(3 // 2, 0))
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=in_fts, out_channels=out_fts[2] // 4, kernel_size=(1, 1))
        )
        self.subbranch2_1 = nn.Sequential(
            nn.Conv2d(in_channels=out_fts[2] // 4, out_channels=out_fts[2], kernel_size=(1, 3), stride=(1, 1),
                      padding=(0, 3 // 2))
        )
        self.subbranch2_2 = nn.Sequential(
            nn.Conv2d(in_channels=out_fts[2] // 4, out_channels=out_fts[3], kernel_size=(3, 1), stride=(1, 1),
                      padding=(3 // 2, 0))
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.Conv2d(in_channels=in_fts, out_channels=out_fts[4], kernel_size=(1, 1), stride=(1, 1))
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels=in_fts, out_channels=out_fts[5], kernel_size=(1, 1), stride=(1, 1))
        )

    def forward(self, input_img):
        o1 = self.branch1(input_img)
        o11 = self.subbranch1_1(o1)
        o12 = self.subbranch1_2(o1)
        o2 = self.branch2(input_img)
        o21 = self.subbranch2_1(o2)
        o22 = self.subbranch2_2(o2)
        o3 = self.branch3(input_img)
        o4 = self.branch4(input_img)
        x = torch.cat([o11, o12, o21, o22, o3, o4], dim=1)
        return x


class AuxClassifier(nn.Module):
    def __init__(self, in_fts, num_classes):
        super(AuxClassifier, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=(5, 5))
        self.conv = nn.Conv2d(in_channels=in_fts, out_channels=128, kernel_size=(1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(5 * 5 * 128, 1024),
            nn.BatchNorm1d(num_features=1024),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        N = x.shape[0]
        x = self.pool(x)
        x = self.conv(x)
        x = x.reshape(N, -1)
        x = self.classifier(x)
        return x


class Inception_v3(nn.Module):
    def __init__(self, in_fts=3, num_classes=nnum_classes):
        super(Inception_v3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_fts, out_channels=32, kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(num_features=32)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(num_features=32)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=64)
        )
        self.pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=80, kernel_size=(3, 3), stride=(1, 1))
        self.conv5 = nn.Conv2d(in_channels=80, out_channels=192, kernel_size=(3, 3), stride=(2, 2))
        self.conv6 = nn.Conv2d(in_channels=192, out_channels=288, kernel_size=(3, 3), stride=(1, 1), padding=1)

        list_incept = [Inceptionx3(in_fts=288, out_fts=[96, 96, 96, 96]),
                       Inceptionx3(in_fts=4 * 96, out_fts=[96, 96, 96, 96]),
                       Inceptionx3(in_fts=4 * 96, out_fts=[96, 96, 96, 96])]

        self.inceptx3 = nn.Sequential(*list_incept)
        self.grid_redn_1 = GridReduction(in_fts=4 * 96, out_fts=384)
        self.aux_classifier = AuxClassifier(768, num_classes)

        list_incept = [Inceptionx5(in_fts=768, out_fts=[160, 160, 160, 160]),
                       Inceptionx5(in_fts=4 * 160, out_fts=[160, 160, 160, 160]),
                       Inceptionx5(in_fts=4 * 160, out_fts=[160, 160, 160, 160]),
                       Inceptionx5(in_fts=4 * 160, out_fts=[160, 160, 160, 160]),
                       Inceptionx5(in_fts=4 * 160, out_fts=[160, 160, 160, 160])]

        self.inceptx5 = nn.Sequential(*list_incept)
        self.grid_redn_2 = GridReduction(in_fts=4 * 160, out_fts=640)

        list_incept = [Inceptionx2(in_fts=1280, out_fts=[256, 256, 192, 192, 64, 64]),
                       Inceptionx2(in_fts=1024, out_fts=[384, 384, 384, 384, 256, 256])]

        self.inceptx2 = nn.Sequential(*list_incept)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, input_img):
        N = input_img.shape[0]
        x = self.conv1(input_img)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.inceptx3(x)
        x = self.grid_redn_1(x)
        aux_out = self.aux_classifier(x)
        x = self.inceptx5(x)
        x = self.grid_redn_2(x)
        x = self.inceptx2(x)
        x = self.avgpool(x)
        x = x.reshape(N, -1)
        x = self.fc(x)
        if self.training:
            return [x, aux_out]
        else:
            return x
