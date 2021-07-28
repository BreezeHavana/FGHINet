import torch.nn as nn
import torch
from torch import autograd
#测试用
import config
import torch.nn.functional as F
import dataset

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_ch, eps=5e-5, affine=True),
            nn.PReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_ch, eps=5e-5, affine=True),
            nn.PReLU()
        )
        self.filtertrans = FeatureWiseTransformation(out_ch)
    def forward(self, input):
        out = self.conv(input)
        # out = self.filtertrans(out) + out
        

        return out
# --- feature-wise transformation layer ---
class FeatureWiseTransformation(nn.BatchNorm2d):
  feature_augment = False
  def __init__(self, num_features, momentum=0.1, track_running_stats=True):
    super(FeatureWiseTransformation, self).__init__(num_features, momentum=momentum, track_running_stats=track_running_stats)
    self.weight.fast = None
    self.bias.fast = None
    if self.track_running_stats:
      self.register_buffer('running_mean', torch.zeros(num_features))
      self.register_buffer('running_var', torch.zeros(num_features))
    if self.feature_augment: # initialize {gamma, beta} with {0.3, 0.5}
      self.gamma = torch.nn.Parameter(torch.ones(1, num_features, 1, 1)*0.3)
      self.beta  = torch.nn.Parameter(torch.ones(1, num_features, 1, 1)*0.5)
    self.reset_parameters()

  def reset_running_stats(self):
    if self.track_running_stats:
      self.running_mean.zero_()
      self.running_var.fill_(1)

  def forward(self, x, step=0):
    if self.weight.fast is not None and self.bias.fast is not None:
      weight = self.weight.fast
      bias = self.bias.fast
    else:
      weight = self.weight
      bias = self.bias
    if self.track_running_stats:
      out = F.batch_norm(x, self.running_mean, self.running_var, weight, bias, training=self.training, momentum=self.momentum)
    else:
      out = F.batch_norm(x, torch.zeros_like(x), torch.ones_like(x), weight, bias, training=True, momentum=1)

    # apply feature-wise transformation
    if self.feature_augment and self.training:
      gamma = (1 + torch.randn(1, self.num_features, 1, 1, dtype=self.gamma.dtype, device=self.gamma.device)*softplus(self.gamma)).expand_as(out)
      beta = (torch.randn(1, self.num_features, 1, 1, dtype=self.beta.dtype, device=self.beta.device)*softplus(self.beta)).expand_as(out)
      out = gamma*out + beta
    return out

class Meta(nn.Module):
    def __init__(self, in_ch=1):
        super(Meta, self).__init__()
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = downsample = nn.Sequential(
                nn.Conv2d(64, 64,
                          kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(64),
            )
        # self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        # self.pool2 = nn.MaxPool2d(2)
        self.pool2 = downsample = nn.Sequential(
                nn.Conv2d(128, 128,
                          kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(128),
            )
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = downsample = nn.Sequential(
                nn.Conv2d(256, 256,
                          kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(256),
            )
        # self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = downsample = nn.Sequential(
                nn.Conv2d(512, 512,
                          kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(512),
            )
        # self.pool4 = nn.MaxPool2d(2)
        self.fc = nn.Linear(512 * 8 * 8, 1024)
        self.logit = nn.Linear(1024, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        fc = p4.view(p4.size(0),-1)
        out = self.fc(fc)
        logit = self.logit(out)

        return out, logit