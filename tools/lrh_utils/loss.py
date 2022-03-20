from torch import nn
import torch


class BCELossWithWeight(nn.Module):
    def __init__(self, weight0, weight1):
        super(BCELossWithWeight, self).__init__()
        self.weight0 = weight0
        self.weight1 = weight1

    def forward(self, input, label):
        """

        :param input: torch.float32 bx1xhxw
        :param label: torch.long    bx1xhxw
        :return:
        """
        loss = -(self.weight0 * torch.mul(torch.log((1 - input).clamp(1e-6, 1)), (1 - label)) +
                 self.weight1 * torch.mul(torch.log(input.clamp(1e-6, 1)), label))
        return torch.mean(loss)


class MBCEloss(torch.nn.Module):
    def __init__(self):
        super(MBCEloss, self).__init__()

    def forward(self, input, label):
        """
        :param input: torch.float32 bx1xhxw
        :param label: torch.long    bx1xhxw
        :return:
        """
        one_mean = torch.sum(torch.mul(input, label), dim=[1, 2, 3]) / torch.sum(label, dim=[1, 2, 3])
        zero_mean = torch.sum(torch.mul(input, 1 - label), dim=[1, 2, 3]) / torch.sum(1 - label, dim=[1, 2, 3])
        loss = -(torch.log(one_mean.clamp(1e-6, 1)) + torch.log((1 - zero_mean).clamp(1e-6, 1))) / 2
        return torch.mean(loss)


class CenterLoss(nn.Module):
    def __init__(self):
        super(CenterLoss, self).__init__()
        self.l2_loss = nn.MSELoss(reduction='sum')

    def forward(self, outputs, targets):
        return self.l2_loss(outputs, targets) / outputs.size(0)
