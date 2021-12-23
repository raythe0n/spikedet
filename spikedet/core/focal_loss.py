import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, focus_param=2, pos_weight=None ):
        super().__init__()

        #self.nll_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
        self.focus_param = focus_param
        self.pos_weight = pos_weight


    def forward(self, input, target):
        pos_weight = self.pos_weight
        if pos_weight is not None:# and pos_weight.shape[-1] != input.shape[-1]:
            pos_weight = pos_weight * torch.ones(target.shape[-1], dtype=input.dtype, device=input.device)

        bce_loss = F.binary_cross_entropy_with_logits(input, target,
                                           None,
                                           pos_weight=pos_weight,
                                           reduction='none')

        #bce_loss = self.nll_loss(input, target)
        # conf loss calculation
        p = torch.sigmoid(input)

        focal = target * torch.pow(1.0 - p, self.focus_param) + \
                (1.0 - target) * torch.pow(p, self.focus_param)

        #torch.sum(bce_loss * focal) / input.shape[0]
        loss = torch.mean(bce_loss * focal)
        return loss


class FocalLossDiffused(nn.Module):
    def __init__(self, focus_param=2, pos_weight=1.0 ):
        super().__init__()

        self.nll_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
        self.focus_param = focus_param
        self.pos_weight = torch.tensor(pos_weight)


    def forward(self, input, target):
        bce_loss = self.nll_loss(input, target)

        if self.pos_weight.device != input.device:
            self.pos_weight = self.pos_weight.to(input.device)

        # conf loss calculation
        p = torch.sigmoid(input)

        log_sig = torch.log(1.0 + torch.exp(-torch.abs(input)))

        zero = torch.tensor(0, dtype=float, device=input.device)

        focal = target * self.pos_weight * torch.pow(1.0 - p, self.focus_param) * (log_sig - input.minimum(zero))  + \
        (1 - target) * torch.pow(p, self.focus_param) * (input.maximum(zero) + log_sig)

        loss = torch.mean(bce_loss * focal)
        return loss
