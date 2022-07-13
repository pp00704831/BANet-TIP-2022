import torch
import torch.nn as nn

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss

class BANet_loss(nn.Module):

    def __init__(self, ):
        super(BANet_loss, self).__init__()

        self.char = CharbonnierLoss()
        self.l1 = nn.L1Loss()
    def forward(self, restore, sharp):

        char = self.char(restore, sharp)
        restore_fft = torch.rfft(restore, signal_ndim=2, normalized=False, onesided=False)
        sharp_fft = torch.rfft(sharp, signal_ndim=2, normalized=False, onesided=False)
        fft_loss = 0.01 * self.l1(restore_fft, sharp_fft)
        loss = char + fft_loss

        return loss

def get_loss(model):
    if model['content_loss'] == 'BANet_loss':
        content_loss = BANet_loss()
    else:
        raise ValueError("ContentLoss [%s] not recognized." % model['content_loss'])
    return content_loss