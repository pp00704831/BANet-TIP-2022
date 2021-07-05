import torch
import torch.autograd as autograd
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from util.image_pool import ImagePool


###############################################################################
# Functions
###############################################################################

class L2_Loss():

    def get_loss(self, fakeIm, realIm):
        return 0.5 * nn.MSELoss()(fakeIm, realIm)

    def __call__(self, fakeIm, realIm):
        return self.get_loss(fakeIm, realIm)


def get_loss(model):
    if model['content_loss'] == 'L2':
        content_loss = L2_Loss()
    else:
        raise ValueError("ContentLoss [%s] not recognized." % model['content_loss'])
    return content_loss