import torch
import torch.nn as nn
import torch.nn.functional as F
import e2cnn
from torch.autograd import Variable

def _upsample_add(self, x, y):
    '''Upsample and add two feature maps:
    x: (Variable) top feature map to be upsampled.
    y: (Variable) lateral feature map.
    '''
    ### now, both x and y are geometric tensors of equal type:
    in_type = x.type
    _,_,H,W = y.size()

    ### mesure the upsampling error
    self.upsample = e2cnn.nn.R2Upsampling(in_type, scale_factor=None, size=(H,W), mode='bilinear')

    output = self.upsample( x  ) + y

    return output
