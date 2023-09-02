import torch
import torch.nn as nn
import torch.nn.functional as F
import e2cnn
from torch.autograd import Variable

def _upsample_add( x, scale ):
    '''Upsample and add two feature maps:
    x: (Variable) top feature map to be upsampled.
    y: (Variable) lateral feature map.
    '''
    ### now, both x and y are geometric tensors of equal type:
    in_type = x.type
    _,_,H,W = x.size()

    ### mesure the upsampling error
    upsample = e2cnn.nn.R2Upsampling(in_type, scale_factor=scale, mode='bilinear')

    output = upsample( x )

    return output


### check for so2 equivarience
so2_gspace = 64
gspace = e2cnn.gspaces.Rot2dOnR2(N=so2_gspace, maximum_frequency=None, fibergroup=None)

### 3 copies of the trivial rep: input images are 3 color channels
rho_triv = e2cnn.nn.FieldType( gspace , [gspace.trivial_repr]*3 )
so2 = gspace.fibergroup


x = torch.rand( 10 , 3 , 32 , 32 )
x = e2cnn.nn.GeometricTensor( x , rho_triv )

### true outputs
y = _upsample_add(x , 4)

for g in so2.elements:


    x_g = x.transform(g)
    y_g = _upsample_add(x_g, 4)


    y_p = y.transform(g)

    print(y_g.shape)

    d = (y_g - y_p ).tensor
    print( torch.norm(d)/torch.norm( y.tensor )  )



