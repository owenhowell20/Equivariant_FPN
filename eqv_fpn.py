'''Equivarient Feature Pyramid Networks in PyTorch. '''

import torch
import torch.nn as nn
import torch.nn.functional as F
import e2cnn
from torch.autograd import Variable
from bottleneck import Equ_Bottleneck


### SO(2)-equivarient feature pyrimid network
class eqv_FPN(nn.Module):
    def __init__(self, so2_gspace, block, num_blocks):
        super(eqv_FPN, self).__init__()

        ### number of bottleneck input dimensions
        self.in_planes = 64

        #### set the so2 discritization, this should always be a power of 2 that is less than or equal to 64:
        self.so2_gspace = so2_gspace
        gspace = e2cnn.gspaces.Rot2dOnR2(N=so2_gspace, maximum_frequency=None, fibergroup=None)

        ### 3 copies of the trivial rep: input images are 3 color channels
        self.rho_triv = e2cnn.nn.FieldType( gspace , [gspace.trivial_repr]*3 )

        ### 64 channel regular features
        rho_first = e2cnn.nn.FieldType( gspace , [gspace.regular_repr]*int(64/so2_gspace) )

        ### first convolutional layer: trivial --> regular
        self.conv_first = e2cnn.nn.R2Conv( self.rho_triv , rho_first , kernel_size=7, stride=2, padding=3, bias=False )
        self.bn_first = e2cnn.nn.InnerBatchNorm( rho_first ) 
        self.relu_first = e2cnn.nn.ReLU( rho_first )

        ### Norm max pool over spatial extent: this should be checked
        self.max_pool2d_layer = e2cnn.nn.NormMaxPool( rho_first , kernel_size=3, stride=2, padding=1 )

        ### Bottom-up layers: so2_gspace, in_planes, planes,  flavor=str,  stride=1 
        self.layer1 = self._make_layer( block, so2_gspace ,  64, num_blocks[0], stride=1 ) ### in_planes=64,  out_planes = 4*num_blocks[0] 
        self.layer2 = self._make_layer( block, so2_gspace , 128, num_blocks[1], stride=2 ) ### in_planes=128, out_planes = 4*num_blocks[1] 
        self.layer3 = self._make_layer( block, so2_gspace , 256, num_blocks[2], stride=2 ) ### in_planes=256, out_planes = 4*num_blocks[2] 
        self.layer4 = self._make_layer( block, so2_gspace , 512, num_blocks[3], stride=2 ) ### in_planes=512, out_planes = 4*num_blocks[3] 
 
        ### Top layer: convs a 2048 reg --> 256 reg
        rho_top_in = e2cnn.nn.FieldType( gspace , [gspace.regular_repr]*int(2048/so2_gspace) )
        rho_top_out = e2cnn.nn.FieldType( gspace , [gspace.regular_repr]*int(256/so2_gspace) )
        self.toplayer = e2cnn.nn.R2Conv( rho_top_in , rho_top_out , kernel_size=1, stride=1, padding=0 )
       
        ### Smooth layers: 3x convs regular 256 --> regular 256
        rho_reg = e2cnn.nn.FieldType( gspace , [gspace.regular_repr]*int(256/so2_gspace) )
        self.conv1 = e2cnn.nn.R2Conv( rho_reg , rho_reg , kernel_size=3, stride=1, padding=1) 
        self.conv2 = e2cnn.nn.R2Conv( rho_reg , rho_reg , kernel_size=3, stride=1, padding=1)
        self.conv3 = e2cnn.nn.R2Conv( rho_reg , rho_reg , kernel_size=3, stride=1, padding=1)
    
        self.bn_smooth = e2cnn.nn.InnerBatchNorm( rho_reg ) 
        self.relu_smooth = e2cnn.nn.ReLU( rho_reg )


        ### Lateral layers, these are all convs of increasing powers of 2
        rho_lat_a = e2cnn.nn.FieldType( gspace , [gspace.regular_repr]*int(1024/so2_gspace) )
        rho_lat_b = e2cnn.nn.FieldType( gspace , [gspace.regular_repr]*int(256/so2_gspace) )

        rho_lat_c = e2cnn.nn.FieldType( gspace , [gspace.regular_repr]*int(512/so2_gspace) )
        rho_lat_d = e2cnn.nn.FieldType( gspace , [gspace.regular_repr]*int(256/so2_gspace) )

        rho_lat_e = e2cnn.nn.FieldType( gspace , [gspace.regular_repr]*int(256/so2_gspace) )
        rho_lat_f = e2cnn.nn.FieldType( gspace , [gspace.regular_repr]*int(256/so2_gspace) )

        ### lateral convolutions
        self.latlayer1 = e2cnn.nn.R2Conv( rho_lat_a , rho_lat_b , kernel_size=1, stride=1, padding=0)
        self.latlayer2 = e2cnn.nn.R2Conv( rho_lat_c , rho_lat_d , kernel_size=1, stride=1, padding=0)
        self.latlayer3 = e2cnn.nn.R2Conv( rho_lat_e, rho_lat_f , kernel_size=1, stride=1, padding=0)


    ###make layer function: block should be an NN with same input and 
    def _make_layer(self, block, so2_gspace , planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1) ### stride should always be power of two for bottleneck
        layers = []

        for stride in strides: ### changes the strides
            layers.append( block( so2_gspace , self.in_planes, planes, stride) ) ### stride should be power of two
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)


    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps:
        x: (Variable) top feature map to be upsampled.
        y: (Variable) lateral feature map.
        '''
        ### now, both x and y are geometric tensors of equal type:
        in_type = x.type
        #_,_,a,b = x.shape
        #_,_,H,W = y.size()

        scale_factor = 2 ### constant 2 scale factor

        ### mesure the upsampling error
        #self.upsample = e2cnn.nn.R2Upsampling(in_type, scale_factor=None, size=(H,W), mode='bilinear')
        self.upsample = e2cnn.nn.R2Upsampling(in_type, scale_factor=scale_factor, mode='bilinear')

       
        output = self.upsample(  x  ) 
        return output + y

    def forward(self, x):

        ### Bottom-up layers
        ### first, convert x to geometric tensor
        x = e2cnn.nn.GeometricTensor( x , self.rho_triv ) ### [: , 3 , : ,:]

        ### first conv
        c1 = self.bn_first( self.conv_first(x) )
        c1 = self.relu_first( c1 )

        ### max pool over spatial dimensions 
        c1 = self.max_pool2d_layer( c1 ) 
    
        ### now up layers
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)  
        c4 = self.layer3(c3)  
        c5 = self.layer4(c4)  
        
        ### top pyrimid layer 
        p5 = self.toplayer( c5 ) 

        ### Top-down layers
        a4 = self.latlayer1(c4)
        p4 = self._upsample_add( p5, a4 )
        a3 = self.latlayer2(c3) 
        p3 = self._upsample_add( p4, a3 ) 
        a2 = self.latlayer3(c2) 
        p2 = self._upsample_add( p3, a2 )
        
        ### Final convolution: all outputs are same dimension and same feature type
        p4 = self.relu_smooth( self.bn_smooth( self.conv1(p4) ) )
        p3 = self.relu_smooth( self.bn_smooth( self.conv2(p3) ) )
        p2 = self.relu_smooth( self.bn_smooth( self.conv3(p2) ) )

        return p2, p3, p4, p5


### so2-equivarient feature pyrimid network
def eqv_FPN101(so2_gspace):
    
    return eqv_FPN( so2_gspace , Equ_Bottleneck, [2,2,2,2] ) 


def test():
    net = eqv_FPN101()
    fms = net(Variable(torch.randn(1,3,600,900)))
    for fm in fms:
        print(fm.size())



if __name__ == "__main__":

    ### check for so2 equivarience
    so2_gspace = 8
    gspace = e2cnn.gspaces.Rot2dOnR2(N=so2_gspace, maximum_frequency=None, fibergroup=None)

    ### 3 copies of the trivial rep: input images are 3 color channels
    rho_triv = e2cnn.nn.FieldType( gspace , [gspace.trivial_repr]*3 )
    so2 = gspace.fibergroup

    x = torch.rand( 10 , 3 , 256 , 256 )
    x = e2cnn.nn.GeometricTensor( x , rho_triv )

    f = eqv_FPN101( so2_gspace )

    ### unchanged y-values:
    y = f( x.tensor )

    ### check that each extracted feature has so2 equivarience
    for g in so2.elements:

        x_rot = x.transform(g)

        ### new inputs
        y_rot = f( x_rot.tensor )

        # ### meausre the differences:
        z0 = y[0].transform(g)
        z1 = y[1].transform(g)
        z2 = y[2].transform(g)
        z3 = y[3].transform(g)


        ### mesure differences
        d0 = z0.tensor - y_rot[0].tensor
        d1 = z1.tensor - y_rot[1].tensor
        d2 = z2.tensor - y_rot[2].tensor
        d3 = z3.tensor - y_rot[3].tensor

        ### take the norm
        print()
        print("group element:" , g)
        print( 'zero percentage error:' ,  torch.norm(d0)/torch.norm( z0.tensor ) ) 
        print( 'one percentage error:' ,  torch.norm(d1)/torch.norm( z1.tensor ) ) 
        print( 'two percentage error:' ,  torch.norm(d2)/torch.norm( z2.tensor ) ) 
        print( 'three percentage error:' ,  torch.norm(d3)/torch.norm( z3.tensor ) ) 
        print()

        ### check types of outputs
        print( y_rot[0].type , y_rot[1].type , y_rot[2].type , y_rot[3].type )


