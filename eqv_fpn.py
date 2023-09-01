'''FPN in PyTorch.

See the paper "Feature Pyramid Networks for Object Detection" for more details.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import e2cnn
from torch.autograd import Variable
from bottleneck import Equ_Bottleneck


### so2-equivarient feature pyrimid network
class eqv_FPN(nn.Module):
    def __init__(self, block, num_blocks):
        super(eqv_FPN, self).__init__()

        ### number of bottleneck input dimensions
        self.in_planes = 64

        #### set the so2 discritization, this should always be a power of 2 that is less than or equal to 64:
        so2_gspace = 8
        self.so2_gspace = so2_gspace
        gspace = e2cnn.gspaces.Rot2dOnR2(N=so2_gspace, maximum_frequency=None, fibergroup=None)

        ### 3 copies of the trivial rep: input images are 3 color channels
        self.rho_triv = e2cnn.nn.FieldType( gspace , [gspace.trivial_repr]*3 )

        ### 64 channel regular features
        rho_first = e2cnn.nn.FieldType( gspace , [gspace.regular_repr]*int(64/so2_gspace) )

        ### first convolutional layer: trivial --> regular
        self.conv_first = e2cnn.nn.R2Conv( self.rho_triv , rho_first , kernel_size=7, stride=2, padding=3, bias=False )
        self.bn_first = e2cnn.nn.GNormBatchNorm( rho_first ) 
        self.relu_first = e2cnn.nn.ReLU( rho_first )

        ### Norm max pool over spatial extent, this should be checked
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
        self.conv1 = e2cnn.nn.R2Conv( rho_reg , rho_reg , kernel_size=3, stride=1, padding=1) ###
        self.conv2 = e2cnn.nn.R2Conv( rho_reg , rho_reg , kernel_size=3, stride=1, padding=1)
        self.conv3 = e2cnn.nn.R2Conv( rho_reg , rho_reg , kernel_size=3, stride=1, padding=1)

        ### Lateral layers, these are all convs of increasing powers of 2
        rho_lat_a = e2cnn.nn.FieldType( gspace , [gspace.regular_repr]*int(1024/so2_gspace) )
        rho_lat_b = e2cnn.nn.FieldType( gspace , [gspace.regular_repr]*int(256/so2_gspace) )

        rho_lat_c = e2cnn.nn.FieldType( gspace , [gspace.regular_repr]*int(512/so2_gspace) )
        rho_lat_d = e2cnn.nn.FieldType( gspace , [gspace.regular_repr]*int(256/so2_gspace) )

        rho_lat_e = e2cnn.nn.FieldType( gspace , [gspace.regular_repr]*int(256/so2_gspace) )
        rho_lat_f = e2cnn.nn.FieldType( gspace , [gspace.regular_repr]*int(256/so2_gspace) )

        ### lateral convolutions
        self.latlayer1 = e2cnn.nn.R2Conv( rho_lat_a , rho_lat_b , kernel_size=1, stride=1, padding=1) ### 1024 --> 
        self.latlayer2 = e2cnn.nn.R2Conv( rho_lat_c , rho_lat_d , kernel_size=1, stride=1, padding=1)
        self.latlayer3 = e2cnn.nn.R2Conv( rho_lat_e, rho_lat_f , kernel_size=1, stride=1, padding=1)


    ###make layer function: block should be an NN with same input and 
    def _make_layer(self, block, so2_gspace , planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides: ### changes the strides
            layers.append( block( so2_gspace , self.in_planes, planes, stride) )
            self.in_planes = planes * block.expansion
            print( 'block_num:',  block.expansion )

        return nn.Sequential(*layers)


    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps:
        x: (Variable) top feature map to be upsampled.
        y: (Variable) lateral feature map.
        '''
        ### now, both x and y are geometric tensors of equal type:
        in_type = x.type
        _,_,H,W = y.size()
        self.upsample = e2cnn.nn.R2Upsampling(in_type, scale_factor=None, size=(H,W), mode='bilinear')

        output = self.upsample( x  ) + y

        return output

    def forward(self, x):

        ### Bottom-up layers
        ### first, convert x to geometric tensor
        print('input:' , x.shape)
        x = e2cnn.nn.GeometricTensor( x , self.rho_triv ) ### [: , 3 , : ,:]

        ### first conv
        c1 = self.conv_first(x) ### [: , 64 , : ,:]
        c1 = self.relu_first( c1 )

        ### max pool over spatial dimensions 
        c1 = self.max_pool2d_layer( c1 ) ### [: , 64 , : ,:]
    

        ### now layers
        print('start layers')
        print('c1:' , c1.shape)
        c2 = self.layer1(c1)  ### 64 -->
        print( "c2" , c2.shape )
        c3 = self.layer2(c2)  
        print( "c3" , c3.shape )
        c4 = self.layer3(c3)  
        print( "c4" , c4.shape )
        c5 = self.layer4(c4)  
        print('passed layer 4')


        print('begin top down layers')
        ### Top-down layers
        p5 = self.toplayer( c5 ) ### top layer, p5: [:,128,:,:]

        ### c4 has shape 512
        a4 = self.latlayer1(c4) ### c4: [;,512,:,:] , latlayer1: 512-->128
        print( a4.shape , p5.shape )
        p4 = self._upsample_add( p5, a4 ) ### p4: 128
       
      
        print('p4:', p4.shape)
        print( 'c3:', c3.shape )

        a3 = self.latlayer2(c3) ### 256 --> 128
        p3 = self._upsample_add( p4, a3 ) ### 128

        print('p3:', p3.shape , c2.shape)

        a2 = self.latlayer3(c2) ### 128 --> 128
        p2 = self._upsample_add( p3, a2 )
        
        ### Final conv: all outputs are same dimension
        p4 = self.conv1(p4)
        p3 = self.conv2(p3)
        p2 = self.conv3(p2)

        return p2, p3, p4, p5


### full equivarient_fpn-101
def eqv_FPN101():
    ### return FPN(Bottleneck, [2,4,23,3])
    return eqv_FPN( Equ_Bottleneck, [2,2,2,2] ) ##eqv_FPN( Equ_Bottleneck, [2,2,2,2] )


def test():
    net = FPN101()
    fms = net(Variable(torch.randn(1,3,600,900)))
    for fm in fms:
        print(fm.size())



