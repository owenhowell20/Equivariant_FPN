'''FPN in PyTorch.

See the paper "Feature Pyramid Networks for Object Detection" for more details.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import e2cnn
from torch.autograd import Variable

### bottleneck layer: always takes regular features to regular features
class Equ_Bottleneck(nn.Module):

    ### expansion factor, the output dimension is always 4 times the input dimension
    expansion = 4

    ### so2_gspace is discritization, in_planes=number input channels, planes = number hidden channels
    def __init__( self, so2_gspace, in_planes, planes,  stride ):
        super(Equ_Bottleneck, self).__init__()

        in_regular_mulplicity = int( in_planes/so2_gspace )
        hidden_regular_mulplicity = int( planes/so2_gspace )
        out_regular_mulplicity = int( self.expansion*in_planes/so2_gspace )

        ### declare gspace
        gspace = e2cnn.gspaces.Rot2dOnR2(N=so2_gspace, maximum_frequency=None, fibergroup=None)

        ### input, hidden and output reps
        rho_in = e2cnn.nn.FieldType( gspace , [gspace.regular_repr]*in_regular_mulplicity )
        rho_hidden = e2cnn.nn.FieldType( gspace , [gspace.regular_repr]*hidden_regular_mulplicity )
        rho_out = e2cnn.nn.FieldType( gspace , [gspace.regular_repr]*out_regular_mulplicity )

        ### for converting to geometric tensor
        self.rho_in = rho_in
        self.rho_out = rho_out

        ### layer 1: in_planes --> planes
        self.conv1 =  e2cnn.nn.R2Conv( rho_in ,  rho_hidden ,  kernel_size=1 , bias=False )

        ### self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False) ### 2d conv
        ### self.bn1 = nn.BatchNorm2d(planes) ### batch norm
        self.bn1 = e2cnn.nn.GNormBatchNorm( rho_in )
        
        ### layer 2: planes --> planes
        self.conv2 =  e2cnn.nn.R2Conv( rho_hidden ,  rho_hidden ,  kernel_size=3, stride=stride, padding=1, bias=False )

        ### self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False) ### 2d conv
        ### self.bn2 = nn.BatchNorm2d(planes) ### batch norm
        self.bn2 = e2cnn.nn.GNormBatchNorm( rho_hidden )
        self.relu_hidden = e2cnn.nn.ReLU( rho_hidden , inplace=False)

        ### layer 3, planes -> self.expansion*planes
        ##self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False) ### 3d conv
        self.conv3 =  e2cnn.nn.R2Conv( rho_hidden ,  rho_out ,  kernel_size=1, bias=False )

        self.bn3 = e2cnn.nn.GNormBatchNorm( rho_out )
        self.relu_out = e2cnn.nn.ReLU( rho_out , inplace=False)
        #self.bn3 = nn.BatchNorm2d(self.expansion*planes) ### batch norm

        ### shortcut layer
        self.shortcut = e2cnn.nn.SequentialModule()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = e2cnn.nn.SequentialModule(
                e2cnn.nn.R2Conv( rho_in ,  rho_out ,  kernel_size=3, stride=stride, padding=1, bias=False ) ,
                #e2cnn.nn.ReLU( rho_out , inplace=False) ### also need batch norm here, e2cnn.nn.GNormBatchNorm( rho_out )
            )


    def forward(self, x):

        ### convert x to a geometric tensor of type rho_in
        ### x = e2cnn.nn.GeometricTensor( x , self.rho_in )

        ### conv/bnorm layer 1
        #out = self.bn1( self.conv1(x) )
        out = self.conv1(x) 
        out = self.relu_hidden( out )

        ### conv/bnorm layer 2
        #out = self.bn2( self.conv2(out) )
        out = self.conv2(out) 
        out = self.relu_hidden( out )

        ### conv/bnorm layer 3
        #out = self.bn3( self.conv3(out) )
        
        ### shortcut addition, need to check equivarience here: so long as representations are 'aligned'
        #print('out:', out.shape)
        passed = self.shortcut(x) 
        
        #print('passed:', passed.shape)
       

        #out = out + passed

        ### final relu
        #out = self.relu_out(out)

        print('done with module')
        ### convert output back to a torch tensor
        ###out = out.tensor

        return out



### so2-equivarient feature pyrimid network
class eqv_FPN(nn.Module):
    def __init__(self, block, num_blocks):
        super(eqv_FPN, self).__init__()

        ### number on input dimensions
        self.in_planes = 64

        #### set the so2 discritization, this should always be a power of 2 that is less than or equal to 64:
        so2_gspace = 2
        self.so2_gspace = so2_gspace
        gspace = e2cnn.gspaces.Rot2dOnR2(N=so2_gspace, maximum_frequency=None, fibergroup=None)

        ### 3 copies of the trivial rep.
        self.rho_triv = e2cnn.nn.FieldType( gspace , [gspace.trivial_repr]*3 )

        ### 64 channel regular
        rho_first = e2cnn.nn.FieldType( gspace , [gspace.regular_repr]*int(64/so2_gspace) )

        ### conv1
        self.conv_first = e2cnn.nn.R2Conv( self.rho_triv , rho_first , kernel_size=7, stride=2, padding=3, bias=False )
        self.bn_first = e2cnn.nn.GNormBatchNorm( rho_first ) 
        self.relu_first = e2cnn.nn.ReLU( rho_first )

        ### Norm max pool over spatial extent
        self.max_pool2d_layer = e2cnn.nn.NormMaxPool( rho_first , kernel_size=3, stride=2, padding=1 )

        ### Bottom-up layers: so2_gspace, in_planes, planes,  flavor=str,  stride=1 
        self.layer1 = self._make_layer( block, so2_gspace ,  64, num_blocks[0], stride=1 ) ### 64 in_planes,  planes = num_blocks[0] 
        self.layer2 = self._make_layer( block, so2_gspace , 128, num_blocks[1], stride=2 ) ### 128 in_planes, planes = num_blocks[1] 
        self.layer3 = self._make_layer( block, so2_gspace , 256, num_blocks[2], stride=2 ) ### 256 in_planes, planes = num_blocks[2] 
        self.layer4 = self._make_layer( block, so2_gspace , 512, num_blocks[3], stride=2 ) ### 512 in_planes, planes = num_blocks[3] 

        ### Top layer: convs a 2048 reg --> 256 reg
        rho_top_in = e2cnn.nn.FieldType( gspace , [gspace.regular_repr]*int(2048/so2_gspace) )
        rho_top_out = e2cnn.nn.FieldType( gspace , [gspace.regular_repr]*int(256/so2_gspace) )
        self.toplayer = e2cnn.nn.R2Conv( rho_top_in , rho_top_out , kernel_size=1, stride=1, padding=0 )
       

        ### Smooth layers: 3x convs regular 256 --> regular 256
        rho_reg = e2cnn.nn.FieldType( gspace , [gspace.regular_repr]*int(256/so2_gspace) )
        self.conv1 = e2cnn.nn.R2Conv( rho_reg , rho_reg , kernel_size=3, stride=1, padding=1) ###
        self.conv2 = e2cnn.nn.R2Conv( rho_reg , rho_reg , kernel_size=3, stride=1, padding=1)
        self.conv3 = e2cnn.nn.R2Conv( rho_reg , rho_reg , kernel_size=3, stride=1, padding=1)

        ### Lateral layers, these are all convs of increasing powers
        rho_lat_1a = e2cnn.nn.FieldType( gspace , [gspace.regular_repr]*int(1024/so2_gspace) )
        rho_lat_1b = e2cnn.nn.FieldType( gspace , [gspace.regular_repr]*int(256/so2_gspace) )

        rho_lat_2a = e2cnn.nn.FieldType( gspace , [gspace.regular_repr]*int(512/so2_gspace) )
        rho_lat_2b = e2cnn.nn.FieldType( gspace , [gspace.regular_repr]*int(256/so2_gspace) )

        ### lateral convolutions
        self.latlayer1 = e2cnn.nn.R2Conv( rho_lat_1a , rho_lat_1b , kernel_size=1, stride=1, padding=1)
        self.latlayer2 = e2cnn.nn.R2Conv( rho_lat_2a , rho_lat_2b , kernel_size=1, stride=1, padding=1)
        self.latlayer3 = e2cnn.nn.R2Conv( rho_reg , rho_reg , kernel_size=1, stride=1, padding=1)


    ###make layer function: block should be an NN with same input and 
    def _make_layer(self, block, so2_gspace , planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append( block( so2_gspace , self.in_planes, planes, stride) )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):

        ### Bottom-up layers
        ### first, convert x to geometric tensor
        print('input:' , x.shape)
        x = e2cnn.nn.GeometricTensor( x , self.rho_triv )

        c1 = self.conv_first(x) 

        print( 'first conv:',  c1.shape)
        #c1 = self.bn_first( c1 )
        c1 = self.relu_first( c1 )


        ### max pool over spatial dimensions 
        c1 = self.max_pool2d_layer( c1 )
        print('maxpool:', c1.shape)

        ### now layers
        c2 = self.layer1(c1)

        print('passed layer 1')
        print(c2.shape)
        c3 = self.layer2(c2)
        print('passed layer 2')
        print(c3.shape)
        c4 = self.layer3(c3)
        print('passed layer 3')
        print(c4.shape)
        c5 = self.layer4(c4)
        print('passed layer 4')
        print(c5.shape)

        ### Top-down layers
        p5 = self.toplayer( c5 )
        p4 = self._upsample_add( p5, self.latlayer1(c4) )
        p3 = self._upsample_add( p4, self.latlayer2(c3) )
        p2 = self._upsample_add( p3, self.latlayer3(c2) )
        
        ### Smooth convs, this is fine
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        return p2, p3, p4, p5


### full equivarient_fpn-101
def eqv_FPN101():
    ### return FPN(Bottleneck, [2,4,23,3])
    return eqv_FPN( Equ_Bottleneck, [2,2,2,2] )


def test():
    net = FPN101()
    fms = net(Variable(torch.randn(1,3,600,900)))
    for fm in fms:
        print(fm.size())


test = eqv_FPN101()
x = torch.rand( 10 , 3 , 250 , 250 )

test(x)

