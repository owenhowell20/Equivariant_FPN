import torch
import torch.nn as nn
import torch.nn.functional as F
import e2cnn
from torch.autograd import Variable

### bottleneck layer: regular features --> regular features
class Equ_Bottleneck( nn.Module ):

    expansion_1 = 4
    expansion_2 = 2

    ### so2_gspace is discritization, in_planes=number input channels, planes = number hidden channels
    def __init__( self, so2_gspace, in_planes, planes,  stride, expansion=4 ):
        super(Equ_Bottleneck, self).__init__()

        self.gspace_dim = 2*so2_gspace

        ### expansion factor, the output dimension is always 4 times the number of hidden planes
        self.expansion = expansion ### i.e output dimension = 4*planes

        ### muplicity of regular representations:
        in_regular_mulplicity = int( in_planes/self.gspace_dim )
        hidden_regular_mulplicity = int( planes/self.gspace_dim )
        out_regular_mulplicity = int( self.expansion*planes/self.gspace_dim )

        ### declare SO(2) gspace
        gspace = e2cnn.gspaces.FlipRot2dOnR2(N=so2_gspace, maximum_frequency=None, fibergroup=None)

        ### input, hidden and output representations
        rho_in = e2cnn.nn.FieldType( gspace , [gspace.regular_repr]*in_regular_mulplicity )
        rho_hidden = e2cnn.nn.FieldType( gspace , [gspace.regular_repr]*hidden_regular_mulplicity )
        rho_out = e2cnn.nn.FieldType( gspace , [gspace.regular_repr]*out_regular_mulplicity )

        ### for converting to geometric tensor
        self.rho_in = rho_in
        self.rho_out = rho_out

        ### layer 1: in_planes --> planes
        self.conv1 =  e2cnn.nn.R2Conv( rho_in ,  rho_hidden ,  kernel_size=1 , bias=False )
        self.bn1 = e2cnn.nn.InnerBatchNorm( rho_hidden )
        
        ### layer 2: planes --> planes
        self.conv2 =  e2cnn.nn.R2Conv( rho_hidden ,  rho_hidden ,  kernel_size=3, stride=stride, padding=1, bias=False )
        self.bn2 = e2cnn.nn.InnerBatchNorm( rho_hidden )
        self.relu_hidden = e2cnn.nn.ReLU( rho_hidden , inplace=False)

        ### layer 3, planes -> self.expansion*planes
        ##self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.conv3 =  e2cnn.nn.R2Conv( rho_hidden ,  rho_out ,  kernel_size=1, bias=False )
        self.bn3 = e2cnn.nn.InnerBatchNorm( rho_out )
        self.relu_out = e2cnn.nn.ReLU( rho_out , inplace=False)

        ### shortcut layer
        if stride != 1 or in_planes != self.expansion*planes:

            self.shortcut_conv = nn.Sequential(   
                e2cnn.nn.R2Conv( rho_in ,  rho_out ,  kernel_size=3, stride=stride, padding=1, bias=False ) , 
                e2cnn.nn.InnerBatchNorm( rho_out ) ,
                e2cnn.nn.ReLU( rho_out , inplace=False)
                )
        else:
            self.shortcut_conv = e2cnn.nn.IdentityModule( rho_out )

    def forward(self, x):

        ### conv/bnorm layer 1
        out = self.relu_hidden(  self.bn1( self.conv1( x ) ) )

        ### conv/bnorm layer 2
        out = self.relu_hidden(  self.bn2( self.conv2( out ) ) )

        ### conv/bnorm layer 3
        out = self.relu_out( self.bn3( self.conv3(out) ) )
                
        ### add signals, they both transform in same way
        out = out + self.shortcut_conv( x ) 

        return out


if __name__ == "__main__":

     ### check neck
    so2_gspace = 4
    in_planes = 64
    planes = 2*in_planes
    stride = 1

    ### so2-equivarient bottleneck
    neck = Equ_Bottleneck(so2_gspace, in_planes, planes,  stride )

    x = torch.rand( 10 , in_planes , 256 , 256 )

    in_regular_mulplicity = int( in_planes/so2_gspace )
    gspace = e2cnn.gspaces.FlipRot2dOnR2(N=so2_gspace, maximum_frequency=None, fibergroup=None)
    rho_in = e2cnn.nn.FieldType( gspace , [gspace.regular_repr]*in_regular_mulplicity )
    x = e2cnn.nn.GeometricTensor( x , rho_in )

    y = neck(x)

    print( 'full output shape:' , y.shape )
