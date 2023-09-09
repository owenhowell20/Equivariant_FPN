import torch
import torch.nn as  nn
import torch.nn.functional as F
import e2cnn
from bottleneck import Equ_Bottleneck


###
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
        
    def forward(self, x):

        identity = x.clone() ### copy x
        x = self.relu(self.batch_norm1(self.conv1(x)))
        
        x = self.relu(self.batch_norm2(self.conv2(x)))
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
        
        #downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        #add identity
        x+=identity
        x=self.relu(x) ## final relu
        
        return x

class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()
       

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
      identity = x.clone()

      x = self.relu(self.batch_norm2(self.conv1(x)))
      x = self.batch_norm2(self.conv2(x))

      if self.i_downsample is not None:
          identity = self.i_downsample(identity)
      

      x += identity
      x = self.relu(x)
      return x


        
class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64 ### 64 input channels
        
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*ResBlock.expansion, num_classes)
        
    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x
        
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*ResBlock.expansion)
            )
            
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers)



class Eqv_ResNet(nn.Module):
    def __init__(self, so2_num, ResBlock, layer_list, num_classes, num_input_channels=3 ):
        super(Eqv_ResNet, self).__init__()
        self.in_channels = 64 ### 64 input channels
        
        #### set the so2 discritization, this should always be a power of 2 that is less than or equal to 64:
        self.so2_gspace = so2_num ### so2_gspace discritization
        gspace = e2cnn.gspaces.Rot2dOnR2(N=so2_gspace, maximum_frequency=None, fibergroup=None)

        ### num_input_channels copies of the trivial rep: input images are num_input_channels invarient features
        self.rho_triv = e2cnn.nn.FieldType( gspace , [gspace.trivial_repr]*num_input_channels )

        ### 64 channel regular features
        rho_first = e2cnn.nn.FieldType( gspace , [gspace.regular_repr]*int(64/so2_gspace) )

        ### first convolutional layer: trivial --> regular
        self.conv1 = e2cnn.nn.R2Conv( self.rho_triv , rho_first , kernel_size=7, stride=2, padding=3, bias=False )
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu1 = e2cnn.nn.ReLU( rho_first )


        ### Norm max pool over spatial extent: this should be checked
        self.max_pool2d_layer = e2cnn.nn.NormMaxPool( rho_first , kernel_size=3, stride=2, padding=1 )
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*ResBlock.expansion, num_classes) ### fully connected layer
        
    def forward(self, x):

        ### convert x to geometric tensor
        x = self.conv1(x)
        #x = self.batch_norm1(x)

        x = self.relu1(x)

        ### max pool
        x = self.max_pool2d_layer(x)

        ### resnet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x
        
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*ResBlock.expansion)
            )
            
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers)

### standard resnets
def ResNet50(num_classes, channels=3):
    return ResNet(Bottleneck, [3,4,6,3], num_classes, channels)
    
def ResNet101(num_classes, channels=3):
    return ResNet(Bottleneck, [3,4,23,3], num_classes, channels)

def ResNet152(num_classes, channels=3):
    return ResNet(Bottleneck, [3,8,36,3], num_classes, channels)


### equivarient resnets
def Eqv_ResNet50( so2_num, num_classes, channels=3):
    return Eqv_ResNet(so2_num, Bottleneck, [3,4,6,3], num_classes, channels)
    
def Eqv_ResNet101(so2_num, num_classes, channels=3):
    return Eqv_ResNet(so2_num, Bottleneck, [3,4,23,3], num_classes, channels)

def Eqv_ResNet152(so2_num, num_classes, channels=3):
    return Eqv_ResNet(so2_num, Bottleneck, [3,8,36,3], num_classes, channels)





if __name__ == "__main__":

    ### check for so2 equivarience
    so2_gspace = 4
    gspace = e2cnn.gspaces.Rot2dOnR2(N=so2_gspace, maximum_frequency=None, fibergroup=None)

    ### 3 copies of the trivial rep: input images are 3 color channels
    rho_triv = e2cnn.nn.FieldType( gspace , [gspace.trivial_repr]*3 )
    so2 = gspace.fibergroup

    x = torch.rand( 10 , 3 , 256 , 256 )
    x = e2cnn.nn.GeometricTensor( x , rho_triv )

    f = Eqv_ResNet50( so2_num=so2_gspace, num_classes=4, channels=3 )

    ### unchanged y-values:
    y = f( x )

    for g in so2.elements:

        1

        # x_rot = x.transform(g)

        # ### new inputs
        # y_rot = f( x_rot.tensor )

        # # ### meausre the differences:
        # z0 = y[0].transform(g)
        # z1 = y[1].transform(g)
        # z2 = y[2].transform(g)
        # z3 = y[3].transform(g)


        # ### mesure differences
        # d0 = z0.tensor - y_rot[0].tensor
        # d1 = z1.tensor - y_rot[1].tensor
        # d2 = z2.tensor - y_rot[2].tensor
        # d3 = z3.tensor - y_rot[3].tensor

        # ### take the norm
        # print()
        # print("group element:" , g)
        # print( 'zero percentage error:' ,  torch.norm(d0)/torch.norm( z0.tensor ) ) 
        # print( 'one percentage error:' ,  torch.norm(d1)/torch.norm( z1.tensor ) ) 
        # print( 'two percentage error:' ,  torch.norm(d2)/torch.norm( z2.tensor ) ) 
        # print( 'three percentage error:' ,  torch.norm(d3)/torch.norm( z3.tensor ) ) 
        # print()

        # ### check types of outputs
        ###print( y_rot[0].type , y_rot[1].type , y_rot[2].type , y_rot[3].type )





