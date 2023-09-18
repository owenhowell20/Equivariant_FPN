import torch
import torch.nn as nn
import torch.nn.functional as F
import e2cnn
from torch.autograd import Variable
from bottleneck import Equ_Bottleneck


class lateral_heads(nn.Module):
	"""docstring for SO(2)-equivarient lateral resnet heads"""
	def __init__(self, so2_gspace, head_number ):
		super( lateral_heads , self ).__init__()

		self.so2_gspace = so2_gspace
		self.gspace_dim = 2*so2_gspace

		expansion=2
		self.neck_1 = Equ_Bottleneck( so2_gspace=self.so2_gspace, in_planes=256, planes=256,  stride=1 , expansion=expansion )
		self.neck_2 = Equ_Bottleneck( so2_gspace=self.so2_gspace, in_planes=expansion*256, planes=expansion*256,  stride=1 , expansion=expansion )

		gspace = e2cnn.gspaces.FlipRot2dOnR2( N=so2_gspace, maximum_frequency=None, fibergroup=None)
		self.rho_output = e2cnn.nn.FieldType( gspace , [gspace.regular_repr]*int(expansion*256/self.gspace_dim)  )
		self.rho = e2cnn.nn.FieldType( gspace , [gspace.regular_repr]*int( expansion*expansion*256/self.gspace_dim ) )

		num_layers = 4 - head_number
		layers = []
		for stride in range(num_layers):

			conv = e2cnn.nn.R2Conv( self.rho_output , self.rho_output , kernel_size=7, stride=2, padding=3, bias=False )
			bn = e2cnn.nn.InnerBatchNorm( self.rho_output ) 
			relu = e2cnn.nn.ReLU( self.rho_output )

			layers.append( conv )
			layers.append( bn   )
			layers.append( relu )

		self.head = nn.Sequential(*layers)
		self.pool = e2cnn.nn.NormMaxPool( self.rho , kernel_size=4 )

	def forward(self,x):

		outputs = self.neck_2( self.head( self.neck_1( x ) ) )
		outputs = self.pool(outputs)

		return outputs ### [b,1024,1,1]


### ResNet lateral heads
class ResNet_lateral_heads(nn.Module):
	"""docstring for SO(2)-equivarient resnet type lateral resnet heads"""
	def __init__(self, so2_gspace, head_number ):
		super( ResNet_lateral_heads , self ).__init__()

		expansion=2
		self.so2_gspace = so2_gspace
		self.gspace_dim = 2*so2_gspace
		gspace = e2cnn.gspaces.FlipRot2dOnR2( N=so2_gspace, maximum_frequency=None, fibergroup=None)
		self.neck_1 = Equ_Bottleneck( so2_gspace=self.so2_gspace, in_planes=256, planes=256,  stride=1 , expansion=2 )
		self.neck_2 = Equ_Bottleneck( so2_gspace=self.so2_gspace, in_planes=expansion*256, planes=expansion*256,  stride=1 , expansion=2 )
		self.rho = e2cnn.nn.FieldType( gspace , [gspace.regular_repr]*int(256*expansion*expansion/self.gspace_dim ) )


		num_layers = 4 - head_number
		layers = []
		current_dim = 256*expansion
		for stride in range(num_layers):

			neck = Equ_Bottleneck( so2_gspace=self.so2_gspace, in_planes=current_dim, planes=current_dim,  stride=2 , expansion=1 )
			layers.append( neck )
			
		self.head = nn.Sequential(*layers)
		self.pool = e2cnn.nn.NormMaxPool( self.rho , kernel_size=4)


	def forward(self,x):

		outputs = self.neck_2( self.head( self.neck_1( x ) ) )
		outputs = self.pool(outputs)

		return outputs ### [b,1024,1,1]



if __name__ == "__main__":

	import numpy as np
	
	so2_gspace = 4
	gspace_dim = 2*so2_gspace
	batch_size = 4

	gspace = e2cnn.gspaces.FlipRot2dOnR2( N=so2_gspace, maximum_frequency=None, fibergroup=None)
	rho_output = e2cnn.nn.FieldType( gspace , [gspace.regular_repr]*int( 256/gspace_dim ) )

	features_0 = torch.rand( batch_size , 256 , 64 , 64 ) 
	features_1 = torch.rand( batch_size , 256 , 32 , 32 )
	features_2 = torch.rand( batch_size , 256 , 16 , 16 )
	features_3 = torch.rand( batch_size , 256 , 8 , 8 ) 


	features_0 = e2cnn.nn.GeometricTensor( features_0 , rho_output )
	features_1 = e2cnn.nn.GeometricTensor( features_1 , rho_output )
	features_2 = e2cnn.nn.GeometricTensor( features_2 , rho_output )
	features_3 = e2cnn.nn.GeometricTensor( features_3 , rho_output )


	### feature heads
	head_0 = lateral_heads( so2_gspace=so2_gspace, head_number=0)
	outputs_0 = head_0(features_0)

	head_1 = lateral_heads( so2_gspace=so2_gspace, head_number=1)
	outputs_1 = head_1(features_1)

	head_2 = lateral_heads( so2_gspace=so2_gspace, head_number=2)
	outputs_2 = head_2(features_2)

	head_3 = lateral_heads( so2_gspace=so2_gspace, head_number=3)
	outputs_3 = head_3(features_3)

	print(outputs_0.shape)
	print(outputs_1.shape)
	print(outputs_2.shape)
	print(outputs_3.shape)


