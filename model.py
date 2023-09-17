
import torch
import torch.nn as nn
import torch.nn.functional as F
import e2cnn
from torch.autograd import Variable


from bottleneck import Equ_Bottleneck
from eqv_fpn import eqv_FPN101, eqv_FPN210
from recombination_module import concat_recombinator, quorum_recombinator


class FPN_predictor(nn.Module):
	"""docstring for SO(2)-equivarient prediction with FPN head"""
	def __init__(self, so2_gspace, num_classes , encoder , recombinator ):
		super( FPN_predictor , self ).__init__()

		self.num_classes = num_classes
		self.so2_gspace = so2_gspace

		### choice of FPN size:
		### equivarient fpn head; input: (b,3,256,256) images --> outputs : [b,256,64,64] , [b,256,32,32] , [b,256,16,16] , [b,256,8,8]
		if encoder=='eqv_fpn101':
			self.fpn = eqv_FPN101( so2_gspace  ) 

		elif encoder =='eqv_fpn210':
			self.fpn = eqv_FPN210( so2_gspace  )

		### choice of feature recombinator:
		if recombinator=='concat':
			self.recombination_layer = concat_recombinator( self.num_classes )
		elif recombinator == 'quorum':
			self.recombination_layer = quorum_recombinator( self.num_classes  )

		#### set the so2 discritization, this should always be a power of 2 that is less than or equal to 64:
		self.so2_gspace = so2_gspace
		gspace = e2cnn.gspaces.Rot2dOnR2( N=so2_gspace, maximum_frequency=None, fibergroup=None)

		### 64 channel regular features
		rho_input = e2cnn.nn.FieldType( gspace , [gspace.regular_repr]*int(256/so2_gspace) )
		rho_output = e2cnn.nn.FieldType( gspace , [gspace.regular_repr]*int(512/so2_gspace) )

		### zero-th convolutional layer: regular --> regular
		self.conv_zero = e2cnn.nn.R2Conv( rho_input , rho_output , kernel_size=7, stride=2, padding=3, bias=False )
		self.bn_zero = e2cnn.nn.InnerBatchNorm( rho_output ) 
		self.relu_zero = e2cnn.nn.ReLU( rho_output )

		### zero-th convolutional layer: regular --> regular
		self.conv_zero_a = e2cnn.nn.R2Conv( rho_output , rho_output , kernel_size=7, stride=2, padding=3, bias=False )
		self.bn_zero_a = e2cnn.nn.InnerBatchNorm( rho_output ) 
		self.relu_zero_a = e2cnn.nn.ReLU( rho_output )

		### zero-th convolutional layer: regular --> regular
		self.conv_zero_b = e2cnn.nn.R2Conv( rho_output , rho_output , kernel_size=7, stride=2, padding=3, bias=False )
		self.bn_zero_b = e2cnn.nn.InnerBatchNorm( rho_output ) 
		self.relu_zero_b = e2cnn.nn.ReLU( rho_output )

		### zero-th convolutional layer: regular --> regular
		self.conv_zero_c = e2cnn.nn.R2Conv( rho_output , rho_output , kernel_size=7, stride=2, padding=3, bias=False )
		self.bn_zero_c = e2cnn.nn.InnerBatchNorm( rho_output ) 
		self.relu_zero_c = e2cnn.nn.ReLU( rho_output )

		### first-th convolutional layer: regular --> regular
		self.conv_one = e2cnn.nn.R2Conv( rho_input , rho_output , kernel_size=7, stride=2, padding=3, bias=False )
		self.bn_one = e2cnn.nn.InnerBatchNorm( rho_output ) 
		self.relu_one = e2cnn.nn.ReLU( rho_output )

		### first-th convolutional layer: regular --> regular
		self.conv_one_a = e2cnn.nn.R2Conv( rho_output , rho_output , kernel_size=7, stride=2, padding=3, bias=False )
		self.bn_one_a = e2cnn.nn.InnerBatchNorm( rho_output ) 
		self.relu_one_a = e2cnn.nn.ReLU( rho_output )

		### first-th convolutional layer: regular --> regular
		self.conv_one_b = e2cnn.nn.R2Conv( rho_output , rho_output , kernel_size=7, stride=2, padding=3, bias=False )
		self.bn_one_b = e2cnn.nn.InnerBatchNorm( rho_output ) 
		self.relu_one_b = e2cnn.nn.ReLU( rho_output )

		### second-th convolutional layer: regular --> regular
		self.conv_two = e2cnn.nn.R2Conv( rho_input , rho_output , kernel_size=7, stride=2, padding=3, bias=False )
		self.bn_two = e2cnn.nn.InnerBatchNorm( rho_output ) 
		self.relu_two = e2cnn.nn.ReLU( rho_output )

		### second-th convolutional layer: regular --> regular
		self.conv_two_a = e2cnn.nn.R2Conv( rho_output , rho_output , kernel_size=7, stride=2, padding=3, bias=False )
		self.bn_two_a = e2cnn.nn.InnerBatchNorm( rho_output ) 
		self.relu_two_a = e2cnn.nn.ReLU( rho_output )

		### third convolutional layer: regular --> regular
		self.conv_three = e2cnn.nn.R2Conv( rho_input , rho_output , kernel_size=7, stride=2, padding=3, bias=False )
		self.bn_three = e2cnn.nn.InnerBatchNorm( rho_output ) 
		self.relu_three = e2cnn.nn.ReLU( rho_output )

		### max_pool layers:
		self.max_pool2d_layer_0 = torch.nn.MaxPool2d( kernel_size=4, stride=None )
		self.max_pool2d_layer_1 = torch.nn.MaxPool2d( kernel_size=4, stride=None )
		self.max_pool2d_layer_2 = torch.nn.MaxPool2d( kernel_size=4, stride=None )
		self.max_pool2d_layer_3 = torch.nn.MaxPool2d( kernel_size=4, stride=None )


	def forward(self,x):

		### SO(2)-Equivarient FPN
		y0, y1, y2, y3 = self.fpn(x)

		### zeroth level
		z0 = self.bn_zero( self.conv_zero(y0)  ) ### [b,512, 32,32  ]
		z0 = self.relu_zero( z0 )

		z0_a = self.bn_zero_a( self.conv_zero_a(z0) ) ### [b,512, 16, 16  ]
		z0_a = self.relu_zero_a( z0_a )

		z0_b = self.relu_zero_b( self.bn_zero_b( self.conv_zero_b(z0_a) ) ) ### [b,512, 8, 8  ]
		z0_c = self.relu_zero_c( self.bn_zero_c( self.conv_zero_b(z0_b) ) )### [b,512, 4, 4  ]

		### first level
		z1 = self.relu_one( self.bn_one( self.conv_one(y1) )  ) ### [b,512,  16, 16 ]
		z1_a = self.relu_one_a( self.bn_one_a(  self.conv_one_a(z1) ) )### [b,512,8,8]
		z1_b = self.relu_one_b( self.bn_one_b( self.conv_one_b(z1_a) ) )### [b,512,4,4]

		### second level
		z2 = self.relu_two( self.bn_two( self.conv_two(y2) )  ) ### [b,512,  8,8  ]
		z2_a = self.relu_two_a(  self.bn_three( self.conv_two_a( z2 ) ) ) ### [b,512,  4,4  ]

		### third level
		z3 = self.relu_three( self.bn_three(  self.conv_three(y3) ) ) ### [b,512, 4,4 ]

		### maybe one additional set of convs here to make all features invarient, then no need for maxpool???
		### i.e. conv: [b,512,4,4] --> torch.squeeze( [b,512,1,1] )

		### maxpool layers
		w0 = torch.squeeze( self.max_pool2d_layer_0( z0_c.tensor ) )
		w1 = torch.squeeze( self.max_pool2d_layer_1( z1_b.tensor ) )
		w2 = torch.squeeze( self.max_pool2d_layer_2( z2_a.tensor ) )
		w3 = torch.squeeze( self.max_pool2d_layer_3( z3.tensor ) )

		### Now, recombine: w0 , w1 , w2 , w3 features, all have shape [b,512]
		outputs = self.recombination_layer( w0 , w1 , w2 , w3 )

		return outputs


	def compute_loss(self, images, labels ):

		outputs = self.forward( images )
		loss = nn.CrossEntropyLoss()( outputs , labels )

		preds = torch.argmax( outputs, 1) ### prediction indices, size: [batch size]

		num_correct = torch.sum( torch.eq( preds , labels ) )

		return loss, num_correct, preds


if __name__ == "__main__":

	import numpy as np
	
	so2_gspace = 4
	num_classes = 10
	batch_size = 10

	outputs = torch.rand(batch_size , num_classes )
	outputs = nn.functional.softmax( outputs , dim=1 )


	images = torch.rand( batch_size , 3 , 256 , 256 )
	labels = torch.randint( num_classes , (batch_size, ) )


	### model
	encoder_str =  'eqv_fpn101' 
	recombinator_str = 'quorum_recombinator'
	f = FPN_predictor( so2_gspace, num_classes , encoder_str , recombinator_str )
	y = f.forward( images )

	loss , num_correct, preds = f.compute_loss( images , labels  )	

	print(preds.shape)
	quit()

	# ### check for so2-invarience of outputs:
	# for g in so2.elements:

	# 	x_rot = x.transform(g)

	# 	### new predictions
	# 	y_rot = f( x_rot.tensor )

	# 	### mesure differences
	# 	d0 = z0.tensor - y_rot
		

	# 	### take the norm
	# 	print()
	# 	print("group element:" , g)
	# 	print( 'zero percentage error:' ,  torch.norm(d0)/torch.norm( z0.tensor ) ) 
	# 	print( 'one percentage error:' ,  torch.norm(d1)/torch.norm( z1.tensor ) ) 
	# 	print( 'two percentage error:' ,  torch.norm(d2)/torch.norm( z2.tensor ) ) 
	# 	print( 'three percentage error:' ,  torch.norm(d3)/torch.norm( z3.tensor ) ) 
	# 	print()

	### check types of outputs
	###print( y_rot[0].type , y_rot[1].type , y_rot[2].type , y_rot[3].type )



