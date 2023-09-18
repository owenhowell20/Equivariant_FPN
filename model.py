
import torch
import torch.nn as nn
import torch.nn.functional as F
import e2cnn
from torch.autograd import Variable

from bottleneck import Equ_Bottleneck
from eqv_fpn import eqv_FPN101, eqv_FPN210
from recombination_module import concat_recombinator, quorum_recombinator, attention_recombinator
from lateral_module import lateral_heads


class FPN_predictor(nn.Module):
	"""docstring for SO(2)-equivarient prediction with FPN head"""
	def __init__(self, so2_gspace, num_classes , encoder , recombinator ):
		super( FPN_predictor , self ).__init__()

		self.num_classes = num_classes
		self.so2_gspace = so2_gspace
		self.gspace_dim =  2*so2_gspace

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
		elif recombinator == 'attention':
			self.recombination_layer = attention_recombinator( self.num_classes  )

		#### set the so2 discritization, this should always be a power of 2 that is less than or equal to 64:
		self.so2_gspace = so2_gspace

		### feature heads
		self.head_0 = lateral_heads( so2_gspace=so2_gspace, head_number=0 )
		self.head_1 = lateral_heads( so2_gspace=so2_gspace, head_number=1 )
		self.head_2 = lateral_heads( so2_gspace=so2_gspace, head_number=2 )
		self.head_3 = lateral_heads( so2_gspace=so2_gspace, head_number=3 )


	def forward(self,x):

		### SO(2)-Equivarient FPN
		y0, y1, y2, y3 = self.fpn(x) ### output shapes: [b,256,64,64],[b,256,32,32],[b,256,16,16],[b,256,8,8]

		### run through the heads
		z0 = self.head_0( y0 )
		z1 = self.head_1( y1 )
		z2 = self.head_2( y2 )
		z3 = self.head_3( y3 )

		### maxpool layers, convert to tensor and squeeze
		w0 = torch.squeeze(  z0.tensor  )
		w1 = torch.squeeze(  z1.tensor  ) 
		w2 = torch.squeeze(  z2.tensor  )
		w3 = torch.squeeze(  z3.tensor  )

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
	recombinator_str = 'quorum'
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



