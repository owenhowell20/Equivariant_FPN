
import torch
import torch.nn as nn
import torch.nn.functional as F
import e2cnn
from torch.autograd import Variable

from bottleneck import Equ_Bottleneck
from eqv_fpn import eqv_FPN101, eqv_FPN210

### concatanation recombination
class concat_recombinator(nn.Module):

	def __init__(self,  num_classes , hidden_linear_size=32 ):
		super( concat_recombinator, self).__init__()

		self.num_classes = num_classes

		### fully connected layers:
		in_features = 512
		out_features = hidden_linear_size
		self.linear_0 = nn.Linear( in_features, out_features, bias=True, device=None, dtype=None)
		self.dropout_0 = nn.Dropout(p=0.2)

		self.linear_1 = nn.Linear( in_features, out_features, bias=True, device=None, dtype=None)
		self.dropout_1 = nn.Dropout(p=0.2)

		self.linear_2 = nn.Linear( in_features, out_features, bias=True, device=None, dtype=None)
		self.dropout_2 = nn.Dropout(p=0.2)

		self.linear_3 = nn.Linear( in_features, out_features, bias=True, device=None, dtype=None)
		self.dropout_3 = nn.Dropout(p=0.2)

		### the final linear layer
		final_in_features = 4*out_features
		final_out_features = self.num_classes
		self.linear_final = nn.Linear( final_in_features, final_out_features, bias=True, device=None, dtype=None)


	def forward(self,w0,w1,w2,w3):

		### now fully connected layers
		w0 = self.dropout_0( self.linear_0( w0 ) )
		w1 = self.dropout_1( self.linear_1( w1 ) )
		w2 = self.dropout_2( self.linear_2( w2 ) )
		w3 = self.dropout_3( self.linear_3( w3 ) )
		
		### concat features and do a final linear layer:
		c = torch.cat( (w0,w1,w2,w3) , 1 )
		outputs = self.linear_final( c )

		### softmax
		outputs = nn.functional.softmax( outputs , dim=1 )

		return outputs


### quorum recombination
class quorum_recombinator(nn.Module):

	def __init__(self, num_classes , hidden_linear_size=128 ):
		super( quorum_recombinator, self).__init__()

		self.num_classes = num_classes

		### fully connected layers:
		in_features = 512
		out_features = hidden_linear_size
		self.linear_0 = nn.Linear( in_features, out_features, bias=True, device=None, dtype=None)
		self.dropout_0 = nn.Dropout(p=0.2)

		self.linear_1 = nn.Linear( in_features, out_features, bias=True, device=None, dtype=None)
		self.dropout_1 = nn.Dropout(p=0.2)

		self.linear_2 = nn.Linear( in_features, out_features, bias=True, device=None, dtype=None)
		self.dropout_2 = nn.Dropout(p=0.2)

		self.linear_3 = nn.Linear( in_features, out_features, bias=True, device=None, dtype=None)
		self.dropout_3 = nn.Dropout(p=0.2)

		### the final linear layers
		final_in_features = out_features
		final_out_features = self.num_classes
		self.linear_final_a = nn.Linear( final_in_features, final_out_features, bias=True, device=None, dtype=None)
		self.linear_final_b = nn.Linear( final_in_features, final_out_features, bias=True, device=None, dtype=None)
		self.linear_final_c = nn.Linear( final_in_features, final_out_features, bias=True, device=None, dtype=None)
		self.linear_final_d = nn.Linear( final_in_features, final_out_features, bias=True, device=None, dtype=None)

	def forward(self,w0,w1,w2,w3):

		### now fully connected layers
		w0 = self.dropout_0( self.linear_0( w0 ) )
		w1 = self.dropout_1( self.linear_1( w1 ) )
		w2 = self.dropout_2( self.linear_2( w2 ) )
		w3 = self.dropout_3( self.linear_3( w3 ) )
		
		### final linear layers:
		outputs_a = self.linear_final_a( w0 )
		outputs_b = self.linear_final_b( w1 )
		outputs_c = self.linear_final_c( w2 )
		outputs_d = self.linear_final_d( w3 )

		### softmax
		outputs_a = nn.functional.softmax( outputs_a , dim=1 )
		outputs_b = nn.functional.softmax( outputs_b , dim=1 )
		outputs_c = nn.functional.softmax( outputs_c , dim=1 )
		outputs_d = nn.functional.softmax( outputs_d , dim=1 )

		### Make each prediction seperatly and do majority vote 
		outputs = 0.25*( outputs_a + outputs_b + outputs_c + outputs_d )

		return outputs


### multi-scale self-attention module
class attention_recombinator(nn.Module):

	def __init__(self, num_classes, hidden_linear_size ):
		super( attention_recombinator, self).__init__()

		self.num_classes = num_classes

		### fully connected layers:
		in_features = 512
		out_features = hidden_linear_size
		self.linear_0 = nn.Linear( in_features, out_features, bias=True, device=None, dtype=None)
		self.dropout_0 = nn.Dropout(p=0.2)

		self.linear_1 = nn.Linear( in_features, out_features, bias=True, device=None, dtype=None)
		self.dropout_1 = nn.Dropout(p=0.2)

		self.linear_2 = nn.Linear( in_features, out_features, bias=True, device=None, dtype=None)
		self.dropout_2 = nn.Dropout(p=0.2)

		self.linear_3 = nn.Linear( in_features, out_features, bias=True, device=None, dtype=None)
		self.dropout_3 = nn.Dropout(p=0.2)

		### multi-head self-attention layers on different scale features:
		self.attention_layer = nn.MultiheadAttention(embed_dim=128, num_heads=4, dropout=0.1, kdim=None, vdim=None, batch_first=False )

		### the final linear layers
		final_in_features = out_features
		final_out_features = self.num_classes
		self.linear_final_a = nn.Linear( final_in_features, final_out_features, bias=True, device=None, dtype=None)
		self.linear_final_b = nn.Linear( final_in_features, final_out_features, bias=True, device=None, dtype=None)
		self.linear_final_c = nn.Linear( final_in_features, final_out_features, bias=True, device=None, dtype=None)
		self.linear_final_d = nn.Linear( final_in_features, final_out_features, bias=True, device=None, dtype=None)

	def forward(self,w0,w1,w2,w3):

		### now fully connected layers: output dim is [b,128]
		w0 = self.dropout_0( self.linear_0( w0 ) )
		w1 = self.dropout_1( self.linear_1( w1 ) )
		w2 = self.dropout_2( self.linear_2( w2 ) )
		w3 = self.dropout_3( self.linear_3( w3 ) )
		
		### self-attention layer
		attn_output, attn_output_weights = self.attention_layer( w0, w0, w0 )

		print(attn_output.shape)
		print( attn_output_weights.shape ) ### this should be [128,128]
		quit()

		### final linear layers:
		outputs_a = self.linear_final_a( w0 )
		outputs_b = self.linear_final_b( w1 )
		outputs_c = self.linear_final_c( w2 )
		outputs_d = self.linear_final_d( w3 )

		### softmax
		outputs_a = nn.functional.softmax( outputs_a , dim=1 )
		outputs_b = nn.functional.softmax( outputs_b , dim=1 )
		outputs_c = nn.functional.softmax( outputs_c , dim=1 )
		outputs_d = nn.functional.softmax( outputs_d , dim=1 )

		### Make each prediction seperatly and do majority vote 
		outputs = 0.25*( outputs_a + outputs_b + outputs_c + outputs_d )

		return outputs


if __name__ == "__main__":

	batch_size = 64
	features = 512
	num_classes = 101

	### FPN features to be recombined: all transform in same SO(2)-feature type
	w0 = torch.rand(batch_size , features )
	w1 = torch.rand(batch_size , features )
	w2 = torch.rand(batch_size , features )
	w3 = torch.rand(batch_size , features )
	
	#recombination_layer = attention_recombinator(num_classes)
	#recombination_layer = quorum_recombinator( num_classes )
	recombination_layer = concat_recombinator( num_classes )
	
	outputs = recombination_layer.forward( w0 , w1 , w2 , w3 )


	print(outputs.shape)



