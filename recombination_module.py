
import torch
import torch.nn as nn
import torch.nn.functional as F
import e2cnn
from torch.autograd import Variable

from bottleneck import Equ_Bottleneck
from eqv_fpn import eqv_FPN101, eqv_FPN210
import numpy as np

### concatanation recombination
class concat_recombinator(nn.Module):

	def __init__(self,  num_classes , in_features,  hidden_linear_size=32 ):
		super( concat_recombinator, self).__init__()

		self.num_classes = num_classes

		### fully connected layers:
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

	def __init__(self, num_classes , in_features, hidden_linear_size=256 ):
		super( quorum_recombinator, self).__init__()

		self.num_classes = num_classes

		### fully connected layers:
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
		w0 = self.linear_final_a( w0 )
		w1 = self.linear_final_b( w1 )
		w2 = self.linear_final_c( w2 )
		w3 = self.linear_final_d( w3 )

		### softmax
		outputs_a = nn.functional.softmax( w0 , dim=1 )
		outputs_b = nn.functional.softmax( w1 , dim=1 )
		outputs_c = nn.functional.softmax( w2 , dim=1 )
		outputs_d = nn.functional.softmax( w3 , dim=1 )

		### Make each prediction seperatly and do majority vote 
		outputs = 0.25*( outputs_a + outputs_b + outputs_c + outputs_d )

		return outputs


### single head multi-scale self-attention module
class SingleHead_attention_recombinator(nn.Module):

	def __init__(self, num_classes, in_features, hidden_linear_size ):
		super( attention_recombinator, self).__init__()

		self.num_classes = num_classes

		### fully connected layers:
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
		### self.attention_layer = nn.MultiheadAttention(embed_dim=128, num_heads=4, dropout=0.1, kdim=None, vdim=None, batch_first=False )

		### attention encoders: each pyrimid level has its own set of features
		f_enc = 10
		self.f_enc = f_enc
		self.Q_linear_0 = nn.Linear( out_features, f_enc, bias=True, device=None, dtype=None)
		self.K_linear_0 = nn.Linear( out_features, f_enc, bias=True, device=None, dtype=None)
		self.V_linear_0 = nn.Linear( out_features, f_enc, bias=True, device=None, dtype=None)

		self.Q_linear_1 = nn.Linear( out_features, f_enc, bias=True, device=None, dtype=None)
		self.K_linear_1 = nn.Linear( out_features, f_enc, bias=True, device=None, dtype=None)
		self.V_linear_1 = nn.Linear( out_features, f_enc, bias=True, device=None, dtype=None)

		self.Q_linear_2 = nn.Linear( out_features, f_enc, bias=True, device=None, dtype=None)
		self.K_linear_2 = nn.Linear( out_features, f_enc, bias=True, device=None, dtype=None)
		self.V_linear_2 = nn.Linear( out_features, f_enc, bias=True, device=None, dtype=None)

		self.Q_linear_3 = nn.Linear( out_features, f_enc, bias=True, device=None, dtype=None)
		self.K_linear_3 = nn.Linear( out_features, f_enc, bias=True, device=None, dtype=None)
		self.V_linear_3 = nn.Linear( out_features, f_enc, bias=True, device=None, dtype=None)

		self.softmax = torch.nn.Softmax(dim=1)


		### the final linear layers
		final_in_features = f_enc
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

		#### encode features
		q_enc_0 = self.Q_linear_0( w0 )
		k_enc_0 = self.K_linear_0( w0 )
		v_enc_0 = self.V_linear_0( w0 )

		q_enc_1 = self.Q_linear_1( w1 )
		k_enc_1 = self.K_linear_1( w1 )
		v_enc_1 = self.V_linear_1( w1 )

		q_enc_2 = self.Q_linear_2( w2 )
		k_enc_2 = self.K_linear_2( w2 )
		v_enc_2 = self.V_linear_2( w2 )

		q_enc_3 = self.Q_linear_3( w3 )
		k_enc_3 = self.K_linear_3( w3 )
		v_enc_3 = self.V_linear_3( w3 )

		### stack encodeings
		q_enc = torch.stack( [q_enc_0,q_enc_1,q_enc_2,q_enc_3] , dim=1 ) ### [b,4,f_enc]
		k_enc = torch.stack( [k_enc_0,k_enc_1,k_enc_2,k_enc_3] , dim=1 ) ### [b,4,f_enc]
		v_enc = torch.stack( [v_enc_0,v_enc_1,v_enc_2,v_enc_3] , dim=1 ) ### [b,4,f_enc]

		
		attn_weights = torch.einsum( 'bij,bil->bjl' , q_enc , k_enc )/np.sqrt( self.f_enc )
		attn_weights = self.softmax( attn_weights )
		attention_output = torch.einsum( 'bnm , bhm -> bhn ' , attn_weights , v_enc )


		### now split
		w0 = attention_output[:,0,:]
		w1 = attention_output[:,0,:]
		w2 = attention_output[:,0,:]
		w3 = attention_output[:,0,:]


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

		return outputs, attn_weights


if __name__ == "__main__":

	batch_size = 64
	features = 1024
	num_classes = 101

	### FPN features to be recombined: all transform in same SO(2)-feature type
	w0 = torch.rand( batch_size , features )
	w1 = torch.rand( batch_size , features )
	w2 = torch.rand( batch_size , features )
	w3 = torch.rand( batch_size , features )
	
	recombination_layer = SingleHead_attention_recombinator(num_classes, in_features=features, hidden_linear_size=128)
	#recombination_layer = quorum_recombinator( num_classes , in_features=features, hidden_linear_size=128 )
	#recombination_layer = concat_recombinator( num_classes , in_features=features, hidden_linear_size=128 )
	
	outputs = recombination_layer.forward( w0 , w1 , w2 , w3 )

	print(outputs.shape)

