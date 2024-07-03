import math
import torch
import numpy as np
import torch.nn as nn
import torchvision.ops
import torch.nn.functional as F


from models import register
from argparse import Namespace
from torch.nn.parameter import Parameter



class CNNEncoder_baseline_light3(nn.Module):
	# less layers, 6 layers in total
	def __init__(self, nc=1, isLastPooling=False):
		super(CNNEncoder_baseline_light3, self).__init__()


		ks = [3, 3, 3, 3, 3, 3, 3, 3, 3]
		ps = [1, 1, 1, 1, 1, 1, 1, 1, 1] 
		ss = [1, 1, 1, 1, 1, 1, 1, 1, 1]
		nm = [16, 16, 16, 32, 32, 32, 64, 64, 64]
		self.isLastPooling = isLastPooling

		self.conv1 = self.conv_bn_relu(nc, nm[0])
		self.conv2 = self.conv_bn_relu(nm[0], nm[1])
		# self.conv3 = self.conv_bn_relu(nm[1], nm[2])
		self.conv4 = self.conv_bn_relu(nm[2], nm[3])
		self.conv5 = self.conv_bn_relu(nm[3], nm[4])
		# self.conv6 = self.conv_bn_relu(nm[4], nm[5])
		self.conv7 = self.conv_bn_relu(nm[5], nm[6])
		self.conv8 = self.conv_bn_relu(nm[6], nm[7])
		# self.conv9 = self.conv_bn_relu(nm[7], nm[8])

		self.pooling = nn.MaxPool2d(2, 2)


	def conv_bn_relu(self,c_in,c_out,kernel=3,stride=1,pad=1):
		layers = nn.Sequential(
			nn.Conv2d(c_in, c_out, kernel, stride, pad),
			nn.BatchNorm2d(c_out),
			nn.ReLU(True))
		return nn.Sequential(*layers)

	def forward(self, input, tsne=False):
		decouple_outs = []
		text_features = []
		out1 = self.conv1(input)
		out2 = self.conv2(out1)
		# out3 = self.conv3(out2)

		out4 = self.conv4(self.pooling(out2))
		out5 = self.conv5(out4)
		# out6 = self.conv6(out5)

		out7 = self.conv7(self.pooling(out5))
		out8 = self.conv8(out7)
		# out9 = self.conv9(out8)
		# text_features.append(out2)
		if self.isLastPooling:
			conv_out = self.pooling(out8)
		else:
			conv_out = out8

		if tsne:
			return conv_out, decouple_outs, text_features
		else:
			return conv_out, text_features

class FCDecoder(nn.Module):
	def __init__(self, nclass, input_dim=512, K =8):
		super(FCDecoder, self).__init__()
		self.input_dim = input_dim
		self.nclass = nclass
		self.fc = nn.Linear(self.input_dim, self.nclass)

	def forward(self, input):
		preds = self.fc(input)

		return preds

class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))
    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)
    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()

class PositionalEncoding_2D(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding_2D, self).__init__()
        self.register_buffer('pos_table_h', self._get_sinusoid_encoding_table(8, int(d_hid/2)))
        self.register_buffer('pos_table_w', self._get_sinusoid_encoding_table(24, int(d_hid/2)))
    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)
    def forward(self, x):
        pos_table_h_br = self.pos_table_h.squeeze().unsqueeze(1).repeat(1, 24, 1)
        pos_table_w_br = self.pos_table_w.repeat(8, 1, 1)

        pos_table_br = torch.cat((pos_table_h_br, pos_table_w_br), 2)
        h,w,c = pos_table_br.size()
        pos_table_br = pos_table_br.view(-1, c).unsqueeze(0)

        return x + pos_table_br[:, :x.size(1)].clone().detach()

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)
    
    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)
        attn = self.softmax(attn)     
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()
        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv
        mask = mask.repeat(n_head, 1, 1) if mask is not None else None # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)
        output = output.view(n_head, sz_b, len_q, d_v)
        attn = attn.view(n_head, sz_b, len_q, len_q)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        return output, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''
    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = x.transpose(1, 2)
        x = self.w_2(F.relu(self.w_1(x)))
        x = x.transpose(1, 2)
        x = self.dropout(x)
        x = self.layer_norm(x + residual)
        return x

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

class Transforme_Encoder(nn.Module):
    def __init__(
            self, d_word_vec=512, n_layers=2, n_head=8, d_k=64, d_v=64,
            d_model=512, d_inner=2048, dropout=0.1, n_position=256):
        super(Transforme_Encoder,self).__init__()
        self.position_enc = PositionalEncoding_2D(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, enc_output, src_mask, return_attns=False):
        enc_output = self.dropout(self.position_enc(enc_output))   # position embeding
        slf_attn_lst = []
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            slf_attn_lst.append(enc_slf_attn)
        enc_output = self.layer_norm(enc_output)
        return slf_attn_lst, enc_output,

class PP_layer(nn.Module):
    def __init__(self,  n_dim=512, N_max_character=25, n_position=256):

        super(PP_layer, self).__init__()
        self.character_len = N_max_character
        self.f0_embedding = nn.Embedding(N_max_character, n_dim)
        self.w0 = nn.Linear(N_max_character, n_position)
        self.wv = nn.Linear(n_dim, n_dim)
        self.we = nn.Linear(n_dim, N_max_character)
        self.active = nn.Tanh()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, enc_output):
        reading_order = torch.arange(self.character_len, dtype=torch.long, device=enc_output.device)
        reading_order = reading_order.unsqueeze(0).expand(enc_output.size(0), -1)    # (S,) -> (B, S)
        reading_order = self.f0_embedding(reading_order)
        # calculate attention
        t = self.w0(reading_order.permute(0,2,1))
        t = self.active(t.permute(0,2,1) + self.wv(enc_output))
        t = self.we(t)
        t = self.softmax(t.permute(0,2,1))
        g_output = torch.bmm(t, enc_output)
        return g_output, t

class Prediction(nn.Module):
    def __init__(self, n_dim=512, n_class=37, N_max_character=25, n_position=256, GSRM_layer=4, nchannel=512 ):
        super(Prediction, self).__init__()
        self.pp = PP_layer(N_max_character=N_max_character, n_position=n_position)
        self.pp_share = PP_layer(N_max_character=N_max_character, n_position=n_position)
        self.w_vrm = nn.Linear(n_dim, n_class)    # output layer
        self.nclass = n_class

    def forward(self, cnn_feature,  f_res, f_sub, Train_is = False, use_mlm = True):
        g_output, attn = self.pp(cnn_feature)
        g_output = self.w_vrm(g_output)
        return g_output

class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1):

        super(DeformableConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        
        # init weight and bias
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        # offset conv
        self.conv_offset_mask = nn.Conv2d(in_channels, 
                                          3 * kernel_size * kernel_size,
                                          kernel_size=kernel_size, 
                                          stride=stride,
                                          padding=self.padding, 
                                          bias=True)
        
        # init        
        self.reset_parameters()
        self._init_weight()


    def reset_parameters(self):
        n = self.in_channels * (self.kernel_size**2)
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()


    def _init_weight(self):
        # init offset_mask conv
        nn.init.constant_(self.conv_offset_mask.weight, 0.)
        nn.init.constant_(self.conv_offset_mask.bias, 0.)


    def forward(self, x):
        out = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        x = torchvision.ops.deform_conv2d(input=x, 
                                          offset=offset, 
                                          weight=self.weight, 
                                          bias=self.bias, 
                                          padding=self.padding,
                                          mask=mask,
                                          stride=self.stride)
        return x

class Deformable_Attention(nn.Module):

    def __init__(self, nc, K=7, downsample=4):
        # Attention_module(nc=128, K=7)
        super(Deformable_Attention, self).__init__()
        self.K = K

        nm = [512,256,128]

        self.fc_k = nn.Linear(192, 192, bias=False)
        self.fc_v = nn.Linear(192, 192, bias=False)

        atten_0_0 = nn.Sequential()
        atten_0_0.add_module('conv_a_0',nn.Conv2d(nc, nm[1], 3, 1, 1))
        atten_0_0.add_module('bn_a_0', nn.BatchNorm2d(nm[1]))
        atten_0_0.add_module('relu_a_0', nn.ReLU(True))
        atten_0_1 = nn.Sequential()
        atten_0_1.add_module('conv_a_1',nn.Conv2d(nm[1], nm[1], 3, 1, 1))
        atten_0_1.add_module('bn_a_1', nn.BatchNorm2d(nm[1]))
        atten_0_1.add_module('relu_a_1', nn.ReLU(True))
        atten_0_2 = nn.Sequential()
        atten_0_2.add_module('conv_a_2',nn.Conv2d(nm[1], nm[1], 3, 1, 1))
        atten_0_2.add_module('bn_a_2', nn.BatchNorm2d(nm[1]))
        atten_0_2.add_module('relu_a_2', nn.ReLU(True))
        atten_0_2.add_module('pooling_a_2',nn.MaxPool2d((2, 2)))
        atten_0_3 = nn.Sequential()
        atten_0_3.add_module('conv_a_3',nn.Conv2d(nm[1], nm[2], 3, 1, 1))
        atten_0_3.add_module('bn_a_3', nn.BatchNorm2d(nm[2]))
        atten_0_3.add_module('relu_a_3', nn.ReLU(True))
        atten_0_4 = nn.Sequential()
        atten_0_4.add_module('conv_a_4',nn.Conv2d(nm[2], nm[2], 3, 1, 1))
        atten_0_4.add_module('bn_a_4', nn.BatchNorm2d(nm[2]))
        atten_0_4.add_module('relu_a_4', nn.ReLU(True))
        atten_0_5 = nn.Sequential()
        atten_0_5.add_module('conv_a_5',nn.Conv2d(nm[2], nm[2], 3, 1, 1))
        atten_0_5.add_module('bn_a_5', nn.BatchNorm2d(nm[2]))
        atten_0_5.add_module('relu_a_5', nn.ReLU(True))
        atten_0_5.add_module('pooling_a_5',nn.MaxPool2d((2, 2)))

        atten_0 = nn.Sequential()
        atten_0.add_module('conv_a_6',DeformableConv2d(nm[2], nm[2], 3, 1, 1))
        atten_0.add_module('bn_a_6', nn.BatchNorm2d(nm[2]))
        atten_0.add_module('relu_a_6', nn.ReLU(True))

        atten_1 = nn.Sequential()
        atten_1.add_module('conv_a_7',DeformableConv2d(nm[2], nm[2], 3, 1, 1))
        atten_1.add_module('bn_a_7', nn.BatchNorm2d(nm[2]))
        atten_1.add_module('relu_a_7', nn.ReLU(True))

        self.atten_0_0 = atten_0_0
        self.atten_0_1 = atten_0_1
        self.atten_0_2 = atten_0_2
        self.atten_0_3 = atten_0_3
        self.atten_0_4 = atten_0_4
        self.atten_0_5 = atten_0_5
        self.atten_0 = atten_0
        self.atten_1 = atten_1

        Fc_dimension = int(96*32/downsample/downsample/16)
        self.atten_fc1 = nn.Linear(Fc_dimension, Fc_dimension)
        self.atten_fc2 = nn.Linear(Fc_dimension, Fc_dimension)

        self.cnn_1_1 = DeformableConv2d(nm[1],64,1,1,0)

        self.relu    = nn.ReLU(inplace=True)
        self.sigmoid    = nn.Sigmoid()

        self.deconv1 = nn.ConvTranspose2d(nm[2], 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, self.K, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(self.K)

    def forward(self, input):
        # conv features
        batch_size = input.size(0)
        conv_out = input

        fc_1 = conv_out.reshape(batch_size, input.size(1), -1)
        conv_k = self.fc_k(fc_1)
        conv_v = self.fc_v(fc_1)
        conv_k = conv_k.reshape(batch_size, input.size(1), input.size(2), input.size(3))
        conv_v = conv_v.reshape(batch_size, input.size(1), input.size(2), input.size(3))

        x00 = self.atten_0_0(conv_k)
        x01 = self.atten_0_1(x00)
        x02 = self.atten_0_2(x01)
        x03 = self.atten_0_3(x02)
        x04 = self.atten_0_4(x03)
        x05 = self.atten_0_5(x04)
        x0 = self.atten_0(x05)
        x1 = self.atten_1(x0)

        channel = x1.size(1)
        height = x1.size(2)
        width = x1.size(3)
        fc_x = x1.view(batch_size, channel, -1)

        fc_atten = self.atten_fc2(self.atten_fc1(fc_x))
        fc_atten = fc_atten.reshape(batch_size, channel, height, width)

        score = self.relu(self.deconv1(fc_atten))
        score = self.bn1(score+self.cnn_1_1(x02))
        atten = self.sigmoid(self.deconv2(score))

        atten_list = torch.chunk(atten, atten.shape[0], 0)
        atten = atten.reshape(batch_size, self.K, -1)
        conv_v = conv_v.reshape(conv_v.size(0), conv_v.size(1), -1)

        conv_v = conv_v.permute(0,2,1)

        atten_out = torch.bmm(atten, conv_v)
        atten_out = atten_out.view(batch_size, self.K, -1)
        
        return atten_list, atten_out

class LPR_model(nn.Module):

    '''

    add GPM between the encoder and decoder.

    head and inner are parameters of GPM, the head number and dim of FFN inner features

    '''

    def __init__(self, args):
        super(LPR_model, self).__init__()
        self.isSeqModel = args.isSeqModel
        self.model_dim = 64
        self.encoder = CNNEncoder_baseline_light3(args.nc)
        self.attention = Deformable_Attention(nc=64, K=args.K, downsample=4)
        self.decoder = FCDecoder(args.nclass, input_dim=64)

        # insert GPM between the encoder and decoder
        if self.isSeqModel:
            key_dim = int(self.model_dim/args.head)
            self.SequenceModeling = Transforme_Encoder(d_word_vec=self.model_dim, n_layers=2, n_head=args.head, d_k=key_dim, d_v=key_dim,
                d_model=self.model_dim, d_inner=args.inner, dropout=0.1, n_position=192)
        self.isl2Norm = args.isl2Norm


    def forward(self, input, isTsne=False, isVis=False, isSAVis=False, isCNN=False):
        conv_out,_ = self.encoder(input)
        if self.isSeqModel:
            b, c, h, w = conv_out.shape
            conv_out = conv_out.permute(0, 1, 3, 2)
            conv_out = conv_out.contiguous().view(b, c, -1)
            conv_out = conv_out.permute(0, 2, 1)
            enc_slf_attn, seq_out = self.SequenceModeling(conv_out, src_mask=None)
            seq_out = seq_out.permute(0, 2, 1).reshape(b, c, w, h).permute(0, 1, 3, 2)
            atten_list, atten_out = self.attention(seq_out)
        else:
            atten_list, atten_out = self.attention(conv_out)
        #l2 norm to attentioned features
        if self.isl2Norm:
            atten_out = F.normalize(atten_out, 2, dim=2)
        text_preds = self.decoder(atten_out)

        return atten_list, text_preds, enc_slf_attn

@register('GPLPR')
def make_GPLPR(alphabet="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ", nc=1, imgW=96, imgH=32,  K=7, isSeqModel=True, head=2, inner=256, isl2Norm=True):
    args = Namespace()
    args.nclass = len(alphabet) + 1
    args.nc = nc
    args.imgW = imgW
    args.imgH = imgH
    args.K = K
    args.isSeqModel = isSeqModel
    args.head = head
    args.inner = inner
    args.isl2Norm = isl2Norm
    
    return LPR_model(args)