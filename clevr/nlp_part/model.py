import math

import torch
import torch.nn as nn
import torch.optim as optim

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 定义编码器
# class Encoder(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(Encoder, self).__init__()
#         self.hidden_size = hidden_size
#         self.lstm = nn.LSTM(input_size, hidden_size)
#
#     def forward(self, input, hidden):
#         output, hidden = self.lstm(input, hidden)
#         return output, hidden
#
#     def initHidden(self):
#         return (torch.zeros(1, 1, self.hidden_size, device=device),
#                 torch.zeros(1, 1, self.hidden_size, device=device))
#
#
# # 定义解码器
# class Decoder(nn.Module):
#     def __init__(self, hidden_size, output_size):
#         super(Decoder, self).__init__()
#         self.hidden_size = hidden_size
#         self.lstm = nn.LSTM(hidden_size, hidden_size)
#         self.out = nn.Linear(hidden_size, output_size)
#         self.softmax = nn.LogSoftmax(dim=1)
#
#     def forward(self, input, hidden):
#         output, hidden = self.lstm(input, hidden)
#         output = self.softmax(self.out(output[0]))
#         return output, hidden
#
#     def initHidden(self):
#         return (torch.zeros(1, 1, self.hidden_size, device=device),
#                 torch.zeros(1, 1, self.hidden_size, device=device))
#
#
# # # 设置超参数
# # input_size = 256  # 输入语言的词汇量
# # hidden_size = 512  # LSTM单元数
# # output_size = 256  # 输出语言的词汇量
# #
# # encoder = Encoder(input_size, hidden_size).to(device)
# # decoder = Decoder(hidden_size, output_size).to(device)
# #
# # # 这里仅展示了模型架构的实现。为了训练这个模型，
# # # 您还需要定义适当的数据处理、损失函数和训练循环。
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import Transformer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.transformer = Transformer(d_model=ninp, nhead=nhead, num_encoder_layers=nlayers,
                                       num_decoder_layers=nlayers, dim_feedforward=nhid, dropout=dropout)
        # self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()
        self.ntoken = ntoken
        self.ninp = ninp
        self.nhead = nhead
        self.nhid = nhid
        self.nlayers = nlayers

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt, src_mask, tgt_mask):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        tgt = self.encoder(tgt) * math.sqrt(self.ninp)
        tgt = self.pos_encoder(tgt)
        output = self.transformer(src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None)
        # output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class CustomTransformerModel(TransformerModel):
    def __init__(self, category_size=25, max_seq_len=25, *args, **kwargs):
        super(CustomTransformerModel, self).__init__(*args, **kwargs)
        # 假设category_size是类别数量
        self.category_decoder = nn.Linear(self.ninp, category_size)
        # self.index_decoders = nn.Linear(self.ninp, max_seq_len)  # 两个索引

    def forward(self, src, tgt, src_mask, tgt_mask):
        output = super().forward(src, tgt, src_mask, tgt_mask)
        category_output = self.category_decoder(output)
        # index_output1 = self.index_decoders(output)  # 第一个索引
        # index_output2 = self.index_decoders(output)  # 第二个索引
        # return category_output, index_output1, index_output2
        return category_output, None, None


class LinearAE(nn.Module):
    def __init__(self, input_size, output_size, max_len, nlayers=6, hidden_size=128, num_heads=8):
        super(LinearAE, self).__init__()
        self.linear_in = nn.Linear(input_size, hidden_size)
        encoder = []
        for n in range(nlayers):
            encoder.append(nn.Linear(hidden_size, hidden_size))
            encoder.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder)
        self.multihead_attention = nn.MultiheadAttention(hidden_size, num_heads)
        decoder = []
        for n in range(nlayers - 1):
            decoder.append(nn.Linear(hidden_size, hidden_size))
            decoder.append(nn.ReLU())
        decoder.append(nn.Linear(hidden_size, output_size))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        x = self.linear_in(x)
        x = self.encoder(x)
        x = x.unsqueeze(0)
        x, _ = self.multihead_attention(x, x, x)
        x = x.squeeze(0)
        return self.decoder(x)
        # result = []
        # for decoder in self.decoders:
        #     result.append(decoder(x))
        # return torch.stack(result, dim=1)
