import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#
# # # 定义编码器
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

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        nn.Transformer
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(device)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


# class TransformerModel(nn.Module):
#     def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
#         super(TransformerModel, self).__init__()
#         from torch.nn import Transformer
#         self.model_type = 'Transformer'
#         self.pos_encoder = PositionalEncoding(ninp, dropout)
#         self.encoder = nn.Embedding(ntoken, ninp)
#         self.transformer = Transformer(d_model=ninp, nhead=nhead, num_encoder_layers=nlayers,
#                                        num_decoder_layers=nlayers, dim_feedforward=nhid, dropout=dropout)
#         # self.decoder = nn.Linear(ninp, ntoken)
#
#         self.init_weights()
#         self.ntoken = ntoken
#         self.ninp = ninp
#         self.nhead = nhead
#         self.nhid = nhid
#         self.nlayers = nlayers
#
#     def generate_square_subsequent_mask(self, sz):
#         mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
#         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#         return mask
#
#     def init_weights(self):
#         initrange = 0.1
#         self.encoder.weight.data.uniform_(-initrange, initrange)
#         # self.decoder.bias.data.zero_()
#         # self.decoder.weight.data.uniform_(-initrange, initrange)
#
#     def forward(self, src, tgt, src_mask, tgt_mask):
#         src = self.encoder(src) * math.sqrt(self.ninp)
#         src = self.pos_encoder(src)
#         tgt = self.encoder(tgt) * math.sqrt(self.ninp)
#         tgt = self.pos_encoder(tgt)
#         output = self.transformer(src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None)
#         # output = self.decoder(output)
#         return output
#
#
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#
#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, 1, d_model)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         x = x + self.pe[:x.size(0)]
#         return self.dropout(x)
#
#
# class CustomTransformerModel(TransformerModel):
#     def __init__(self, category_size=25, max_seq_len=25, *args, **kwargs):
#         super(CustomTransformerModel, self).__init__(*args, **kwargs)
#         # 假设category_size是类别数量
#         self.category_decoder = nn.Linear(self.ninp, category_size)
#         # self.index_decoders = nn.Linear(self.ninp, max_seq_len)  # 两个索引
#
#     def forward(self, src, tgt, src_mask, tgt_mask):
#         output = super().forward(src, tgt, src_mask, tgt_mask)
#         category_output = self.category_decoder(output)
#         # index_output1 = self.index_decoders(output)  # 第一个索引
#         # index_output2 = self.index_decoders(output)  # 第二个索引
#         # return category_output, index_output1, index_output2
#         return category_output, None, None


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


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, max_seq_len, num_class, vocab_size, num_layers=32):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 512)
        self.lstm = nn.LSTM(512, hidden_size, num_layers, batch_first=True)
        self.linear = nn.ModuleList(
            [nn.Sequential(nn.BatchNorm1d(hidden_size), nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden_size, num_class)) for _ in range(max_seq_len)])

    def forward(self, x):
        embedded = self.embedding(x.to(torch.int64))
        x, _ = self.lstm(embedded)
        # to seq first
        x = x.transpose(1, 0)
        result = []
        for i, linear in enumerate(self.linear):
            result.append(linear(x[0]))
        return torch.stack(result, dim=1)


class TranslationModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers):
        super(TranslationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers)
        self.decoder = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        output = self.transformer(src, tgt, src_mask, tgt_mask, None, src_padding_mask, tgt_padding_mask, None)
        output = self.decoder(output)
        return output


class BaseRNN(nn.Module):
    """Base RNN module"""

    def __init__(self, vocab_size, max_len, hidden_size, input_dropout_p,
                 dropout_p, n_layers, rnn_cell):
        super(BaseRNN, self).__init__()

        self.vocab_size = vocab_size
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_dropout_p = input_dropout_p
        self.dropout_p = dropout_p

        if rnn_cell == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError('Unsupported RNN Cell: %s' % rnn_cell)

        self.input_dropout = nn.Dropout(p=input_dropout_p)

    def forward(self, *args, **kwargs):
        raise NotImplementedError()


class Attention(nn.Module):
    """Attention layer"""

    def __init__(self, dim, use_weight=False, hidden_size=512):
        super(Attention, self).__init__()
        self.use_weight = use_weight
        self.hidden_size = hidden_size
        if use_weight:
            print('| using weighted attention layer')
            self.attn_weight = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_out = nn.Linear(2 * dim, dim)

    def forward(self, output, context):
        """
        - args
        output : Tensor
            decoder output, dim (batch_size, output_size, hidden_size)
        context : Tensor
            context vector from encoder, dim (batch_size, input_size, hidden_size)
        - returns
        output : Tensor
            attention layer output, dim (batch_size, output_size, hidden_size)
        attn : Tensor
            attention map, dim (batch_size, output_size, input_size)
        """
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)

        if self.use_weight:
            output = self.attn_weight(output.contiguous().view(-1, hidden_size)).view(batch_size, -1, hidden_size)

        attn = torch.bmm(output, context.transpose(1, 2))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)  # (batch_size, output_size, input_size)

        mix = torch.bmm(attn, context)  # (batch_size, output_size, hidden_size)
        comb = torch.cat((mix, output), dim=2)  # (batch_size, output_size, 2*hidden_size)
        output = F.tanh(self.linear_out(comb.view(-1, 2 * hidden_size)).view(batch_size, -1, hidden_size))  # (batch_size, output_size, hidden_size)

        return output, attn


class Decoder(BaseRNN):
    """Decoder RNN module
    To do: add docstring to methods
    """

    def __init__(self, vocab_size, max_len, word_vec_dim, hidden_size,
                 n_layers, start_id=1, end_id=2, rnn_cell='lstm',
                 bidirectional=False, input_dropout_p=0,
                 dropout_p=0, use_attention=False):
        super(Decoder, self).__init__(vocab_size, max_len, hidden_size,
                                      input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.max_length = max_len
        self.output_size = vocab_size
        self.hidden_size = hidden_size
        self.word_vec_dim = word_vec_dim
        self.bidirectional_encoder = bidirectional
        if bidirectional:
            self.hidden_size *= 2
        self.use_attention = use_attention
        self.start_id = start_id
        self.end_id = end_id

        self.embedding = nn.Embedding(self.output_size, self.word_vec_dim)
        self.rnn = self.rnn_cell(self.word_vec_dim, self.hidden_size, n_layers, batch_first=True, dropout=dropout_p)
        self.out_linear = nn.Linear(self.hidden_size, self.output_size)
        if use_attention:
            self.attention = Attention(self.hidden_size)

    def forward_step(self, input_var, hidden, encoder_outputs):
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        output, hidden = self.rnn(embedded, hidden)

        attn = None
        if self.use_attention:
            output, attn = self.attention(output, encoder_outputs)

        output = self.out_linear(output.contiguous().view(-1, self.hidden_size))
        predicted_softmax = F.log_softmax(output.view(batch_size, output_size, -1), 2)
        return predicted_softmax, hidden, attn

    def forward(self, y, encoder_outputs, encoder_hidden):
        decoder_hidden = self._init_state(encoder_hidden)
        decoder_outputs, decoder_hidden, attn = self.forward_step(y, decoder_hidden, encoder_outputs)
        return decoder_outputs, decoder_hidden

    def forward_sample(self, encoder_outputs, encoder_hidden, reinforce_sample=False):
        if isinstance(encoder_hidden, tuple):
            batch_size = encoder_hidden[0].size(1)
            use_cuda = encoder_hidden[0].is_cuda
        else:
            batch_size = encoder_hidden.size(1)
            use_cuda = encoder_hidden.is_cuda
        decoder_hidden = self._init_state(encoder_hidden)
        decoder_input = Variable(torch.LongTensor(batch_size, 1).fill_(self.start_id))
        if use_cuda:
            decoder_input = decoder_input.cuda()

        output_logprobs = []
        output_symbols = []
        output_lengths = np.array([self.max_length] * batch_size)

        def decode(i, output, reinforce_sample=reinforce_sample):
            output_logprobs.append(output.squeeze())
            if reinforce_sample:
                dist = torch.distributions.Categorical(probs=torch.exp(output.view(batch_size, -1)))  # better initialize with logits
                symbols = dist.sample().unsqueeze(1)
            else:
                symbols = output.topk(1)[1].view(batch_size, -1)
            output_symbols.append(symbols.squeeze())

            eos_batches = symbols.data.eq(self.end_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((output_lengths > i) & eos_batches) != 0
                output_lengths[update_idx] = len(output_symbols)

            return symbols

        for i in range(self.max_length):
            decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
            decoder_input = decode(i, decoder_output)

        return output_symbols, output_logprobs

    def _init_state(self, encoder_hidden):
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h


class Encoder(BaseRNN):
    """Encoder RNN module"""

    def __init__(self, vocab_size, max_len, word_vec_dim, hidden_size, n_layers,
                 input_dropout_p=0, dropout_p=0, bidirectional=False, rnn_cell='lstm',
                 variable_lengths=False, word2vec=None, fix_embedding=False):
        super(Encoder, self).__init__(vocab_size, max_len, hidden_size, input_dropout_p, dropout_p, n_layers, rnn_cell)
        self.variable_lengths = variable_lengths
        if word2vec is not None:
            assert word2vec.size(0) == vocab_size
            self.word_vec_dim = word2vec.size(1)
            self.embedding = nn.Embedding(vocab_size, self.word_vec_dim)
            self.embedding.weight = nn.Parameter(word2vec)
        else:
            self.word_vec_dim = word_vec_dim
            self.embedding = nn.Embedding(vocab_size, word_vec_dim)
        if fix_embedding:
            self.embedding.weight.requires_grad = False
        self.rnn = self.rnn_cell(self.word_vec_dim, hidden_size, n_layers,
                                 batch_first=True, bidirectional=bidirectional, dropout=dropout_p)

    def forward(self, input_var, input_lengths=None):
        """
        To do: add input, output dimensions to docstring
        """
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        output, hidden = self.rnn(embedded)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden


class Seq2seq(nn.Module):
    """Seq2seq model module
    To do: add docstring to methods
    """

    def __init__(self, encoder, decoder):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, y, input_lengths=None):
        encoder_outputs, encoder_hidden = self.encoder(x, input_lengths)
        decoder_outputs, decoder_hidden = self.decoder(y, encoder_outputs, encoder_hidden)
        return decoder_outputs

    def sample_output(self, x, input_lengths=None):
        encoder_outputs, encoder_hidden = self.encoder(x, input_lengths)
        output_symbols, _ = self.decoder.forward_sample(encoder_outputs, encoder_hidden)
        return torch.stack(output_symbols).transpose(0, 1)

    def reinforce_forward(self, x, input_lengths=None):
        encoder_outputs, encoder_hidden = self.encoder(x, input_lengths)
        self.output_symbols, self.output_logprobs = self.decoder.forward_sample(encoder_outputs, encoder_hidden, reinforce_sample=True)
        return torch.stack(self.output_symbols).transpose(0, 1)

    def reinforce_backward(self, reward, entropy_factor=0.0):
        assert self.output_logprobs is not None and self.output_symbols is not None, 'must call reinforce_forward first'
        losses = []
        grad_output = []
        for i, symbol in enumerate(self.output_symbols):
            if len(self.output_symbols[0].shape) == 1:
                loss = - torch.diag(torch.index_select(self.output_logprobs[i], 1, symbol)).sum() * reward \
                       + entropy_factor * (self.output_logprobs[i] * torch.exp(self.output_logprobs[i])).sum()
            else:
                loss = - self.output_logprobs[i] * reward
            losses.append(loss.sum())
            grad_output.append(None)
        torch.autograd.backward(losses, grad_output, retain_graph=True)


class Seq2seqParser():
    """Model interface for seq2seq parser"""

    def __init__(self, opt):
        self.opt = opt
        self.vocab = get_vocab(opt)
        if opt.load_checkpoint_path is not None:
            self.load_checkpoint(opt.load_checkpoint_path)
        else:
            print('| creating new network')
            self.net_params = self._get_net_params(self.opt, self.vocab)
            self.seq2seq = create_seq2seq_net(**self.net_params)
        self.variable_lengths = self.net_params['variable_lengths']
        self.end_id = self.net_params['end_id']
        self.gpu_ids = opt.gpu_ids
        self.criterion = nn.NLLLoss()
        if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
            self.seq2seq.cuda(opt.gpu_ids[0])

    def load_checkpoint(self, load_path):
        print('| loading checkpoint from %s' % load_path)
        checkpoint = torch.load(load_path)
        self.net_params = checkpoint['net_params']
        if 'fix_embedding' in vars(self.opt):  # To do: change condition input to run mode
            self.net_params['fix_embedding'] = self.opt.fix_embedding
        self.seq2seq = create_seq2seq_net(**self.net_params)
        self.seq2seq.load_state_dict(checkpoint['net_state'])

    def save_checkpoint(self, save_path):
        checkpoint = {
            'net_params': self.net_params,
            'net_state': self.seq2seq.cpu().state_dict()
        }
        torch.save(checkpoint, save_path)
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            self.seq2seq.cuda(self.gpu_ids[0])

    def set_input(self, x, y=None):
        input_lengths, idx_sorted = None, None
        if self.variable_lengths:
            x, y, input_lengths, idx_sorted = self._sort_batch(x, y)
        self.x = self._to_var(x)
        if y is not None:
            self.y = self._to_var(y)
        else:
            self.y = None
        self.input_lengths = input_lengths
        self.idx_sorted = idx_sorted

    def set_reward(self, reward):
        self.reward = reward

    def supervised_forward(self):
        assert self.y is not None, 'Must set y value'
        output_logprob = self.seq2seq(self.x, self.y, self.input_lengths)
        self.loss = self.criterion(output_logprob[:, :-1, :].contiguous().view(-1, output_logprob.size(2)), self.y[:, 1:].contiguous().view(-1))
        return self._to_numpy(self.loss).sum()

    def supervised_backward(self):
        assert self.loss is not None, 'Loss not defined, must call supervised_forward first'
        self.loss.backward()

    def reinforce_forward(self):
        self.rl_seq = self.seq2seq.reinforce_forward(self.x, self.input_lengths)
        self.rl_seq = self._restore_order(self.rl_seq.data.cpu())
        self.reward = None  # Need to recompute reward from environment each time a new sequence is sampled
        return self.rl_seq

    def reinforce_backward(self, entropy_factor=0.0):
        assert self.reward is not None, 'Must run forward sampling and set reward before REINFORCE'
        self.seq2seq.reinforce_backward(self.reward, entropy_factor)

    def parse(self):
        output_sequence = self.seq2seq.sample_output(self.x, self.input_lengths)
        output_sequence = self._restore_order(output_sequence.data.cpu())
        return output_sequence

    def _get_net_params(self, opt, vocab):
        net_params = {
            'input_vocab_size': len(vocab['question_token_to_idx']),
            'output_vocab_size': len(vocab['program_token_to_idx']),
            'hidden_size': opt.hidden_size,
            'word_vec_dim': opt.word_vec_dim,
            'n_layers': opt.n_layers,
            'bidirectional': opt.bidirectional,
            'variable_lengths': opt.variable_lengths,
            'use_attention': opt.use_attention,
            'encoder_max_len': opt.encoder_max_len,
            'decoder_max_len': opt.decoder_max_len,
            'start_id': opt.start_id,
            'end_id': opt.end_id,
            'word2vec_path': opt.word2vec_path,
            'fix_embedding': opt.fix_embedding,
        }
        return net_params

    def _sort_batch(self, x, y):
        _, lengths = torch.eq(x, self.end_id).max(1)
        lengths += 1
        lengths_sorted, idx_sorted = lengths.sort(0, descending=True)
        x_sorted = x[idx_sorted]
        y_sorted = None
        if y is not None:
            y_sorted = y[idx_sorted]
        lengths_list = lengths_sorted.numpy()
        return x_sorted, y_sorted, lengths_list, idx_sorted

    def _restore_order(self, x):
        if self.idx_sorted is not None:
            inv_idxs = self.idx_sorted.clone()
            inv_idxs.scatter_(0, self.idx_sorted, torch.arange(x.size(0)).long())
            return x[inv_idxs]
        return x

    def _to_var(self, x):
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    def _to_numpy(self, x):
        return x.data.cpu().numpy().astype(float)
