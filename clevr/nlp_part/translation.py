from loguru import logger
from rich.logging import RichHandler
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from typing import Iterable, List

from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm.rich import tqdm
from timeit import default_timer as timer
from dataset import ClevrQuestionTranslationDataset

logger.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])
# save log to logdir
logger.add("logdir/log_{time}.log")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PAD_IDX = None
BOS_IDX = None
EOS_IDX = None
text_transform = None
SRC_LANGUAGE = None
TGT_LANGUAGE = None
loss_fn = None
vocab_transform = None
BATCH_SIZE = 256


# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
            self.tgt_tok_emb(tgt)), memory,
            tgt_mask)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))


# function to collate data samples into batch tensors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch


def cfn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)

    return torch.stack(src_batch).permute(1, 0), torch.stack(tgt_batch).permute(1, 0)


def train_epoch(model, optimizer, train_dataloader):
    model.train()
    losses = 0
    accuracies = 0
    # train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    # train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in train_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        accuracies += (torch.argmax(logits, dim=2) == tgt_out).sum().item() / (BATCH_SIZE * tgt_out.shape[0])
        losses += loss.item()

    return losses / len(list(train_dataloader)), accuracies / len(list(train_dataloader))


def evaluate(model):
    model.eval()
    losses = 0

    val_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(list(val_dataloader))


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len - 1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model, src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")


# def prepare_data():
#     global PAD_IDX, BOS_IDX, EOS_IDX, text_transform, SRC_LANGUAGE, TGT_LANGUAGE, loss_fn, vocab_transform
#     # We need to modify the URLs for the dataset since the links to the original dataset are broken
#     # Refer to https://github.com/pytorch/text/issues/1756#issuecomment-1163664163 for more info
#     multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
#     multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"
#
#     SRC_LANGUAGE = 'de'
#     TGT_LANGUAGE = 'en'
#
#     # Place-holders
#     token_transform = {}
#     vocab_transform = {}
#
#     token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
#     token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')
#
#     # helper function to yield list of tokens
#     def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
#         language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}
#
#         for data_sample in data_iter:
#             yield token_transform[language](data_sample[language_index[language]])
#
#     # Define special symbols and indices
#     UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
#     # Make sure the tokens are in order of their indices to properly insert them in vocab
#     special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
#
#     for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
#         # Training data Iterator
#         train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
#         # Create torchtext's Vocab object
#         vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
#                                                         min_freq=1,
#                                                         specials=special_symbols,
#                                                         special_first=True)
#
#     # Set ``UNK_IDX`` as the default index. This index is returned when the token is not found.
#     # If not set, it throws ``RuntimeError`` when the queried token is not found in the Vocabulary.
#     for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
#         vocab_transform[ln].set_default_index(UNK_IDX)
#
#     return vocab_transform, SRC_LANGUAGE, TGT_LANGUAGE, PAD_IDX, token_transform

def prepare_data(question_path, max_seq_len=32):
    global PAD_IDX, BOS_IDX, EOS_IDX, text_transform, SRC_LANGUAGE, TGT_LANGUAGE, vocab_transform

    SRC_LANGUAGE = 'question'
    TGT_LANGUAGE = 'program'

    # 初始化自定义数据集
    clevr_dataset = ClevrQuestionTranslationDataset(question_path, max_seq_len=max_seq_len)

    # 获取特殊符号的索引
    special_symbols_indices = clevr_dataset.get_special_symbol_indices()
    UNK_IDX = special_symbols_indices['<unk>']
    PAD_IDX = special_symbols_indices['<pad>']
    BOS_IDX = special_symbols_indices['<bos>']
    EOS_IDX = special_symbols_indices['<eos>']

    # 初始化词汇转换和令牌转换的字典
    vocab_transform = {SRC_LANGUAGE: clevr_dataset.vocab, TGT_LANGUAGE: clevr_dataset.vocab}
    text_transform = {SRC_LANGUAGE: clevr_dataset.tokenizer, TGT_LANGUAGE: clevr_dataset.tokenizer}

    return vocab_transform, SRC_LANGUAGE, TGT_LANGUAGE, PAD_IDX, text_transform


def main():
    dataset = ClevrQuestionTranslationDataset('../test_data/questions_t.json')
    symbols = dataset.get_special_symbol_indices_list()
    x, y = dataset[0]
    global PAD_IDX, BOS_IDX, EOS_IDX, text_transform, SRC_LANGUAGE, TGT_LANGUAGE, loss_fn, vocab_transform
    vocab_transform, SRC_LANGUAGE, TGT_LANGUAGE, PAD_IDX, token_transform = prepare_data('../test_data/questions_t.json')

    torch.manual_seed(0)

    SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
    TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
    EMB_SIZE = 512
    NHEAD = 8
    FFN_HID_DIM = 1024
    BATCH_SIZE = 512
    NUM_ENCODER_LAYERS = 6
    NUM_DECODER_LAYERS = 6

    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                     NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM, dropout=0.25)
    clevr_dataset = ClevrQuestionTranslationDataset('../test_data/questions_t.json', max_seq_len=64)
    train_dataloader = DataLoader(clevr_dataset, batch_size=BATCH_SIZE, collate_fn=cfn)
    # print model info and dataset info
    logger.info(
        f"src vocab size: {SRC_VOCAB_SIZE}, tgt vocab size: {TGT_VOCAB_SIZE}, embedding size: {EMB_SIZE}, "f"num encoder layers: {NUM_ENCODER_LAYERS}, num decoder layers: {NUM_DECODER_LAYERS}, "f"ffn hid dim: {FFN_HID_DIM}, batch size: {BATCH_SIZE}")
    logger.info(f"train dataset size: {len(clevr_dataset)}")

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = transformer.to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    text_transform = {}
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        text_transform[ln] = sequential_transforms(token_transform[ln],  # Tokenization
                                                   vocab_transform[ln],  # Numericalization
                                                   tensor_transform)  # Add BOS/EOS and create tensor

    NUM_EPOCHS = 50

    for epoch in (progress := tqdm(range(1, NUM_EPOCHS + 1), desc="epoch")):
        start_time = timer()
        train_loss, train_acc = train_epoch(transformer, optimizer, train_dataloader)
        end_time = timer()
        # val_loss = evaluate(transformer)
        # logger.info((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
        progress.set_description(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Train Accuracy: {train_acc * 100:.3f}%, "f"Epoch time = {(end_time - start_time):.3f}s")
        if epoch % 5 == 0:
            logger.info(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Train Accuracy: {train_acc * 100:.3f}%, "f"Epoch time = {(end_time - start_time):.3f}s")
            torch.save(transformer.state_dict(), f"logdir/transformer_epoch_{epoch}.pth")
        logger.info("Translate to: " + translate(transformer, "Are there the same number of large blue objects and large green rubber things?"))

    logger.info("Final to: " + translate(transformer, "Are there the same number of large blue objects and large green rubber things?"))


if __name__ == "__main__":
    main()
