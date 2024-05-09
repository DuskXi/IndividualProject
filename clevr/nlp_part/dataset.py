import torch
from loguru import logger
from torch import nn
from torch.utils.data import Dataset
import json

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

program_types = [
    'scene',
    'filter_color',
    'filter_shape',
    'filter_material',
    'filter_size',
    'filter_objectcategory',
    'unique',
    'relate',
    'union',
    'intersect',
    'count',
    'query_color',
    'query_shape',
    'query_material',
    'query_size',
    'exist',
    'equal_color',
    'equal_shape',
    'equal_integer',
    'equal_material',
    'equal_size',
    'equal_object',
    'less_than',
    'greater_than',
    'same_color',
    'same_shape',
    'same_size',
    'same_material',
]


class ClevrQuestionDataset(Dataset):
    def __init__(self, question_path, ntokens=25):
        self.question_path = question_path
        self.ntokens = ntokens
        with open(question_path, 'r') as f:
            d = json.load(f)
        self.questions = d['questions']
        question_text = [q['question'] for q in self.questions]
        self.tokenizer = get_tokenizer('basic_english')
        self.vocab = build_vocab_from_iterator(map(self.tokenizer, question_text), specials=["<unk>"])
        self.vocab.set_default_index(self.vocab["<unk>"])

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        question_text = question['question']
        program = question['program']
        question_tensor = torch.tensor([self.vocab[token] for token in self.tokenizer(question_text)])
        if len(question_tensor) < self.ntokens:
            question_tensor = torch.cat([question_tensor, torch.zeros(self.ntokens - len(question_tensor))])
            # question_mask = torch.cat([torch.ones(len(question_tensor)), torch.zeros(self.ntokens - len(question_tensor))])
        else:
            question_tensor = question_tensor[:self.ntokens]
            # question_mask = torch.ones(self.ntokens)
        program_tensor = torch.zeros(self.ntokens, len(program_types))
        program_mask = torch.zeros(self.ntokens)
        for i, p in enumerate(program):
            program_tensor[i, program_types.index(p['type'])] = 1
            program_mask[i] = 1
        question_tensor = question_tensor.float() / len(self.vocab)
        program_tensor = torch.argmax(program_tensor, dim=-1).float()
        # question_mask = question_mask.bool()
        program_mask = program_mask.bool()
        return question_tensor, program_tensor, program_mask


class ClevrQuestionS2SDataset(Dataset):
    def __init__(self, question_path, max_seq_len=25, num_limit=15000):
        self.question_path = question_path
        with open(question_path, 'r') as f:
            d = json.load(f)
        self.questions = d['questions'][:max(num_limit, len(d['questions']))]
        self.question_text, self.program_text = self.post_process()
        self.tokenizer = get_tokenizer('basic_english')
        self.vocab = build_vocab_from_iterator(map(self.tokenizer, self.question_text + self.program_text), specials=["<unk>"])
        self.vocab.set_default_index(self.vocab["<unk>"])
        self.inverse_vocab = {index: token for token, index in self.vocab.get_stoi().items()}
        self.max_seq_len = len(self.vocab)

    def post_process(self):
        question_texts = [q['question'] for q in self.questions]
        programs = [q['program'] for q in self.questions]
        program_texts = []
        for p in programs:
            program_text = []
            for item in p:
                p_type = item['type']
                p_value = item.get('value_inputs', [])
                if len(p_value) > 0:
                    program_text.append(f'{p_type}[{",".join(p_value)}]')
                else:
                    program_text.append(p_type)
            program_texts.append(' '.join(program_text))
        return question_texts, program_texts

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question_text = self.question_text[idx]
        program_text = self.program_text[idx]

        question_tensor = torch.tensor([self.vocab[token] for token in self.tokenizer(question_text)])
        program_tensor = torch.tensor([self.vocab[token] for token in self.tokenizer(program_text)])

        question_padding_mask = torch.ones(self.max_seq_len, dtype=torch.bool)
        program_padding_mask = torch.ones(self.max_seq_len, dtype=torch.bool)

        # 处理问题（question）张量和掩码
        if len(question_tensor) < self.max_seq_len:
            padding_len = self.max_seq_len - len(question_tensor)
            question_padding_mask[len(question_tensor):] = False
            question_tensor = torch.cat([question_tensor, torch.zeros(padding_len, dtype=torch.long)])
        else:
            question_tensor = question_tensor[:self.max_seq_len]

        # 处理程序（program）张量和掩码
        if len(program_tensor) < self.max_seq_len:
            padding_len = self.max_seq_len - len(program_tensor)
            program_padding_mask[len(program_tensor):] = False
            program_tensor = torch.cat([program_tensor, torch.zeros(padding_len, dtype=torch.long)])
        else:
            program_tensor = program_tensor[:self.max_seq_len]

        # return question_tensor, question_padding_mask, program_tensor, program_padding_mask
        return question_tensor, question_padding_mask, program_tensor, program_padding_mask


class ClevrQuestionTranslationDataset(Dataset):
    def __init__(self, question_path, max_seq_len=32):
        self.special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
        self.question_path = question_path
        with open(question_path, 'r') as f:
            d = json.load(f)
        self.questions = d['questions']
        self.question_text, self.program_text = self.post_process()
        self.tokenizer = get_tokenizer('basic_english')
        self.vocab = build_vocab_from_iterator(map(self.tokenizer, self.question_text + self.program_text), specials=self.special_symbols)
        self.vocab.set_default_index(self.vocab["<unk>"])
        self.inverse_vocab = {index: token for token, index in self.vocab.get_stoi().items()}
        self.max_seq_len = max_seq_len

    def post_process(self):
        question_texts = [q['question'] for q in self.questions]
        programs = [q['program'] for q in self.questions]
        program_texts = []
        for p in programs:
            program_text = []
            for item in p:
                p_type = item['type']
                p_value = item.get('value_inputs', [])
                if len(p_value) > 0:
                    program_text.append(f'{p_type}[{",".join(p_value)}]')
                else:
                    program_text.append(p_type)
            program_texts.append(' '.join(program_text))
        return question_texts, program_texts

    def get_special_symbol_indices(self):
        """返回特殊符号的索引字典"""
        return {symbol: self.vocab[symbol] for symbol in self.special_symbols}

    def get_special_symbol_indices_list(self):
        return [self.vocab[symbol] for symbol in self.special_symbols]

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question_text = ['<bos>'] + self.tokenizer(self.question_text[idx]) + ['<eos>']
        program_text = ['<bos>'] + self.tokenizer(self.program_text[idx]) + ['<eos>']

        question_tensor = torch.tensor([self.vocab[token] for token in question_text])
        program_tensor = torch.tensor([self.vocab[token] for token in program_text])

        # Adjust padding for max sequence length
        if len(question_tensor) < self.max_seq_len:
            padding_len = self.max_seq_len - len(question_tensor)
            question_tensor = torch.cat([question_tensor, torch.full((padding_len,), self.vocab['<pad>'], dtype=torch.long)])
        else:
            logger.warning(f"Question tensor length exceeds max sequence length: {len(question_tensor)}")
            question_tensor = question_tensor[:self.max_seq_len]

        if len(program_tensor) < self.max_seq_len:
            padding_len = self.max_seq_len - len(program_tensor)
            program_tensor = torch.cat([program_tensor, torch.full((padding_len,), self.vocab['<pad>'], dtype=torch.long)])
        else:
            logger.warning(f"Program tensor length exceeds max sequence length: {len(program_tensor)}")
            program_tensor = program_tensor[:self.max_seq_len]

        return question_tensor, program_tensor
