import torch
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
