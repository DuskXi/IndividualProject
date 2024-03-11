import unittest

from loguru import logger
from rich.logging import RichHandler
from torch.utils.data import DataLoader
from tqdm.rich import tqdm

from clevr.nlp_part.dataset import ClevrQuestionDataset, program_types
from model import *

logger.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_fn(batch):
    src, tgt, src_mask, tgt_mask = [], [], [], []
    for i in range(len(batch)):
        a, b, c, d = batch[i]
        src.append(a)
        tgt.append(b)
        src_mask.append(c)
        tgt_mask.append(d)
    src = torch.stack(src)
    tgt = torch.stack(tgt)
    src_mask = torch.stack(src_mask)
    tgt_mask = torch.stack(tgt_mask)
    return src.to(device), tgt.to(device), src_mask.to(device), tgt_mask.to(device)


class NLPTest(unittest.TestCase):
    def test_model_structure(self):
        self.assertEqual(True, True)  # add assertion here

    def test_dataset(self):
        dataset = ClevrQuestionDataset('../test_data/questions_t.json')
        question_tensor, program_tensor, question_mask, program_mask = dataset[0]

        self.assertEqual(len(question_tensor), dataset.ntokens)

    def test_model(self):
        dataset = ClevrQuestionDataset('../test_data/questions_t.json')
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=load_fn)
        vocab_size = len(dataset.vocab)
        ntokens = dataset.ntokens
        ninp = 200  # 嵌入层的维度
        nhead = 2
        nhid = 200
        nlayers = 2
        dropout = 0.2

        model = CustomTransformerModel(len(program_types), ntokens, vocab_size, ninp, nhead, nhid, nlayers, dropout)
        model.init_weights()
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())

        for epoch in range(10):
            for batch in (progress := tqdm(dataloader)):
                src, tgt, src_mask, tgt_mask = batch
                category_tgt, index1_tgt, index2_tgt = tgt[:, :, 0], tgt[:, :, 1], tgt[:, :, 2]

                category_output, index_output1, index_output2 = model(src, category_tgt, src_mask.transpose(0, 1), tgt_mask.transpose(0, 1))
                category_output = torch.argmax(category_output, dim=-1).float()
                loss = criterion(category_output, category_tgt.float())  # 分类损失
                # loss += criterion(index_output1, index1_tgt) + criterion(index_output2, index2_tgt)  # 索引损失
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                logger.info(f'epoch {epoch} loss: {loss.item()}')
                progress.set_description(f'epoch {epoch} loss: {loss.item()}')

        self.assertEqual(True, True)

    def test_ae(self):
        dataset = ClevrQuestionDataset('../test_data/questions_t.json')
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=load_fn)
        vocab_size = len(dataset.vocab)
        max_seq_len = dataset.ntokens

        model = LinearAE(max_seq_len, max_seq_len)
        model.to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())

        for epoch in range(20):
            for batch in (progress := tqdm(dataloader)):
                src, tgt, src_mask, tgt_mask = batch
                category_tgt, index1_tgt, index2_tgt = tgt[:, :, 0], tgt[:, :, 1], tgt[:, :, 2]

                pred = model(src)
                loss = criterion(pred, category_tgt)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                logger.info(f'epoch {epoch} loss: {loss.item()}')
                progress.set_description(f'epoch {epoch} loss: {loss.item()}')

        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
