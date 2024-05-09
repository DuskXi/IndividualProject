import traceback

import numpy as np
import torch
from loguru import logger
from rich.logging import RichHandler
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import ClevrQuestionDataset, program_types, ClevrQuestionS2SDataset
from model import *
from tqdm.rich import tqdm
import transfromer

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

def main_4():
    tensorboard_path = 'runs/transformer'
    tensorboard_writer = SummaryWriter(tensorboard_path)
    dataset = ClevrQuestionS2SDataset('../test_data/questions_t.json')
    batch_size = 1
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    ntokens = len(dataset.vocab)
    emsize = 200  # embedding dimension
    d_hid = 200  # dimension of the feedforward network model in ``nn.TransformerEncoder``
    nlayers = 2  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
    nhead = 2  # number of heads in ``nn.MultiheadAttention``
    dropout = 0.2  # dropout probability
    model = transfromer.TransformerModel(ntokens)
    model.to(device)
    model.init_weights()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=35, gamma=0.5)
    losses = []
    accuracies = []
    window_size = 30
    for epoch in range(200):
        for i, batch in enumerate(progress := tqdm(dataloader)):
            if type(batch) not in (list, tuple):
                continue
            try:
                src, question_padding_mask, tgt, program_padding_mask = batch
                src = src.to(device)
                tgt = tgt.to(device)
                question_padding_mask = question_padding_mask.to(device)
                program_padding_mask = program_padding_mask.to(device)
                # make seq first
                src = src.permute(1, 0)
                tgt = tgt.permute(1, 0)
                pred = model(src, tgt, src_mask=question_padding_mask, tgt_mask=program_padding_mask)

                # output_logits = pred.view(-1,  len(dataset.vocab))
                # loss = criterion(output_logits, torch.flatten(tgt))
                pred = pred.permute(1, 0, 2)
                tgt = tgt.permute(1, 0)
                tgt = F.one_hot(tgt, len(dataset.vocab)).float()
                loss = criterion(pred, tgt)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                compare = (pred == tgt)[program_padding_mask]
                accuracy = compare.sum().item() / compare.size(0) * 100 if compare.size(0) > 0 else 0
                accuracies.append(accuracy)
                progress.set_description(f'epoch {epoch} loss: {np.mean(losses):.4f} accuracy: {np.mean(accuracies):.4f}%')
                losses.append(loss.item())
                tensorboard_writer.add_scalar('loss', np.mean(losses), epoch * len(dataloader) * batch_size + i * batch_size)
                tensorboard_writer.add_scalar('accuracy', np.mean(accuracies), epoch * len(dataloader) * batch_size + i * batch_size)

                pred = torch.argmax(pred, dim=2)
                text = []
                for i in range(pred.size(0)):
                    text.append(' '.join([dataset.inverse_vocab[index.item()] for index in pred[i]]))
                # print(text)

                if len(losses) > window_size:
                    losses.pop(0)
                    accuracies.pop(0)
            except Exception as e:
                traceback.print_exc()
                logger.error(e)
        tensorboard_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        lr_scheduler.step()

def main_3():
    tensorboard_path = 'runs/transformer'
    tensorboard_writer = SummaryWriter(tensorboard_path)
    dataset = ClevrQuestionDataset('../test_data/questions_t.json')
    batch_size = 256
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    vocab_size = len(dataset.vocab)
    max_seq_len = dataset.ntokens

    # model = LinearAE(max_seq_len, len(program_types), max_seq_len)
    encoder = Encoder(vocab_size, max_seq_len, 256, 256, 8)
    decoder = Decoder(vocab_size, max_seq_len, 256, 256, 8)
    model = Seq2seq(encoder, decoder)
    model.to(device)
    criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=35, gamma=0.5)
    losses = []
    accuracies = []
    window_size = 30
    for epoch in range(200):
        for i, batch in enumerate(progress := tqdm(dataloader)):
            if type(batch) not in (list, tuple):
                continue
            try:
                src, tgt, label_mask = batch
                src = src.to(device)
                tgt = tgt.to(device)
                label_mask = label_mask.to(device)
                # category_tgt, index1_tgt, index2_tgt = tgt[:, :, 0], tgt[:, :, 1], tgt[:, :, 2]

                pred = model(src)
                # pred = torch.argmax(pred, dim=-1)
                loss = criterion(pred, tgt)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # logger.info(f'epoch {epoch} loss: {loss.item()}')
                compare = (pred.to(torch.int64) == tgt.to(torch.int64))[label_mask]
                accuracy = compare.sum().item() / compare.size(0) * 100
                accuracies.append(accuracy)
                progress.set_description(f'epoch {epoch} loss: {np.mean(losses):.4f} accuracy: {np.mean(accuracies):.4f}%')
                losses.append(loss.item())
                tensorboard_writer.add_scalar('loss', np.mean(losses), epoch * len(dataloader) * batch_size + i * batch_size)
                tensorboard_writer.add_scalar('accuracy', np.mean(accuracies), epoch * len(dataloader) * batch_size + i * batch_size)

                if len(losses) > window_size:
                    losses.pop(0)
                    accuracies.pop(0)
            except Exception as e:
                traceback.print_exc()
                logger.error(e)
        tensorboard_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        lr_scheduler.step()


def main_2():
    tensorboard_path = 'runs/transformer'
    tensorboard_writer = SummaryWriter(tensorboard_path)
    dataset = ClevrQuestionDataset('../test_data/questions_t.json')
    batch_size = 256
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    vocab_size = len(dataset.vocab)
    max_seq_len = dataset.ntokens

    # model = LinearAE(max_seq_len, len(program_types), max_seq_len)
    model = LSTMModel(max_seq_len, 512, max_seq_len, len(program_types), vocab_size, 12)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=35, gamma=0.5)
    losses = []
    accuracies = []
    window_size = 30
    for epoch in range(200):
        for i, batch in enumerate(progress := tqdm(dataloader)):
            if type(batch) not in (list, tuple):
                continue
            try:
                src, tgt, label_mask = batch
                src = src.to(device)
                tgt = tgt.to(device, dtype=torch.int64)
                # current shape: (batch, max_seq_len)
                # need one hot to (batch, max_seq_len, num_classes)
                tgt = F.one_hot(tgt, len(program_types)).float()
                label_mask = label_mask.to(device)
                # category_tgt, index1_tgt, index2_tgt = tgt[:, :, 0], tgt[:, :, 1], tgt[:, :, 2]

                pred = model(src)
                # pred = torch.argmax(pred, dim=-1)
                loss = criterion(pred, tgt)
                masked_loss = loss[label_mask]
                loss = masked_loss.mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # logger.info(f'epoch {epoch} loss: {loss.item()}')
                pred = torch.argmax(pred, dim=2)
                tgt = torch.argmax(tgt, dim=2)
                compare = (pred == tgt)[label_mask]
                accuracy = compare.sum().item() / compare.size(0) * 100
                accuracies.append(accuracy)
                progress.set_description(f'epoch {epoch} loss: {np.mean(losses):.4f} accuracy: {np.mean(accuracies):.4f}%')
                losses.append(loss.item())
                tensorboard_writer.add_scalar('loss', np.mean(losses), epoch * len(dataloader) * batch_size + i * batch_size)
                tensorboard_writer.add_scalar('accuracy', np.mean(accuracies), epoch * len(dataloader) * batch_size + i * batch_size)

                if len(losses) > window_size:
                    losses.pop(0)
                    accuracies.pop(0)
            except Exception as e:
                traceback.print_exc()
                # logger.error(e)
        tensorboard_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        lr_scheduler.step()


def main_1():
    tensorboard_path = 'runs/transformer'
    tensorboard_writer = SummaryWriter(tensorboard_path)
    dataset = ClevrQuestionDataset('../test_data/questions_t.json')
    batch_size = 256
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    vocab_size = len(dataset.vocab)
    max_seq_len = dataset.ntokens

    # model = LinearAE(max_seq_len, len(program_types), max_seq_len)
    model = LinearAE(max_seq_len, max_seq_len, max_seq_len, 12, 512, 16)
    model.to(device)
    criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=35, gamma=0.5)
    losses = []
    accuracies = []
    window_size = 30
    for epoch in range(200):
        for i, batch in enumerate(progress := tqdm(dataloader)):
            if type(batch) not in (list, tuple):
                continue
            try:
                src, tgt, label_mask = batch
                src = src.to(device)
                tgt = tgt.to(device)
                label_mask = label_mask.to(device)
                # category_tgt, index1_tgt, index2_tgt = tgt[:, :, 0], tgt[:, :, 1], tgt[:, :, 2]

                pred = model(src)
                # pred = torch.argmax(pred, dim=-1)
                loss = criterion(pred, tgt)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # logger.info(f'epoch {epoch} loss: {loss.item()}')
                compare = (pred.to(torch.int64) == tgt.to(torch.int64))[label_mask]
                accuracy = compare.sum().item() / compare.size(0) * 100
                accuracies.append(accuracy)
                progress.set_description(f'epoch {epoch} loss: {np.mean(losses):.4f} accuracy: {np.mean(accuracies):.4f}%')
                losses.append(loss.item())
                tensorboard_writer.add_scalar('loss', np.mean(losses), epoch * len(dataloader) * batch_size + i * batch_size)
                tensorboard_writer.add_scalar('accuracy', np.mean(accuracies), epoch * len(dataloader) * batch_size + i * batch_size)

                if len(losses) > window_size:
                    losses.pop(0)
                    accuracies.pop(0)
            except Exception as e:
                traceback.print_exc()
                logger.error(e)
        tensorboard_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        lr_scheduler.step()


if __name__ == '__main__':
    main_4()
