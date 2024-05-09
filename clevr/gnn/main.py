import torch
from loguru import logger
from tqdm.rich import tqdm

from dataset import GNNDataset
from torch_geometric.data import DataLoader
from model import GNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    dataset = GNNDataset(r"W:\projects\IndividualProject\clevr\clevr-dataset-gen\question_generation\questions_t.json",
                         r"W:\projects\IndividualProject\clevr\Dataset Generation\output\clevr_scenes.json")
    dataset.questions = list(filter(lambda x: type(x['answer']) == bool, dataset.questions))
    # dataset.questions = dataset.questions[:100]
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = GNN(num_classes=2)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    losses = []
    accuracies = []
    window = 25
    result_cache = []
    for epoch in range(10):
        for i, data in enumerate(progress := tqdm(dataloader, desc=f"Epoch {epoch}")):
            optimizer.zero_grad()
            result = model(data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device))
            loss = criterion(result, data[5].to(device))
            loss.backward()
            optimizer.step()
            result_cache.append(torch.argmax(result, dim=1).cpu())
            accuracy = torch.sum(torch.argmax(result, dim=1) == data[5].to(device)).item() / len(data[5])
            losses.append(loss.item())
            accuracies.append(accuracy)
            if len(losses) > window:
                losses.pop(0)
                accuracies.pop(0)
                result_cache.pop(0)
            avg_loss = sum(losses) / len(losses)
            avg_acc = sum(accuracies) / len(accuracies)
            progress.set_description(f"Epoch {epoch} Loss {loss.item():.4f} Avg Loss {avg_loss:.4f} Accuracy {accuracy * 100:.4f}% Avg Acc {avg_acc * 100:.4f}%")
            if i % 100 == 0:
                logger.info(f"Epoch {epoch} Loss {loss.item():.4f} Avg Loss {avg_loss:.4f} Accuracy {accuracy * 100:.4f}% Avg Acc {avg_acc * 100:.4f}%")


if __name__ == '__main__':
    main()
