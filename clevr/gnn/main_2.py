import torch
from loguru import logger
from rich.logging import RichHandler
# from torch.utils.data import DataLoader
from tqdm.rich import tqdm

from dataset import GNN_CNNDataset
from torch_geometric.data import DataLoader
from model import CNN_GNN
import torch.utils.data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# set loguru handler to rich handler
logger.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


def evaluate(model, dataloader):
    model.eval()
    accuracy = 0
    for i, data in enumerate(progress := tqdm(dataloader, desc="Evaluating")):
        relation, attribute_features, question_tree, label = data
        relation = relation.to(device)
        attribute_features = attribute_features.to(device)
        question_tree = question_tree.to(device)
        label = label.to(device)
        result = model(relation, attribute_features, question_tree)
        accuracy += torch.sum(torch.argmax(result, dim=1) == label).item() / label.shape[0]
    accuracy = accuracy / len(dataloader)
    logger.info(f"Accuracy {accuracy * 100:.4f}%")
    return accuracy


def main():
    dataset = GNN_CNNDataset(r"W:\projects\IndividualProject\clevr\clevr-dataset-gen\question_generation\questions_big.json",
                             r"W:\projects\IndividualProject\clevr\Dataset Generation\output\clevr_scenes_2.json")
    dataset.questions = list(filter(lambda x: type(x['answer']) == bool, dataset.questions))
    logger.info(f"Dataset size: {len(dataset)}")
    # dataset.questions = dataset.questions[:100]
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    model = CNN_GNN(num_classes=2)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)


    losses = []
    accuracies = []
    window = 25
    result_cache = []
    eval_accs = []
    last_eval_acc = 0
    for epoch in range(50):
        for i, data in enumerate(progress := tqdm(train_dataloader, desc=f"Epoch {epoch}")):
            relation, attribute_features, question_tree, label = data
            relation = relation.to(device)
            attribute_features = attribute_features.to(device)
            question_tree = question_tree.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            result = model(relation, attribute_features, question_tree)
            loss = criterion(result, label)
            loss.backward()
            optimizer.step()
            result_cache.append(torch.argmax(result, dim=1).cpu())
            accuracy = torch.sum(torch.argmax(result, dim=1) == label).item() / label.shape[0]
            losses.append(loss.item())
            accuracies.append(accuracy)
            if len(losses) > window:
                losses.pop(0)
                accuracies.pop(0)
                result_cache.pop(0)
            avg_loss = sum(losses) / len(losses)
            avg_acc = sum(accuracies) / len(accuracies)
            progress.set_description(f"Epoch {epoch} Loss {loss.item():.4f} Avg Loss {avg_loss:.4f} Accuracy {accuracy * 100:.4f}% Avg Acc {avg_acc * 100:.4f}%")
            if i % (len(train_dataloader) // 4) == 0:
                logger.info(f"Epoch {epoch} Loss {loss.item():.4f} Avg Loss {avg_loss:.4f} Accuracy {accuracy * 100:.4f}% Avg Acc {avg_acc * 100:.4f}%")

        eval_acc = evaluate(model, test_dataloader)
        eval_accs.append(eval_acc)
        if eval_acc > last_eval_acc:
            last_eval_acc = eval_acc
            torch.save(model.state_dict(), f"logdir/model_gnn_cnn_{epoch}_{eval_acc * 100:.4f}%.pt")
            logger.info("Model saved")


if __name__ == '__main__':
    main()
