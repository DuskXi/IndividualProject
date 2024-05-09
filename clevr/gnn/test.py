import json
import unittest

import torch
from tqdm import tqdm

from model import GNN
from torch_geometric.data import Data as GData
from dataset import GNNDataset
from torch_geometric.data import DataLoader


class TestModel(unittest.TestCase):
    def test_test_model(self):
        relation_edge_index_left = torch.tensor([[0, 1, 1, 2, 3], [1, 0, 2, 1, 0]], dtype=torch.long)
        relation_edge_index_right = torch.tensor([[0, 1, 1, 2, 3], [1, 0, 2, 1, 0]], dtype=torch.long)
        relation_edge_index_front = torch.tensor([[0, 1, 1, 2, 3], [1, 0, 2, 1, 0]], dtype=torch.long)
        relation_edge_index_behind = torch.tensor([[0, 1, 1, 2, 3], [1, 0, 2, 1, 0]], dtype=torch.long)
        relation_node_feature = torch.tensor([[1], [1], [1], [1]], dtype=torch.float32)
        relation_data = [
            GData(x=relation_node_feature, edge_index=relation_edge_index_left),
            GData(x=relation_node_feature, edge_index=relation_edge_index_right),
            GData(x=relation_node_feature, edge_index=relation_edge_index_front),
            GData(x=relation_node_feature, edge_index=relation_edge_index_behind),
        ]

        question_tree_edge_index = torch.tensor([[0, 1, 1, 2, 3], [1, 0, 2, 1, 0]], dtype=torch.long)
        question_tree_node_feature = torch.tensor([[1, 0], [1, 0], [1, 0], [1, 0]], dtype=torch.float32)
        question_tree_data = GData(x=question_tree_node_feature, edge_index=question_tree_edge_index)

        model = GNN(num_classes=10)
        result = model(relation_data[0], relation_data[1], relation_data[2], relation_data[3], question_tree_data)
        self.assertEqual(True, True)  # add assertion here

    def test_dataset(self):
        dataset = GNNDataset(r"W:\projects\IndividualProject\clevr\clevr-dataset-gen\question_generation\questions.json",
                             r"W:\projects\IndividualProject\clevr\Dataset Generation\output\clevr_scenes.json")
        data = dataset[0]
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        data = next(iter(dataloader))
        model = GNN(num_classes=10)
        result = model(data[0], data[1], data[2], data[3], data[4])
        self.assertEqual(True, True)

    def test_temp(self):
        dataset = GNNDataset(r"W:\projects\IndividualProject\clevr\clevr-dataset-gen\question_generation\questions_t.json",
                             r"W:\projects\IndividualProject\clevr\Dataset Generation\output\clevr_scenes.json")
        questions = dataset.questions
        print(len(questions))
        ans = {

        }
        for question in questions:
            if question['answer'] not in ans:
                ans[question['answer']] = 0
            ans[question['answer']] += 1

        print(json.dumps(ans, indent=4))
        self.assertEqual(True, True)

    def test_train(self):
        dataset = GNNDataset(r"W:\projects\IndividualProject\clevr\clevr-dataset-gen\question_generation\questions_t.json",
                             r"W:\projects\IndividualProject\clevr\Dataset Generation\output\clevr_scenes.json")
        dataset.questions = list(filter(lambda x: type(x['answer']) == bool, dataset.questions))
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        model = GNN(num_classes=2)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(10):
            losses = []
            accuracies = []
            window = 100
            for data in tqdm(dataloader):
                optimizer.zero_grad()
                result = model(data[0], data[1], data[2], data[3], data[4])
                label = torch.nn.functional.one_hot(data[5], 2).squeeze(1).to(torch.float32)
                loss = criterion(result, data[5])
                loss.backward()
                optimizer.step()
                accuracy = torch.sum(torch.argmax(result, dim=1) == torch.argmax(label, dim=1)).item() / len(label)
                losses.append(loss.item())
                accuracies.append(accuracy)
                if len(losses) > window:
                    losses.pop(0)
                    accuracies.pop(0)
                avg_loss = sum(losses) / len(losses)
                avg_acc = sum(accuracies) / len(accuracies)
                print(f"Epoch {epoch} Loss {loss.item():.4f} Avg Loss {avg_loss:.4f} Accuracy {accuracy * 100:.4f}% Avg Acc {avg_acc * 100:.4f}%")




if __name__ == '__main__':
    unittest.main()
