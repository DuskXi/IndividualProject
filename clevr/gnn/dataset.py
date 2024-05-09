import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data as GData
import json
from torch.nn import functional as F

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

color_types = [
    'gray',
    'blue',
    'brown',
    'cyan',
    'yellow',
    'green',
    'purple',
    'red'
]

material_types = [
    'rubber',
    'metal'
]

shape_types = [
    'cube',
    'cylinder',
    'sphere'
]

size_types = [
    'small',
    'large'
]

direction_types = [
    'left',
    'right',
    'front',
    'behind'
]


class GNNDataset(Dataset):
    def __init__(self, question_path, scene_path):
        self.question_path = question_path
        self.scene_path = scene_path
        self.questions = self.load_questions()
        self.scenes = self.load_scenes()

    def load_questions(self):
        with open(self.question_path, 'r') as f:
            d = json.load(f)
        return d['questions']

    def load_scenes(self):
        with open(self.scene_path, 'r') as f:
            d = json.load(f)
        return d['scenes']

    def build_tree_data(self, idx):
        # node feature: [0]: type of program, [1]: filter target if available else -1, [2]: 0 = single input, 1 = double input
        question = self.questions[idx]
        programs = question['program']
        node_features = []
        edge_index = []
        for i, program in enumerate(programs):
            type_idx = program_types.index(program['type'])
            input_type = 0 if len(program['inputs']) == 1 else 1
            if len(program["value_inputs"]) > 0:
                if program["value_inputs"][0] in color_types:
                    value_input = color_types.index(program["value_inputs"][0]) + 1
                elif program["value_inputs"][0] in material_types:
                    value_input = material_types.index(program["value_inputs"][0]) + 1 + len(color_types)
                elif program["value_inputs"][0] in shape_types:
                    value_input = shape_types.index(program["value_inputs"][0]) + 1 + len(color_types) + len(material_types)
                elif program["value_inputs"][0] in size_types:
                    value_input = size_types.index(program["value_inputs"][0]) + 1 + len(color_types) + len(material_types) + len(shape_types)
                elif program["value_inputs"][0] in direction_types:
                    value_input = direction_types.index(program["value_inputs"][0]) + 1 + len(color_types) + len(material_types) + len(shape_types) + len(size_types)
                else:
                    raise ValueError(f"Unknown value input {program['value_inputs'][0]}")
            else:
                value_input = 0
            type_idx_ohe = F.one_hot(torch.tensor(type_idx), num_classes=len(program_types))
            # value_input_ohe = F.one_hot(torch.tensor(value_input), num_classes=max(len(color_types), len(material_types), len(shape_types), len(size_types), len(direction_types)) + 1)
            value_input_ohe = F.one_hot(torch.tensor(value_input), num_classes=len(color_types) + len(material_types) + len(shape_types) + len(size_types) + len(direction_types) + 1)
            input_type_ohe = F.one_hot(torch.tensor(input_type), num_classes=2)
            feature = torch.cat((type_idx_ohe, value_input_ohe, input_type_ohe), dim=0).unsqueeze(0)
            node_features.append(feature)
            if len(program["inputs"]):
                for input_idx in program["inputs"]:
                    edge_index.append([i, input_idx])
        edge_index = list(zip(*edge_index))
        return GData(x=torch.cat(node_features, dim=0).to(dtype=torch.float32), edge_index=torch.tensor(edge_index, dtype=torch.long))

    def build_relation(self, idx):
        # node feature: [0]: color, [1]: shape, [2]: material, [3]: size
        idx = self.questions[idx]['image_index']
        scene = self.scenes[idx]
        relationships = scene['relationships']
        node_features = []
        for obj in scene['objects']:
            color_ohe = F.one_hot(torch.tensor(color_types.index(obj['color'])), num_classes=len(color_types))
            shape_ohe = F.one_hot(torch.tensor(shape_types.index(obj['shape'])), num_classes=len(shape_types))
            material_ohe = F.one_hot(torch.tensor(material_types.index(obj['material'])), num_classes=len(material_types))
            size_ohe = F.one_hot(torch.tensor(size_types.index(obj['size'])), num_classes=len(size_types))
            node_features.append(torch.cat((color_ohe, shape_ohe, material_ohe, size_ohe), dim=0).unsqueeze(0))
            # node_features.append([color_types.index(obj['color']), shape_types.index(obj['shape']), material_types.index(obj['material']), size_types.index(obj['size'])])
        edge_indexes = {
            'left': [],
            'right': [],
            'front': [],
            'behind': []
        }
        for relationship, adj_list in relationships.items():
            for i, indexes in enumerate(adj_list):
                for adj in indexes:
                    edge_indexes[relationship].append([i, adj])
        for key in edge_indexes.keys():
            edge_indexes[key] = list(zip(*edge_indexes[key]))
        return [GData(x=torch.cat(node_features, dim=0).to(dtype=torch.float32), edge_index=torch.tensor(edge_indexes[key], dtype=torch.long)) for key in edge_indexes.keys()]

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question_tree = self.build_tree_data(idx)
        relation_left, relation_right, relation_front, relation_behind = self.build_relation(idx)
        answer = self.questions[idx]['answer']
        if answer:
            label = torch.tensor(1, dtype=torch.long)
        else:
            label = torch.tensor(0, dtype=torch.long)
        return relation_left, relation_right, relation_front, relation_behind, question_tree, label


class GNN_CNNDataset(GNNDataset):
    def __init__(self, question_path, scene_path):
        super().__init__(question_path, scene_path)
        self.attribute_labels = self.build_attribute_labels()

    def build_attribute_labels(self):
        labels = []
        for shape in shape_types:
            for color in color_types:
                for material in material_types:
                    for size in size_types:
                        labels.append(f"{size}_{color}_{material}_{shape}")
        return labels

    def build_relation(self, idx):
        idx = self.questions[idx]['image_index']
        scene = self.scenes[idx]
        relationships = scene['relationships']
        attribute_features = []
        for obj in scene['objects']:
            label = self.attribute_labels.index(f"{obj['size']}_{obj['color']}_{obj['material']}_{obj['shape']}")
            if label == -1:
                raise ValueError(f"Unknown label {obj['size']}_{obj['color']}_{obj['material']}_{obj['shape']}")
            attribute_features.append(F.one_hot(torch.tensor(label), num_classes=len(self.attribute_labels)))
        relation = np.zeros((4, len(attribute_features), len(attribute_features)))
        for d, (relationship, adj_list) in enumerate(relationships.items()):
            for i, indexes in enumerate(adj_list):
                for adj in indexes:
                    relation[d, i, adj] = 1
        relation = torch.tensor(relation, dtype=torch.float32)
        attribute_features = torch.stack(attribute_features).to(dtype=torch.float32)
        return relation, attribute_features

    def __getitem__(self, idx):
        question_tree = self.build_tree_data(idx)
        relation, attribute_features = self.build_relation(idx)
        answer = self.questions[idx]['answer']
        if answer:
            label = torch.tensor(1, dtype=torch.long)
        else:
            label = torch.tensor(0, dtype=torch.long)
        return relation, attribute_features, question_tree, label


class GNN_FULL_CNNDataset(GNNDataset):
    def __init__(self, question_path, scene_path):
        super().__init__(question_path, scene_path)
        self.attribute_labels = self.build_attribute_labels()

    def build_attribute_labels(self):
        labels = []
        for shape in shape_types:
            for color in color_types:
                for material in material_types:
                    for size in size_types:
                        labels.append(f"{size}_{color}_{material}_{shape}")
        return labels

    def build_relation(self, idx):
        idx = self.questions[idx]['image_index']
        scene = self.scenes[idx]
        relationships = scene['relationships']
        attribute_features = []
        for obj in scene['objects']:
            label = self.attribute_labels.index(f"{obj['size']}_{obj['color']}_{obj['material']}_{obj['shape']}")
            if label == -1:
                raise ValueError(f"Unknown label {obj['size']}_{obj['color']}_{obj['material']}_{obj['shape']}")
            attribute_features.append(F.one_hot(torch.tensor(label), num_classes=len(self.attribute_labels)))
        relation = np.zeros((4, len(attribute_features), len(attribute_features)))
        for d, (relationship, adj_list) in enumerate(relationships.items()):
            for i, indexes in enumerate(adj_list):
                for adj in indexes:
                    relation[d, i, adj] = 1
        relation = torch.tensor(relation, dtype=torch.float32)
        attribute_features = torch.stack(attribute_features).to(dtype=torch.float32)
        return relation, attribute_features

    def __getitem__(self, idx):
        question_tree = self.build_tree_data(idx)
        question_adj_matrix = torch.zeros((question_tree.x.shape[1], len(question_tree.x), len(question_tree.x)))
        for i, j in question_tree.edge_index.T:
            question_adj_matrix[:, i, j] = question_tree.x[j]
            question_adj_matrix[:, j, i] = question_tree.x[i]
        relation, attribute_features = self.build_relation(idx)
        answer = self.questions[idx]['answer']
        if answer:
            label = torch.tensor(1, dtype=torch.long)
        else:
            label = torch.tensor(0, dtype=torch.long)
        return relation, attribute_features, question_tree, label
