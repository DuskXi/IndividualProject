import json
import random
from typing import Literal, Optional


class SceneObject:
    def __init__(self,
                 size: Literal['big', 'small'],
                 shape: Literal['cylinder', 'sphere', 'cube'],
                 material: Literal['rubber', 'metal'],
                 color: Literal['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']):
        self.size = size
        self.shape = shape
        self.material = material
        self.color = color

    def __str__(self):
        return f'A {self.size} {self.color} {self.material} {self.shape}'

    def __eq__(self, other):
        return self.size == other.size and self.shape == other.shape and self.material == other.material and self.color == other.color

    def __hash__(self):
        return hash((self.size, self.shape, self.material, self.color))

    def __repr__(self):
        return str(self)


class Side:
    def __init__(self, size: list[Literal['big', 'small']] = [], shape: list[Literal['cylinder', 'sphere', 'cube']] = [], material: list[Literal['rubber', 'metal']] = [],
                 color: list[Literal['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']] = [], number: int = 4, number_limit_type: Literal['less', 'more', 'equal'] = 'equal'):
        self.size = size
        self.shape = shape
        self.material = material
        self.color = color
        if number <= 0 and number_limit_type == 'equal':
            raise ValueError('Number of objects must be greater than 0 if number_limit_type is not equal')
        self.number = number
        self.number_limit_type = number_limit_type

    def is_in_condition(self, obj: SceneObject):
        return ((obj.size in self.size or len(self.size) == 0) and
                (obj.shape in self.shape or len(self.shape) == 0) and
                (obj.material in self.material or len(self.material) == 0) and
                (obj.color in self.color or len(self.color) == 0))

    def is_number_in_condition(self, num: int):
        if self.number_limit_type == 'less':
            return num < self.number
        if self.number_limit_type == 'more':
            return num > self.number
        return num == self.number

    @staticmethod
    def arr_str(arr):
        if len(arr) == 1:
            return arr[0]
        elif len(arr) > 1:
            text = f"{arr[0]}"
            if len(arr) == 1:
                return text
            for i in range(1, len(arr)):
                text += f' or {arr[i]}'
            return text

    def to_label(self):
        return f'{self.arr_str(self.size)}_{self.arr_str(self.shape)}_{self.arr_str(self.material)}_{self.arr_str(self.color)}'

    def to_json(self):
        return json.dumps({'size': self.size, 'shape': self.shape, 'material': self.material, 'color': self.color})

    def to_prolog_query(self, c='B'):
        contents = []
        for size in range(2):
            for shape in range(3):
                for material in range(2):
                    for color in range(8):
                        if len(self.size) <= size and len(self.shape) <= shape and len(self.material) <= material and len(self.color) <= color:
                            continue
                        if (len(self.size) <= size and len(self.size) > 0) or (len(self.shape) <= shape and len(self.shape) > 0) or (len(self.material) <= material and len(self.material) > 0) or (len(self.color) <= color and len(self.color) > 0):
                            continue
                        content = ""
                        if len(self.size) > size:
                            content += ('' if content == '' else ', ') + f'size({c}, {self.size[size]})'
                        if len(self.shape) > shape:
                            content += ('' if content == '' else ', ') + f'shape({c}, {self.shape[shape]})'
                        if len(self.material) > material:
                            content += ('' if content == '' else ', ') + f'material({c}, {self.material[material]})'
                        if len(self.color) > color:
                            content += ('' if content == '' else ', ') + f'color({c}, {self.color[color]})'
                        if content not in contents:
                            contents.append(content)
        return contents

    def __str__(self):
        content = "if the object"
        if len(self.size) > 0:
            content += f' has size [{self.arr_str(self.size)}], '
        if len(self.shape) > 0:
            content += f' has shape [{self.arr_str(self.shape)}], '
        if len(self.material) > 0:
            content += f' has material [{self.arr_str(self.material)}], '
        if len(self.color) > 0:
            content += f' has color [{self.arr_str(self.color)}], '
        content += f'and the number of objects were in limit muse be {self.number_limit_type} {"than" if self.number_limit_type != "equal" else "to"} {self.number}'
        return content

    def __eq__(self, other):
        return self.size == other.size and self.shape == other.shape and self.material == other.material and self.color == other.color and self.number == other.number and self.number_limit_type == other.number_limit_type


class RelationCondition:
    def __init__(self, direction: Literal['left', 'right', 'front', 'behind']):
        self.direction = direction


class Rule:
    def __init__(self, left: Side, right: Side):
        self.left = left
        self.right = right

    def identify_scene(self, scene: list[SceneObject], force=True):
        # num = len(scene)
        left = list(filter(lambda x: self.left.is_in_condition(x), scene))
        right = list(filter(lambda x: self.right.is_in_condition(x), scene))
        if not force:
            if self.left.is_number_in_condition(len(left)):
                return 'left'
            else:
                return 'right'
        if self.left.is_number_in_condition(len(left)) and not self.right.is_number_in_condition(len(right)):
            return 'left'
        if not self.left.is_number_in_condition(len(left)) and self.right.is_number_in_condition(len(right)):
            return 'right'
        return 'all'

    def __str__(self):
        return f'If {self.left} for positive rule, then {self.right} for negative rule.'

    def __eq__(self, other):
        return self.left == other.left and self.right == other.right


class RelatedRuleSingle:
    def __init__(self, rule_sequence):
        self.rule_sequence = rule_sequence

    def identify_scene(self, scene: list[SceneObject], relation_adj_list: dict[str, list[list[int]]]):
        for i in range(len(scene)):
            if self.object_in_condition(scene, relation_adj_list, i, self.rule_sequence):
                return True
        return False

    def object_in_condition(self, scene: list[SceneObject], relation_adj_list: dict[str, list[list[int]]], idx_object, rule_seq, visited: list[int] = [],
                            no_visited: bool = False):
        rule = rule_seq[0]
        if isinstance(rule, Side):
            return rule.is_in_condition(scene[idx_object]) and (
                self.object_in_condition(scene, relation_adj_list, idx_object, rule_seq[1:], visited + [idx_object], no_visited) if len(rule_seq) > 1 else True)
        elif isinstance(rule, RelationCondition):
            adj_list = relation_adj_list[rule.direction]
            # connected = [i for i in range(len(adj_list[idx_object])) if adj_list[idx_object][i] == 1]
            connected = adj_list[idx_object]
            # for i in range(len(adj_list[idx_object])):
            #     connections = adj_list[idx_object]
            #     if i in connections:
            #         connected.append(i)
            if no_visited:
                connected = list(filter(lambda x: x not in visited, connected))
            for i in connected:
                if self.object_in_condition(scene, relation_adj_list, i, rule_seq[1:], visited + [idx_object], no_visited):
                    return True
            return False
        else:
            raise ValueError('Invalid rule sequence')

    def __str__(self):
        text = "The scene must have following condition: \n"
        contents = []
        for rule in self.rule_sequence:
            if isinstance(rule, Side):
                contents.append(f"the object must follow the condition: {rule}")
            elif isinstance(rule, RelationCondition):
                contents.append(f"has a object {rule.direction} of it,")
        return text + "\nand ".join(contents)

    def to_prolog_query(self):
        char_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        query_head = "query_object(A) :-\n\tcontains(B, A)"
        contents = []
        last_init_var = 'B'
        queue_rule = [(0, last_init_var, query_head)]
        while len(queue_rule) > 0:
            rule_index, c, prev = queue_rule.pop(0)
            if rule_index >= len(self.rule_sequence):
                if prev not in contents:
                    contents.append(prev)
                continue
            rule = self.rule_sequence[rule_index]
            if isinstance(rule, Side):
                rule_query = rule.to_prolog_query(c)
                for rule_q in rule_query:
                    query = f'{prev}, {rule_q}'
                    # contents.append(query)
                    queue_rule.append((rule_index + 1,c, query))
            elif isinstance(rule, RelationCondition):
                i = char_list.index(c)
                next_char = char_list[i + 1]
                query = f'{prev}, {rule.direction}({c}, {next_char})'
                # contents.append(query)
                queue_rule.append((rule_index + 1, next_char, query))
        return contents


class RuleGenerator:
    def __init__(self):
        self.rules = []
        self.size = ['big', 'small']
        self.shape = ['cylinder', 'sphere', 'cube']
        self.material = ['rubber', 'metal']
        self.color = ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']

    def rand_rule(self, num_objects: int, no_number=False) -> tuple[Rule, dict[str, float]]:
        size_rand = random.randint(0, 2)
        shape_rand = random.randint(0, 3)
        material_rand = random.randint(0, 2)
        color_rand = random.randint(0, 8)
        left = {
            'size': random.sample(self.size, size_rand),
            'shape': random.sample(self.shape, shape_rand),
            'material': random.sample(self.material, material_rand),
            'color': random.sample(self.color, color_rand)
        }
        size_rand = random.randint(0, 2)
        shape_rand = random.randint(0, 3)
        material_rand = random.randint(0, 2)
        color_rand = random.randint(0, 8)
        right = {
            'size': random.sample(self.size, size_rand),
            'shape': random.sample(self.shape, shape_rand),
            'material': random.sample(self.material, material_rand),
            'color': random.sample(self.color, color_rand)
        }
        similar = {}
        for key in left:
            left[key].sort()
            right[key].sort()
            set1, set2 = set(left[key]), set(right[key])
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            similar[key] = (intersection / union) if union != 0 else 0

        if not no_number:
            left_number_limit = random.choice(['less', 'more', 'equal'])
            right_number_limit = random.choice(['less', 'more', 'equal'])
            left_range = (1, num_objects) if left_number_limit == 'equal' else ((2, num_objects) if left_number_limit == 'less' else (1, num_objects - 1))
            right_range = (1, num_objects) if right_number_limit == 'equal' else ((2, num_objects) if right_number_limit == 'less' else (1, num_objects - 1))
            left_number = random.randint(*left_range)
            right_number = random.randint(*right_range)
        else:
            left_number_limit = 'more'
            right_number_limit = 'more'
            left_number = 0
            right_number = 0

        return Rule(Side(left['size'], left['shape'], left['material'], left['color'], left_number, left_number_limit),
                    Side(right['size'], right['shape'], right['material'], right['color'], right_number, right_number_limit)), similar
