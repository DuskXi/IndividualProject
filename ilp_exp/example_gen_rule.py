import argparse
import json
import multiprocessing
import os
import random
import time

import numpy as np
import pandas as pd
from loguru import logger
from rich.logging import RichHandler
from tqdm.rich import tqdm

logger.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

from rule_gen import Rule, Side, SceneObject, RelatedRuleSingle, RelationCondition, RuleGenerator

text_head = """
:- discontiguous has_color/2.
:- discontiguous has_material/2.
:- discontiguous has_size/2.
:- discontiguous has_shape/2.
:- discontiguous contains/2.

:- discontiguous left_of/2.
:- discontiguous right_of/2.
:- discontiguous front_of/2.
:- discontiguous behind_of/2.

color(gray).
color(blue).
color(brown).
color(yellow).
color(red).
color(green).
color(purple).
color(cyan).
material(rubber).
material(metal).
size(small).
size(large).
shape(cube).
shape(sphere).
shape(cylinder).

"""

labels = {
    'shape': ['cube', 'sphere', 'cylinder'],
    'material': ['rubber', 'metal'],
    'color': ['gray', 'blue', 'brown', 'yellow', 'red', 'green', 'purple', 'cyan'],
    'size': ['small', 'large']
}


def load_scene(path):
    with open(path, 'r') as f:
        return json.load(f)


#
# def main_question_text(args):
#     scene = load_scene(args.path)
#     question = load_scene(args.question)
#     content = ""
#     questions = list(filter(lambda x: type(x['answer']) == bool, question['questions']))[:args.number]
#     idx_scene = list(map(lambda x: x['image_index'], questions))
#     words_table = set()
#     for q in questions:
#         words = q['question'].replace('?', '').replace(';', '').replace(',', '').split(' ')
#         for word in words:
#             words_table.add(word)
#     words_table = list(words_table)
#     words_table.sort()
#
#     for


def main_question(args):
    scene = load_scene(args.path)
    question = load_scene(args.question)
    content = ""
    questions = list(filter(lambda x: type(x['answer']) == bool and len(x['program'][-1]['inputs']) != 2, question['questions']))[:args.number]
    programs = list(map(lambda x: merge_programs(x['program']), questions))

    q1 = questions[0]


def main(args):
    scene = load_scene(args.path)
    content = ""
    content, examples, raw_data, examples_scene, adj_lists = init_examples(args, content, scene)

    max_len = max([len(x) for x in raw_data])
    col_names = ['shape', 'material', 'color', 'size']
    columns = ['label'] + [f'{col_names[i % 4]}_{i // 4}' for i in range(max_len - 1)]
    # columns += ['relation_' + str(i) for i in range(len(raw_data[0]) - len(columns))]
    # if max_len > 20 + 1:
    #     raw_data = raw_data_expand(raw_data)
    df = pd.DataFrame(raw_data, columns=columns)

    for example in examples:
        for i, obj in enumerate(examples[example]):
            content += f"contains({obj}, {example}).\n"
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # bk
    with open(f"{args.output}/{args.name}.bk", 'w') as f:
        f.write(text_head)
        f.write(content)

    # n and f
    if args.type == 'random':
        positive = []
        negative = []
        p_index = []
        for i, example in enumerate(examples):
            if random.random() > 0.5:
                positive.append(f"true_class({example}).")
                p_index.append(i)
            else:
                negative.append(f"true_class({example}).")

        df['label'] = df['label'].apply(lambda x: 1 if x in p_index else 0)

        df.to_csv(f"{args.output}/{args.name}.csv", index=False)

        with open(f"{args.output}/{args.name}.n", 'w') as f:
            f.write("\n".join(positive))

        with open(f"{args.output}/{args.name}.f", 'w') as f:
            f.write("\n".join(negative))
    elif args.type == 'random_rule':
        logger.info("Random Rule")
        rule = Rule(left=Side(shape=['cube'], material=['rubber'], number=0, number_limit_type='more'),
                    right=Side(shape=['cylinder'], material=['metal'], number=0, number_limit_type='more'))
        # rule = Rule(left=Side(shape=['cube'], material=['rubber'], number=1, number_limit_type='equal'),
        #             right=Side(shape=['cube'], material=['rubber'], number=2, number_limit_type='equal'))
        logger.info(f"Rule left: {rule.left.to_prolog_query('A')}")
        logger.info(f"Rule right: {rule.right.to_prolog_query('B')}")
        positive, negative, positive_index, negative_index = split_by_rule(examples_scene, rule)
        positive = list(map(lambda x: f"true_class({x}).", positive))
        negative = list(map(lambda x: f"true_class({x}).", negative))

        df['label'] = df['label'].apply(lambda x: 1 if x in positive_index else 0)
        # remove the line not in positive and negative
        available_index = positive_index + negative_index
        available_index.sort()
        df = df.loc[available_index]
        df.to_csv(f"{args.output}/{args.name}.csv", index=False)

        with open(f"{args.output}/{args.name}.n", 'w') as f:
            f.write("\n".join(positive))

        with open(f"{args.output}/{args.name}.f", 'w') as f:
            f.write("\n".join(negative))

    elif args.type == 'random_rule_relation':
        logger.info("Random Rule Relation")
        # rule = RelatedRuleSingle(
        #     [
        #         Side(material=['rubber']),
        #         RelationCondition('left'),
        #         Side(shape=['cube', 'cylinder']),
        #     ]
        # )

        # true_class(A) :- contains(B,A), has_material(B, rubber), left_of(C,B), has_shape(C, cube).
        # true_class(A) :- contains(B,A), has_material(B, rubber), left_of(C,B), has_shape(C, cylinder).
        rule = RelatedRuleSingle(
            [
                Side(material=['metal']),
                RelationCondition('left'),
                Side(shape=['cube'], material=['rubber', 'metal']),
            ]
        )
        # true_class(A) :- contains(B,A), has_material(B, metal), left_of(C,B), has_shape(C, cube), has_material(C, rubber).
        for query in rule.to_prolog_query():
            logger.info(query)

        positive, negative, positive_index = split_by_rule_relation(examples_scene, adj_lists, rule)
        positive = list(map(lambda x: f"true_class({x}).", positive))
        negative = list(map(lambda x: f"true_class({x}).", negative))

        df['label'] = df['label'].apply(lambda x: 1 if x in positive_index else 0)
        df.to_csv(f"{args.output}/{args.name}.csv", index=False)

        with open(f"{args.output}/{args.name}.f", 'w') as f:
            f.write("\n".join(positive))

        with open(f"{args.output}/{args.name}.n", 'w') as f:
            f.write("\n".join(negative))
    elif args.type == 'rand_rule_batch':
        logger.info("Random Rule Batch")
        split_ratio_threshold = 0.3
        manager = multiprocessing.Manager()
        lock = manager.Lock()
        dataset = manager.list()
        cache_rule = manager.list()
        generator = RuleGenerator()
        with multiprocessing.Pool(processes=20, initializer=init, initargs=(lock,)) as pool:
            results = []
            for i in tqdm(range(args.batch), desc='CreateMission', total=args.batch):
                # worker(cache_rule, dataset, examples_scene, generator, split_ratio_threshold)
                results.append(pool.apply_async(worker, (cache_rule, dataset, examples_scene, generator, split_ratio_threshold, lock)))
                time.sleep(0.01)
            for result in tqdm(results, desc='Generation'):
                result.get()

        dataset = list(filter(lambda x: 1 - split_ratio_threshold > x['split_ratio'] > split_ratio_threshold, dataset))
        data = []
        for i, d in enumerate(dataset):
            data.append([d['rule'].left.to_json(), d['rule'].right.to_json(), json.dumps(d['positive']), json.dumps(d['negative']), d['split_ratio']])
        dfs = pd.DataFrame(data, columns=['positive', 'negative', 'positive_index', 'negative_index', 'split_ratio'])
        dfs.to_csv(f"{args.output}/{args.name}_spilt.csv", index=False)

        df.to_csv(f"{args.output}/{args.name}_data.csv", index=False)


def init(l):
    global lock
    lock = l


def worker(cache_rule, dataset, examples_scene, generator, split_ratio_threshold, lock):
    while True:
        rule, similar = generator.rand_rule(5)
        with lock:
            if rule in cache_rule or sum(similar.values()) / len(similar) > 0.75:
                continue
            cache_rule.append(rule)
        positive, negative, positive_index, negative_index = split_by_rule(examples_scene, rule, log_output=False)
        data = {
            'rule': rule,
            'positive': positive,
            'negative': negative,
            'positive_index': positive_index,
            'negative_index': negative_index,
            'split_ratio': len(positive) / (len(positive) + len(negative)) if len(positive) + len(negative) > 0 else 0,
        }
        if 1 - split_ratio_threshold > data['split_ratio'] > split_ratio_threshold:
            with lock:
                dataset.append(data)
            break


def split_by_rule(examples: dict[str, list[SceneObject]], rule: Rule, log_output=True):
    if log_output:
        logger.info(f"Rule: {rule}")
    left = []
    right = []
    positive_index = []
    negative_index = []
    all_count = 0
    for (i, example) in enumerate(examples):
        t = rule.identify_scene(examples[example], True)
        if t == 'left':
            left.append(example)
            positive_index.append(i)
        elif t == 'right':
            right.append(example)
            negative_index.append(i)
        else:
            # right.append(example)
            all_count += 1
    if log_output:
        logger.info(f"Left: {len(left)}, Right: {len(right)}, All: {all_count}")
    return left, right, positive_index, negative_index


def split_by_rule_relation(examples: dict[str, list[SceneObject]], adj_lists, rule: RelatedRuleSingle):
    logger.info(f"Rule: {rule}")
    positive = []
    negative = []
    positive_index = []
    l = 5
    for i, (example, adj_list) in enumerate(zip(examples, adj_lists)):
        # if i != 9:
        #     continue
        t = rule.identify_scene(examples[example], adj_list)
        if i < l:
            logger.info(f"Example: {examples[example]},\n Result: {t}")
        if t:
            positive.append(example)
            positive_index.append(i)
        else:
            negative.append(example)
    logger.info(f"Pos: {len(positive)}, Neg: {len(negative)}")
    return positive, negative, positive_index


def raw_data_expand(raw_data):
    new_data = []
    for data in raw_data:
        label, attrs, relations = data[0], data[1:1 + 5 * 4], data[1 + 5 * 4:]
        relations = np.array(relations).reshape(4, 5, 5)
        attrs = np.array(attrs).reshape(5, 4)
        for i in range(1, 5):
            shift_attrs = np.roll(attrs, i, axis=0)
            shift_relations = np.zeros((4, 5, 5))
            for d, matrix in enumerate(relations):
                relations[d] = np.roll(np.roll(matrix, 1, axis=0), 1, axis=1)
            new_data.append([label] + shift_attrs.reshape(-1).tolist() + shift_relations.reshape(-1).tolist())

    return new_data


def merge_programs(programs):
    merged_set = []
    cache = []
    for item in programs:
        if len(item['inputs']) < 2:
            if item['type'] == 'scene':
                if len(cache) > 0:
                    merged_set.append(cache)
                    cache = []
            cache.append(item['type'] + (f"-{item['value_inputs'][0]}" if len(item['value_inputs']) > 0 else ''))
        else:
            if len(cache) > 0:
                merged_set.append(cache)
                cache = []
            if len(merged_set) == 2:
                temp_set = {
                    'type': item['type'],
                    'inputs': [merged_set[-2], merged_set[-1]]
                }
                merged_set.pop()
                merged_set.pop()
                merged_set.append(temp_set)
            else:
                raise ValueError("Invalid program")
    if len(cache) > 0:
        merged_set.append(cache)
    return merged_set


def word_table(content, questions):
    words_set = set()
    for question in questions:
        question_text = question['question']
        words = question_text.split(' ')
        for word in words:
            words_set.add(word)
    words_list = list(words_set)
    words_list.sort()
    for i, word in enumerate(words_list):
        pass


def gen_questions(content, questions):
    for question in questions:
        program = question['program']
        pass


def init_examples(args, content, scene, indexes: list[int] = None):
    obj_count = 0
    examples = {}
    examples_scene = {}
    raw_data = []
    adj_lists = []
    for i in range(min(args.number, len(scene['scenes']))) if indexes is None else indexes:
        # if i != 4:
        #     continue
        item = scene['scenes'][i]
        objects = item['objects']
        examples_scene[f"example_{i}"] = []
        cache = []
        cache_raw = [i]
        for obj in objects:
            mtl = obj['material']
            color = obj['color_name']
            shape = obj['shape']
            size = obj['size']
            name = f"object_{obj_count}"
            cache.append(name)
            examples_scene[f"example_{i}"].append(SceneObject(size, shape, mtl, color))
            content += f"has_shape({name}, {shape}).\n"
            content += f"has_material({name}, {mtl}).\n"
            content += f"has_color({name}, {color}).\n"
            content += f"has_size({name}, {size}).\n"
            content += "\n"
            obj_count += 1
            cache_raw += [labels['shape'].index(shape), labels['material'].index(mtl), labels['color'].index(color), labels['size'].index(size)]

        adj_lists.append(item['relationships'])
        relation_info = [[[0] * 5] * 5] * 4
        relation_info = np.array(relation_info)
        for ki, key in enumerate(['left', 'right', 'front', 'behind']):
            adj_list = item['relationships'][key]
            for obj_i, indexes in enumerate(adj_list):
                for j in indexes:
                    # content += f"{key}_of({cache[obj_i]}, {cache[j]}).\n"
                    content += f"{key}_of({cache[j]}, {cache[obj_i]}).\n"
                    # relation_info[ki][obj_i][j] = 1
                    relation_info[ki][j][obj_i] = 1
            content += "\n"
        content += "\n"

        # flatten
        relation_info = relation_info.reshape(-1)
        if args.type == 'random_rule_relation':
            cache_raw += relation_info.tolist()

        raw_data.append(cache_raw)
        examples[f"example_{i}"] = cache
    return content, examples, raw_data, examples_scene, adj_lists


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--number', type=int, default=50)
    parser.add_argument('--batch', type=int, default=50)
    parser.add_argument('-p', '--path', type=str, required=True)
    parser.add_argument('-q', '--question', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-t', '--type', type=str, choices=['random', 'random_rule', 'random_rule_relation', 'rand_rule_batch'], default='random')
    parser.add_argument('--name', type=str, default='example')

    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
