import argparse
import json
import os
import random
import pandas as pd

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
:- discontiguous word/3.
:- discontiguous def_question/3.

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
    content, examples, raw_data = init_examples(args, content, scene)

    max_len = max([len(x) for x in raw_data])
    col_names = ['shape', 'material', 'color', 'size']
    columns = ['label'] + [f'{col_names[i % 4]}_{i // 4}' for i in range(max_len - 1)]
    df = pd.DataFrame(raw_data, columns=columns)

    for example in examples:
        for obj in examples[example]:
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
    raw_data = []
    for i in range(min(args.number, len(scene['scenes']))) if indexes is None else indexes:
        item = scene['scenes'][i]
        objects = item['objects']
        cache = []
        cache_raw = [i]
        for obj in objects:
            mtl = obj['material']
            color = obj['color_name']
            shape = obj['shape']
            size = obj['size']
            name = f"object_{obj_count}"
            cache.append(name)
            content += f"has_shape({name}, {shape}).\n"
            content += f"has_material({name}, {mtl}).\n"
            content += f"has_color({name}, {color}).\n"
            content += f"has_size({name}, {size}).\n"
            content += "\n"
            obj_count += 1
            cache_raw += [labels['shape'].index(shape), labels['material'].index(mtl), labels['color'].index(color), labels['size'].index(size)]

        for key in ['left', 'right', 'front', 'behind']:
            adj_list = item['relationships'][key]
            for obj_i, indexes in enumerate(adj_list):
                for j in indexes:
                    content += f"{key}_of({cache[obj_i]}, {cache[j]}).\n"
                    # TODO: add raw data
            content += "\n"
        content += "\n"

        raw_data.append(cache_raw)
        examples[f"example_{i}"] = cache
    return content, examples, raw_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--number', type=int, default=50)
    parser.add_argument('-p', '--path', type=str, required=True)
    parser.add_argument('-q', '--question', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-t', '--type', type=str, choices=['random'], default='random')
    parser.add_argument('--name', type=str, default='example')

    return parser.parse_args()


if __name__ == '__main__':
    main_question(parse_args())
