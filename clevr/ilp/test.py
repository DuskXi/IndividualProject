import gc
import json
import unittest
import os
from pprint import pprint
import numpy as np
from loguru import logger
from prolog import *

if os.environ.get('SWI_HOME_DIR', None) is None:
    raise Exception('SWI_HOME_DIR environment variable must be set')
from pyswip import Prolog


def load_scene(scene_json_path, questions_json_path, idx=0):
    with open(questions_json_path, 'r') as f:
        question = json.load(f)['questions'][idx]
    with open(scene_json_path, 'r') as f:
        scene = json.load(f)['scenes'][question['image_index']]
    return scene, question


def get_example_question(questions_json_path, target_type):
    with open(questions_json_path, 'r') as f:
        questions = json.load(f)['questions']
    for question in questions:
        for program in question['program']:
            if program['type'] == target_type:
                return question
    return None


def program_merge_dfs(programs, idx=-1):
    if idx == -1:
        idx = len(programs) - 1
    program = programs[idx]
    if len(program['inputs']) == 0:
        return [program]
    if len(program['inputs']) == 1 and idx < len(programs) - 1:
        return program_merge_dfs(programs, program['inputs'][0]) + [program]
    elif len(program['inputs']) == 1 and idx == len(programs) - 1:
        merge = {"merge": program_merge_dfs(programs, program['inputs'][0])}
        merge["filter"] = get_conditions(merge["merge"])
        return [{**program, "_inputs": [merge]}]
    elif len(program['inputs']) == 2:
        merge = [{"merge": program_merge_dfs(programs, program['inputs'][0])}, {"merge": program_merge_dfs(programs, program['inputs'][1])}]
        for m in merge:
            m["filter"] = get_conditions(m["merge"])
        return [{**program, "_inputs": merge}]
    print(f'Invalid program: {program}')


def get_conditions(merge_list: list):
    new_list = merge_list.copy()
    for i in range(len(new_list)):
        if i > 0 and new_list[i].get('type') == 'relate' and new_list[i - 1].get('type') == 'unique':
            temp = new_list[i]
            new_list[i] = new_list[i - 1]
            new_list[i - 1] = temp
    result = []
    buffer = {}
    for m in new_list:
        if m['type'] in ['filter_color', 'filter_material', 'filter_shape', 'filter_size', 'relate']:
            buffer[m['type'].replace('filter_', '')] = m['value_inputs']
        elif m['type'] in ['scene']:
            continue
        # elif m['type'] in ['same_color', 'same_material', 'same_shape', 'same_size']:
        #     buffer['type'] = m['type']
        else:
            if len(list(buffer.keys())) > 0:
                result.append(buffer)
                buffer = {}
            result.append(m)
    if len(list(buffer.keys())) > 0:
        result.append(buffer)

    return result


def clean_program(program):
    remove_keys = ['value_inputs', 'inputs', 'merge', '_output']
    queue = [program]
    while len(queue) > 0:
        p = queue.pop(0)
        for k in remove_keys:
            if k in p:
                del p[k]
        for key, value in p.items():
            if type(value) == list:
                queue += filter(lambda x: type(x) == dict, value)
            elif type(value) == dict:
                queue.append(value)
    return program


def run_query(engine: Engine, postprocess_program: dict):
    if "_inputs" in postprocess_program:
        if len(postprocess_program["_inputs"]) == 2:
            result_1 = run_query(engine, postprocess_program["_inputs"][0])
            result_2 = run_query(engine, postprocess_program["_inputs"][1])
            logger.debug(f'Result 1: {result_1}')
            logger.debug(f'Result 2: {result_2}')
            if postprocess_program['type'].startswith('equal_') and postprocess_program['type'] != 'equal_integer':
                if len(result_1) != len(result_2):
                    return False
                if len(result_1) == 0 or len(result_2) == 0:
                    return False
                result = True
                for r1 in result_1:
                    for r2 in result_2:
                        if r1 != r2:
                            result = False
                            break
                    if not result:
                        break
                return result
            elif postprocess_program['type'] == 'equal_integer':
                return result_1 == result_2
            elif postprocess_program['type'].startswith('same_'):
                force_attr = postprocess_program['type'].replace('same_', '')
                if type(result_1) == list and (len(result_1) == 0 or len(result_2) == 0):
                    return False
                # if len(result_1) != len(result_2):
                #     return False
                result = True
                for r1 in result_1:
                    for r2 in result_2:
                        if r1[force_attr] != r2[force_attr]:
                            result = False
                            break
                    if not result:
                        break
                return result
            elif postprocess_program['type'] == 'less_than':
                return len(result_1) < len(result_2) if type(result_1) == list else result_1 < result_2
            elif postprocess_program['type'] == 'greater_than':
                return len(result_1) > len(result_2) if type(result_1) == list else result_1 > result_2
            elif postprocess_program['type'] == 'intersect':
                if type(result_1) == list and type(result_2) == list:
                    return list(set(result_1).intersection(set(result_2)))
            elif postprocess_program['type'] == 'union':
                if type(result_1) == list and type(result_2) == list:
                    return list(set(result_1).union(set(result_2)))
        elif len(postprocess_program["_inputs"]) == 1:
            result = run_query(engine, postprocess_program["_inputs"][0])
            if postprocess_program['type'] == 'count':
                if type(result) not in [list, int]:
                    return False
                return len(result) if type(result) == list else result
            elif postprocess_program['type'] == 'exist':
                return len(result) > 0
            elif postprocess_program['type'].startswith('query_'):
                if postprocess_program['type'] == 'query_color':
                    query_attrs = {'color': 'Attr'}
                elif postprocess_program['type'] == 'query_material':
                    query_attrs = {'material': 'Attr'}
                elif postprocess_program['type'] == 'query_shape':
                    query_attrs = {'shape': 'Attr'}
                elif postprocess_program['type'] == 'query_size':
                    query_attrs = {'size': 'Attr'}
                else:
                    query_attrs = {}

                if type(result) == list:
                    _result = []
                    for r in result:
                        query_attrs['name'] = r
                        _result += [d['Attr'] for d in engine.query_object(**query_attrs)]
                    result = _result
                if len(result) > 0:
                    return result[0]
                else:
                    return False
            else:
                return result
    else:
        if 'filter' in postprocess_program:
            f = postprocess_program['filter']
            result = ['Name']
            for filter_ in f:
                if '_inputs' in filter_:
                    result = run_query(engine, filter_)
                    continue
                if 'type' not in filter_:
                    _result = []
                    for item in result:
                        attrs = {k: v[0] for k, v in filter_.items() if k in ['color', 'material', 'shape', 'size']}
                        if 'relate' in filter_:
                            attrs['direction'] = filter_['relate'][0]
                            _result += list(set([d['Target'] for d in engine.query(name=item, **attrs)]))
                        else:
                            if item != 'Name':
                                if item in list(set([d['Name'] for d in engine.query_object(**attrs)])):
                                    _result.append(item)
                            else:
                                _result += list(set([d['Name'] for d in engine.query_object(**attrs)]))
                    result = _result
                elif filter_['type'] == 'unique':
                    continue
                else:
                    if filter_['type'].startswith('query_'):
                        if filter_['type'] == 'query_color':
                            query_attrs = {'color': 'Attr'}
                        elif filter_['type'] == 'query_material':
                            query_attrs = {'material': 'Attr'}
                        elif filter_['type'] == 'query_shape':
                            query_attrs = {'shape': 'Attr'}
                        elif filter_['type'] == 'query_size':
                            query_attrs = {'size': 'Attr'}
                        else:
                            query_attrs = {}
                        _result = []
                        for item in result:
                            query_attrs['name'] = item
                            r = [d['Attr'] for d in engine.query_object(**query_attrs)]
                            if len(r) > 0:
                                _result.append(r[0])
                        result = _result
                    elif filter_['type'] == 'count':
                        result = len(result)
                    elif filter_['type'].startswith('same_'):
                        _target_attr = filter_['type'].replace('same_', '')
                        target_attr = 'Color' if _target_attr == 'color' else _target_attr
                        target_attr = 'Material' if _target_attr == 'material' else target_attr
                        target_attr = 'Shape' if _target_attr == 'shape' else target_attr
                        target_attr = 'Size' if _target_attr == 'size' else target_attr
                        _result = []
                        for item in result:
                            _result += [x for x in [d['Name'] for d in engine.query_same_attr(name=item, **{_target_attr: target_attr})] if x != item]
                        result = _result
                # print()
            return result
        else:
            raise Exception(f'Invalid program: {postprocess_program}')


class MainTest(unittest.TestCase):
    # def test_pyswip(self):
    #     engine = Prolog()
    #
    #     rules = """
    #     loves(vincent, mia).
    #     loves(marcellus, mia).
    #     loves(pumpkin, honey_bunny).
    #     loves(honey_bunny, pumpkin).
    #
    #     jealous(X, Y) :-
    #         loves(X, Z),
    #         loves(Y, Z).
    #     """
    #     for rule in rules.split('.\n'):
    #         if rule != '':
    #             logger.debug(f'Asserting rule: {rule}')
    #             engine.assertz(rule)
    #     result = list(engine.query('jealous(X, Y)'))
    #     logger.debug(f'Result: {result}')
    #     self.assertEqual(True, True)

    def test_scene(self):
        scene, question = load_scene(r'clevr/test_data/clevr_scenes.json',
                                     r'clevr/test_data/questions_t.json')
        # logger.info(f'Scene: {scene}')
        # logger.info(f'Question: {question}')
        rules = []
        engine = Prolog()
        for obj in scene['objects']:
            rules += set_object(engine, obj)
        for key, value in scene['relationships'].items():
            rules += set_relationship(engine, adj_list_to_matrix(value, len(scene['objects'])), scene['objects'], key)
        # define the prolog finder
        rules.sort()
        with open('clevr_test.pl', 'w', encoding="utf-8") as f:
            f.write(".\n".join(rules) + '.')
        # engine.assertz(prolog_finder)

        # find all small yellow objects
        result = query(engine, color='yellow')
        # for r in result:
        #     logger.info(f'Result: {r}')
        logger.info("Result: \n" + str(result_to_dataframe(result)))
        self.assertEqual(True, True)

    def test_engine(self):
        engine = Engine()
        scene, question = load_scene(r'../test_data/clevr_scenes.json',
                                     r'../test_data/questions_t.json')
        # logger.info(f'Scene: \n{json.dumps(question, indent=4)}')
        engine.auto_write(scene['objects'], scene['relationships'])
        result_1 = engine.query(material="rubber", shape="sphere")
        result_2 = engine.query(material="metal")
        logger.info("Result 1: \n" + str(result_to_dataframe(result_1)))
        logger.info("Result 2: \n" + str(result_to_dataframe(result_2)))
        # logger.info("Result: \n" + str(result_to_dataframe(result)))
        self.assertEqual(True, True)

    def test_list_all_component(self):
        q_path = r'../test_data/questions_t.json'
        logger.info(f'Loading questions from: {q_path}')
        with open(q_path, 'r') as f:
            questions = json.load(f)['questions']

        logger.info(f'Loaded {len(questions)} questions')

        result = []
        for q in questions:
            for p in q['program']:
                name = p['type']
                if name not in result:
                    result.append(name)

        for r in result:
            logger.info(f'Name: {r}')
        self.assertEqual(True, True)

    def test_example_question(self):
        q_path = r'../test_data/questions_t.json'
        logger.info(f'Loading questions from: {q_path}')
        with open(q_path, 'r') as f:
            questions = json.load(f)['questions']

        logger.info(f'Loaded {len(questions)} questions')

        result = get_example_question(q_path, 'intersect')
        # logger.info(f'Example question: {json.dumps(result, indent=4)}')
        pprint(result)
        self.assertEqual(True, True)

    def test_count_question(self):
        q_path = r'../test_data/questions_t.json'
        logger.info(f'Loading questions from: {q_path}')
        with open(q_path, 'r') as f:
            questions = json.load(f)['questions']

        logger.info(f'Loaded {len(questions)} questions')

        counts = []
        for q in questions:
            counts.append(len(q['program']))
            # if len(q['program']) > 15:
            #     logger.info(f'Question: {json.dumps(q, indent=4)}')
        logger.info(f'Counts: {len(counts)}')
        logger.info(f'Min: {min(counts)}, Max: {max(counts)}')
        logger.info(f'Average: {sum(counts) / len(counts)}')
        self.assertEqual(True, True)

    def test_merge_program(self):
        q_path = r'../test_data/questions_t.json'
        logger.info(f'Loading questions from: {q_path}')
        with open(q_path, 'r') as f:
            questions = json.load(f)['questions']

        logger.info(f'Loaded {len(questions)} questions')

        result = get_example_question(q_path, 'query_color')
        # logger.info(f'Example question: {json.dumps(result, indent=4)}')
        # pprint(result)
        program = result['program']
        merged_program = program_merge_dfs(program)
        # pprint(merged_program)
        # logger.info(f'Merged program: {json.dumps(merged_program, indent=4)}')
        self.assertEqual(True, True)

    def test_target_program(self):
        # 33589
        q_path = r'../test_data/questions_t.json'
        logger.info(f'Loading questions from: {q_path}')
        with open(q_path, 'r') as f:
            questions = json.load(f)['questions']

        logger.info(f'Loaded {len(questions)} questions')

        program = questions[33589]['program']
        merged_program = program_merge_dfs(program)
        merged_program = clean_program(merged_program[0])
        # pprint(merged_program)
        # logger.info(f'Merged program: {json.dumps(merged_program, indent=4)}')
        self.assertEqual(True, True)

    def test_run_question(self):
        engine = Engine()
        scene, question = load_scene(r'../test_data/clevr_scenes.json',
                                     r'../test_data/questions_t.json', 70)
        # logger.info(f'Scene: \n{json.dumps(question, indent=4)}')
        engine.auto_write(scene['objects'], scene['relationships'])
        program = question['program']
        merged_program = program_merge_dfs(program)
        merged_program = clean_program(merged_program[0])
        logger.info(f'Question: {question["question"]}')
        logger.info(f'Image: {question["image_filename"]}')
        result = run_query(engine, merged_program)
        logger.info(f'Query Result: {result}, Answer: {question["answer"]}')
        self.assertEqual(result, question["answer"])

    def test_run_questions(self):
        task_result = []
        engine = Engine()
        for i in range(50, 100):
            try:  # 56, 58, 59, 64, 65, 67, 68, 69, 70, 71
                if i == 10:
                    pass
                logger.warning(f'Running question: {i}')
                scene, question = load_scene(r'../test_data/clevr_scenes.json',
                                             r'../test_data/questions_t.json', i)
                engine.auto_write(scene['objects'], scene['relationships'])
                program = question['program']
                merged_program = program_merge_dfs(program)
                merged_program = clean_program(merged_program[0])
                logger.info(f'Question: {question["question"]}')
                logger.info(f'Image: {question["image_filename"]}')
                result = run_query(engine, merged_program)
                logger.info(f'Query Result: {result}, Answer: {question["answer"]}')
                task_result.append(result == question["answer"])
                engine.reset()
                # del engine
                gc.collect()
            except Exception as e:
                logger.error(f'Error: {e}, on index: {i}')
                task_result.append(False)
        logger.info(f'Task Result: {task_result}')
        logger.info(f'Correct Rate: {sum(task_result) / len(task_result) * 100:.4f}%')
        logger.info(f'Correct Count: {sum(task_result)}/{len(task_result)}')
        logger.info(f'Error questions: {np.where(np.array(task_result) == False)}')
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
