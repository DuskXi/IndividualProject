import json
import unittest
import os
from loguru import logger
from prolog import *

if os.environ.get('SWI_HOME_DIR', None) is None:
    raise Exception('SWI_HOME_DIR environment variable must be set')
from pyswip import Prolog


def load_scene(scene_json_path, questions_json_path, idx=0):
    with open(scene_json_path, 'r') as f:
        scene = json.load(f)['scenes'][idx]
    with open(questions_json_path, 'r') as f:
        question = json.load(f)['questions'][idx]
    return scene, question


class MainTest(unittest.TestCase):
    def test_pyswip(self):
        engine = Prolog()

        rules = """
        loves(vincent, mia).
        loves(marcellus, mia).
        loves(pumpkin, honey_bunny).
        loves(honey_bunny, pumpkin).
        
        jealous(X, Y) :-
            loves(X, Z),
            loves(Y, Z).
        """
        for rule in rules.split('.\n'):
            if rule != '':
                logger.debug(f'Asserting rule: {rule}')
                engine.assertz(rule)
        result = list(engine.query('jealous(X, Y)'))
        logger.debug(f'Result: {result}')
        self.assertEqual(True, True)

    def test_scene(self):
        scene, question = load_scene(r'D:\projects\IndividualProject\clevr\Dataset Generation\output\clevr_scenes.json',
                                     r'D:\projects\IndividualProject\clevr\clevr-dataset-gen\question_generation\questions_t.json')
        logger.info(f'Scene: {scene}')
        logger.info(f'Question: {question}')
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
        scene, question = load_scene(r'D:\projects\IndividualProject\clevr\Dataset Generation\output\clevr_scenes.json',
                                     r'D:\projects\IndividualProject\clevr\clevr-dataset-gen\question_generation\questions_t.json')
        engine.auto_write(scene['objects'], scene['relationships'])
        result = engine.query(name="obj_SmoothCube_v2_0", shape="cube")
        logger.info("Result: \n" + str(result_to_dataframe(result)))
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
