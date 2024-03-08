from loguru import logger
from pyswip import Prolog
import pandas as pd


def set_object(engine: Prolog, obj):
    name = obj['name']
    size = obj['size']
    material = obj['material']
    color = obj['color']
    shape = obj['shape']
    prolog_str = f'object({name}, {size}, {material}, {color}, {shape})'
    logger.debug(f'Asserting: {prolog_str}')
    engine.assertz(prolog_str)
    return [prolog_str]


def set_relationship(engine: Prolog, adj_matrix, object_list, direction="left"):
    if direction not in ["left", "right", "behind", "front"]:
        raise Exception(f'Invalid direction: {direction}')

    t = []

    for i in range(len(object_list)):
        for j in range(len(object_list)):
            if adj_matrix[i][j] == 1:
                prolog_str = f'relationship({object_list[i]["name"]}, {object_list[j]["name"]}, {direction})'
                logger.debug(f'Asserting: {prolog_str}')
                engine.assertz(prolog_str)
                t.append(prolog_str)

    return t


def adj_list_to_matrix(adj_list, n):
    adj_matrix = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in adj_list[i]:
            adj_matrix[i][j] = 1
    return adj_matrix


def query(engine: Prolog, name="Name", size="_", material="_", color="_", shape="_", direction="_"):
    prolog_str = f'object({name}, {size}, {material}, {color}, {shape}), relationship({name}, Target, {direction}).'
    logger.debug(f'Query: {prolog_str}')
    result = list(engine.query(prolog_str))
    return result


def result_to_dataframe(result):
    headers = list(result[0].keys())
    data = []
    # 需要合并同类项
    for r in result:
        d = [r[header] for header in headers]
        if d not in data:
            data.append(d)
    return pd.DataFrame(data, columns=headers)


class Engine:
    def __init__(self):
        self.engine = Prolog()
        self.rules = []

    def reset(self):
        del self.engine
        self.engine = Prolog()
        self.rules = []

    def write_object(self, obj):
        self.rules += set_object(self.engine, obj)

    def write_relationship(self, adj_matrix, object_list, direction="left"):
        self.rules += set_relationship(self.engine, adj_matrix, object_list, direction)

    def auto_write(self, objects, relationships):
        for obj in objects:
            self.write_object(obj)
        for key, value in relationships.items():
            self.write_relationship(adj_list_to_matrix(value, len(objects)), objects, key)

    def query(self, name="Name", size="_", material="_", color="_", shape="_", direction="_"):
        return query(self.engine, name, size, material, color, shape, direction)
