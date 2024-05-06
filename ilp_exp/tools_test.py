import json
import os
import unittest

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec


def for_scene(scene_file=r"W:\projects\IndividualProject\clevr\Dataset Generation\output\clevr_scenes.json", start: int = 0, num: int = 10):
    with open(scene_file, 'r') as f:
        scenes = json.load(f)['scenes']
    for i in range(start, start + num):
        yield scenes[i]


class MyTestCase(unittest.TestCase):
    def test_print_scene_relation(self):
        for scene in for_scene():
            pass
        self.assertEqual(True, True)

    def test_display(self):
        width = 480
        height = 320
        image_directory = r"W:\projects\IndividualProject\clevr\Dataset Generation\output"
        offset_by_scene_index = True
        for scene in for_scene(start=0, num=10):
            fig = plt.figure(figsize=(10, 10))
            gs = gridspec.GridSpec(4, 2)
            ax1 = fig.add_subplot(gs[:2, :])
            ax1.imshow(Image.open(os.path.join(image_directory, scene['image_filename'])))
            ax1.set_title('Main Image')
            plt.suptitle(f"scene {scene['image_index']}")
            objects = []
            adjective_matrices = []
            for direction in ['left', 'right', 'front', 'behind']:
                adjective_matrix = [[0 for _ in range(5)] for _ in range(5)]
                adjective_list = scene["relationships"][direction]
                for i, adj in enumerate(adjective_list):
                    for j in adj:
                        adjective_matrix[i][j] = 1
                adjective_matrices.append(np.array(adjective_matrix))

            for i, obj_info in enumerate(scene['objects']):
                obj = {"index": i, "draw_index": i + (scene['image_index'] * 5 if offset_by_scene_index else 0)}
                xmin = ((obj_info['bounding_box']['xmin'] + 1) / 2) * width
                ymin = ((obj_info['bounding_box']['ymin'] + 1) / 2) * height
                xmax = ((obj_info['bounding_box']['xmax'] + 1) / 2) * width
                ymax = ((obj_info['bounding_box']['ymax'] + 1) / 2) * height
                obj['bounding_box'] = [xmin, ymin, xmax, ymax]
                obj['center'] = [(xmin + xmax) / 2, (ymin + ymax) / 2]
                objects.append(obj)

                # draw index in the center of the object

                ax1.text(obj['center'][0], obj['center'][1], str(obj['draw_index']),
                         horizontalalignment='center', verticalalignment='center', fontsize=12, color='white')

            for index, adj_matrix in enumerate(adjective_matrices):
                ax = fig.add_subplot(gs[index // 2 + 2, index % 2])
                cax = ax.matshow(adj_matrix, cmap='viridis')
                for (i, j), val in np.ndenumerate(adj_matrix):
                    ax.text(j, i, str(val), ha='center', va='center', color='white' if val < 0.5 else 'black')
                ax.set_title(f"Relation {['left', 'right', 'front', 'behind'][index]}")
                ax.set_xticks(range(5))
                ax.set_yticks(range(5))
                ax.set_xticklabels(map(lambda x: str(objects[x]["draw_index"]), range(5)))
                ax.set_yticklabels(map(lambda x: str(objects[x]["draw_index"]), range(5)))
                # fig.colorbar(cax)

            plt.tight_layout()
            plt.show()

        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
