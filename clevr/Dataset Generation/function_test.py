import unittest

# from geometry_tools import calculate_bounding_box
from render import Render
import bpy
from geometry_tools import calculate_bounding_box


class TestFunction(unittest.TestCase):
    def test_bounding_box(self):
        render = Render('data/base_scene.blend', 'data/shapes', 'data/materials')
        render.init_render()
        render.load_object_auto("SmoothCube_v2", "SmoothCube_v2")
        obj = render.get_object("SmoothCube_v2")
        model_matrix = obj.matrix_world
        view_matrix = bpy.context.scene.camera.matrix_world.inverted()
        projection_matrix = bpy.context.scene.camera.calc_matrix_camera(
            bpy.context.evaluated_depsgraph_get(),
            x=480,
            y=320)
        mvp_matrix = projection_matrix @ view_matrix @ model_matrix
        xmin, xmax, ymin, ymax, zmin, zmax = calculate_bounding_box(obj, mvp_matrix, True)
        self.assertEqual(True, xmin < xmax)
        self.assertEqual(True, ymin < ymax)
        self.assertEqual(True, zmin < zmax)

    def test_config(self):
        from config import Config
        import os
        config = Config.from_file('_config.json')
        self.assertEqual(True, os.path.exists('_config.json'))
        try:
            os.remove('_config.json')
        except:
            self.assertEqual(True, False)

    def test_image_tool(self):
        import os
        import json
        from utils import show_bounding_box_image
        if os.path.exists("output/scenes.json") and os.path.isfile("output/scenes.json"):
            with open("output/scenes.json", 'r') as f:
                data = json.load(f)
            if "scenes" in data and len(data["scenes"]) > 0:
                show_bounding_box_image("output", 0, True, "temp.png")
                self.assertEqual(True, os.path.exists("temp.png"))
                try:
                    os.remove("temp.png")
                except:
                    self.assertEqual(True, False)
            self.assertEqual(True, True)
        else:
            self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
