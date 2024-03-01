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
        self.assertEqual(True, True)  # add assertion here


if __name__ == '__main__':
    unittest.main()
