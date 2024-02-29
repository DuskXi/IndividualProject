import os
import sys
import traceback
import unittest
from loguru import logger


class DatasetGeneration(unittest.TestCase):
    def test_blender_path(self):
        import bpy
        import shutil

        blender_bin = shutil.which("blender")
        if blender_bin:
            logger.info("Found:", blender_bin)
            bpy.app.binary_path = blender_bin
        else:
            logger.info("Unable to find blender!")
        self.assertEqual(True, True)

    def test_load_blend(self):
        file = 'data/base_scene.blend'

        import bpy
        import shutil

        blender_bin = shutil.which("blender")
        if blender_bin:
            logger.info("Found:", blender_bin)
            bpy.app.binary_path = blender_bin
            logger.info("Blender binary loaded")
            bpy.ops.wm.open_mainfile(filepath=file)
        else:
            logger.info("Unable to find blender!")
        self.assertEqual(True, True)

    def test_a_simple_render_example(self):
        runtime_path = os.path.abspath(os.getcwd())
        scene_blend = 'data/base_scene.blend'
        object_blend = 'data/shapes/SmoothCube_v2.blend'
        mtl_blend = 'data/materials/Rubber.blend'
        scene_blend = os.path.abspath(scene_blend)
        object_blend = os.path.abspath(object_blend)
        mtl_blend = os.path.abspath(mtl_blend)

        import bpy
        import shutil

        try:
            blender_bin = shutil.which("blender")
            if blender_bin:
                logger.info("Found:", blender_bin)
                bpy.app.binary_path = blender_bin
                # set blender work dir to script dir
                # bpy.context.workspace = bpy.context.copy()
                # bpy.context.workspace.blend_data.filepath = runtime_path
                logger.info("Blender binary loaded")

                logger.info("Load sense files")

                # with bpy.data.libraries.load(scene_blend, link=False) as (data_from, data_to):
                #     data_to.scenes = [name for name in data_from.scenes]
                #
                # bpy.context.window.scene = bpy.data.scenes[0]

                # set resolution

                bpy.ops.wm.open_mainfile(filepath=scene_blend)
                bpy.context.scene.render.resolution_x = 480 * 2
                bpy.context.scene.render.resolution_y = 320 * 2

                logger.info("Load object and material files")

                #
                with bpy.data.libraries.load(object_blend, link=False) as (data_from, data_to):
                    data_to.objects = [name for name in data_from.objects]

                #
                for obj in data_to.objects:
                    if obj is not None:
                        bpy.context.collection.objects.link(obj)

                #
                with bpy.data.libraries.load(mtl_blend, link=False) as (data_from, data_to):
                    data_to.materials = [name for name in data_from.materials]

                logger.info("Apply material to object")

                #
                material = bpy.data.materials.get('BMD_Rubber_0004')
                if material:
                    obj = bpy.data.objects.get("SmoothCube_v2")
                    if obj:
                        # change coordinates
                        obj.location.z = 1
                        if obj.data.materials:
                            obj.data.materials[0] = material
                        else:
                            obj.data.materials.append(material)

                logger.info("Set render parameters")

                # set render parameters
                bpy.context.scene.render.engine = 'CYCLES'
                bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'

                bpy.context.preferences.addons['cycles'].preferences.get_devices()
                for device in bpy.context.preferences.addons['cycles'].preferences.devices:
                    if device.type == 'CUDA':
                        device.use = True

                bpy.context.scene.cycles.device = 'GPU'

                bpy.context.scene.render.image_settings.file_format = 'PNG'
                image_path = os.path.join(runtime_path, "output.png")
                bpy.context.scene.render.filepath = image_path

                logger.info("Render")

                # render
                bpy.ops.render.render(write_still=True)
            else:
                logger.info("Unable to find blender!")
            self.assertEqual(True, True)
        except Exception as e:
            traceback.print_exc()
            logger.error(e)
            self.assertEqual(True, False)

    def test_render(self):
        # test same scene with different object location, and without reinit whole blender bpy
        # this test is for optimize the render process different from the clevr-dataset-gen version
        runtime_path = os.path.abspath(os.getcwd())
        scene_blend = 'data/base_scene.blend'
        object_blend = 'data/shapes/SmoothCube_v2.blend'
        # mtl_blend = 'data/materials/Rubber.blend'

        mtl_name = 'BMD_Rubber_0004'
        obj_name = 'SmoothCube_v2'

        from gen import Render

        # import logging
        # logging.basicConfig(level=logging.WARNING)

        render = Render(scene_blend, 'data/shapes', 'data/materials')
        render.init_render()
        logger.info("Objects before loaded:")
        logger.debug(render.list_objects())
        render.set_render_args(resolution=(480 * 2, 320 * 2))
        # first render round
        render.load_object(os.path.abspath(object_blend))
        render.apply_material(obj_name, mtl_name)
        render.offset_object_location(obj_name, (0, 0, 1))
        image_path = os.path.join(runtime_path, "output-1.png")
        render.set_render_output(image_path)
        # display the scene
        # logger.info("Render")
        # logger.info("Loaded objects:")
        # logger.debug(render.loaded_objects)
        # logger.info("Loaded materials:")
        # logger.debug(render.loaded_materials)
        # logger.info("Blender objects:")
        # logger.debug(render.list_objects())
        # logger.info("Blender materials:")
        # logger.debug(render.list_materials())
        # render.print_objects()
        render.render_scene()
        logger.info(f"Objects before unloaded: {render.list_object_names()}")
        render.unload_objects()
        logger.info(f"Objects after unloaded: {render.list_object_names()}")

        logger.info("Start second round render:")
        render.get_material(mtl_name).diffuse_color = (1, 0, 0, 1)
        render.load_object(os.path.abspath(object_blend))
        render.apply_material(obj_name, mtl_name)
        render.offset_object_location(obj_name, (1, 1, 1))
        image_path = os.path.join(runtime_path, "output-2.png")
        render.set_render_output(image_path)
        render.render_scene()
        self.assertEqual(True, True)

    def test_duplicate_and_change_color_for_mtl(self):
        runtime_path = os.path.abspath(os.getcwd())
        scene_blend = 'data/base_scene.blend'
        object_blend = 'data/shapes/SmoothCube_v2.blend'
        # mtl_blend = 'data/materials/Rubber.blend'

        mtl_name = 'BMD_Rubber_0004'
        mtl_name = 'Material'
        obj_name = 'SmoothCube_v2'

        import bpy
        from gen import Render

        # import logging
        # logging.basicConfig(level=logging.WARNING)

        render = Render(scene_blend, 'data/shapes', 'data/materials')
        render.init_render()

        mtls = list(render.list_materials())
        material = render.get_material(mtl_name)
        material = material.copy()
        if material is not None and material.node_tree is not None:
            # Get the nodes in the material
            new_mat = material.copy()
            new_mat.name = 'n_mtl'

            nodes = new_mat.node_tree.nodes

            # Find the Principled BSDF node
            for node in nodes:
                if node.name in ['Rubber', 'Group']:
                    # ns = list(node.node_tree.nodes)
                    ns = list(node.inputs)
                    for n in ns:
                        if n.type == 'RGBA':
                            n.default_value = (1, 0, 0, 1)  # Red color
                    node.update()

            # bpy.data.materials.append(material)

            # Update the material
            # material.update()

        logger.info("Objects before loaded:")
        render.load_object(os.path.abspath(object_blend))
        render.apply_material(obj_name, 'n_mtl')
        render.offset_object_location(obj_name, (1, 1, 1))
        image_path = os.path.join(runtime_path, "output-3.png")
        render.set_render_args(samples=1024)
        render.set_render_output(image_path)
        render.render_scene()
        self.assertEqual(True, True)

    def test_auto_mtl_color(self):
        runtime_path = os.path.abspath(os.getcwd())
        scene_blend = 'data/base_scene.blend'
        object_blend = 'data/shapes/SmoothCube_v2.blend'
        # mtl_blend = 'data/materials/Rubber.blend'

        mtl_name = 'BMD_Rubber_0004'
        # mtl_name = 'Material'
        obj_name = 'SmoothCube_v2'

        import bpy
        from gen import Render

        # import logging
        # logging.basicConfig(level=logging.WARNING)

        render = Render(scene_blend, 'data/shapes', 'data/materials')
        render.init_render()

        # render.load_object(os.path.abspath(object_blend))
        render.load_object_auto(obj_name, 'box1')
        render.load_object_auto(obj_name, 'box2')
        render.apply_material('box1', mtl_name, (1, 0, 0, 1))
        render.apply_material('box2', mtl_name, (1, 1, 0, 1))
        render.offset_object_location('box1', (1, 1, 1))
        render.offset_object_location('box2', (-1, -1, 1))
        image_path = os.path.join(runtime_path, "output-4.png")
        render.set_render_args(samples=1024)
        render.set_render_output(image_path)
        render.render_scene()
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
