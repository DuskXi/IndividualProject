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
        sense_blend = 'data/base_scene.blend'
        object_blend = 'data/shapes/SmoothCube_v2.blend'
        mtl_blend = 'data/materials/Rubber.blend'
        sense_blend = os.path.abspath(sense_blend)
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

                # with bpy.data.libraries.load(sense_blend, link=False) as (data_from, data_to):
                #     data_to.scenes = [name for name in data_from.scenes]
                #
                # bpy.context.window.scene = bpy.data.scenes[0]

                # set resolution

                bpy.ops.wm.open_mainfile(filepath=sense_blend)
                bpy.context.scene.render.resolution_x = 512
                bpy.context.scene.render.resolution_y = 512

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


if __name__ == '__main__':
    unittest.main()
