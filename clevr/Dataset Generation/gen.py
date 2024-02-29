import os

import bpy
import shutil

import numpy as np

from loguru import logger
from rich.logging import RichHandler

logger.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


class Render:
    def __init__(self, scene_file, object_dir, material_dir, blender_bin=None):
        if blender_bin:
            self.blender_bin = blender_bin
        else:
            self.blender_bin = shutil.which("blender")
        if self.blender_bin:
            logger.info("Found:", self.blender_bin)
            bpy.app.binary_path = self.blender_bin
            logger.info("Blender binary loaded")
        else:
            logger.info("Unable to find blender!")
            raise FileNotFoundError("Blender not found")

        if not os.path.isfile(scene_file):
            raise FileNotFoundError("Scene file not found")
        if not os.path.isdir(object_dir):
            raise FileNotFoundError("Object directory not found")
        if not os.path.isdir(material_dir):
            raise FileNotFoundError("Material directory not found")

        self.scene_file = os.path.abspath(scene_file)
        self.object_dir = os.path.abspath(object_dir)
        self.material_dir = os.path.abspath(material_dir)
        self.loaded_objects = []
        self.loaded_materials = []

    def init_render(self):
        self.load_scene()
        self.load_mtl()

    def load_scene(self):
        bpy.ops.wm.open_mainfile(filepath=self.scene_file)

    def load_mtl(self):
        material_files = os.listdir(self.material_dir)
        material_files = [os.path.join(self.material_dir, f) for f in material_files if f.endswith(".blend")]
        for mtl_file in material_files:
            with bpy.data.libraries.load(mtl_file, link=False) as (data_from, data_to):
                data_to.materials = [name for name in data_from.materials]
                self.loaded_materials.extend(data_to.materials)

    def load_object(self, obj_file):
        if type(obj_file) == str:
            with bpy.data.libraries.load(obj_file, link=False) as (data_from, data_to):
                data_to.objects = [name for name in data_from.objects]
                self.loaded_objects.extend(data_to.objects)
            for obj in data_to.objects:
                if obj is not None:
                    bpy.context.collection.objects.link(obj)
        elif type(obj_file) == int:
            obj_files = os.listdir(self.object_dir)
            obj_files = [os.path.join(self.object_dir, f) for f in obj_files if f.endswith(".blend")]
            if len(obj_files) <= obj_file:
                raise FileNotFoundError("Object file not found")
            object_blend = obj_files[obj_file]
            with bpy.data.libraries.load(object_blend, link=False) as (data_from, data_to):
                data_to.objects = [name for name in data_from.objects]
                self.loaded_objects.extend(data_to.objects)
            for obj in data_to.objects:
                if obj is not None:
                    bpy.context.collection.objects.link(obj)
        else:
            raise ValueError("Invalid object file")

    def load_object_auto(self, target_name, new_name):
        obj_files = os.listdir(self.object_dir)
        obj_files = [os.path.join(self.object_dir, f) for f in obj_files if f.endswith(".blend")]
        for obj_file in obj_files:
            with bpy.data.libraries.load(obj_file, link=False) as (data_from, data_to):
                data_to.objects = [name for name in data_from.objects]
            for obj in data_to.objects:
                if obj is not None and obj.name.startswith(target_name):
                    bpy.context.collection.objects.link(obj)
                    obj.name = new_name
            self.loaded_objects.extend(data_to.objects)

    def apply_material(self, obj: str, mtl: str, color: tuple = None):
        # if obj not in self.loaded_objects:
        #     raise ValueError("Object not found")
        # if mtl not in self.loaded_materials:
        #     raise ValueError("Material not found")
        material = self.get_mtl(mtl, color)
        object_3d = bpy.data.objects[obj]
        if object_3d.data.materials:
            object_3d.data.materials[0] = material
        else:
            object_3d.data.materials.append(material)

    @staticmethod
    def rgba_to_str(color: tuple):
        return f"rgba_{color[0]}_{color[1]}_{color[2]}_{color[3]})"

    def get_mtl(self, mtl: str, color: tuple = None):
        if color is None:
            return bpy.data.materials[mtl]
        else:
            material = bpy.data.materials[mtl].copy()
            material.name = f"{material.name}_{self.rgba_to_str(color)}"
            nodes = material.node_tree.nodes
            for node in nodes:
                if node.name in ['Rubber', 'Group']:
                    ns = list(node.inputs)
                    for n in ns:
                        if n.type == 'RGBA':
                            n.default_value = color
                    node.update()
            return material

    def set_object_location(self, obj: str, location: tuple):
        # if obj not in self.loaded_objects:
        #     raise ValueError("Object not found")
        if len(location) != 3:
            raise ValueError("Invalid location")
        object_3d = bpy.data.objects[obj]
        object_3d.location = location

    def offset_object_location(self, obj: str, offset: tuple):
        # if obj not in self.loaded_objects:
        #     raise ValueError("Object not found")
        if len(offset) != 3:
            raise ValueError("Invalid offset")
        object_3d = bpy.data.objects[obj]
        object_3d.location = np.add(object_3d.location, offset)

    def zoom_object(self, obj: str, zoom: float):
        # if obj not in self.loaded_objects:
        #     raise ValueError("Object not found")
        object_3d = bpy.data.objects[obj]
        object_3d.scale = (zoom, zoom, zoom)

    def rotate_object(self, obj: str, rotation: tuple):
        # if obj not in self.loaded_objects:
        #     raise ValueError("Object not found")
        if len(rotation) != 3:
            raise ValueError("Invalid rotation")
        object_3d = bpy.data.objects[obj]
        object_3d.rotation_euler = rotation

    def set_camera_location(self, location: tuple):
        if len(location) != 3:
            raise ValueError("Invalid location")
        bpy.data.objects["Camera"].location = location

    def move_camera(self, offset: tuple):
        if len(offset) != 3:
            raise ValueError("Invalid offset")
        camera = bpy.data.objects["Camera"]
        camera.location = np.add(camera.location, offset)

    def set_camera_rotation(self, rotation: tuple):
        if len(rotation) != 3:
            raise ValueError("Invalid rotation")
        camera = bpy.data.objects["Camera"]
        camera.rotation_euler = rotation

    def unload_objects(self):
        for obj in self.loaded_objects:
            bpy.data.objects.remove(bpy.data.objects[obj])

    @staticmethod
    def reset_scene():
        bpy.ops.wm.read_factory_settings()

    @staticmethod
    def set_render_args(engine="CYCLES", resolution=(480 * 2, 320 * 2), samples=128, use_gpu=True):
        bpy.context.scene.render.engine = engine
        bpy.context.scene.render.resolution_x = resolution[0]
        bpy.context.scene.render.resolution_y = resolution[1]
        bpy.context.scene.cycles.samples = samples
        if use_gpu:
            bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
            bpy.context.preferences.addons['cycles'].preferences.get_devices()
            for device in bpy.context.preferences.addons['cycles'].preferences.devices:
                if device.type == 'CUDA':
                    device.use = True
            bpy.context.scene.cycles.device = "GPU"

    @staticmethod
    def render_scene():
        bpy.ops.render.render(write_still=True)

    @staticmethod
    def set_render_output(output_path="output.png", file_format="PNG"):
        bpy.context.scene.render.image_settings.file_format = file_format
        bpy.context.scene.render.filepath = output_path

    @staticmethod
    def list_objects():
        return bpy.data.objects

    @staticmethod
    def list_object_names():
        return [obj.name for obj in bpy.data.objects]

    @staticmethod
    def list_materials():
        return bpy.data.materials

    @staticmethod
    def get_material(name):
        return bpy.data.materials.get(name)

    @staticmethod
    def print_objects():
        logger.info("Objects in scene:")
        for obj in bpy.data.objects:
            logger.info("\t" + obj.name)


class SceneObject(dict):
    model_name: str
    name: str
    mtl_name: str
    color: tuple
    scale: float
    location: tuple
    rotation: tuple

    def __init__(self, model_name: str, name: str, mtl_name: str, color: tuple, scale: float = 1.0, location: tuple = (0, 0, 0), rotation: tuple = (0, 0, 0)):
        data_dict = {
            "model_name": model_name,
            "name": name,
            "mtl_name": mtl_name,
            "color": color,
            "scale": scale,
            "location": location,
            "rotation": rotation
        }
        super().__init__(data_dict)
        for k, v in data_dict.items():
            setattr(self, k, v)


class Scene(dict):
    objects: list[SceneObject]
    camera_location: tuple
    camera_rotation: tuple

    def __init__(self, objects: list[SceneObject], camera_location: tuple = None, camera_rotation: tuple = None):
        data_dict = {
            "objects": objects,
            "camera_location": camera_location,
            "camera_rotation": camera_rotation
        }
        super().__init__(data_dict)
        for k, v in data_dict.items():
            setattr(self, k, v)


def render_scene(scene: Scene, render: Render, output_path="output.png", file_format="PNG"):
    for obj in scene.objects:
        render.load_object_auto(obj.model_name, obj.name)
        render.apply_material(obj.name, obj.mtl_name, obj.color)
        render.set_object_location(obj.name, obj.location)
        render.rotate_object(obj.name, obj.rotation)
        render.zoom_object(obj.name, obj.scale)

    if scene.camera_location is not None:
        render.set_camera_location(scene.camera_location)
    if scene.camera_rotation is not None:
        render.set_camera_rotation(scene.camera_rotation)
    render.set_render_output(output_path, file_format)

    render.render_scene()
