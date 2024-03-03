import os
import shutil
import sys
from contextlib import contextmanager

import numpy as np
from loguru import logger
from multiprocessing import current_process

if current_process().name == 'MainProcess':
    import bpy

from mathutils import Vector


@contextmanager
def redirect_stdout(enable: bool = True, target=os.devnull):
    if not enable:
        # If redirection is not enabled, do nothing and exit the context manager
        yield
        return

    original_stdout_fd = sys.stdout.fileno()

    def _redirect_stdout(to_target):
        sys.stdout.close()
        os.dup2(to_target.fileno(), original_stdout_fd)
        sys.stdout = os.fdopen(original_stdout_fd, 'w')

    with os.fdopen(os.dup(original_stdout_fd), 'w') as old_stdout:
        with open(target, 'w') as new_stdout:
            _redirect_stdout(to_target=new_stdout)
            try:
                yield
            finally:
                _redirect_stdout(to_target=old_stdout)


class Render:
    def __init__(self, scene_file, object_dir, material_dir, blender_bin=None, blender_log_suppress=False):
        self.blender_log_suppress = blender_log_suppress
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
                    self.loaded_objects.append(obj.name)
        elif type(obj_file) == int:
            obj_files = os.listdir(self.object_dir)
            obj_files = [os.path.join(self.object_dir, f) for f in obj_files if f.endswith(".blend")]
            if len(obj_files) <= obj_file:
                raise FileNotFoundError("Object file not found")
            object_blend = obj_files[obj_file]
            with bpy.data.libraries.load(object_blend, link=False) as (data_from, data_to):
                data_to.objects = [name for name in data_from.objects]
                # self.loaded_objects.extend(data_to.objects)
            for obj in data_to.objects:
                if obj is not None:
                    bpy.context.collection.objects.link(obj)
                    self.loaded_objects.append(obj.name)
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
                if obj is not None:
                    self.loaded_objects.append(obj.name)

    def apply_material(self, obj: str, mtl: str, color: tuple = None):
        if obj not in self.loaded_objects:
            raise ValueError("Object not found")
        if mtl not in self.loaded_materials:
            raise ValueError("Material not found")
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
        if obj not in self.loaded_objects:
            raise ValueError("Object not found")
        if len(location) != 3:
            raise ValueError("Invalid location")
        object_3d = bpy.data.objects[obj]
        object_3d.location = location

    def offset_object_location(self, obj: str, offset: tuple):
        if obj not in self.loaded_objects:
            raise ValueError("Object not found")
        if len(offset) != 3:
            raise ValueError("Invalid offset")
        object_3d = bpy.data.objects[obj]
        object_3d.location = np.add(object_3d.location, offset)

    def zoom_object(self, obj: str, zoom: float):
        if obj not in self.loaded_objects:
            raise ValueError("Object not found")
        object_3d = bpy.data.objects[obj]
        object_3d.scale = (zoom, zoom, zoom)

    def rotate_object(self, obj: str, rotation: tuple):
        if obj not in self.loaded_objects:
            raise ValueError("Object not found")
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
        self.loaded_objects = []

    @staticmethod
    def reset_scene():
        bpy.ops.wm.read_factory_settings()

    @staticmethod
    def set_render_args(engine="CYCLES", resolution=(480, 320), resolution_percentage=100, samples=128, use_gpu=True, use_adaptive_sampling=False):
        bpy.context.scene.render.engine = engine
        bpy.context.scene.render.resolution_x = resolution[0]
        bpy.context.scene.render.resolution_y = resolution[1]
        bpy.context.scene.render.resolution_percentage = resolution_percentage
        bpy.context.scene.cycles.use_adaptive_sampling = use_adaptive_sampling
        bpy.context.scene.cycles.samples = samples
        if use_gpu:
            bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
            bpy.context.preferences.addons['cycles'].preferences.get_devices()
            for device in bpy.context.preferences.addons['cycles'].preferences.devices:
                if device.type == 'CUDA':
                    device.use = True
            bpy.context.scene.cycles.device = "GPU"

    def render_scene(self):
        with redirect_stdout(self.blender_log_suppress):
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
    def get_object(name):
        return bpy.data.objects.get(name)

    @staticmethod
    def print_objects():
        logger.info("Objects in scene:")
        for obj in bpy.data.objects:
            logger.info("\t" + obj.name)

    @staticmethod
    def calculate_plane():
        # bpy.ops.mesh.primitive_plane_add(radius=5)
        # plane = bpy.context.object
        camera = bpy.data.objects["Camera"]
        # plane_normal = plane.data.vertices[0].normal
        plane_normal = Vector((0, 0, 1))
        cam_behind = camera.matrix_world.to_quaternion() @ Vector((0, 0, -1))
        cam_left = camera.matrix_world.to_quaternion() @ Vector((-1, 0, 0))
        cam_up = camera.matrix_world.to_quaternion() @ Vector((0, 1, 0))
        plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
        plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
        plane_up = cam_up.project(plane_normal).normalized()

        # bpy.data.objects.remove(plane)
        return {
            "behind": tuple(plane_behind),
            "front": tuple(-plane_behind),
            "left": tuple(plane_left),
            "right": tuple(-plane_left),
            "above": tuple(plane_up),
            "below": tuple(-plane_up)
        }
