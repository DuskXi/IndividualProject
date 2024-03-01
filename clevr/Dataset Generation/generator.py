import os

import numpy as np

from loguru import logger

from config import Config
from data_middleware import SceneObject, Scene
from geometry_tools import calculate_horizontal_max_radius, check_collision
from render import Render
from tqdm.rich import trange, tqdm


class Generator:
    scene: Scene

    def __init__(self, config: Config):
        self.config = config
        self.render = Render(config.scene_blend_file, config.object_dir, config.material_dir, blender_log_suppress=config.blender_log_suppress)
        self.render.init_render()
        self.objects: list[SceneObject] = []

    def generate_scene(self, num_objects, shape_heights, shape_radius):
        list_object_type = ["SmoothCube_v2", "SmoothCylinder", "Sphere"]
        list_material = ["BMD_Rubber_0004", "Material"]
        colors = {
            "gray": [87, 87, 87], "red": [173, 35, 35], "blue": [42, 75, 215],
            "green": [29, 105, 20], "brown": [129, 74, 25], "purple": [129, 38, 192],
            "cyan": [41, 208, 208], "yellow": [255, 238, 51]
        }
        sizes = {"large": 0.7, "small": 0.35}
        horizontal_offset_range = (-3, 3)
        objects: list[SceneObject] = []
        for i in range(num_objects):
            for j in range(20):
                obj_type = np.random.choice(list_object_type)
                mtl = np.random.choice(list_material)
                color = np.random.choice(list(colors.keys()))
                size = np.random.choice(list(sizes.keys()))
                location = (np.random.uniform(*horizontal_offset_range), np.random.uniform(*horizontal_offset_range), (shape_heights[obj_type] * sizes[size]) / 2)
                rotation = (0, 0, np.random.uniform(0, 360))
                obj = SceneObject(obj_type, f"obj_{obj_type}_{i}", mtl, (*(np.array(colors[color]) / 255), 1), sizes[size], location, rotation)
                if not check_collision(objects, obj, shape_radius):
                    break
                if j >= 19:
                    return self.generate_scene(num_objects, shape_heights, shape_radius)
            objects.append(obj)
        self.objects = objects
        self.scene = Scene(objects)

    def render_scene(self, output_path="output.png", file_format="PNG"):
        for obj in self.scene.objects:
            self.render.load_object_auto(obj.model_name, obj.name)
            self.render.apply_material(obj.name, obj.mtl_name, obj.color)
            self.render.set_object_location(obj.name, obj.location)
            self.render.rotate_object(obj.name, obj.rotation)
            self.render.zoom_object(obj.name, obj.scale)
            # bpy.context.view_layer.update()

        if self.scene.camera_location is not None:
            self.render.set_camera_location(self.scene.camera_location)
        if self.scene.camera_rotation is not None:
            self.render.set_camera_rotation(self.scene.camera_rotation)
        self.render.set_render_output(output_path, file_format)

        self.render.render_scene()

    @staticmethod
    def calculate_all_shape_height(render: Render, list_object_type=None):
        if list_object_type is None:
            list_object_type = ["SmoothCube_v2", "SmoothCylinder", "Sphere"]

        result = {}
        for obj_type in list_object_type:
            render.load_object_auto(obj_type, obj_type)
            obj = render.get_object(obj_type)
            result[obj_type] = obj.dimensions.z
            render.unload_objects()
        return result

    @staticmethod
    def calculate_shape_radius(render: Render, list_object_type=None):
        if list_object_type is None:
            list_object_type = ["SmoothCube_v2", "SmoothCylinder", "Sphere"]

        result = {}
        for obj_type in list_object_type:
            render.load_object_auto(obj_type, obj_type)
            obj = render.get_object(obj_type)
            result[obj_type] = calculate_horizontal_max_radius(obj)
            render.unload_objects()

        return result

    def run(self):
        shape_heights = self.calculate_all_shape_height(self.render)
        shape_radius = self.calculate_shape_radius(self.render)
        self.render.set_render_args(self.config.engine, self.config.resolution, self.config.resolution_percentage, self.config.samples, self.config.use_gpu, self.config.use_adaptive_sampling)
        for i in tqdm(range(self.config.num_images), desc="Rendering"):
            self.generate_scene(self.config.num_objects, shape_heights, shape_radius)
            self.render_scene(os.path.abspath(os.path.join(self.config.output_dir, f"output_{i}.png")))
            self.render.unload_objects()
        logger.info(f"Rendered {self.config.num_images} images")


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


def random_scene(num_objects, shape_heights):
    list_object_type = ["SmoothCube_v2", "SmoothCylinder", "Sphere"]
    list_material = ["BMD_Rubber_0004", "Material"]
    colors = {
        "gray": [87, 87, 87], "red": [173, 35, 35], "blue": [42, 75, 215],
        "green": [29, 105, 20], "brown": [129, 74, 25], "purple": [129, 38, 192],
        "cyan": [41, 208, 208], "yellow": [255, 238, 51]
    }
    sizes = {"large": 0.7, "small": 0.35}
    horizontal_offset_range = (-3, 3)
    objects: list[SceneObject] = []
    for i in range(num_objects):
        obj_type = np.random.choice(list_object_type)
        mtl = np.random.choice(list_material)
        color = np.random.choice(list(colors.keys()))
        size = np.random.choice(list(sizes.keys()))
        location = (np.random.uniform(*horizontal_offset_range), np.random.uniform(*horizontal_offset_range), (shape_heights[obj_type] * sizes[size]) / 2)
        rotation = (0, 0, np.random.uniform(0, 360))
        objects.append(SceneObject(obj_type, f"obj_{obj_type}_{i}", mtl, (*(np.array(colors[color]) / 255), 1), sizes[size], location, rotation))

    return Scene(objects)
