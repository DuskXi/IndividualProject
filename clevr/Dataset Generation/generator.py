import json
import os
import shutil
import threading

from multiprocessing import current_process
from multiprocessing.pool import ThreadPool

if current_process().name == 'MainProcess':
    import bpy
import numpy as np

from loguru import logger

from config import Config
from data_middleware import SceneObject, Scene
from geometry_tools import calculate_horizontal_max_radius, check_collision, simple_rasterization, calculate_bounding_box, calculate_bounding_box_dict, simple_rasterization_multiprocess
from render import Render
from tqdm.rich import trange, tqdm

from timer import Timer


class Generator:
    scene: Scene

    def __init__(self, config: Config):
        self.config = config
        self.render = Render(config.scene_blend_file, config.object_dir, config.material_dir, blender_log_suppress=config.blender_log_suppress)
        self.render.init_render()
        self.objects: list[SceneObject] = []

    def generate_scene(self, num_objects, shape_heights, shape_radius):
        shape_names = {"SmoothCube_v2": "cube", "SmoothCylinder": "cylinder", "Sphere": "sphere"}
        mtl_names = {"BMD_Rubber_0004": "rubber", "Material": "metal"}
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
                obj = SceneObject(obj_type, f"obj_{obj_type}_{i}", mtl, (*(np.array(colors[color]) / 255), 1), sizes[size], location, rotation, color, size, shape_names[obj_type], mtl_names[mtl])
                if not check_collision(objects, obj, shape_radius):
                    break
                if j >= 19:
                    return self.generate_scene(num_objects, shape_heights, shape_radius)
            objects.append(obj)
        self.objects = objects
        self.scene = Scene(objects)

    def render_scene(self, output_path="output.png", file_format="PNG"):
        timer = Timer().start()
        for obj in self.scene.objects:
            self.render.load_object_auto(obj.model_name, obj.name)
            self.render.apply_material(obj.name, obj.mtl_name, obj.color)
            self.render.set_object_location(obj.name, obj.location)
            self.render.rotate_object(obj.name, obj.rotation)
            self.render.zoom_object(obj.name, obj.scale)
        timer.stop().print("Apply scene to blender", level="debug").reset().start()

        bpy.context.view_layer.update()

        if self.scene.camera_location is not None:
            self.render.set_camera_location(self.scene.camera_location)
        if self.scene.camera_rotation is not None:
            self.render.set_camera_rotation(self.scene.camera_rotation)
        bpy.context.view_layer.update()
        timer.stop().print("Update blender and setup cam", level="debug").reset().start()
        objects = [bpy.data.objects[obj.name] for obj in self.scene.objects]
        mvp_matrices = [self.calculate_mvp(obj) for obj in objects]
        timer.stop().print("Prepare simple rasterization data", level="debug").reset().start()
        # object_reverse_occlusion_rate = simple_rasterization(objects, mvp_matrices, sample_size=32)
        pool = ThreadPool(processes=1)

        def execute_simple_rasterization(o, m, sample_size):
            timer_rasterization = Timer().start()
            result = simple_rasterization(o, m, sample_size)
            timer_rasterization.stop().print("Rasterization", level="debug").reset().start()
            return result

        async_result = pool.apply_async(execute_simple_rasterization, (objects, mvp_matrices, 32))
        # thread_rasterization = threading.Thread(target=simple_rasterization, args=(objects, mvp_matrices, 32))
        # timer.stop().print("Rasterization", level="debug").reset().start()

        self.render.set_render_output(output_path, file_format)

        self.render.render_scene()
        timer.stop().print("Run Render", level="debug").reset().start()
        bounding_boxes = [calculate_bounding_box_dict(obj, mvp_matrix, reverseY=True) for i, (obj, mvp_matrix) in enumerate(zip(objects, mvp_matrices))]
        async_result.wait()
        object_reverse_occlusion_rate = async_result.get()
        timer.stop().print("Calculate bounding boxes", level="debug").reset().start()

        return object_reverse_occlusion_rate, bounding_boxes

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

    @staticmethod
    def calculate_mvp(obj):
        model_matrix = obj.matrix_world
        view_matrix = bpy.context.scene.camera.matrix_world.inverted()
        projection_matrix = bpy.context.scene.camera.calc_matrix_camera(
            bpy.context.evaluated_depsgraph_get(),
            x=480,
            y=320)
        return projection_matrix @ view_matrix @ model_matrix

    @staticmethod
    def scene_to_clevr_json_dict(scene: Scene, directions, object_reverse_occlusion_rate):
        data = scene.to_dict()
        for i, obj in enumerate(data['objects']):
            logger.debug(
                f"Shape[{i}]: {obj['shape']}; Color: {obj['color_name']}; Material: {obj['material']}; Size: {obj['size_name']}; Pixel Occlusion Rate: {object_reverse_occlusion_rate[obj['name']] * 100:.2f}%")
        logger.debug("Relationship data:")
        relationships = Generator.compute_all_relationships(data, directions)
        data['relationships'] = relationships
        data['directions'] = directions
        return data

    @staticmethod
    def compute_all_relationships(scene_struct, directions, eps=0.2):
        all_relationships = {}
        for name, direction_vec in directions.items():
            if name == 'above' or name == 'below': continue
            all_relationships[name] = []
            for i, obj1 in enumerate(scene_struct['objects']):
                coords1 = obj1['location']
                related = set()
                for j, obj2 in enumerate(scene_struct['objects']):
                    if obj1 == obj2: continue
                    coords2 = obj2['location']
                    diff = [coords2[k] - coords1[k] for k in [0, 1, 2]]
                    dot = sum(diff[k] * direction_vec[k] for k in [0, 1, 2])
                    if dot > eps:
                        related.add(j)
                all_relationships[name].append(sorted(list(related)))
        return all_relationships

    def run(self):
        shape_heights = self.calculate_all_shape_height(self.render)
        shape_radius = self.calculate_shape_radius(self.render)
        timer = Timer()
        scenes = {
            "scenes": []
        }
        if os.path.exists(os.path.join(self.config.output_dir, "scenes.json")):
            try:
                with open(os.path.join(self.config.output_dir, "scenes.json"), "r") as f:
                    scenes = json.load(f)
                    if "scenes" not in scenes:
                        raise json.JSONDecodeError
            except json.JSONDecodeError:
                logger.warning("Failed to load scenes.json")
                shutil.copy(os.path.join(self.config.output_dir, "scenes_bak.json"), os.path.join(self.config.output_dir, "scenes.json"))
                with open(os.path.join(self.config.output_dir, "scenes.json"), "r") as f:
                    scenes = json.load(f)
        start_index = len(scenes["scenes"])
        for i in tqdm(range(start_index, self.config.num_images), initial=start_index, total=self.config.num_images, desc="Rendering"):
            self.render.set_render_args(self.config.engine, self.config.resolution, self.config.resolution_percentage, self.config.samples, self.config.use_gpu, self.config.use_adaptive_sampling)
            timer.reset().start()
            self.generate_scene(self.config.num_objects, shape_heights, shape_radius)
            timer.stop().print("Generate Scene").reset().start()
            object_reverse_occlusion_rate, bounding_boxes = self.render_scene(os.path.abspath(os.path.join(self.config.output_dir, f"output_{i}.png")))
            timer.stop().print("Render").reset().start()
            directions = self.render.calculate_plane()
            scene_dict = self.scene_to_clevr_json_dict(self.scene, directions, object_reverse_occlusion_rate)
            scene_dict['image_index'] = i
            scene_dict['image_filename'] = f"output_{i}.png"
            scene_dict['object_reverse_occlusion_rate'] = object_reverse_occlusion_rate if object_reverse_occlusion_rate is not None else {obj['name']: 0 for obj in scene_dict['objects']}
            for j in range(len(scene_dict['objects'])):
                scene_dict['objects'][j]['bounding_box'] = bounding_boxes[j]
            scenes["scenes"].append(scene_dict)
            self.render.unload_objects()
            if i % 10 == 0:
                logger.info(f"Resetting blender scene after {i} images rendered")
                self.render.reset_scene()
                self.render = Render(self.config.scene_blend_file, self.config.object_dir, self.config.material_dir, blender_log_suppress=self.config.blender_log_suppress)
                self.render.init_render()
                # copy scenes to bak
                if os.path.exists(os.path.join(self.config.output_dir, "scenes.json")):
                    shutil.copy(os.path.join(self.config.output_dir, "scenes.json"), os.path.join(self.config.output_dir, "scenes_bak.json"))
                with open(os.path.join(self.config.output_dir, "scenes.json"), "w") as f:
                    json_str = json.dumps(scenes, indent=4)
                    f.write(json_str)
        logger.info(f"Rendered {self.config.num_images} images")
        with open(os.path.join(self.config.output_dir, "scenes.json"), "w") as f:
            json_str = json.dumps(scenes, indent=4)
            f.write(json_str)
        logger.info(f"Saved scenes to {os.path.join(self.config.output_dir, 'scenes.json')}")


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
