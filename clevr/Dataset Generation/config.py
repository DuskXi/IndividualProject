import json


class Config(dict):
    properties_file: str = 'data/properties.json'
    output_dir: str = 'output'

    scene_blend_file: str = 'data/base_scene.blend'
    object_dir: str = 'data/shapes'
    material_dir: str = 'data/materials'

    # render options
    engine: str = 'CYCLES'
    resolution: list = [480, 320]
    resolution_percentage: int = 100
    samples: int = 256
    use_gpu: bool = True
    use_adaptive_sampling: bool = False

    # generation options
    num_objects: int = 5
    num_images: int = 10

    def __init__(self, **kwargs):
        data_dict = {
            'properties_file': kwargs.get('properties_file', self.properties_file),
            'output_dir': kwargs.get('output_dir', self.output_dir),
            'scene_blend_file': kwargs.get('scene_blend_file', self.scene_blend_file),
            'object_dir': kwargs.get('object_dir', self.object_dir),
            'material_dir': kwargs.get('material_dir', self.material_dir),

            'engine': kwargs.get('engine', self.engine),
            'resolution': kwargs.get('resolution', self.resolution),
            'resolution_percentage': kwargs.get('resolution_percentage', self.resolution_percentage),
            'samples': kwargs.get('samples', self.samples),
            'use_gpu': kwargs.get('use_gpu', self.use_gpu),
            'use_adaptive_sampling': kwargs.get('use_adaptive_sampling', self.use_adaptive_sampling),

            'num_objects': kwargs.get('num_objects', self.num_objects),
            'num_images': kwargs.get('num_images', self.num_images),
        }
        super().__init__(data_dict)
        for k, v in data_dict.items():
            setattr(self, k, v)

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self, f, indent=4)
        return self

    def to_dict(self):
        return dict(self)

    def to_json(self):
        return json.dumps(self)

    @staticmethod
    def from_json(s):
        return Config(**json.loads(s))
