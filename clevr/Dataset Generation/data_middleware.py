from geometry_tools import check_collision


class SceneObject(dict):
    model_name: str
    name: str
    mtl_name: str
    color: tuple
    scale: float
    location: tuple
    rotation: tuple
    color_name: str
    size_name: str
    shape: str
    material: str

    def __init__(self, model_name: str, name: str, mtl_name: str, color: tuple, scale: float = 1.0, location: tuple = (0, 0, 0), rotation: tuple = (0, 0, 0), color_name: str = None,
                 size_name: str = None, shape: str = None, material: str = None):
        data_dict = {
            "model_name": model_name,
            "name": name,
            "mtl_name": mtl_name,
            "color": color,
            "scale": scale,
            "location": location,
            "rotation": rotation,
            "color_name": color_name,
            "size_name": size_name,
            "shape": shape,
            "material": material
        }
        super().__init__(data_dict)
        for k, v in data_dict.items():
            setattr(self, k, v)

    def to_dict(self):
        return {
            "model_name": self.model_name,
            "name": self.name,
            "mtl_name": self.mtl_name,
            "color": self.color,
            "scale": self.scale,
            "location": self.location,
            "rotation": self.rotation,
            "color_name": self.color_name,
            "size_name": self.size_name,
            "shape": self.shape,
            "material": self.material
        }


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

    def to_dict(self):
        return {
            "objects": [obj.to_dict() for obj in self.objects],
            "camera_location": self.camera_location,
            "camera_rotation": self.camera_rotation
        }
