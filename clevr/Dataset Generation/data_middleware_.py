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
