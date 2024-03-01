import math

from mathutils import Vector


def projection(obj, mvp_matrix):
    vertices = [Vector((v.co.x, v.co.y, v.co.z, 1.0)) for v in obj.data.vertices]
    ndc_vertices = [mvp_matrix @ v for v in vertices]
    ndc_vertices = [(v / v.w) for v in ndc_vertices]
    return ndc_vertices


def calculate_bounding_box(obj, mvp_matrix, reserve=False):
    ndc_vertices = projection(obj, mvp_matrix)
    xmin, xmax = min(ndc_vertices, key=lambda v: v.x).x, max(ndc_vertices, key=lambda v: v.x).x
    ymin, ymax = min(ndc_vertices, key=lambda v: v.y).y, max(ndc_vertices, key=lambda v: v.y).y
    zmin, zmax = min(ndc_vertices, key=lambda v: v.z).z, max(ndc_vertices, key=lambda v: v.z).z
    if reserve:
        return xmax * -1, xmin * -1, ymax * -1, ymin * -1, zmin, zmax
    return xmin, xmax, ymin, ymax, zmin, zmax


# TODO: rasterization

def calculate_horizontal_max_radius(obj):
    if obj and hasattr(obj, 'bound_box'):
        bound_box = obj.bound_box
        max_dist = 0

        for i in range(8):
            for j in range(i + 1, 8):
                dist_x = abs(bound_box[i][0] - bound_box[j][0])
                dist_y = abs(bound_box[i][1] - bound_box[j][1])
                dist = math.sqrt(dist_x ** 2 + dist_y ** 2)
                if dist > max_dist:
                    max_dist = dist

        max_radius = max_dist / 2
        return max_radius
    else:
        return None


def check_collision(objects, target, shape_radius: dict, tolerance=1.25):
    for obj in objects:
        if obj.name != target.name:
            distance = math.sqrt((obj.location[0] - target.location[0]) ** 2 + (obj.location[1] - target.location[1]) ** 2)
            if distance + tolerance < shape_radius[obj.model_name] + shape_radius[target.model_name]:
                return True
    return False
