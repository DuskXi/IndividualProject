import concurrent.futures
import math
import time

# import numba
import numpy as np
from mathutils import Vector
from loguru import logger
import gc


def projection(obj, mvp_matrix):
    vertices = [Vector((v.co.x, v.co.y, v.co.z, 1.0)) for v in obj.data.vertices]
    ndc_vertices = [mvp_matrix @ v for v in vertices]
    ndc_vertices = [(v / v.w) for v in ndc_vertices]
    return ndc_vertices


def get_bounding_box(ndc_vertices):
    xmin, xmax = min(ndc_vertices, key=lambda v: v.x).x, max(ndc_vertices, key=lambda v: v.x).x
    ymin, ymax = min(ndc_vertices, key=lambda v: v.y).y, max(ndc_vertices, key=lambda v: v.y).y
    zmin, zmax = min(ndc_vertices, key=lambda v: v.z).z, max(ndc_vertices, key=lambda v: v.z).z
    return xmin, xmax, ymin, ymax, zmin, zmax


def calculate_bounding_box(obj, mvp_matrix, reverseX=False, reverseY=False, reverseZ=False):
    ndc_vertices = projection(obj, mvp_matrix)
    xmin, xmax, ymin, ymax, zmin, zmax = get_bounding_box(ndc_vertices)
    if reverseY:
        ymin, ymax = -ymax, -ymin
    if reverseX:
        xmin, xmax = -xmax, -xmin
    if reverseZ:
        zmin, zmax = -zmax, -zmin
    return xmin, xmax, ymin, ymax, zmin, zmax


def calculate_bounding_box_dict(obj, mvp_matrix, reverseX=False, reverseY=False, reverseZ=False):
    xmin, xmax, ymin, ymax, zmin, zmax = calculate_bounding_box(obj, mvp_matrix, reverseX, reverseY, reverseZ)
    return {
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "zmin": zmin,
        "zmax": zmax
    }


def quad_to_triangles(quad):
    # 分割四边形为两个三角形
    return [(quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])]


def simple_rasterization_multiprocess(objects, mvp_matrices, sample_size=32, retry=3):
    # just use subprocess to call simple_rasterization

    for i in range(retry):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            task = executor.submit(simple_rasterization, objects, mvp_matrices, sample_size)

            try:
                result = task.result()
                return result
            except Exception as e:
                logger.warning(f"Retry {i + 1} failed")
                logger.error(e)
                continue

    return None


def simple_rasterization(objects, mvp_matrices, sample_size=32):
    # projection to ndc
    width = sample_size
    height = sample_size
    ndc_vertices = {}
    for obj, mvp_matrix in zip(objects, mvp_matrices):
        ndc_vertices[obj.name] = projection(obj, mvp_matrix)
        for v in ndc_vertices[obj.name]:
            v.x = (v.x + 1) * 0.5 * width
            v.y = (v.y + 1) * 0.5 * height

    zBuffer = np.zeros((height, width))
    zBuffer.fill(float('-inf'))
    frameBuffer = np.zeros((height, width, 1), dtype=np.int32)
    frameBuffer.fill(-1)
    for i, obj in enumerate(objects):
        mesh = obj.data
        triangles = []
        for face in mesh.polygons:
            vertex_indices = face.vertices[:]
            if len(vertex_indices) == 3:
                triangles.append((*[np.array(ndc_vertices[obj.name][i], dtype=np.float64) for i in vertex_indices], i))
            elif len(vertex_indices) == 4:
                quads = [np.array(ndc_vertices[obj.name][i], dtype=np.float64) for i in vertex_indices]
                for triangle in quad_to_triangles(quads):
                    triangles.append((*triangle, i))
        try:
            draw_triangles(frameBuffer, zBuffer, triangles, width, height)
        finally:
            pass
    # statistics
    result = {}
    for i, obj in enumerate(objects):
        xmin, xmax, ymin, ymax, zmin, zmax = calculate_bounding_box(obj, mvp_matrices[i])
        screen_box = np.array([(xmin * 0.5 + 0.5) * width, (xmax * 0.5 + 0.5) * width, (ymin * 0.5 + 0.5) * height, (ymax * 0.5 + 0.5) * height], dtype=np.int32)
        # count pixels in screen_box that value equals i
        area = frameBuffer[screen_box[2]:screen_box[3], screen_box[0]:screen_box[1]]
        count_obj = np.sum(area == i)
        count_background = np.sum(area == -1)
        # count_background = np.sum(frameBuffer[screen_box[2]:screen_box[3], screen_box[0]:screen_box[1]] == -1)
        count_other = (screen_box[1] - screen_box[0]) * (screen_box[3] - screen_box[2]) - count_background - count_obj
        pixels_percentage = (count_obj / (count_obj + count_other)) if (count_obj + count_other) > 0 else 0
        result[obj.name] = pixels_percentage
    gc.collect()

    return result


def edge_function(v0, v1, p):
    return (p[0] - v0[0]) * (v1[1] - v0[1]) - (p[1] - v0[1]) * (v1[0] - v0[0])


def barycentric_numpy(a: np.ndarray, b: np.ndarray, c: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return u, v, w


def draw_triangles(frame, z_buffer, triangles, width, height):
    for triangle in triangles:
        v1, v2, v3, color = triangle
        if len(v1) < 3 or len(v1) != len(v2) or len(v1) != len(v3):
            continue
        min_x = max(int(min(v1[0], v2[0], v3[0])), 0)
        max_x = min(int(max(v1[0], v2[0], v3[0])), width - 1)
        min_y = max(int(min(v1[1], v2[1], v3[1])), 0)
        max_y = min(int(max(v1[1], v2[1], v3[1])), height - 1)

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                p = np.array([x, y])
                w0 = edge_function(v2, v3, p)
                w1 = edge_function(v3, v1, p)
                w2 = edge_function(v1, v2, p)
                if w0 >= 0 and w1 >= 0 and w2 >= 0:
                    u, v, w = barycentric_numpy(v1[:3], v2[:3], v3[:3], np.array([x, y, 0]))
                    z = v1[2] * u + v2[2] * v + v3[2] * w
                    if z > z_buffer[y, x]:
                        z_buffer[y, x] = z
                        frame[y, x] = color


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