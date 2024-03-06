import json
import os.path

from PIL import Image, ImageDraw
from matplotlib import pyplot as plt


def show_bounding_box_image(output_path, index, headless=False, headless_target_path='temp.png'):
    with open(os.path.join(output_path, 'scenes.json'), 'r') as f:
        data = json.load(f)
        scenes = data['scenes']
        scene = scenes[index]
        image_filename = scene['image_filename']
        object_reverse_occlusion_rate = scene['object_reverse_occlusion_rate']
        image = Image.open(os.path.join(output_path, image_filename))
        width, height = image.size
        draw = ImageDraw.Draw(image)
        for obj in scene['objects']:
            x1 = (obj['bounding_box']['xmin'] + 1) * width / 2
            y1 = (obj['bounding_box']['ymin'] + 1) * height / 2
            x2 = (obj['bounding_box']['xmax'] + 1) * width / 2
            y2 = (obj['bounding_box']['ymax'] + 1) * height / 2
            draw.rectangle([x1, y1, x2, y2], outline='red')
            pixel_percentage = object_reverse_occlusion_rate[obj['name']]
            rect_text = f"{pixel_percentage * 100:.1f}%"
            draw.text((x1, y1), rect_text, fill='white', align='bottom')
        plt.imshow(image)
        if not headless:
            plt.show()
        else:
            plt.savefig(headless_target_path)


def scene_to_clevr_scene(src, dest, encoding='utf-8'):
    with open(src, "r", encoding=encoding) as f:
        data = json.load(f)
    scenes = data['scenes']
    for scene in scenes:
        for obj in scene['objects']:
            obj['_color'] = obj['color']
            obj['color'] = obj['color_name']
            obj['size'] = obj['size_name']
            obj['3d_coords'] = obj['location']

    with open(dest, "w", encoding=encoding) as f:
        json.dump(data, f)
