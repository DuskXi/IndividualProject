import json
import os.path

from PIL import Image, ImageDraw
from matplotlib import pyplot as plt


def show_bounding_box_image(output_path, index):
    with open(os.path.join(output_path, 'scenes.json'), 'r') as f:
        data = json.load(f)
        scenes = data['scenes']
        scene = scenes[index]
        image_filename = scene['image_filename']
        image = Image.open(os.path.join(output_path, image_filename))
        width, height = image.size
        draw = ImageDraw.Draw(image)
        for obj in scene['objects']:
            x1 = (obj['bounding_box']['xmin'] + 1) * width / 2
            y1 = (obj['bounding_box']['ymin'] + 1) * height / 2
            x2 = (obj['bounding_box']['xmax'] + 1) * width / 2
            y2 = (obj['bounding_box']['ymax'] + 1) * height / 2
            draw.rectangle([x1, y1, x2, y2], outline='red')
        plt.imshow(image)
        plt.show()
