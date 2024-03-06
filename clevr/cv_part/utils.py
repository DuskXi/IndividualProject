import numpy as np
from matplotlib import pyplot as plt
import cv2


def display_image(image, y, label_map, dpi=100, draw_width=1080, title=""):
    src_image = image.permute(1, 2, 0).cpu().numpy()
    if src_image.mean() < 1.2:
        src_image = (src_image * 255).astype(np.uint8)
    # zoom to 1080p
    src_image = cv2.resize(src_image, (int((draw_width / src_image.shape[0]) * src_image.shape[1]), draw_width))
    d = []
    for box, label in zip(y["boxes"], y["labels"]):
        d.append({
            "box": box / image.shape[1] * draw_width,
            "text": label_map[label]
        })
    # opencv rgb -> bgr
    src_image = cv2.cvtColor(src_image, cv2.COLOR_RGB2BGR)
    draw_boxes(src_image, d)
    src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(10, 10), dpi=dpi)
    plt.title(title)
    plt.imshow(src_image)
    plt.show()


def draw_boxes(image: np.ndarray, data, color=(0, 255, 0), thickness=2):
    for d in data:
        box = d["box"]
        text = d.get("text", "")
        c = d.get("color", color)
        draw_box(image, box, text, c, thickness)
    return image


def draw_box(image: np.ndarray, box: np.ndarray, text: str = "", color=(0, 255, 0), thickness=5, font_scale_factor=0.7):
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # 根据图像的分辨率和提供的比例因子自适应调整字体大小
    img_height, img_width = image.shape[:2]
    font_scale = font_scale_factor * (img_width * img_height) ** 0.5 / 1000  # 调整字体大小

    # 计算文本的大小
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

    # 在文本下方绘制矩形背景
    cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)

    # 绘制文本
    cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

    # 绘制边界框
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    return image
