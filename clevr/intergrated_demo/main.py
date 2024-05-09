import gradio as gr
from PIL import Image
from loguru import logger
from torchvision.transforms import transforms

from model import ModelLoader
import torch
import pandas as pd

from utils import display_image


def attributes_to_label(attributes):
    return f"{attributes['size']}_{attributes['color']}_{attributes['material']}_{attributes['shape']}"


def get_labels_class():
    sizes = ["small", "large"]
    colors = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
    materials = ["rubber", "metal"]
    shapes = ["cube", "cylinder", "sphere"]
    labels = []
    for size in sizes:
        for color in colors:
            for material in materials:
                for shape in shapes:
                    labels.append(attributes_to_label({
                        "size": size,
                        "color": color,
                        "material": material,
                        "shape": shape
                    }))
    return labels


def post_process_objects(boxes, width, height):
    data = []
    for i, box in enumerate(boxes):
        if i == 5:
            break
        x1, y1, x2, y2 = box.cpu().detach().numpy()
        x1 = (x1 / width) * 2 - 1
        x2 = (x2 / width) * 2 - 1
        y1 = (y1 / height) * 2 - 1
        y2 = (y2 / height) * 2 - 1
        data.append([x1, x2, y1, y2])
    return data


transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])


def predict(object_detector, relationship_detector, image, labels_class):
    image = Image.open(image).convert('RGB')
    image = transform(image).unsqueeze(0).cuda()
    result = object_detector(image)
    boxes = result[0]["boxes"]
    labels = [labels_class[x] for x in result[0]["labels"]]
    data = post_process_objects(boxes, image.shape[2], image.shape[3])
    data = torch.tensor(data, dtype=torch.float32).cuda().unsqueeze(0)
    relationship = relationship_detector(data).squeeze(0)
    result_binary = torch.where(relationship >= 0.5, torch.tensor(1.0), torch.tensor(0.0))
    relationship = result_binary.detach().cpu().numpy()
    dfs = []
    for i, name in enumerate(["left", "right", "front", "behind"]):
        data = relationship[i]
        df = pd.DataFrame(data, columns=labels)
        # copy index to column at first
        df["index"] = labels
        # move index to the first column, not setting index
        df = df[["index"] + [col for col in df.columns if col != 'index']]
        # remove
        dfs.append(df)

    draw_image = display_image(image[0], result[0], labels_class, headless=True)
    return draw_image, dfs[1], dfs[0], dfs[3], dfs[2]


def main():
    print("init")
    relationship_path = r"D:\projects\IndividualProject\clevr\location_part\model\model_40_acc_97.pth"
    fasterrcnn_path = r"D:\projects\IndividualProject\clevr\cv_part\model_4_acc_98.pth"

    logger.info("Loading models")
    relationship_model = ModelLoader.load_relationship(relationship_path, 5)
    fasterrcnn_model = ModelLoader.load_faster_rcnn(fasterrcnn_path)
    relationship_model.cuda()
    relationship_model.eval()
    fasterrcnn_model.cuda()
    fasterrcnn_model.eval()

    logger.info("Launching UI")

    with gr.Blocks() as blocks:
        image_in = gr.File(label="Image")
        image_out = gr.Plot(label="Image")
        # display dataframe index
        dataframe_out_left = gr.Dataframe(label="Dataframe Left")
        dataframe_out_right = gr.Dataframe(label="Dataframe Right")
        dataframe_out_front = gr.Dataframe(label="Dataframe Front")
        dataframe_out_behind = gr.Dataframe(label="Dataframe Behind")
        image_in.upload(fn=lambda x: predict(fasterrcnn_model, relationship_model, x, get_labels_class()), inputs=[image_in], outputs=[image_out, dataframe_out_left, dataframe_out_right, dataframe_out_front, dataframe_out_behind])

    blocks.launch()


if __name__ == "__main__":
    main()
