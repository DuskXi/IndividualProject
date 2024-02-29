# Log for Individual Project Recording (CLEVR part)

### 2024-02-20:

---

Implement the neural network structure of the CV part and the trainer

---

### 2024-02-21:

---

It was found that in order to improve the training of the object detection model, it is still necessary to know the range of the model in the picture, that is, the bounding box. Because most object detection models have RPN (Region Proposal Network), training a separate image->CV neural network of object attribute information on the Clevr data set still cannot avoid using bounding box data.

---

Obtaining the bounding box of an object in the Clevr dataset does not require manual labeling. You only need to obtain the rendering information, write a python program to perform the entire transformation process, from model transformation, view transformation, perspective projection transformation, and then calculate the bounding box of the object based on the position of the model vertices in the NDC space. This is feasible. Unfortunately, the scene files included in the Clevr dataset lose a lot of information, including model scaling and camera position, which is difficult to reconstruct. Therefore, the focus of work needs to shift to modifying the generation code, generating the data set by oneself, and outputting the corresponding bounding box data.

---


### 2024-02-22:

---

I encountered some troubles. The image_generation code given by the clevr-dataset-gen project is problematic. It can only support blender versions before 2.8 because the blender API has changed a lot since 2.8.

I tried another code that tried to upgrade the open source repository of image_generation, but still failed. Although these codes can be run, they cannot adapt to advanced NVIDIA GPUs. Even if CUDA8/9/10 is installed, lower versions of blender still cannot call the graphics card. Acceleration without using graphics card acceleration to generate tens of thousands of images is unacceptable.

---

In this case, we can only turn our goal to migrating and refactoring the image generation code to blender 3.6.

---

### **before this line is the recording that I have done before the date of start recording**

---

### 2024-02-28:

---
#### Start Recording

---

#### Plan Created
Plan Name[0]: clevr dataset generation code migration work
Plan created[0]: rewrite the clevr generate code from blender < 2.8 to blender 3.6
Consider[0]: due to the code (`facebookresearch/clevr-dataset-gen/blob/main/image_generation/render_images.py`) only supprot blender < 2.8, and blender < 2.8 is to old, cannot support cuda >= 11.0. And old cuda (such as 8.0 or 9.0 is not stable for RTX 40X0). So I need to rewrite the code to support blender 3.6

---

#### Plan execute
1. Log[0]: Successfully test the code in blender 3.6 to render a image, and use cuda device
2. Log[0]: test same scene with different object location, and without reinit whole blender bpy. This test is for optimize the render process different from the clevr-dataset-gen version
