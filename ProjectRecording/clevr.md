# Log for Individual Project Recording (CLEVR part)

### 2024-02-20:

---

Implement the neural network structure of the CV part and the trainer

---

### 2024-02-21:

---

It was found that in order to improve the training of the object detection model, it is still necessary to know the range of the model in the picture, that is, the bounding box. Because most object detection models have RPN (Region Proposal Network), training a separate image->CV neural network of object attribute information on the Clevr data set still cannot avoid using bounding box data.

---

Obtaining the bounding box of an object in the Clevr dataset does not require manual labeling. I need to obtain the rendering information, write a python program to perform the entire transformation process, from model transformation, view transformation, perspective projection transformation, and then calculate the bounding box of the object based on the position of the model vertices in the NDC space, this is feasible. Unfortunately, the scene files included in the Clevr dataset lose a lot of information, including model scaling and camera position, which is difficult to reconstruct. Therefore, the focus of work needs to shift to modifying the generation code, generating the data set by oneself, and outputting the corresponding bounding box data.

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

### 2024-02-29:

---

#### Plan execute
1. Log[0]: Successfully test the code in blender 3.6 to render a image, and use cuda device
2. Log[0]: test same scene with different object location, and without reinit whole blender bpy. This test is for optimize the render process different from the clevr-dataset-gen version
3. Log[0]: Successfully supports the use of SceneObject lists and Scene objects for rendering. Scene and SceneObject are inherited from dictionaries, so they can be directly converted to json, and rendering also supports different materials and reuse of different colors.

---

### 2024-03-01:

---

#### Plan execute
1. Log[0]: Successfully tested the entire random scene generation rendering process
2. Log[0]: Continue to wrap the entire generation task into the generator and equip it with Config processing tools.
3. Log[0]: Phase summary: Now the entire migration/rewriting work has no blind spots from a technical perspective. What still needs to be implemented: In the random process, before rendering, filter out problems where objects are completely occluded by other objects, or when two objects intersect in 3D space.
4. Log[0]: Phased goal: Implement CG MVP projection calculation and simple rasterization calculation in python to calculate depth field occlusion information and object bounding box.
5. Log[0]: Successfully conducted technical verification of the bounding box algorithm.
6. Log[0]: Implements collision detection during scene generation, reduces unnecessary calculations to a minimum, and avoids clevr's original generation code that requires blender to run additional unnecessary calculations while rendering.
7. Log[0]: It realizes the calculation of the proportion of non-background pixels of each object in its own frame based on its own rasterization results, so that the object's occlusion rate can be calculated. This can be used as a threshold to filter out inappropriate data.
8. Log[0]: Successfully migrated relative position calculation code to new code.
9. Log[0]: Implemented bounding box data comes with output, as well as a bounding box drawing tool (mainly to visually verify the accuracy of the bounding box position). The bounding box takes and outputs the original center coordinate ndc data (including the z-axis).

### 2024-03-02:

---

#### Plan Status
1. Log[0]: The image generation code of the clevr dataset has been refactored to blender 3.6, and more information and object bounding box data output have been added.

### 2024-03-03:

---

#### Plan Created
Plan Name[1]: Implementing the CV part of Clevr

Consider[1]: The object detection network should be modified, because the task of the CV part is to first identify the object and then identify the object's attributes.

---

#### Plan execute
1. Log[1]: First read the source code of the object detection neural network (Faster RCNN/SSD)

---

### 2024-03-04:

---

#### Plan Created

Plan Name[2]: Implement a neural network that can calculate the relative position based on the position of the object's bounding box.

Consider[2]: After a small end-to-end neural network experiment (trying to implement a counting model for the number of red objects with a specified color in Clevr pictures, which failed), I realized a problem. Maybe Clevr’s CV task needs to be disassembled and implemented, and cannot be directly implemented. Implement an end-to-end (that is, the relationship matrix data between pictures and objects) neural network? After my preliminary research, it may be that the optimization method of gradient descent is inherently problematic. For complex multi-layer logical relationships, Data set, gradient descent is not good at optimizing model weights. This means that the creation of this task is necessary. Because this task is to disassemble Clevr's CV task, the end-to-end model is split into object detection tasks and relationship calculation tasks.

---

### 2024-03-05:

#### Plan execute
1. Log[2]: A neural network based on a fully connected layer-convolution layer is implemented. This neural network receives a two-dimensional tensor, which is (number of objects [5], bounding box data [4]), and the output is an adjacency matrix between objects. Currently, a neural network can only calculate one direction, such as the left side or the back side.
2. Log[2]: The experimental result is that the accuracy can be approximated to 80%, but still needs to be improved.

---

### 2024-03-06:

---

#### Plan execute
1. Log[2]: Copy the neural network to RelationshipV2, add dropout and BatchNorm2d layers, and continue training. The accuracy can be increased to 85% (only 20-30epoch is needed)
2. Log[2]: The neural network is modified to use MSE loss, and the accuracy can be increased to 95% (only 20-30epoch is needed).
3. Log[2]: Add a test set (which has no intersection with the training set) to increase the credibility of the model. The situation is now very optimistic, and the accuracy can be as high as 98% (35 epoch).
4. Log[2]: The model is implemented to simultaneously calculate the relationship adjacency matrix in four directions(30 epoch 97%).
5. Log[1]: Train a Faster RCNN on the Clevr dataset.
5. Log[1]: Successfully implemented Fast RCNN's object detection for Clevr, and supports attributes such as color, shape, size, material, etc.

---

#### Plan Created
Plan Name[3]: Merge object detection network, and relative position network.
Consider[3]: Mainly to test model performance and experimental properties.

---

#### Plan execute
1. Log[3]: Successfully built a suggestion interface implemented using Gradio to connect two neural networks.
