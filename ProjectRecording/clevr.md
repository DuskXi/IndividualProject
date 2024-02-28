# Log for Individual Project Recording (CLEVR part)


---

### **before this line is the recording that I have done before the date of start recording**

---

### 2023-02-28:

---
#### Start Recording

---

#### Plan Created
Plan Name[0]: clevr dataset generation code migration work
Plan created[0]: rewrite the clevr generate code from blender < 2.8 to blender 3.6
Consider[0]: due to the code (`facebookresearch/clevr-dataset-gen/blob/main/image_generation/render_images.py`) only supprot blender < 2.8, and blender < 2.8 is to old, cannot support cuda >= 11.0. And old cuda (such as 8.0 or 9.0 is not stable for RTX 40X0). So I need to rewrite the code to support blender 3.6

---

#### Plan execute
Log[0]: Successfully test the code in blender 3.6 to render a image, and use cuda device
