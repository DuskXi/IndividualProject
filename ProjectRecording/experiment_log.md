# Chinese Version

## Experiment log

### 实验: 端到端的简化Clevr测试

- 实验概要：构建一个CNN-FC的模型，分类目标数据是Clevr图片中，指定颜色物体的数量（为了尽可能简单化，颜色是固定的，其他颜色是无意义的，也就是一个模型只管一个颜色，这是为了尽可能的控制变量和简化问题），这是一个很简单的分类任务，因为我生成的数据集的物体上限为5个，所以num_classes=6
- 实验结果：失败
- 结果分析：猜测是因为计数也是一种带有一定逻辑性质的任务导致的失败
- 总结：既然如此简单的指定颜色都实验失败，就更别说实现端到端训练整个Clevr以及对应问题了
- 延展：应，设计更加简化的试验，以在最小单元上测试神经网络的逻辑能力，验证线性层组成的深度网络是否具有计数等最简单的逻辑能力
- 实验时间：2024-03-02

### 实验：对最小神经网络单元的逻辑能力验证实验

- 实验概要：构建一个双线性层神经网络，用随机数生成，输入是一个数组，其中有0-1的随机数，要求统计数组内在0.5-0.6区间的个数
- 实验结果：成功，在几百个epoch后收敛到100%的准确率
- 结果分析：在数据简化到极致，有着明显逻辑关系的情况下，线性神经网络层组成的神经网络具有统计数量的能力，初步具备逻辑能力
- 总结：逻辑能力验证实验成功，下一步可以尝试更复杂的逻辑任务
- 延展：新的实验应该围绕着在上个失败实验的目的上进行修改，比如说使用AutoEncoder来实现CNN特征提取部分的预训练，然后再单独训练线性层的计数能力，同时应该锁住CNN的参数，保证只对线性层参数进行优化。
- 实验时间：2024-03-03


### 实验: 进行包围框->物体位置关系的神经网络实现测试

- 实验概要：构建一个模型，输入是图片中的对象的包围框数据，输出则是num_objects*num_objects的邻接矩阵，表示关系，然后为了简化，一个模型只管一个方向的检测，比如说模型检测left，邻接矩阵中的连接就代表该物体是否左侧有目标物体
- 实验细节:
  - 在进行模型训练的时候发现，纯粹的FC-CNN结构貌似有些不稳定，对训练超参数非常的敏感。
  - 在加入dropout和BatchNorm2d层以后，可以达到85%的准确率
  - 不要轻易使用CrossEntropyLoss，多用MSE，现在准确率高达95%
- 实验结果：成功
- 结果分析：模型的准确率达到了97%，可以认为是成功的，原因嘛，大概率是因为根据包围框数据计算关系，或则说在一个指定方向上，对于其他几个物体是否存在这个方向的关系。是一个复杂线性映射，并没有逻辑推理关系。
- 实验时间：2024-03-06

### 实验: 尝试直接将问题用线性网络处理成结构化问题

- 实验概要：主要尝试将分词后的文本问题直接经过线性网络处理输出为结构化的问题以供ILP使用
- 实验结果：失败，准确率仅20%
- 结果分析：可能需要序列网络
- 实验时间：2024-03-11

### 试验记录: 对于多种模式分割的数据集的正反例子学习应用于ILP与神经网络：

#### 属性分割（仅限于依靠：大小，形状，颜色，材质这四个属性进行分割的数据集）

##### ILP:
- 准确率高达100%，直接输出规则，应该是这种分割方式确实非常容易
##### 神经网络
- 待测试，但是预期也很高

#### 属性分割+数量分割，将问题简化成对于符合属性分割的物体的数量有要求，例如正例中必须有两个红色的圆，负例则不然
##### ILP:
- 尝试以后不太行，就算将数字条件从至少2个改为等于2个（因为ILP背景知识并没有写比较，只有统计），但是效果比较随机，学习到的75%准确率有些规则也是噪声。
##### 神经网络:
- 神经网络的泛用型确实比较好，在数据集处理较好的时候，不需要修改什么就可以适应复杂的逻辑变动，神经网络的成绩为：500例子72%验证集准确率，3000例子83%验证集准确率

#### 属性分割+关系分割，要求关系信息关联的两个物体分别符合对应的属性分割条件。
##### ILP:
- 尝试有时候会成功(75%-80%准确率)，学习到了对应的关系规则，但是依然会受到噪声影响，以及一些受到噪声影响而输出的无意义规则。
- 修正：修复了背景知识的bug以后貌似可以解决问题了：
- 比如说分割方式为: 一个rubber物体的前面是一个cube或者cylinder，学习出来的规则是：
- [Rule 1] [Pos cover = 25 Neg cover = 8]
  true_class(A) :-
  contains(B,A), has_shape(B,sphere), right_of(B,C), front_of(C,B).
##### 神经网络:
- 神经网络在这方面依然好用，但是相比于数字分割，带有关系情况的数据对神经网络挑战较高，因为这种分割方式更加容易引入噪声问题，在实施了不同的分割方向后得到了不同的验证集准确率：
- 1. 左右分割下，如果要求符合条件1的物体的的右边是符合条件2的物体，准确率在75%-80%跳动。如果将这个关系改为左边，准确率变为87%。很奇怪，应该是数据集在左分割和右分割时出现的噪声不一样。
- 2. 上下分割下，如果分割的关系是front, 准确率在85%左右，如果是behind，准确率在88%左右，这个差距不及左右，
- 3. 在排除数据集泛化不足的情况下，是一个值得分析的点，为什么纵向分割的准确率比横向分割的准确率高，或则说稳定，是如何避免噪声的。

# English Version

## Experiment log

### Experiment: End-to-end simplified Clevr testing

- Experiment summary: Construct a CNN-FC model. The classification target data is the number of specified color objects in Clevr pictures (in order to simplify it as much as possible, the color is fixed, and other colors are meaningless, that is, one model only cares about one Color, this is to control variables and simplify the problem as much as possible), this is a very simple classification task, because the upper limit of the objects in the data set I generated is 5, so num_classes=6
- Experiment result: Failure
- Result analysis: It is speculated that the failure is caused by counting, which is also a task with certain logical nature.
- Summary: Since such a simple experiment of specifying colors has failed, let alone achieving end-to-end training of the entire Clevr and corresponding issues.
- Extension: Design more simplified experiments to test the logical capabilities of the neural network on the smallest unit and verify whether the deep network composed of linear layers has the simplest logical capabilities such as counting.
- Experiment time: 2024-03-02

### Experiment: Logical capability verification experiment on the smallest neural network unit

- Experiment summary: Construct a bilinear layer neural network and generate it with random numbers. The input is an array, which contains random numbers from 0 to 1. It is required to count the number of numbers in the range of 0.5-0.6 in the array.
- Experimental results: Successful, converging to 100% accuracy after hundreds of epochs
- Result analysis: When the data is simplified to the extreme and there are obvious logical relationships, the neural network composed of linear neural network layers has the ability to count quantities and initially has logical capabilities.
- Summary: The logic ability verification experiment was successful. You can try more complex logic tasks next.
- Extension: The new experiment should focus on modifying the purpose of the previous failed experiment, such as using AutoEncoder to achieve pre-training of the feature extraction part of the CNN, and then separately training the counting ability of the linear layer, and at the same time locking the CNN Parameters, ensuring that only linear layer parameters are optimized.
- Experiment time: 2024-03-03


### Experiment: Test the neural network implementation of the bounding box->object position relationship

- Experiment summary: Construct a model. The input is the bounding box data of the object in the picture, and the output is the adjacency matrix of num_objects*num_objects, which represents the relationship. Then, in order to simplify, a model only detects one direction, for example, the model detects left. The connection in the adjacency matrix represents whether there is a target object on the left side of the object
- Experimental details:
  - During model training, I found that the pure FC-CNN structure seems to be a bit unstable and is very sensitive to training hyperparameters.
  - After adding dropout and BatchNorm2d layers, an accuracy of 85% can be achieved
  - Don’t use CrossEntropyLoss lightly, use MSE more often, the accuracy is now as high as 95%


### Experiment: Try to directly use a linear network to process the problem into a structured problem

- Experiment summary: The main attempt is to directly process the text questions after word segmentation through linear network processing and output them into structured questions for ILP use.
- Experiment result: Failure, accuracy rate is only 20%
- Result analysis: sequence network may be needed
- Experiment time: 2024-03-11
