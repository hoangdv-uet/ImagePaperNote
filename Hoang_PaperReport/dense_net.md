[DenseNet paper](http://openaccess.thecvf.com/content_cvpr_2017/html/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.html)

## **1. Main contributions**
Propose a new CNN architecture (Dense Convolutional Network - direct connections between any two layers with the same feature-map size)

## **2. Abstract**

**Observation**: Recent work has shown that conv-net can be substantially **deeper, more accurate, and efficient to train** if they contain shorter connections between layers close to input and layers close to output.

**Traditional Network**: $L$ layers have $L$ connections.

**Proposed DenseNet**:

- Connects each layer to every other layer in a feed-forward fashion.

- So $L$ layers hayve $\frac{L(L+1)}{2}$ direct connections.

- Use **all the feature-maps of preceding layers** as input to next layer.

- Advantages:
    - Reduce the vanishing-gradient problem.
    - Strenghtne feature propagation
    - Encourage feature reuse
    - Reduce number of parameters

## **3. Introduction**

**Problems of deep CNNs**:
- Input's information passes through many layers, it can **vanish** by the time it reaches the end of the network
- Many recent (2017) publications try to avoid that by creating network with short paths from early layers to later layers
  
**Proposed method:**
- Connect *all layers* directly with each others to ensure maximum information flow between layers.
![example](samplenet.png)
- Combine feature-map method: **concatenate**
- Big advantage: improve flow of information and gradients throughout the network (easy to train). Each layer has direct access to the gradients from the loss function and the original input signal, leading to an implicit deep supersion
- **Obsevation**: Dense connections have a regularizing effect

## **4. DenseNets**

**Dense connectivity**:

$$x_l = H_l([x_0, x_1,..., x_{l-1}])$$
where $[x_0, x_1,..., x_{l-1}]$ is concatenation of $l$ preceeded feature-maps

**Composite function**: composed of 3 consecutive operations: batch normalization (BN), ReLU and a 3x3 Conv

**Pooling layers**: To facilitate down-sampling, authors devide the network into multiple densely connected *dense blocks*
![sample](densenet.png)
The layers between blocks are refered as *transition layers*, which do convolution & pooling

**Growth rate**: 
- If each layer produce $k$ feature-maps, it follows that the $l^{th}$ layer has $input_{channel} + k(l-1)$ inputs feature-maps
- **Important difference**: DenseNet can have very narrow layers, e.g. $k=12$
- $k$ is denoted as **growth rate**

**Bottleneck layers**: Because each layer produces $k$ output feature-maps but has many more inputs $\to$ bottleneck
- Introduce $1 \times 1$ convolution as *bottleneck* layer to reduce number of input feature map
- DenseNet with BN-ReLU-Conv(1)-BN-ReLu-Conv(3) is refered as DenseNet-B
  
**Compression**:
- If a Dense block has $m$ feature-maps
- Let the following transition layer generate $[\theta m]$ feature-maps, where $0 < \theta < 1$
- Refer DenseNet with $\theta < 1$ as DenseNet-C
- In experiments, authors use $\theta = 0.5$

**Implementation on ImageNet**
![sample](densenets.png)

## **5. Experiments**
![exp](densenet_exp.png)

## **6. Discussion**

**Model compactness**: Dense connections encourage feature reuse throughout the network, and leads to more compact models

**Implicit Deep Supervision**: Individual layers receive additional supervision from the loss function through the shorter connections
