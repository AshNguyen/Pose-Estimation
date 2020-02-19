## Classification \& Generation Of Human Pose Sequences

### 1. Abstract

The capstone project investigates the effect of mapping a dataset into a latent space using human pose data, via the two tasks of classification and generation. Classically, some machine learning methods implicitly or explicitly transform the original dataset into an embedded space with a different dimension, sometimes even infinite-dimensional space, such as SVMs, and this yields good results in classification or regression as downstream tasks. This paper explores the usage of variational autoencoders (VAEs) in this embedding process on a complex dataset, namely human activity poses, and the data in latent spaces's  performance in the task of sequence modelings, such as classification of activity from poses and auto-generation of probable subsequent poses. Using RNNs, the results include a positive performance change in the classification task, although there is a notion of optimal dimension, and a negative impact on the generation task. Further explorations are necessary to concretely determine the cause of this negative effect on secondary tasks after mapping, and the notion of optimality in the number of dimensions in the latent space. This problem, from both a practical standpoint, like for autoencoders training, and a theoretical standpoint, like for deciding the depth of hierarchical probabilistic models, are very interesting and should be investigated at lengths.

### 2. Generation results

**Realistic generation results**

![](results/g1.gif) 

![](results/g2.gif)

**Mode collapse**

![](results/collapse1.gif) 

![](results/collapse2.gif)

**Drifting**

![](results/move1.gif) 

![](results/move2.gif)

**Long sequences**

What the network sees: 
![](results/long0.gif)

+5 generated frame:
![](results/long5.gif)

+15 generated frame:
![](results/long15.gif)

+30 generated frame:
![](results/long30.gif)

+50 generated frame: 
![](results/long50.gif)

**More generated sequences:** https://drive.google.com/drive/u/0/folders/1T4I-VpoKf2Ae96JpF101RDN6eWr6k-dU


