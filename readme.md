# Alternative Neural Networks.

## Intro

In this repo we dive into alternative neural network structures for image classification on the Mnist Fashion test data set. The alternatives we will test are AdderNet which replaces matrix multiplication in convolutional layers with simplified addition operations, and spiking neural networks which aim to mirror biological neurons where information passes thru the network via binary spike firings.
- Base code was applied from: https://github.com/pytorch/ignite/blob/master/examples/notebooks/FashionMNIST.ipynb
- We apply basic pytorch classification with a convolutional neural network.
- We then apply AdderNet: https://github.com/huawei-noah/AdderNet
- We then apply SNNTorch to the problem: https://snntorch.readthedocs.io/en/latest/
- Finally we apply AdderNet with SNNTorch
- We use Average Accuracy for validtation on train and test data sets

### Here is a nice animation which gives a visual to help conceptualize how spiking neural networks operate.
[![IMAGE ALT TEXT](https://github.com/jnesfield/Alternative-Neural-Networks/blob/main/fda0a98ec7c2460b94d5eacf2d2b0be9.gif)](https://youtu.be/3JQ3hYko51Y?si=nrvmjCGcJh2fq0qB "Neural Network Animations")

Animation taken from: https://youtu.be/3JQ3hYko51Y?si=91eotxorvnvh6qj3

## Basic Comparisons:

| Experiment        | Steps | Final Training Validation | Final Test Validation | Link                                                                                                                                                                                                                   |
| ----------------- | ----- | ------------------------- | --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ConvNet           | n/a   | 96.06                     | 90.72                 | [https://github.com/jnesfield/Alternative-Neural-Networks/blob/main/FashionMNIST.ipynb](https://github.com/jnesfield/Alternative-Neural-Networks/blob/main/FashionMNIST.ipynb)                                         |
| AdderNet          | n/a   | 92.77                     | 89.21                 | [https://github.com/jnesfield/Alternative-Neural-Networks/blob/main/FashionMNIST-AdderNet.ipynb](https://github.com/jnesfield/Alternative-Neural-Networks/blob/main/FashionMNIST-AdderNet.ipynb)                       |
| SNNTorch          | 20    | 91.59                     | 89.5                  | [https://github.com/jnesfield/Alternative-Neural-Networks/blob/main/FashionMNIST-SNNTorch-4.ipynb](https://github.com/jnesfield/Alternative-Neural-Networks/blob/main/FashionMNIST-SNNTorch-4.ipynb)                   |
| SNNTorch          | 15    | 89.6                      | 88.14                 | [https://github.com/jnesfield/Alternative-Neural-Networks/blob/main/FashionMNIST-SNNTorch-3.ipynb](https://github.com/jnesfield/Alternative-Neural-Networks/blob/main/FashionMNIST-SNNTorch-3.ipynb)                   |
| SNNTorch          | 5     | 87.24                     | 85.72                 | [https://github.com/jnesfield/Alternative-Neural-Networks/blob/main/FashionMNIST-SNNTorch-2.ipynb](https://github.com/jnesfield/Alternative-Neural-Networks/blob/main/FashionMNIST-SNNTorch-2.ipynb)                   |
| SNNTorch          | 10    | 90.18                     | 88.71                 | [https://github.com/jnesfield/Alternative-Neural-Networks/blob/main/FashionMNIST-SNNTorch.ipynb](https://github.com/jnesfield/Alternative-Neural-Networks/blob/main/FashionMNIST-SNNTorch.ipynb)                       |
| SNNTorch+AdderNet | 20    | 87.95                     | 86.4                  | [https://github.com/jnesfield/Alternative-Neural-Networks/blob/main/FashionMNIST-SNNTorch-AdderNet-4.ipynb](https://github.com/jnesfield/Alternative-Neural-Networks/blob/main/FashionMNIST-SNNTorch-AdderNet-4.ipynb) |
| SNNTorch+AdderNet | 15    | 84.8                      | 83.68                 | [https://github.com/jnesfield/Alternative-Neural-Networks/blob/main/FashionMNIST-SNNTorch-AdderNet-4.ipynb](https://github.com/jnesfield/Alternative-Neural-Networks/blob/main/FashionMNIST-SNNTorch-AdderNet-4.ipynb) |
| SNNTorch+AdderNet | 5     | 73.36                     | 72.98                 | [https://github.com/jnesfield/Alternative-Neural-Networks/blob/main/FashionMNIST-SNNTorch-AdderNet-2.ipynb](https://github.com/jnesfield/Alternative-Neural-Networks/blob/main/FashionMNIST-SNNTorch-AdderNet-2.ipynb) |
| SNNTorch+AdderNet | 10    | 79.58                     | 78.82                 | [https://github.com/jnesfield/Alternative-Neural-Networks/blob/main/FashionMNIST-SNNTorch-AdderNet.ipynb](https://github.com/jnesfield/Alternative-Neural-Networks/blob/main/FashionMNIST-SNNTorch-AdderNet.ipynb)     |

## Performance Over Epochs:

This shows the training performance over each epoch for each variants champion!

### ConvNet 
<img src="https://github.com/jnesfield/Alternative-Neural-Networks/blob/main/ConvNetMnist.png" alt="Convolution Network Training" style="width: 563px; height:432px;"/>

### SNNTorch
<img src="https://github.com/jnesfield/Alternative-Neural-Networks/blob/main/SnnMnist.png" alt="SNNTorch Training" style="width: 563px; height:432px;"/>

### AdderNet
<img src="https://github.com/jnesfield/Alternative-Neural-Networks/blob/main/AdderMnist.png" alt="AdderNet Training" style="width: 563px; height:432px;"/>

### SNNTorch + AdderNet
<img src="https://github.com/jnesfield/Alternative-Neural-Networks/blob/main/Snn_AdderMnist.png" alt="SNNTorch + AdderNet Training" style="width: 563px; height:432px;"/>

### Findings:

SNNTorch performs the best with test performance close to traditional Convolutional Networks. AdderNets as well as AdderNets with SNNTorch also perform well!

Very interesting to observe how SNNTorch has narrow gap between training and test performance over time as opposed to AdderNets and Convolutional Networks which seem to overlearn and over time have a montonically increasing gap between train and test validation on average accuracy.
