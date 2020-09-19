# Receptive Fields and Network Architecture - Resnet18

In this exercise we are going to run a Resnet18 model on Cifar10 dataset

The model reaches a maximum accuracy of **85.74%** in **37 epochs** .

## Resnet18 Architecture

![resnet18](images/resnet18.png)

## Residual block

Residual blocks are basically a special case of highway networks without any gates in their skip connections. Essentially, residual blocks allows the flow of memory (or information) from initial layers to last layers.

![resnet](images/resnet.png)

## Validation Loss

![loss_change](loss_change.png)

## Validation Accuracy

![accuracy_change](accuracy_change.png)

## Missclassified Images

![incorrect_predictions](images/incorrect_predictions.png)





