---
layout: post
title: "Loss Landscapes"
tags: [optimization, generalizability, loss]
authors: Jacob Hansen, Christian Cmehil-Warn
---


## Lost in the Loss

Loss is the primary way of measuring progress when training deep learning models. Simply put, the higher the value the worse the model is, and the loss should be decreasing as the model is trained and improved upon. However, the actual values of loss depend greatly on the task that the deep learning model is performing.


For example, a classification task might use cross-entropy loss, which is calculated from the differences between predicted values and the ground truth labels. Yet, the loss could also be measured through KL Divergence, a measure of the difference between a model's output distribution and the distribution of a test set. Trying to compare either of those loss values is just as useless as comparing either to the mean-squared error loss used for a regression task. 

# TODO: Different types of Loss Chart

Another case where loss cannot be compared between models is when they use different optimizers. As show in the charts below, its possible for a model with a lower loss and higher training accuracy to have a a worse test accuracy than a model with higher loss and lower training accuracy. Even though the dark green model has "better" training measures, its a model with over-fitting issues.

# TODO: that one chart with the two models where one has

Given that deep learning models are heavily over-parameterized, over-fitting is a serious concern. The above example shows that a more robust and generalizable model can have a higher loss; however, even validation accuracy has its limits as a measure when a model is going to be deployed to the real world. However, there's an important tool in understanding a model's generalizability: loss landscapes.


## Exploring the Loss Landscape




## Optimizers: Terraforming Loss Landscapes
