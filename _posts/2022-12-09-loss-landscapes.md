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

Another case where loss cannot be compared between models is when they use different optimizers. As shown in the charts below, its possible for a model with a lower loss and higher training accuracy to have a a worse test accuracy than a model with higher loss and lower training accuracy. Even though the dark green model has "better" training measures, its a model with over-fitting issues.

![]({{ site.url }}/public/images/2022-12-09-loss-landscapes/loss-vs-accuracy.png)

Given that deep learning models are heavily over-parameterized, over-fitting is a serious concern. The above example shows that a more robust and generalizable model can have a higher loss; however, even validation accuracy has its limits as a measure when a model is going to be deployed to the real world. However, there's an important tool in understanding a model's generalizability: loss landscapes.

![Picture from https://medium.com/greyatom/what-is-underfitting-and-overfitting-in-machine-learning-and-how-to-deal-with-it-6803a989c76]({{ site.url }}/public/images/2022-12-09-loss-landscapes/overfitting.png)


## Exploring the Loss Landscape

A loss landscape is a visualization of the loss function where each point is the loss of a model with a different set of parameters. Given that deep learning models often have thousands, if not millions, of parameters, the visualization is made after dimensionality reduction. Also to note that the visualizations are a subset of the landscape, often where a trained model ends up. 


![]({{ site.url }}/public/images/2022-12-09-loss-landscapes/visualizing-loss.png)

When looking at a loss landscape, the local minima of the function are very apparent as well as how sharp they are. Sharper minima indicate that the model is less generalizable (ie more overfit) models that are less robust in real would situations.

The shape of loss landscapes near the final minima are completely dependent on what optimizer the model uses. Some research has found that added skip connections results in a much smoother and easier to navigate loss landscape (https://arxiv.org/abs/1712.09913). Other new research has found that training on smaller batches of data results in models ending up in minima with a much wider "opening" at the top of the descent, resulting in better performace on unseen data. (https://arxiv.org/abs/1609.04836) 

In fact, recent literature has found great success in designing optimizers around finding smoother areas of the loss landscape. Designed by reserachers at Google, Sharpness-Aware Minimization, or SAM, is an optimizer that minimizes both loss and loss sharpness. While more computationally expensive, the model trained achieved state of the art results on image labeling benchmarks.

![]({{ site.url }}/public/images/2022-12-09-loss-landscapes/our-sam.png)

Loss landscapes are an important tool in making informed decisions about model architecture for generalization. Even if models don't seem to be improving on testing or validation sets, checking the loss landscape can be important to checking for overall generalizability. 

