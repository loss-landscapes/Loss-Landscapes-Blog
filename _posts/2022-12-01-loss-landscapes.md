---
layout: post
title: "Loss Landscapes"
tags: [optimization, generalizability, loss]
authors: Jacob Hansen, Christian Cmehil-Warn
---


## Loss and Over-fitting

Loss is the primary way of measuring progress when training deep learning models. The higher the loss value the worse the model performs, so when training the model, the loss should be decreasing. However, the actual loss values _depend greatly_ on the deep learning model's task.

Loss cannot be compared between models when they use different optimizers. As shown in the charts below, a model with a lower loss and higher training accuracy can have a worse test accuracy than a model with higher loss and lower training accuracy. Even though the dark green model has "better" training measures, it has _over-fitting_ issues.

![]({{ site.url }}/public/images/2022-12-01-loss-landscapes/loss-vs-accuracy.png)

Over-fitting is a serious concern because deep learning models have more parameters than training points. The above example shows that a more robust and generalizable model can have a higher loss; however, even validation accuracy has limits as a measure when condsidering real-world deployment. However, there's an important tool in understanding a model's generalizability: loss landscapes.


![]({{ site.url }}/public/images/2022-12-01-loss-landscapes/overfitting.png)
Picture from [here](https://medium.com/greyatom/what-is-underfitting-and-overfitting-in-machine-learning-and-how-to-deal-with-it-6803a989c76)



## Loss Landscapes Informing Architecture

A loss landscape visualizes the loss function across different sets of parameters. Given the potentially millions of parameters, the dimensions of the landscape are reduced. The visualization subsets the landscape, often the subset where a trained model ends up. 


![]({{ site.url }}/public/images/2022-12-01-loss-landscapes/visualizing-loss.png)

In loss landscapes, the local minima of the function are very apparent as well as how sharp they are. Sharper minima indicate that the model is less generalizable (ie more overfit) models that are less robust in real would situations.

The shape of loss landscapes near the final minima completely depends on what optimizer the model uses. Some [research](https://arxiv.org/abs/1712.09913) found that added skip connections results in a much smoother loss landscape. Other new [research](https://arxiv.org/abs/1609.04836)  has found that training on smaller batches of data results in models ending up in minima with a much wider "opening" at the top of the descent, resulting in better performace on unseen data. 

In fact, recent recent found great success designing optimizers around finding smoother areas of the loss landscape. Sharpness-Aware Minimization, or SAM, is an optimizer that minimizes both loss and loss sharpness. While more computationally expensive, the model trained achieved state of the art results on image labeling benchmarks.

![]({{ site.url }}/public/images/2022-12-01-loss-landscapes/our-sam.png)


