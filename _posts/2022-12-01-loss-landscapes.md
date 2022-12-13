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

The loss landscape **graphically** represents how the model's loss function changes as the model parameters change. Exmining the loss landscape's width and smoothness provides meaningful insight into model performance.

![]({{ site.url }}/public/images/2022-12-01-loss-landscapes/visualizing-loss.png)
Image from [Foret et al](https://arxiv.org/abs/2010.01412) Depecting loss landscapes before and after training with SAM

Previous work has shown that structure of the loss landscape foretells the generalizability and robustness on a model solution ([Keskar et al.](https://arxiv.org/abs/1609.04836)). Keskar et al explores how optimizing CNNs on small batches of data (e.g. stochastic gradient descent) vs large batches of data affect the loss landscape of models. They find that small-batch training results in loss landscapes that have a minima with a wider opening at the top, resulting in more generalizable models


Recent research ([Foret et al](https://arxiv.org/abs/2010.01412)) has found great success designing optimizers around finding smoother areas of the loss landscape. Sharpness-Aware Minimization, or SAM, is an optimizer that minimizes both loss and loss sharpness. While more computationally expensive, a ResNet-101 model trained on ImageNet using SAM had a 3.3% error decrease compared to a equivalent model without SAM. Similarly our Visual Transformer model showed a 10% improvement on ImageNet-100 using SAM.

![]({{ site.url }}/public/images/2022-12-01-loss-landscapes/our-sam.jpeg)


