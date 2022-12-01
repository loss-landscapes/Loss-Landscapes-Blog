---
layout: post
title: "Deep Neural Nets: 33 years ago and 33 years from now (Invited Post)"
tags: [optimization, generalizability, loss]
authors: Jacob Hansen, Christian Cmehil-Warn
---


## Why Care about Loss?

Loss is the primary way of measuring progress when training deep learning models. In each epoch, the loss value decreasing is seen as a sign of progress and that the model is improving; however, the loss value alone tells us nothing about the model generalizablity. Given that deep learning models are often significantly overparameterized (i.e. there's many more parameters than data points), generalizabliity is an important goal. 

In spite of this, there is little empircal understanding of how any why models generalize (https://arxiv.org/abs/1611.03530). Research has show that one seemingly important factor in model generalizability is the "sharpness" of the loss minima that the optimized model settles on (https://arxiv.org/abs/1609.04836) and there have been new optimization techniques that have taken advantage of this fact to much success (https://arxiv.org/abs/2010.01412). 

In this blog post we will explore the shape of the loss minima by exmining reduced dimensionality visualzatoins of the loss function, or the "loss landscape" and experiment with how different optimizers and parameters affect the sharpness.  

## Loss vs Accuracy



## Exploring the Loss Landscape


## Optimizers: Terraforming Loss Landscapes
