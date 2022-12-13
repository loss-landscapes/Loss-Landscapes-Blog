
---
layout: post
title: "Loss Landscapes"
tags: [optimization, generalizability, loss]
authors: Jacob Hansen, Christian Cmehil-Warn
---
# Visualizing the Loss Landscape 
The loss landscape graphically represents the model's loss function, a measure of how well the model can make predictions on a given dataset. Previous work has shown that structure of the loss landscape foretells the generalizability and robustness on a model solution \[ref...\]. Furthermore, recent optimization methods leverage local loss information to traverse the loss landscape and lead to drastic training improvements \[ref SAM\]. Most papers use loss visualization to validate model performance and provide comparison between solutions. In contrast, here we describe methods and heuristics for analyzing the loss landscape with the intention to improve model architecture, adjust training hyperparameters, and gain insight into the training process of large models. 
</br></br>
## A Deeper Background Into Loss
In deep learning, loss measures how well a model's predictions match the ground truth or expected output for a given input data. 
By minimizing loss through the optimization process, we improve the model's ability to make accurate predictions on unseen data. 
</br></br>
Deep learning differs from traditional optimization in that the process for obtaining a minima matters more than the actual value of the minima acheived.
For example, the figure below depicts a vision transformer trained with two different optimization techniques. Despite one model having substantially lower loss and higher training accuracy, its test accuracy is 10% lower. 
</br>**ExampleImage**</br></br>
This problem initially appears to be due to overfitting. Classical machine learning describes overfitting as the point in training where the model begins to have high training performance by learning features specific to the training set. Overfitting is typically overcome by adding regularization. 
</br></br> 
Though the models above contain far sufficiet paramaters to overfit, they cannot be overcome by regularization techniques alone. As seen below, increasing the regularization (as either weight decay or gaussian noise) did not improve the model test performance. 
</br></br> 
Rather, the model was improved using local loss information to smoothen the loss landscape in a process called Sharpness Aware Minimization.
**ExampleImage**
</br></br> 
## Sharpness Aware Minimization (SAM)
Sharpness Aware Minimization (SAM) analyzes the geometry of the parameter space induced by the symmetries of deep learning models, allowing for the identification of regions of the loss landscape where the minima are flatter. This information can be used to guide the training process and improve the performance of the model. SAM does not just identify when a good solution is reached, but forces the model in directions with smoother loss landscapes. 
</br></br> 
SAM directs the model training by reprametrizing the loss function. Let $f(x)$ be a function with parameters $\theta$, and let $g(\theta)$ be a reparametrization of $f(x)$. The relationship between $f(x)$ and $g(\theta)$ is given by:

$g(\theta) = f(x(\theta))$

The geometry of the parameter space can be changed by choosing a different reparametrization $g(\theta)$, resulting in a new set of parameters $\theta'$ for the function $f(x)$. This allows for the analysis of the relationship between flatness and generalization in deep learning models.


SAM also allows for the reparametrization of functions, which can be used to change the geometry of the parameter space and identify new regions of the loss landscape that may be more conducive to good generalization.






Mathematically, this can be represented as follows:


</br></br> 
## Heuristics for Loss Analysis
The loss landscape graphically represents the model's loss as a function of its parameters, which can provide insight into the training process of the model and its final test performance. 
The loss landscape is typically visualized as a one or two-dimensional plot, an approximation of the true high-dimentional function landscape. Thus, the loss landscape only represents a small slice of the full function space, 
with the dimensions and axes of the plot being determined by the specific parameters that are being visualized. 
</br></br> 
Since the loss landscape selects only a slice of the true function space, it is essential to understand the various techniques for slicing as well as heuristics for improving loss landscape reliability and interpretability. 
As seen in the example below, even modifications to the distance to traverse can lead to large, unintuitive changes in the loss landscape visualization. Here, we'd expect the loss landscape traversed over a 2x2 traversal grid would encompass an identical 1x1 traversal grid within the graph. Unintuitively, we see that changing the size of the traversal grid leads to a new slice and exploration that is substantially smoother. 
</br></br> 
**IMAGEofLOSSchanging**
</br></br> 
Despite this limitation, the loss landscape can still be a useful tool for understanding how the model is training and identifying potential problems or issues with the optimization process. 
For example, wide exploration techniques with large batch sizes may be used to evaluate data augmentation techniques. A smoothly loss landscape likely indicates that the dataset is well patched and continous. In contrast, narrow loss exploration techniques provides insight into hyperparameterization, evaluating how fast a model is converging and to what type of solution.   </br></br> 
</br></br> 
To identify loss landscape plot strategies that best reveal characteristics of interest, it is important to first identify a test case and as base model. We started with the base optimizer, AdamW, and a Vision Transformer. After initial tuning, develop a loss plot strategy that reflects an average case. A single plot by itself can not be used to gain information, rather provide a base condition. By generating a plot representing an average case (a relatively smooth, but variable loss landscape), the test case can then be evaluated using the same strategy to validate that differences are observed. If no differences are observed, reconsider the plot strategy. If the differences are unexpected, then also do so, but with a new test case. </br></br> 

When designing loss plots, we recommend considering traversal strategies, generation metrics, and exploration distance. Furthermore, it is important develop interpretable plot scaling and color to ensure proper evaluation of the loss landscapes. Here, demonstrate these considerations when using loss-landscapes as provided by {Marcello de Bernardi}[https://github.com/marcellodebernardi/loss-landscapes/blob/8d3461045f317bc0f4ba35e552fb22f3242647ff/loss_landscapes/main.py] (pip package loss-landscapes). 

</br></br>
__Traversal Strategies__ </br> 
A loss landscape traversal strategy, as defined here, determines how to vary the parameters of the model in order identify the loss of nearby model paramater settings. One could arbitrarily increase the loss by adding noise to all the paramaters of the model, but this reveals little about the model properties other than it's robustness to parameter pertubation. Thus, more powerful techniques include selecting specific axis of pertubation applied either to the whole model, layers individually, or with a filter strategy developed by \[ref loss for NN paper\]. In the plots below, we compare these three traversal strategies on a Vision Transformer trained with a small learning rate, causing it to quickly converge into a poor local minima. </br> 
**Traversal.png**
</br> 
As seen above, we found that pertubation directions applied to the model as a whole were more reliable for evaluting vision transformers. In contrast, layer or filter pertubation techniques often reveal more information for highly structured models stuch as CNN's. 
</br> </br> 

Topics to Cover 
 -Plot Strategy
 -Uniform Generation Metrics
 -Initial Base Plot Exploration
 -Interpretable Scale and Color
 -Model Traversal Strategies
 -Python Packages








## Further Exploration of Loss Landscapes


</br></br>
The loss landscape is typically plotted on a graph, with the x-axis representing the number of training iterations (also known as epochs) and the y-axis representing the value of the loss function. The resulting graph will typically have a distinctive "valley" shape, with the loss decreasing as the model is trained and reaching a minimum at the end of training. This minimum value is the point at which the model has achieved the best possible performance on the given dataset.
</br></br>
The math behind the loss landscape involves a number of complex concepts from machine learning and optimization theory. At a high level, the loss function used in the loss landscape is typically a measure of the difference between the model's predicted output and the true output. This difference is often quantified using a metric such as mean squared error or cross-entropy loss. The gradient descent algorithm used to train the model then adjusts the model's parameters in order to minimize this loss function, ultimately leading to a model that is able to make accurate predictions.
</br></br>
After training a machine learning model, one can visualize the loss landscape by using various techniques that reduce the high dimensionality of the model's parameter space and the data space to a two-dimensional surface. One such technique is 1-dimensional linear interpolation, which involves projecting the high-dimensional space onto a one-dimensional line and then interpolating the points on this line to create a two-dimensional surface.
</br></br>
Another technique is to use contour plots, which involve dividing the two-dimensional surface into a series of concentric contours, each representing a different level of the loss function. This creates a visual representation of the shape of the loss landscape and allows one to see how the loss changes as a function of the model's parameters.
</br></br>
Finally, one can also use random directions to visualize the loss landscape. This involves selecting a random direction in the high-dimensional space and then projecting the points onto this direction to create a two-dimensional surface. This can provide a different perspective on the loss landscape and can help to identify areas of the space where the loss is particularly high or low.
</br></br>
Overall, these techniques can provide valuable insights into the performance of a machine learning model and can help to identify potential improvements or issues with the model.
</br></br>







