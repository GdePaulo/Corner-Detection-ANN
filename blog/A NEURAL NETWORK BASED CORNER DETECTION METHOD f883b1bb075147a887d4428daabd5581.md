# A NEURAL NETWORK BASED CORNER DETECTION METHOD

Authors: Gedeon d' Abreu de Paulo and Jimmy Vlekke

Corner detection sounds like a trivial task these days, however, back in 1995, this was more of a challenge. Doing so yourself sounds like a trivial task, especially given the current state of deep learning and the ease with which one could train a simple feedforward network. But we would like to challenge you to reverse engineer corner detections by solving the image below: 

![A%20NEURAL%20NETWORK%20BASED%20CORNER%20DETECTION%20METHOD%20f883b1bb075147a887d4428daabd5581/image.png_output.png](A%20NEURAL%20NETWORK%20BASED%20CORNER%20DETECTION%20METHOD%20f883b1bb075147a887d4428daabd5581/image.png_output.png)

Mystery image

What could the original shapes have been? We give you two hints; the shapes are capitalised letters and it is a 5 letter word with an additional punctuation character at the end.

As the title of this blog already hints, we will be discussing a neural network based method of detecting corners. Yes, indeed, we will let the computer do the learning 😉

But we will not only discuss and reproduce the proposed method of the paper, we will also explore an alternative technique to that of the feedforward network, this might bring some change to the detection of corners.

> *A change may be just around the corner ~ Roy T. Bennett*

What this change actually is, you might be wondering? Hang tight as you will find out by reading this blog!

# Intro

In this blog, we will be showcasing our reproduction efforts of the "A Neural Network Based Corner Detection Method" paper. In this paper, published in 1995, the authors improved upon the existing methods at the time by applying a neural network to detect corners in 2D images. Specifically, the authors employed a simple feedforward neural network with one hidden layer to classify 2D 8x8 images as corner containing images, and images with no corner.
The paper is very sparse on details and only contained 5 pages. The authors do not give many details regarding how the data was generated, exactly what data they used, how the model was trained or how they validated the model. Thus, we had to perform an independent reproduction of this paper which required us to generate our own data and fill in the blanks for many of the aspects of the experimental design. In the end, we aimed to reproduce their results regarding the classification of noise-free images with corners and multiples of 45°:

![A%20NEURAL%20NETWORK%20BASED%20CORNER%20DETECTION%20METHOD%20f883b1bb075147a887d4428daabd5581/Untitled.png](A%20NEURAL%20NETWORK%20BASED%20CORNER%20DETECTION%20METHOD%20f883b1bb075147a887d4428daabd5581/Untitled.png)

Table 1 from the original paper, showing the NNSIC performance results

We will describe how we went about reproducing this result by describing the data we used and how it was generated.

# Methods

As we already mentioned, the authors did not provide much detail about the data they used to train the model. They do specify that they only used binary and grey level images as input. Specifically, they used 8x8 sub-images which were labelled as "image with corner" or "image without corner". For images with corners, the corner was only allowed to be located within the central 4x4 portion of the 8x8 sub-image, as can be seen below.

![A%20NEURAL%20NETWORK%20BASED%20CORNER%20DETECTION%20METHOD%20f883b1bb075147a887d4428daabd5581/Untitled%201.png](A%20NEURAL%20NETWORK%20BASED%20CORNER%20DETECTION%20METHOD%20f883b1bb075147a887d4428daabd5581/Untitled%201.png)

Figure 1 from the original paper, depicting the central 4x4 portion of a 8x8 sub-image

To generate these sub-images, they created larger test images and then moved an 8x8 sub window throughout these images to obtain input data. In the paper, the authors include a few examples of these test images, which we show below:

![A%20NEURAL%20NETWORK%20BASED%20CORNER%20DETECTION%20METHOD%20f883b1bb075147a887d4428daabd5581/Untitled%202.png](A%20NEURAL%20NETWORK%20BASED%20CORNER%20DETECTION%20METHOD%20f883b1bb075147a887d4428daabd5581/Untitled%202.png)

Figure 4A from the original paper

![A%20NEURAL%20NETWORK%20BASED%20CORNER%20DETECTION%20METHOD%20f883b1bb075147a887d4428daabd5581/Untitled%203.png](A%20NEURAL%20NETWORK%20BASED%20CORNER%20DETECTION%20METHOD%20f883b1bb075147a887d4428daabd5581/Untitled%203.png)

Figure 5A from the original paper

Unfortunately, they did not include much more than this. Thus, to generate the input images for the training of the model, we attempted to replicate their data by creating similar test images with various shapes. These shapes were programmatically drawn on an empty black canvas. The corner points of each shape were calculated or retrieved and saved into a list. As described in the paper, we then moved an 8x8 window through these test images, to generate 8x8 sub-image input data. We then used the location of the stored corner points to see whether the sub-images contained a corner and classify them accordingly. The two main inputs test images we used can be seen below:

![A%20NEURAL%20NETWORK%20BASED%20CORNER%20DETECTION%20METHOD%20f883b1bb075147a887d4428daabd5581/test_1.png](A%20NEURAL%20NETWORK%20BASED%20CORNER%20DETECTION%20METHOD%20f883b1bb075147a887d4428daabd5581/test_1.png)

Generated test image 1

![A%20NEURAL%20NETWORK%20BASED%20CORNER%20DETECTION%20METHOD%20f883b1bb075147a887d4428daabd5581/test_1_2.png](A%20NEURAL%20NETWORK%20BASED%20CORNER%20DETECTION%20METHOD%20f883b1bb075147a887d4428daabd5581/test_1_2.png)

Generated test image 2

Additionally, we also handcrafted some images and used these images, as well as their rotations, to augment the input data. Some examples can be seen below:

![A%20NEURAL%20NETWORK%20BASED%20CORNER%20DETECTION%20METHOD%20f883b1bb075147a887d4428daabd5581/Untitled%204.png](A%20NEURAL%20NETWORK%20BASED%20CORNER%20DETECTION%20METHOD%20f883b1bb075147a887d4428daabd5581/Untitled%204.png)

Example of one of our handcrafted 8x8 sub-images where the central 4x4 portion has been highlighted in red (this image has been scaled for the sake of the reader)

In the end, we were able to generate 2155 images, of which 316 were corner images. We only included binary images and made sure to remove duplicate images from the input data.

For the reproduction, we used a feedforward neural network, as described in the paper. This was a simple 3 layer neural network containing only one hidden layer. It contained 64 input features, one for each pixel of the 8x8 sub-image input. The hidden layer consisted of 16 nodes, while the output layer had only one node. The overall architecture of this model can be seen below:

![A%20NEURAL%20NETWORK%20BASED%20CORNER%20DETECTION%20METHOD%20f883b1bb075147a887d4428daabd5581/Untitled%205.png](A%20NEURAL%20NETWORK%20BASED%20CORNER%20DETECTION%20METHOD%20f883b1bb075147a887d4428daabd5581/Untitled%205.png)

Figure of the architecture of the feedforward network

In addition to the feedforward network, we also created a convolutional neural network to investigate how it performed when compared to the feedforward network which was used in the original paper. As this is an image classification task, we hypothesized that the CNN would do much better. In order to compare, we designed the CNN to match the complexity of the feedforward network as much as possible. To this end, we tried to match the amount of parameters and layers of the networks. The CNN we designed had 900 parameters (vs 1024 for the feedforward network), 1 hidden convolutional layer which was followed by max pooling, and 1 fully connected layer which consisted of the single output node.

![A%20NEURAL%20NETWORK%20BASED%20CORNER%20DETECTION%20METHOD%20f883b1bb075147a887d4428daabd5581/Untitled%206.png](A%20NEURAL%20NETWORK%20BASED%20CORNER%20DETECTION%20METHOD%20f883b1bb075147a887d4428daabd5581/Untitled%206.png)

Figure of the architecture of the convolutional network

Furthermore, for both models, we used the ReLU activation function in the hidden layer, followed by a sigmoid in the output layer. The train the model, as this is a binary classification task,
we used the binary cross entropy loss function. Additionally, we performed the optimization with Adam gradient descent.

To validate the model, we used 10-fold cross-validation. As only about 15% of the data set consisted of corner images, we used stratified cross-validation to ensure that there was an equal amount of images for each label in every fold. We likely did not have nearly as much data as the original authors used, so we aimed to create a learning curve to investigate how the models performed with more data. For every data amount size, we performed 10 fold stratified cross validation to obtain an accurate estimate of the training in the testing performance of the model. Finally, we also trained the models on 80% of the whole dataset and tested on the other 20% to further investigate how the models fit the data.

# Results

For the neural network, we managed to achieve an 90% accuracy on the test set. This is quite far from the 97.55% which was found in the original paper. As the model is relatively simple, we believe the difference comes from the validation method and the input dataset that was used. The authors did not specify how they validated their model, whether they used cross validation, or how they ensured a proper split between training and testing data. Thus, this could clarify a part of the difference in accuracy. However, we believe the biggest difference comes from the input data. We do not have as much data as the original authors, which would affect how many different types of input images the model can learn to classify and how well it can generalize.

We made numerous plots to investigate how much the model was overfitting on the training data. The plot for the training and testing accuracy and loss for the feedforward network can be seen below:

![A%20NEURAL%20NETWORK%20BASED%20CORNER%20DETECTION%20METHOD%20f883b1bb075147a887d4428daabd5581/accuracy.png](A%20NEURAL%20NETWORK%20BASED%20CORNER%20DETECTION%20METHOD%20f883b1bb075147a887d4428daabd5581/accuracy.png)

Accuracy for each epoch for training and testing our neural network

![A%20NEURAL%20NETWORK%20BASED%20CORNER%20DETECTION%20METHOD%20f883b1bb075147a887d4428daabd5581/loss.png](A%20NEURAL%20NETWORK%20BASED%20CORNER%20DETECTION%20METHOD%20f883b1bb075147a887d4428daabd5581/loss.png)

Loss for each epoch for training and testing our neural network 

As can be seen, the model fits well on the training set, but there is a large gap between the performance on the training set and the testing set, indicating that the model has problems generalizing beyond the training set. Below, we also show the plots for the CNN:

![A%20NEURAL%20NETWORK%20BASED%20CORNER%20DETECTION%20METHOD%20f883b1bb075147a887d4428daabd5581/accuracy%201.png](A%20NEURAL%20NETWORK%20BASED%20CORNER%20DETECTION%20METHOD%20f883b1bb075147a887d4428daabd5581/accuracy%201.png)

Accuracy for each epoch for training and testing our convolutional neural network 

![A%20NEURAL%20NETWORK%20BASED%20CORNER%20DETECTION%20METHOD%20f883b1bb075147a887d4428daabd5581/loss%201.png](A%20NEURAL%20NETWORK%20BASED%20CORNER%20DETECTION%20METHOD%20f883b1bb075147a887d4428daabd5581/loss%201.png)

Loss for each epoch for training and testing our convolutional neural network

The CNN obtains a much higher accuracy with the same amount of data and has much fewer problems overfitting during training. This shows the power of CNNs, as the parameter sharing
of the network restricts its capacity to overfit.

In order to further investigate the effect of data size on the model's performance, we plotted the performance of the feedforward network with different data sizes. This evaluation
was done by using stratified 10-fold cross-validation.

![A%20NEURAL%20NETWORK%20BASED%20CORNER%20DETECTION%20METHOD%20f883b1bb075147a887d4428daabd5581/accuracy%202.png](A%20NEURAL%20NETWORK%20BASED%20CORNER%20DETECTION%20METHOD%20f883b1bb075147a887d4428daabd5581/accuracy%202.png)

Accuracy for training and testing our neural network on different data set sizes for our feedforward network

![A%20NEURAL%20NETWORK%20BASED%20CORNER%20DETECTION%20METHOD%20f883b1bb075147a887d4428daabd5581/loss%202.png](A%20NEURAL%20NETWORK%20BASED%20CORNER%20DETECTION%20METHOD%20f883b1bb075147a887d4428daabd5581/loss%202.png)

Loss for training and testing our neural network on different data set sizes for our feedforward network

As can be seen, there is a large gap between the training and testing accuracy, which indicates overfitting. However, as the data set increases, this gap becomes smaller. With the full data set, the model achieves an accuracy of around 90%. We also ran the same experiment on the CNN:

![A%20NEURAL%20NETWORK%20BASED%20CORNER%20DETECTION%20METHOD%20f883b1bb075147a887d4428daabd5581/accuracy%203.png](A%20NEURAL%20NETWORK%20BASED%20CORNER%20DETECTION%20METHOD%20f883b1bb075147a887d4428daabd5581/accuracy%203.png)

Accuracy for training and testing our neural network on different data set sizes for our convolutional neural network

![A%20NEURAL%20NETWORK%20BASED%20CORNER%20DETECTION%20METHOD%20f883b1bb075147a887d4428daabd5581/loss%203.png](A%20NEURAL%20NETWORK%20BASED%20CORNER%20DETECTION%20METHOD%20f883b1bb075147a887d4428daabd5581/loss%203.png)

Loss for training and testing our neural network on different data set sizes for our convolutional neural network

As can be seen, even with 500 input images, the CNN performs better than the feedforward network trained on > 2000 images. This is even more interesting when considering that the CNN has fewer parameters and has a similarly sized architecture with only one convolutional layer. This further confirms the common knowledge that the initial layers of a CNN capture low-level features (i.e. edges). In the end, the CNN obtains a 96% accuracy when trained and validated on the whole data set.

# Conclusion

To conclude, we did not manage to completely reproduce the original paper. Training and validating a feedforward network on our generated data set of 2155 images, we managed to achieve a 90% accuracy. This is about 8% lower than the accuracy which was claimed in the original paper. We believe and have provided evidence that a large part of this difference is most likely due to a difference in the input data used, which made it harder for our model to generalize. We also tested and compared this model to a similarly-sized shallow CNN with only one convolutional layer. This model managed to obtain a 96% accuracy with a low amount of input data.

To relate back to the beginning of this blog, we presented the following quote, "A change may be just around the corner." Based on the results of this reproduction, we believe that this change, in this case, refers to convolutional neural networks. By simply changing the model, you can obtain a much greater performance with the same amount of information!

To end this blog, we will now reveal the answer to the challenge which was presented in the beginning of the paper. Did you figure out what the word was?

![A%20NEURAL%20NETWORK%20BASED%20CORNER%20DETECTION%20METHOD%20f883b1bb075147a887d4428daabd5581/image.png](A%20NEURAL%20NETWORK%20BASED%20CORNER%20DETECTION%20METHOD%20f883b1bb075147a887d4428daabd5581/image.png)

The actual input image of our trained model for corner detection. Were you able to infer this text from the corner highlighted image from the beginning of this blog?