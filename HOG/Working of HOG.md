# Working of HOG

# HOG (Histogram of Oriented Gradients

- We can determine gradient of an image. The orientations of the gradients give us the image’s features.
- HOG is a feature descriptor used in CV and image processing for purpose of object detection.
- We determine gradient orientations in localised portions of an image and we also count the occurrences of gradient orientation in those localised portions.
- The distribution of intensity gradients or edge directions help in describing the local object appearance and shape within an image.

### What are Gradients and Edges?

- Gradient in an image measures how the intensity of pixel changes, that is, it shows how light or dark the image is changing at any point. *It is the change in intensity at any particular pixel*
- An edge is where there is a significant change in intensity, like the border of an object.


### How object detection will be done here

![Untitled](Computer-Vision---Basic-Applications/HOG/Documentation Resources/Untitled/roadmap.png)

- We first create a dataset which will contain images of pedestrians in different environments, so that our model will be able to learn what pedestrians look like. This forms the foundation for training of the model.
- We will then encode the images into feature spaces.
(`*Feature space*` refers to a new representation of the image where each image is described by a set of features (numerical values) rather than raw pixel values.)
HOG converts an image into feature space that captures the essential shapes and appearances by focusing on the edges and gradients.
- We will then learn binary classifier using for example, linear SVM which will finally help in the object detection, here, detection of a pedestrian. 
(A `*binary classifier*` is a type of ML model that distinguishes between two classes, in our case : *Pedestrian and `Not` Pedestrian)*
(`*SVM*` is a popular tool for classification tasks. It learns from the feature vectors in out dataset[feature vectors = images in feature space] and adjusts its parameters to classify as many examples as possible)

## *Procedure in HOG*

### Pre-Processing

First, resize the image into `64x128` pixels (This is a standard value, it can be anything. But take it 64x128 in usual).

The 64x128 image in our example be : 

![Untitled](Working%20of%20HOG%20a6f89c9b047845c0bcf3ef204a6e00ad/Untitled%201.png)

### Gradient and Gradient Orientation Calculation

<aside>
💡 To calculate *Gradient* and *Gradient Orientation* for a pixel, we use the following formulae : 
Apply Sobel Filters to get - 
***Gradient in x direction :***

![Untitled](Working%20of%20HOG%20a6f89c9b047845c0bcf3ef204a6e00ad/Untitled%202.png)

***Gradient in y direction :*** 

![Untitled](Working%20of%20HOG%20a6f89c9b047845c0bcf3ef204a6e00ad/Untitled%203.png)

***Total gradient is then given by :*** 

![Untitled](Working%20of%20HOG%20a6f89c9b047845c0bcf3ef204a6e00ad/Untitled%204.png)

The gradient orientation is given by : 

![Untitled](Working%20of%20HOG%20a6f89c9b047845c0bcf3ef204a6e00ad/Untitled%205.png)

</aside>

### Orientation Binning

- You have the 64x128 / 128x64 image. This means it has total 64x128 pixels : 64 rows and 128 columns.
- Now, we will take `cells` of *size* `8x8 pixels`. This means the number of cells is (64x128)/(8x8) = `128 cells`

The image with the 128 cells can be shown as : 

![Untitled](Working%20of%20HOG%20a6f89c9b047845c0bcf3ef204a6e00ad/Untitled%206.png)

- This is a grid with 8x16 cells. Each cell is 8x8 pixels.

Each cell (0f 8x8 pixel) can be shown as : 

![Untitled](Working%20of%20HOG%20a6f89c9b047845c0bcf3ef204a6e00ad/Untitled%207.png)

Now, in each cell, we need to find the *gradient magnitude* and *gradient orientation* for `each` pixel. 

In the above diagram, we have calculate these for the highlighted pixel with pixel value 60. 

→ To get the Gradients in X and Y direction for any pixel, we need to subtract the the adjacent values.

For example, for the highlighted pixel of value 60, the gradient in x direction will be |40-70| = 30 and gradient in y direction will be |20-70| = 50.

→ To get the gradient orientation, we will use the formula mentioned. Arctan(Gy/Gx)

For that highlighted pixel, the gradient magnitude came 58 and gradient orientation came 30. 

In this way, we will find the gradient magnitude and orientation for each pixel in the cell. 

Now, for each cell, we will plot a histogram.

Rules while plotting the histogram : 

- Create `9` bins of size `20` each. This is because the orientations can be from `0 to 180`.
- The histogram can be shown as :

![Untitled](Working%20of%20HOG%20a6f89c9b047845c0bcf3ef204a6e00ad/Untitled%208.png)

Each `bar` here is of 0-20, 20-40, 40-60, 60-80, …. 160-180. 

### The magnitude of each bar is calculated as follows :

- For each pixel, see it’s orientation value. For example, in our diagram, the orientation for the highlighted pixel came 30.
- Then, we will put the magnitude of this pixel (which is 58) into the 20-40 degree bin.
- In this way, we will see the orientation of each pixel in the cell and then put their gradient magnitudes in the respective bins. In this way, we will get the length of each bar.

Hence, in this way, ***we’ve created histograms for each cell***. In total, we had 128 cells. Hence we have 128 histograms now. 

We can say that for each cell, we have a single histogram such that it has 9 bins. We can say that each histogram is a vector of size 9. `Each histogram is a feature vector which is a matrix with values equal to the magnitudes of the bars in the histogram`, hence the feature vector is of size 9. 

Example : If the bars in our histogram have values 50,135,105,….; then our feature vector can be shown as : FV = [50,135,105,…..] (Size 9)

## Concatenation of cells into `blocks` :

Till now, we’ve created the feature vectors (histograms) for the 128 cells. 

Now, we will create *blocks* each of size `2x2` cells. For each of the block, we will concatenate the 4 feature vectors into a single feature vector of size 36. (Each feature vector is of size 9).

The 2x2 cells will be formed in such a manner : 

![Untitled](Working%20of%20HOG%20a6f89c9b047845c0bcf3ef204a6e00ad/Untitled%209.png)

We will start creating the blocks from the top left corner and then keep sliding it as shown.

In total, we will have 7 x 15 = `105 blocks`. 

We will hence create 105 such feature vectors of size 36. 

The next procedure will be the normalisation of these vectors. 

### Block Normalisation

Normalise each block (each feature vector of size 36). The procedure of normalisation can be shown as : 

![Untitled](Working%20of%20HOG%20a6f89c9b047845c0bcf3ef204a6e00ad/Untitled%2010.png)

### Descriptor Formation

Concatenate all the normalised histograms from all blocks into a single feature vector that represents the entire image. This final feature vector is used as an input to our ML model.
