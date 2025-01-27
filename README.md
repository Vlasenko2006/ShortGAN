# FastGAN (Generating Handwritten Digits)
This software is an ilustrative example of how Generative Adversarial Network (GAN) generate realistic handwritten digits based on the MNIST dataset. It includes generator pretraining to enhance performance and stabilize GAN training. It Perfectly  fits for educational purposes.

![Sample Output](gan_generated_image_epoch.jpeg)

## **Prerequisites**
1. Linux
2. Python 3

---

## **Overview**
The neural network consists of:
1. A **Generator** network that generates synthetic images of digits.
2. A **Discriminator** network that classifies images as real or fake.
3. **Pretraining** the generator with real MNIST samples to ensure Generator training.
4. Training the GAN where the generator and discriminator compete against each other.

---

## **1. How the Generator Works**

The **Generator** creates synthetic images from random noise vectors. Here's how its layers function:
- **Input Layer**: Accepts a noise vector of size `noise_dim` (e.g., 128).
- **Fully Connected Layer 1**: A linear transformation followed by a ReLU activation. This projects the noise into a higher-dimensional space (256 units).
- **Convolutional Block**:
  - `Unflatten`: Reshapes the data into a feature map of size `(1, 16, 16)` to work with convolutional layers.
  - `Conv2d` Layers: Adds spatial context to the feature maps, progressively learning localized patterns.
  - `BatchNorm2d`: Normalizes activations, speeding up convergence.
  - `ConvTranspose2d`: Upsamples the feature maps, reconstructing a higher-resolution image.
- **Fully Connected Layer 2**:
  - Maps the convolutional outputs to a 28x28 grayscale image.
  - Includes a **Tanh** activation to normalize pixel values to \([-1, 1]\), matching the MNIST normalization.

### **Peculiarities in the Generator**
- **Spectral Normalization**: Ensures smoother gradients and prevents the generator from overproducing high-frequency details.
- **Convolution and Transpose Convolution**: Learn to downsample and upsample, preserving realistic image structure.

---

## **2. How Pretraining Works**

Pretraining is mandatory for the FastGAN convergence. In a nutshell, in a conventional GAN neural network (NN) without pretraining, the generator learns simultaneously with 
the discriminator. The discriminator usually trains much faster than the generator and becomes overconfident: whatever untrained generator creates, the discriminator classifies it as close to 100% 'fake'.
From the training perspective it looks as: in whatever direction the generator starts its training, the discriminator indicates the wrong direction. Thus, the generator makes quite a little progress in
training over several epochs. Pretraining helps the generator to break this vicious circle by producing images vaugeely resembling MNIST digits before full GAN training (no discriminator on this stage!). 
Here's how it works:


1. **Objective**: Minimize the difference between generated images and real MNIST samples using the **Smooth L1 Loss** (a regression loss function).
2. **Procedure**:
   - Feed random noise to the generator to produce fake images.
   - Compare generated images to real images (from the MNIST dataset).
   - Backpropagate the loss to update the generator’s parameters.
3. **Purpose**:
   - Initializes the generator close to the image manifold, avoiding mode collapse during GAN training.
   - Speeds up convergence during adversarial training.

---

## **3. How GAN Training Works**

Once generator is pretrained, GAN training starts, alternatively improving **Discriminator** and the **Generator** estimates:
1. **Discriminator Training**:
   - Input: A mix of real images (labeled as 1) and fake images from the generator (labeled as 0).
   - Loss Function: **Binary Cross-Entropy Loss (BCELoss)**.
   - Goal: Maximize the discriminator's ability to distinguish real from fake samples.

2. **Generator Training**:
   - Input: Random noise vectors.
   - Loss Function: BCELoss, where the generator aims to fool the discriminator into classifying fake images as real.
   - Goal: Minimize the discriminator’s success rate, forcing the generator to produce more realistic images.

3. **Epoch Flow**:
   - Update the discriminator.
   - Train the generator twice as often as the discriminator to counteract discriminator overfitting.
   - Add **Gaussian noise** to real images to make the discriminator robust to noisy inputs.
   - Smooth the labels for real samples (e.g., use values like 0.9 instead of 1.0) to prevent the discriminator from becoming overly confident.

---

## **4. Cost Function and Performance**
The GAN uses **Binary Cross-Entropy Loss**:
- **Discriminator Loss**:
 ```math
  \mathcal{L}_D = -\frac{1}{N} \sum_{i=1}^N [y_i \log(D(x_i)) + (1 - y_i) \log(1 - D(G(z_i)))]
   ```
  
  - $$\( y_i \)$$: Real labels for real images $$(\(\approx 1\))$$ or fake images $$(\(\approx 0\))$$.
  - $$\( D(x_i) \)$$: Discriminator’s confidence score for an input.

- **Generator Loss**:
```math
  \mathcal{L}_G = -\frac{1}{N} \sum_{i=1}^N \log(D(G(z_i)))
```
  - $$\( G(z_i) \)$$: Generated images.

### **Good Performance Reference**
- **Discriminator**: $$\( \mathcal{L}_D \approx 0.5 - 1.5 \)$$ indicates balanced performance.
- **Generator**: $$\( \mathcal{L}_G \approx 0.7 - 1.2 \)$$ suggests the generator is producing plausible samples.

### **Scientific References**
1. Goodfellow, I., et al. (2014). *Generative Adversarial Networks*. [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)
2. Miyato, T., et al. (2018). *Spectral Normalization for Generative Adversarial Networks*. [arXiv:1802.05957](https://arxiv.org/abs/1802.05957)

---

## **Installation Instructions**

Here is a `gan_env.yaml` file to set up the required Python environment using Anaconda. Run ```conda env create -f gan_env.yaml``` in your terminal to get the packages. 

### **gan_env.yaml**
```yaml
name: gan_env
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.9
  - pytorch=1.13.0
  - torchvision=0.14.0
  - numpy
  - matplotlib
  - tqdm
  - pip
  - pip:
      - torch-summary
```

---

## **Results**
- The generated images improve progressively over epochs.
- Example output after sufficient training:
![Generated Digits](gan_generated_image_epoch_1000.jpeg)

---

## **Usage**
1. Run the code to pretrain the generator.
2. Train the GAN.
3. Visualize the generated images at different stages of training.

Feel free to experiment with the hyperparameters or architecture to further improve the results!

