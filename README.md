CIFAR-100 Image Classification with PyTorch
Project Overview
This project implements a deep convolutional neural network (CNN) using PyTorch to classify images from the CIFAR-100 dataset, which contains 60,000 32x32 color images across 100 classes. The model leverages a ResNet-inspired architecture with advanced features like residual connections, batch normalization, dropout, and data augmentation to achieve an accuracy above 80% on the test set. The code includes training, evaluation, and inference on custom images, with detailed logging to track execution progress.
This README provides a comprehensive guide, from setting up the environment to understanding the advanced techniques used, making it accessible for beginners and informative for experienced users.
Table of Contents

Prerequisites
Installation
Dataset
Model Architecture
Training Process
Evaluation
Inference on Custom Images
Advanced Features
Usage
File Structure
Troubleshooting
Contributing
License

Prerequisites
To run this project, you need the following:

Python: Version 3.8 or higher.
PyTorch: Version 1.9 or higher, with torchvision.
Hardware: A GPU (CUDA-compatible) is recommended for faster training, but the code will fall back to CPU if no GPU is available.
Dependencies: Listed in the Installation section.
Storage: Approximately 500 MB for the CIFAR-100 dataset and model weights.

Installation

Clone the Repository (if applicable):
git clone <repository-url>
cd cifar100-classification


Set Up a Virtual Environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:Install the required Python packages using pip:
pip install torch torchvision numpy pillow


Verify PyTorch Installation:Run the following to ensure PyTorch is installed correctly:
import torch
print(torch.__version__)
print(torch.cuda.is_available())  # Should return True if a GPU is available



Dataset
The CIFAR-100 dataset is automatically downloaded by the script to the ./data directory when you run it for the first time. It consists of:

Training Set: 50,000 images.
Test Set: 10,000 images.
Classes: 100 distinct classes (e.g., apple, bicycle, dog).
Image Size: 32x32 pixels, RGB format.

The dataset is preprocessed with:

Training Transformations: Random horizontal flips, rotations, and crops for data augmentation, followed by normalization using CIFAR-100 mean and standard deviation ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)).
Test Transformations: Only normalization is applied to ensure consistent evaluation.

Model Architecture
The model is a ResNet-inspired CNN designed for CIFAR-100 classification, incorporating advanced deep learning techniques to achieve high accuracy and efficiency.
Key Components

Initial Convolution: A 3x3 convolutional layer with 64 filters to extract low-level features.
Residual Blocks: Three layers of residual blocks (inspired by ResNet) with increasing channel sizes (64, 128, 256). Each block includes:
Two 3x3 convolutions with batch normalization and ReLU activation.
A shortcut connection to mitigate vanishing gradients.
Stride adjustments for downsampling in deeper layers.


Pooling: Adaptive average pooling to reduce spatial dimensions to 1x1.
Dropout: 50% dropout before the final layer to prevent overfitting.
Fully Connected Layer: Maps the 256-channel features to 100 classes.

Residual Block Details
Each residual block consists of:

Two convolutional layers with batch normalization and ReLU.
A shortcut connection that either matches dimensions (identity) or uses a 1x1 convolution for dimension alignment.

Model Summary

Layers: Initial conv + 3 residual layers (each with 2 blocks) + pooling + dropout + fully connected.
Parameters: Approximately 1.5 million, optimized for CIFAR-100's complexity.
Output: Logits for 100 classes.

Training Process
The model is trained for 50 epochs with the following settings:

Batch Size: 128 images for efficient GPU utilization.
Optimizer: AdamW with a learning rate of 0.001 and weight decay of 0.01 for regularization.
Loss Function: Cross-Entropy Loss for multi-class classification.
Learning Rate Scheduler: Cosine Annealing to gradually reduce the learning rate over 50 epochs, improving convergence.
Data Loading: Uses 4 worker threads and pin_memory=True for faster data transfer to GPU.

Logging
The script includes print statements to track:

Script initialization.
Dataset loading and transformations.
Model and layer initialization.
Training progress (epoch and batch-level).
Evaluation and inference steps.

The training loss is printed after each epoch, and the model weights are saved to cifar100_net.pth.
Evaluation
After training, the model is evaluated on the CIFAR-100 test set (10,000 images). The evaluation process:

Runs in inference mode (net.eval()).
Disables gradient computation for efficiency.
Computes accuracy by comparing predicted class indices with ground truth labels.

Expected test accuracy: >80%, achieved through the combination of residual connections, data augmentation, and advanced optimization.
Inference on Custom Images
The script supports classifying custom images using the trained model. To use this feature:

Place your images (e.g., image1.jpg, image2.jpg) in the ../images/ directory.
The script applies the same test transformations (resize to 32x32, normalize) and predicts the class index (0–99).
If an image is not found, a warning is printed.

Example output:
image1.jpg predicted as class 42

Note: CIFAR-100 class names are not explicitly labeled in the script (they are class_0 to class_99). Refer to the CIFAR-100 documentation for specific class names.
Advanced Features
This project incorporates several advanced techniques to achieve high accuracy and efficiency:

Residual Connections: Prevent vanishing gradients and allow deeper networks.
Batch Normalization: Normalizes layer outputs to stabilize and accelerate training.
Dropout: Reduces overfitting by randomly disabling 50% of neurons in the final layer.
Data Augmentation: Random flips, rotations, and crops increase dataset diversity, improving generalization.
AdamW Optimizer: Combines adaptive learning rates with weight decay for better convergence.
Cosine Annealing Scheduler: Gradually reduces the learning rate to fine-tune the model in later epochs.
Efficient Data Loading: Uses pin_memory and multiple workers to minimize data transfer bottlenecks.

These features make the model robust and capable of achieving state-of-the-art performance on CIFAR-100.
Usage

Run the Script:Ensure the dependencies are installed, then execute:
python cifar100_cnn.py

This will:

Download the CIFAR-100 dataset (if not already present).
Train the model for 50 epochs.
Save the model weights to cifar100_net.pth.
Evaluate the model on the test set.
Predict classes for custom images (if provided).


Custom Image Prediction:Place images in the ../images/ directory and update the image_paths list in the script. For example:
image_paths = ['../images/your_image.jpg']


Monitor Progress:The script prints detailed logs for:

Initialization of components.
Epoch and batch progress during training.
Evaluation and inference steps.



File Structure
cifar100-classification/
├── cifar100_cnn.py       # Main script for training, evaluation, and inference
├── data/                 # Directory for CIFAR-100 dataset (auto-downloaded)
├── cifar100_net.pth      # Saved model weights
├── images/               # Directory for custom images (user-provided)
└── README.md             # This file

Troubleshooting

GPU Not Found: If torch.cuda.is_available() returns False, ensure CUDA is installed and compatible with your GPU. Alternatively, the script will use the CPU.
Out of Memory: Reduce the batch size (e.g., from 128 to 64) if you encounter GPU memory issues.
File Not Found: Ensure custom images exist in the specified paths. Update image_paths if needed.
Low Accuracy: Verify that the dataset is downloaded correctly and that training runs for all 50 epochs. Check for overfitting by comparing training and test losses.

For further assistance, consult the PyTorch documentation or raise an issue in the repository.
Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a feature branch (git checkout -b feature-name).
Commit your changes (git commit -m "Add feature").
Push to the branch (git push origin feature-name).
Open a pull request.

Please ensure your code follows PEP 8 style guidelines and includes appropriate tests.
License
This project is licensed under the MIT License. See the LICENSE file for details.

This project was developed to demonstrate a high-performance image classification pipeline using PyTorch. For questions or feedback, contact the repository maintainer or open an issue.
