# Sparse Autoencoder and Anomaly Detection in Images

This project consists of two Python files, `sparse_autoencoder.py` and `anomalies.py`, which implement a Sparse Autoencoder (SAE) and anomaly detection in images. In my use case, I trained the SAE on a dataset of **images of the same object**, and then used the SAE to find anomalies in the dataset.

## Files

### sparse_autoencoder.py

This file contains the implementation of the Sparse Autoencoder

### anomalies.py

This file uses the Sparse Autoencoder to find anomalies in the dataset

## Usage

To use this project, first ensure that you have the necessary dependencies installed. These include PyTorch, numpy, matplotlib, PIL, and wandb.

Next, run the `sparse_autoencoder.py` script to train the Sparse Autoencoder on your dataset. This will print the loss on the training and validation sets every 200 epochs, and will plot some sample reconstructions.

Finally, you can run the `anomalies.py` script to find anomalies in your dataset. This will print the smallest and largest losses, and will plot the images with the smallest and largest losses.

## Configuration

You can configure the following parameters:

- `D`: The dimension of the images (D x D x 3).
- `PATH`: The path to the images.
- `niters`: The number of iterations for training (default is 5000).
- `learning_rate`: The learning rate for the Adam optimizer (default is 0.001).

You may also want to change the loss function or use a different optimizer, etc.

## Note

This project uses the wandb library for logging metrics. You will need to log in to your wandb account before running the scripts.
