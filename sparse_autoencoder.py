import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import wandb

# The dimension of the images: D x D x 3
D = 32
# The path to the images
PATH = 'images'


def read_jpg(path):
    '''read a jpg image from the disk and return it as a DxDx3 numpy array.'''
    img = Image.open(path)
    img = img.resize((D,D), Image.ANTIALIAS)
    img = np.asarray(img)
    return img


def read_dir(directory):
    '''read all jpg images in a directory and return them as a numpy array.'''
    imgs = []

    for file in os.listdir(directory):
        if file.endswith('.jpg'):
            img = read_jpg(os.path.join(directory, file))
            imgs.append(img)
            
    return np.array(imgs)
    

imgs = read_dir(PATH)

# print('Dimensions of the image data:', imgs.shape)
# print('Example image:')
# plt.imshow(imgs[17])
# plt.show()

# Convert numpy array to torch tensor
imgs = torch.from_numpy(imgs)

# flatten the images
# Set second dimension to -1, which is a placeholder PyTorch will infer the size based on the input
imgs = torch.reshape(imgs, (imgs.shape[0], -1))

# normalize color values of the images
imgs = imgs / 255.0


def plot_reconstructions(imgs, recs):
    '''helper function to plot original and reconstructed images.'''

    # Define the number of images to display
    NUM_IMAGES = 8

    # Calculate the number of rows and columns for the subplot grid
    N = int(np.ceil(math.sqrt(2 * NUM_IMAGES)))

    # Create the subplot grid
    fig, axarr = plt.subplots(nrows=N, ncols=N, figsize=(18, 18))

    for i in range(NUM_IMAGES):
        # Display the original image
        img_ax = axarr[2*i // N, 2*i % N]
        img_ax.imshow(imgs[i].reshape((D, D, 3)), interpolation='nearest')

        # Display the reconstructed image
        rec_ax = axarr[(2*i + 1) // N, (2*i + 1) % N]
        rec_ax.imshow(recs[i].reshape((D, D, 3)), interpolation='nearest')

    fig.tight_layout()
    plt.show()
    plt.close()


class SAE(nn.Module):
    def __init__(self, D):
        super(SAE, self).__init__()

        # define the sparse autoencoder network as a encoder and decoder
        self.encoder = nn.Sequential(
            nn.Linear(D*D*3, 1000),
            nn.Sigmoid(),
            nn.Linear(1000, 100),
            nn.Sigmoid(),
            nn.Linear(100, 50),
            nn.Sigmoid())
        
        self.decoder = nn.Sequential(
            nn.Linear(50, 100),
            nn.Sigmoid(),
            nn.Linear(100, 1000),
            nn.Sigmoid(),
            nn.Linear(1000, D*D*3),
            nn.Sigmoid())

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded


# The loss function is the mean squared error loss
loss_function = nn.MSELoss()

# Split the dataset into training, validation and test set
imgs_train = imgs[:-500]
imgs_val = imgs[-100:]
imgs_test = imgs[-500:-100]


def run_training(net, imgs_train, imgs_val, niters=5000, learning_rate=0.001):
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
    # We feed the network all images in the dataset on every epoch
    for epoch in range(niters):
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward + backward + optimize
        y_train = net(imgs_train)
        y_val = net(imgs_val)
        # We use the flattened images as targets
        loss_train = loss_function(y_train, imgs_train)
        loss_val = loss_function(y_val, imgs_val)

        loss_train.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            print('Epoch %d, loss on training set: %.4f, loss on validation set: %.4f' %(epoch, loss_train.item(), loss_val.item()))

            # Log metrics to wandb
            wandb.log({"training loss": loss_train.item(), "valdation loss": loss_val.item(), "epoch": epoch})

            # Sample 8 random images from the training and validation data sets
            indices_train = torch.randint(0, len(imgs_train), (4,))
            indices_val = torch.randint(0, len(imgs_val), (4,))

            sample_imgs = np.concatenate((imgs_train[indices_train].cpu().numpy(), imgs_val[indices_val].cpu().numpy()), axis=0)
            sample_recs = np.concatenate((y_train[indices_train].detach().cpu().numpy(), y_val[indices_val].detach().cpu().numpy()), axis=0)

            plot_reconstructions(sample_imgs[:4], sample_recs[:4], epoch)
            plot_reconstructions(sample_imgs[4:], sample_recs[4:], epoch)
    
    print('Finished Training')


if __name__ == "__main__":
    wandb.login()
    wandb.init(
        project="sae",
        
        config={
            "learning_rate": 0.001,
            "architecture": "Autoencoder",
            "dataset": "some dataset",
            "epochs": 10000,
        }
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)

    imgs_train = imgs_train.to(device)
    imgs_val = imgs_val.to(device)

    net = SAE(D=D).to(device)
    # Set the network to training mode
    net.train()
    run_training(net, imgs_train, imgs_val)
    wandb.finish()

    # Save the trained network
    # torch.save(net.state_dict(), 'sae.pth')
