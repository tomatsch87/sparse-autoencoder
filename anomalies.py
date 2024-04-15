import torch
import torch.optim as optim
import torch.nn as nn
from sparse_autoencoder import SAE, imgs, D, plot_reconstructions

# This script can be used to find anomalies in the dataset. It uses a different training loop and prints some of the smallest and largest losses and also plots 20 images with the smallest and largest losses.
def find_anomalies(net, imgs, niters=5000, learning_rate=0.001):
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # Define the loss function with reduction='none' to get a loss for each image 
    loss_function = nn.MSELoss(reduction='none')
    
    # We feed the network all images in the dataset on every epoch
    for epoch in range(niters):
        optimizer.zero_grad()
        
        # Forward + backward + optimize
        y = net(imgs)
        # We use the flattened images as targets and compute the loss for each image individually
        loss = loss_function(y, imgs)
        # Compute the mean loss across all images
        mean_loss = loss.mean()
        # Get the mean loss across the dimensions of the image, shape (len(imgs),)
        dim_loss = loss.mean(dim=1)

        mean_loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            print('Epoch %d, loss on training set: %.4f' %(epoch, mean_loss.item()))

            dim_loss.detach_()
            dim_loss = dim_loss.cpu().numpy()

            # Create a list of tuples (loss, index) and sort it in ascending order
            dim_loss = [(loss, idx) for idx, loss in enumerate(dim_loss)]
            dim_loss.sort(key=lambda x: x[0])

            # Print the 5 smallest and largest losses
            print('Smallest losses:', dim_loss[:5])
            print('Largest losses:', dim_loss[-5:])

            # Get the indices of 20 images with the highest and lowest losses
            indices_min = [idx for _, idx in dim_loss[:20]]
            indices_max = [idx for _, idx in dim_loss[-20:]]

            # Convert the indices to a torch tensor with concatenated indices, first the min indices, then the max indices
            indices = torch.cat((torch.tensor(indices_min, dtype=torch.int64), torch.tensor(indices_max, dtype=torch.int64)))

            # Get the corresponding images and reconstructions and convert them to numpy arrays
            anomaly_imgs = imgs[indices].cpu().numpy()
            anomaly_recs = y[indices].detach().cpu().numpy()

            plot_reconstructions(anomaly_imgs[:20], anomaly_recs[:20], epoch)
            plot_reconstructions(anomaly_imgs[20:], anomaly_recs[20:], epoch)
    
    print('Finished Training')


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)

    imgs = imgs.to(device)
    net = SAE(D=D).to(device)
    find_anomalies(net, imgs)
    