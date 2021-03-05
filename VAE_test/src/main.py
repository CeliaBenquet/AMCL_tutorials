import argparse
import matplotlib
import torch.optim as optim
import torch.nn as nn
import torch
import torchvision

import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from utils import prepare_data, final_loss
import net

matplotlib.style.use('ggplot')


# construct the argument parser and parser the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', default=10, type=int,
                    help='number of epochs to train the VAE for')

parser.add_argument('--training', action = 'store_true', default = False,
                        help = 'Train the model on the MNIST dataset') 

parser.add_argument('--testing', action = 'store_true', default = False, 
                        help = 'Testing the model on the MNIST dataset')

args = parser.parse_args()


# learning parameters
epochs = args.epochs
batch_size = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare data
train_data, val_data, train_loader, val_loader = prepare_data(batch_size)

# Initialization
model = net.LinearVAE().to(device)
criterion = nn.BCELoss(reduction='sum') # to calculate reconstruction loss


# ===================  execute training and testing functions =================== #

train_loss = []
val_loss = []

for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")


    if args.training:
        # optimizer initilization 
        lr = 0.0001
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # start training mode 
        model.train()
        running_loss = 0.0

        for i, data in tqdm(enumerate(train_loader), total=int(len(train_data)/train_loader.batch_size)):
            data, _ = data
            data = data.to(device)
            data = data.view(data.size(0), -1)

            optimizer.zero_grad()
            reconstruction, mu, logvar = model(data)

            # Losses 
            bce_loss = criterion(reconstruction, data)
            loss = final_loss(bce_loss, mu, logvar)
            running_loss += loss.item() #batch loss

            #Backward pass
            loss.backward()
            optimizer.step()

        train_epoch_loss = running_loss/len(train_loader.dataset) #mean 

        # stock and display 
        train_loss.append(train_epoch_loss)
        print(f"Train Loss: {train_epoch_loss:.4f}")
    
    if args.testing: 
        model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for i, data in tqdm(enumerate(val_loader), total=int(len(val_data)/val_loader.batch_size)):
                data, _ = data
                data = data.to(device)
                data = data.view(data.size(0), -1)

                reconstruction, mu, logvar = model(data)

                bce_loss = criterion(reconstruction, data)
                loss = final_loss(bce_loss, mu, logvar)
                running_loss += loss.item()
            
                # save the last batch input and output of every epoch
                if i == int(len(data)/val_loader.batch_size) - 1:
                    num_rows = 8
                    both = torch.cat((data.view(val_loader.batch_size, 1, 28, 28)[:8], 
                                    reconstruction.view(val_loader.batch_size, 1, 28, 28)[:8]))
                    save_image(both.cpu(), f"../outputs/output{epoch}.png", nrow=num_rows)
        
        val_epoch_loss = running_loss/len(val_loader.dataset)

        # stock and display         
        val_loss.append(val_epoch_loss)
        print(f"Val Loss: {val_epoch_loss:.4f}")


