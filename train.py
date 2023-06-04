import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.utils.data as dutils
from LSTM import LSTM
import warnings
warnings.simplefilter("ignore")

# Load data
train_set = torch.load('cafef_data/train_set.pt')
val_set = torch.load('cafef_data/val_set.pt')

# Dataloader
batch_size = 16
train_loader = dutils.DataLoader(train_set, batch_size = batch_size, shuffle = True, drop_last = True)
val_loader = dutils.DataLoader(val_set, batch_size = batch_size, shuffle = False, drop_last = False)

# Define device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define model
model = LSTM(1, 1, 1)

# Define visualize_losses function
def visualize_losses(train_losses, val_losses):
    plt.plot(train_losses, color="green", linewidth=1)
    plt.plot(val_losses, color="purple", linestyle="--", linewidth=1)
    plt.xlabel('epoch')
    plt.ylabel('losses')
    plt.legend(['Train', 'Valid'])
    plt.title('Train - valid losses')

    plt.show()

# Define loss_function
lossf = nn.MSELoss()

# Define train_fumction
def train(model, train_loader, val_loader, lr, epochs):
  # Put model on device
  model = model.to(device)
  # Optimization algorithm
  optimizer = optim.Adam(lr = lr, params = model.parameters())

  train_losses = []
  val_losses = []

  best_val_loss = 1e100
  best_state_dict = None

  for ei in tqdm(range(epochs)):
    train_lossi = []
    for bi, (xi, yi) in enumerate(train_loader):
      optimizer.zero_grad()
      xi = xi.type(torch.FloatTensor)
      xi = xi.to(device)
      yi = yi.type(torch.FloatTensor)
      yi = yi.to(device)
      yi_hat = model(xi)
      lossi = lossf(yi_hat, yi)
      lossi.backward()
      optimizer.step()
      train_lossi.append(lossi.item())

    val_lossi = []

    with torch.no_grad():
      for bvi, (xvi, yvi) in enumerate(val_loader):
        xvi = xvi.type(torch.FloatTensor)
        xvi = xvi.to(device)
        yvi = yvi.type(torch.FloatTensor)
        yvi = yvi.to(device)
        yvi_hat = model(xvi)
        lossvi = lossf(yvi_hat, yvi)
        val_lossi.append(lossvi.item())

    train_losses.append(torch.FloatTensor(train_lossi).mean().item())
    val_losses.append(torch.FloatTensor(val_lossi).mean().item())
    if val_losses[-1] < best_val_loss:
      best_val_loss = val_losses[-1]
      best_state_dict = model.state_dict()
    tqdm.write(" train_loss %.4f - val_loss %.4f" % (train_losses[-1], val_losses[-1]))
  model.load_state_dict(best_state_dict)  # parameters of moded

  return model, train_losses, val_losses

lr = 0.05
epochs = 50
model, train_losses, val_losses = train(model, train_loader, val_loader, lr = lr, epochs = epochs)

# Visualize output
visualize_losses(train_losses, val_losses)

# Save model
torch.save(model, "model.pt")