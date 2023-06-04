import pandas as pd
import torch
import warnings
warnings.simplefilter("ignore")

# Define device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Get y_max, y_min
data = pd.read_csv("cafef_data/FPT_data.csv", index_col = 0)
data = data.iloc[::-1].reset_index(drop = True)
y_max = data["Gia_dieu_chinh"].max()
y_min = data["Gia_dieu_chinh"].min()

model = torch.load("model.pt")
test_set = torch.load("cafef_data/test_set.pt")

def predict(x_test):
  with torch.no_grad():
    predict = model(x_test.unsqueeze(0))
    return (predict[0][0].item())

x_test = test_set[10][0].to(device)
y_predict = predict(x_test)
y_predict = round((y_predict*(y_max - y_min) + y_min), 2)
y_real = round((test_set[10][1].item()*(y_max - y_min) + y_min), 2)
print("y_predict:",y_predict)
print("ground truth:",y_real)

