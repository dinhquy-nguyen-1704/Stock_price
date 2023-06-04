import pandas as pd
import torch
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.simplefilter("ignore")

# Read data_file
data = pd.read_csv("cafef_data/FPT_data.csv", index_col = 0)
data = data.iloc[::-1].reset_index(drop = True)

# Get 2 columns "Ngay" and "Gia_dieu_chinh"
data = data[['Ngay', 'Gia_dieu_chinh']]
# String to datetime type
data['Ngay'] = pd.to_datetime(data['Ngay'], dayfirst = True)

def prepare_data(df, n_back):
    df = deepcopy(df)
    df.set_index("Ngay", inplace = True)

    for i in range(1, (n_back + 1)):
        df[f'Gia_dieu_chinh(t-{i})'] = df['Gia_dieu_chinh'].shift(i)

    df.dropna(inplace = True)

    return df

# Set new_df
n_back = 5
new_df = prepare_data(data, n_back)

# Get max and min of first column
y_max = new_df["Gia_dieu_chinh"].max()
y_min = new_df["Gia_dieu_chinh"].min()

# Dataframe to numpy
new_df_np = new_df.to_numpy()

# MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
new_df_np = scaler.fit_transform(new_df_np)

# Get X, y for traing
X = new_df_np[:,1:]
y = new_df_np[:,0]

# Train_test_split
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size = 0.2, shuffle = False)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size = 0.25, shuffle = False)

# Reshape X for trainng
X_train = X_train.reshape((-1, n_back, 1))
X_val = X_val.reshape((-1, n_back, 1))
X_test = X_test.reshape((-1, n_back, 1))
# Reshape y for trainng
y_train = y_train.reshape((-1, 1))
y_val = y_val.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))

# Tensor type for X, y
X_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train).float()
X_val = torch.tensor(X_val).float()
y_val = torch.tensor(y_val).float()
X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).float()

# Combine X, y
train_set = list(zip(X_train, y_train))
val_set = list(zip(X_val, y_val))
test_set = list(zip(X_test, y_test))

torch.save(train_set, 'cafef_data/train_set.pt')
torch.save(val_set, 'cafef_data/val_set.pt')
torch.save(test_set, 'cafef_data/test_set.pt')




