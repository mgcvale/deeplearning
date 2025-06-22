import pandas as pd
from sklearn.model_selection import train_test_split
from zyn.core import BaseModel, Dense
from zyn.func.loss import CategoricalCrossEntropy
from zyn.optim import SGD
from zyn.func.activation import ReLU, Softmax, Tanh
import numpy as np

rs = 42

df = pd.read_csv('Iris.csv')
y = pd.get_dummies(df['Species'], dtype=int).values
X = df.drop(['Id', 'Species'], axis=1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=rs)

X_train = np.array(X_train)
X_val = np.array(X_val)
y_train = np.array(y_train)
y_val = np.array(y_val)

# build the model
class Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.fc1 = Dense(in_shape=X.shape[1], out_shape=8, activation=ReLU())
        self.fc2 = Dense(in_shape=8, out_shape=8, activation=ReLU())
        self.out = Dense(in_shape=8, out_shape=y.shape[1], activation=Softmax())

        self.layers = [self.fc1, self.fc2, self.out]

    def forward(self, x):
        y = x
        for layer in self.layers:
            y = layer.forward(y)

        return y


epochs = 300
batch_size = 8
model = Model()
cce = CategoricalCrossEntropy()
optim = SGD(model, loss=cce, lr=0.005)

for i in range(epochs):
    if i == 50:
        optim.lr = 0.001
    if i == 100:
        optim.lr = 0.0005
    indicies = np.random.choice(len(X_train), size=batch_size, replace=False)
    bx = X_train[indicies]
    by = y_train[indicies]

    epoch_loss = 0
    for bxx, byy in zip(bx, by):
        y_pred = model.forward(bxx)
        loss = cce.forward(byy, y_pred)
        epoch_loss += loss

        w_grads, b_grads = optim.backpropagate(byy, y_pred)
        optim.adjust(w_grads, b_grads)

    print(f"[EPOCH {i}: loss: {epoch_loss / len(bx)}")

for x, y in zip(X_val, y_val):
    prediction = model.forward(x)
    pred_index = np.argmax(prediction)
    print(f"Prediction for {x}: {pred_index}, was {np.argmax(y)}")



# train loops



