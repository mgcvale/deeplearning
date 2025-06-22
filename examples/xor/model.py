from zyn.core import BaseModel, Dense
from zyn.func.activation import Tanh, Sigmoid
from zyn.func.loss import BinaryCrossEntropy
from zyn.optim import SGD


class Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.fc1 = Dense(in_shape=2, out_shape=4, activation=Tanh())
        self.fc2 = Dense(in_shape=4, out_shape=2, activation=Tanh())
        self.out = Dense(in_shape=2, out_shape=1, activation=Sigmoid())
        self.layers = [self.fc1, self.fc2, self.out]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)

        return x[0]

dataset = [
        [[0, 0], 0],
        [[0, 1], 1],
        [[1, 0], 1],
        [[1, 1], 0]
]

epochs = 1500
model = Model()
bce = BinaryCrossEntropy()
optimizer = SGD(model, bce, lr=0.05)

# train the model
for i in range(epochs):
    total_loss = 0

    for j in range(len(dataset)):
        x, y = dataset[j][0], dataset[j][1]
        y_pred = model.forward(x)
        loss = bce.forward(y, y_pred)
        total_loss += loss

        w_grads, b_grads = optimizer.backpropagate(y, y_pred)
        optimizer.adjust(w_grads, b_grads)

    print(f"[ EPOCH {i} ] - BCE: {total_loss / len(dataset)}")

for data in dataset:
    prediction = model.forward(data[0])
    print(f"Prediction for {data[0]}: {prediction}, was {data[1]}")

