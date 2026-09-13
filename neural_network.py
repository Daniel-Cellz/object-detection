import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load the flower dataset
data = load_iris()
X = data.data          # measurements (petal length, width, etc.)
y = data.target        # flower type (0, 1, or 2)

# 2. Split into training data and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scale the numbers so the network learns better
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Convert data into PyTorch's format (tensors)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# 5. Define the neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 16)   # 4 inputs -> 16 neurons
        self.layer2 = nn.Linear(16, 8)   # 16 -> 8 neurons
        self.layer3 = nn.Linear(8, 3)    # 8 -> 3 outputs (flower types)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

model = SimpleNet()

# 6. Set up the learning process
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 7. Train the network (show it examples 100 times)
print("Training started...\n")
for epoch in range(100):
    optimizer.zero_grad()
    predictions = model(X_train)
    loss = loss_function(predictions, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/100 - Loss: {loss.item():.4f}")

# 8. Test how well it learned
with torch.no_grad():
    test_predictions = model(X_test)
    predicted_labels = torch.argmax(test_predictions, dim=1)
    accuracy = (predicted_labels == y_test).float().mean()
    print(f"\nTraining complete! Test Accuracy: {accuracy.item()*100:.2f}%")