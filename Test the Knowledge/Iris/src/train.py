import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import os


# Load Data
iris = load_iris()
X = iris.data
y = iris.target


def data_prepro(X,y):
    # standardisiert deine Eingabedaten, indem er Mittelwert = 0 und Standardabweichung = 1 für jede Feature-Spalte herstellt.
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train / Eval / Test Split (60 / 20 / 20)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
    X_eval, X_test, y_eval, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # 3. Konvertiere zu Torch-Tensoren
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_eval = torch.tensor(X_eval, dtype=torch.float32)
    y_eval = torch.tensor(y_eval, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return X_train, y_train, X_eval, y_eval, X_test, y_test


# Model definieren
class IrisModel(nn.Module):
    def __init__(self, in_features=4, h1=16, h2=8, out_features=3 ):
        super(IrisModel, self).__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x




if __name__ == "__main__":
    # Daten laden
    X_train, y_train, X_eval, y_eval, X_test, y_test = data_prepro(X=X, y=y)
    # print(f"Train: {X_train.shape} Eval: {X_eval.shape} Text: {X_test.shape}")
    # print(f"Train: {y_train.shape} Eval: {y_eval.shape} Text: {y_test.shape}")

    # Model initialisieren
    model = IrisModel(in_features=4, h1=16, h2= 8, out_features=3)
    # print(model.parameters)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Erstelle Ordner für Model
    os.makedirs("models", exist_ok=True)
    model_path = "models/iris_model.pth"


    # Training
    epochs = 100
    train_losses = []
    eval_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            eval_out = model(X_eval)
            eval_loss = criterion(eval_out, y_eval)
            eval_losses.append(eval_loss.item())

            test_out = model(X_test)
            test_preds = torch.argmax(test_out, dim=1)
            acc = accuracy_score(y_test, test_preds)
            test_accuracies.append(acc)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {loss.item():.4f} | Eval Loss: {eval_loss.item():.4f} | Test Acc: {acc:.4f}")

    # Model Speichen 
    torch.save(model.state_dict(), model_path)

    # Model hochladen
    loaded_model = IrisModel()
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.eval()


    sample_input = X_test[0].unsqueeze(0)
    pred = torch.argmax(loaded_model(sample_input), dim=1).item()
    print(f"\nBeispielvorhersage: {pred} | Tasächliche: {y_test[0].item()}")



    # Plot
    plt.plot(train_losses, label="Train Loss")
    plt.plot(eval_loss, label="Eval Loss")
    plt.plot(test_accuracies, label="Test Accuracy")
    plt.title("Training / Eval Loss & Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()






    
