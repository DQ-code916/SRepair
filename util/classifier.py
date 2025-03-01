import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

def Classifier(file_path,IN,Ig):
    data = pd.read_csv(file_path, index_col=0) 
    X = data.iloc[:, :-1].values  
    y = data.iloc[:, -1].values  
    if isinstance(y[0], str):
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        num_classes = len(label_encoder.classes_)
    else:
        y_encoded = y
        num_classes = len(set(y))  
    X_train = X[Ig]
    y_train = y_encoded[Ig]
    X_test = X[IN]
    y_test = y_encoded[IN]
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)


    class SimpleClassifier(torch.nn.Module):
        def __init__(self, input_size, num_classes):
            super(SimpleClassifier, self).__init__()
            self.fc1 = torch.nn.Linear(input_size, 50)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(50, num_classes)
        
        def forward(self, x):
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)
            return out

    input_size = X.shape[1]
    model = SimpleClassifier(input_size, num_classes)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 50
    for epoch in range(num_epochs):
        for batch_idx, (features, labels) in enumerate(train_loader):
            outputs = model(features)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()  
    with torch.no_grad():  
        correct = 0
        total = 0
        for features, labels in test_loader:
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    if total!=0:
        pre= correct / total
    else :
        pre=0
    return pre

