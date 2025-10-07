## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

<img width="756" height="809" alt="489861149-0509383a-d594-4eb5-aa28-237dc84ae6d5" src="https://github.com/user-attachments/assets/9aa34031-97ec-42d1-9866-721149ecb9e7" />

## DESIGN STEPS

### STEP 1:
Import necessary libraries and load the dataset.

### STEP 2:
Encode categorical variables and normalize numerical features.

### STEP 3:
Split the dataset into training and testing subsets.

### STEP 4:
Design a multi-layer neural network with appropriate activation functions.

### STEP 5:
Train the model using an optimizer and loss function.

### STEP 6:
Evaluate the model and generate a confusion matrix.

### STEP 7:
Use the trained model to classify new data samples.

### STEP 8:
Display the confusion matrix, classification report, and predictions.

## PROGRAM

### Name: GOKUL PRAKASH M
### Register Number:212223240041

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 4)
    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=F.relu(self.fc4(x))
        x=self.fc5(x)
        return x

```
```python
# Initialize the Model, Loss Function, and Optimizer
model =PeopleClassifier(input_size=X_train.shape[1])
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)

train_model(model,train_loader,criterion,optimizer,epochs=100)

```
```python
def train_model(model,train_loader,criterion,optimizer,epochs):
  for epoch in range(epochs):
    model.train()
    for X_batch,y_batch in train_loader:
      optimizer.zero_grad()
      outputs=model(X_batch)
      loss=criterion(outputs,y_batch)
      loss.backward()
      optimizer.step()

  if(epoch+1)%10==0:
    print(f'Epoch [{epoch+1}/{epochs}],Loss:{loss.item():.4f}')
```



## Dataset Information

<img width="933" height="213" alt="image" src="https://github.com/user-attachments/assets/a50990aa-e0e9-4021-8a10-8d169c6167dc" />

## OUTPUT



### Confusion Matrix

<img width="539" height="455" alt="download" src="https://github.com/user-attachments/assets/018a3b90-bfea-41e3-bc23-3f9d0de82f76" />


### Classification Report

<img width="392" height="324" alt="image" src="https://github.com/user-attachments/assets/5241f941-53ec-448d-9dbb-a649767f1cea" />


### New Sample Data Prediction

<img width="751" height="269" alt="image" src="https://github.com/user-attachments/assets/4d2a1ca0-aac4-48bb-a53e-556c7826a1fd" />

## RESULT
Thus, a neural network classification model for the given dataset as been created successfully.
