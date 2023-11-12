import torch
import numpy as np
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

Device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu") 
   
# Define the LSTM text classifier
class LSTMTextClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMTextClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        hidden_state = None
    # Move input tensor to the same device as the LSTM parameters
        input = input.to(self.lstm.weight_ih_l0.device)
        output, (hidden_state, _) = self.lstm(input, hidden_state)
        output = self.fc(torch.cat((hidden_state[-2,:,:], hidden_state[-1,:,:]), dim=1))
        output = self.softmax(output)
        return output


# Define the training data
def set_seed(seed=2008):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
set_seed()
# Load the BERT model and tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(Device)

# Set up the LSTM text classifier with BERT embeddings
input_size = 768 # BERT base model outputs 768-dimensional embeddings
hidden_size = 128
output_size = 2 # Binary classification task (e.g. sentiment analysis)
lstm_classifier = LSTMTextClassifier(input_size, hidden_size, output_size)
lstm_classifier=lstm_classifier.to(Device)

# Define the loss function and optimizer
criterion = nn.NLLLoss()
optimizer = torch.optim.AdamW(lstm_classifier.parameters(), lr=0.01)

# Set up training parameters related to Nuno config
num_epochs = 100
batch_size = 256
lstm_classifier.train()
# Define the training data
df = pd.read_csv('all_data1.csv', delimiter='\t', header=None)
df = df.sample(frac=1).reset_index(drop=True)
df = df.loc[df[4] != 'relevant']
#train_data, test_data = train_test_split(df, train_size=0.7, test_size=0.3)
#val_data, train_data = train_test_split(train_data, val_size=0.2, train_size=0.8)

train_data, test_data = train_test_split(df, test_size=0.3, random_state=42)
# Split the training data into training and validation sets
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

labels1 = train_data[4]
text=train_data[3]
training_data = list(zip(text,labels1))
for i in range(len(training_data)):
    if training_data[i][1].isdigit():
       training_data[i] = (training_data[i][0], int(training_data[i][1]))
# Define the data loader
def collate_fn(data):
    sentences, labels = zip(*data)
    input_ids = bert_tokenizer.batch_encode_plus(sentences, padding=True, truncation=True, return_tensors='pt')['input_ids']
    input_ids=input_ids.to(Device)
    embeddings = bert_model(input_ids)[0]
    return embeddings, torch.tensor(labels, dtype=torch.long).squeeze()

data_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, collate_fn=collate_fn)
file_path = "bert-validation-accuracy.txt"
lstm_classifier.train()

# Example training loop
for epoch in range(num_epochs):
    for embeddings, labels in tqdm(data_loader):
        embeddings=embeddings.to(Device)
        labels=labels.to(Device)
        # Pass BERT embeddings through the LSTM text classifier
        lstm_classifier.zero_grad()
        lstm_classifier.train()  # set the model to training mode
        output = lstm_classifier(embeddings)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

    # Evaluate your LSTM model on the validation set after each epoch
    labels2 = val_data[4]
    text = val_data[3]
    validation_data = list(zip(text, labels2))
    for j in range(len(validation_data)):
        if validation_data[j][1].isdigit():
            validation_data[j] = (validation_data[j][0], int(validation_data[j][1]))

    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, collate_fn=collate_fn)
    lstm_classifier.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient tracking to save memory
        correct = 0
        total = 0
        for embeddings, labels in validation_loader:
            output = lstm_classifier(embeddings)
            predictions = torch.argmax(output, dim=1).to(labels.device)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Epoch {epoch}: Validation accuracy: {accuracy:.2f}")
    with open(file_path, "a") as f:
        f.write(f"Epoch {epoch}: Validation accuracy: {accuracy:.2f}\n")
# Evaluate your LSTM model on the test set

labels3 = test_data[4]
text=test_data[3]
test_data = list(zip(text,labels3))
for k in range(len(test_data)):
    if test_data[k][1].isdigit():
        test_data[k] = (test_data[k][0], int(test_data[k][1]))

test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn)
lstm_classifier.eval()  # Set the model to evaluation mode

with torch.no_grad():  # Disable gradient tracking to save memory
    correct = 0
    total = 0
    for embeddings, labels in test_loader:
        output = lstm_classifier(embeddings)
        predictions = torch.argmax(output, dim=1).to(labels.device)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Test accuracy: {accuracy:.2f}")

#First result with 5 epoch and 32 batch size, Validation accuracy: 0.86
#Test accuracy: 0.87
print(f"bert-test-accuracy: {accuracy:.2f}")
file_path = "bert-test-accuracy.txt"
with open(file_path, "a") as f:
        f.write(f"Epoch {epoch}: test accuracy: {accuracy:.2f}\n")


    
