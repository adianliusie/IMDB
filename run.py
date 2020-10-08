import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.optim as optim

from models import *
from data_processor import text_processor

#Preprocess and load data
T = text_processor()
T.prepare_data("IMDB Dataset.csv")

dataset_size = len(T.review_ids)
train_size = int(dataset_size*0.8)
dev_size = dataset_size - train_size

training_reviews = torch.LongTensor(T.review_ids[:train_size])
training_labels = torch.LongTensor(T.labels[:train_size])
training_masks = torch.FloatTensor(T.masks[:train_size])

train_data_set = TensorDataset(training_reviews, training_labels, training_masks)
train_data_loader = DataLoader(train_data_set, batch_size=256, shuffle=True)

dev_reviews = torch.LongTensor(T.review_ids[train_size:])
dev_labels = torch.LongTensor(T.labels[train_size:])
dev_masks = torch.FloatTensor(T.masks[train_size:])

dev_data_set = TensorDataset(dev_reviews, dev_labels, dev_masks)
dev_data_loader = DataLoader(dev_data_set, batch_size=256, shuffle=False)


#Set up model and training criterions
model = Classifier(len(T.word_dict),100)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.8)
loss = nn.CrossEntropyLoss()

for epoch in range(20):
    epoch_loss = 0
    train_accuracy = [0,0]
    i = 0
    for review, labels, mask in train_data_loader:
        y_pred = model(review, mask)
        predictions = torch.argmax(y_pred, dim=1)
        output = loss(y_pred, labels)
        optimizer.zero_grad()
        output.backward()
        optimizer.step()
        epoch_loss += output
        train_accuracy = [train_accuracy[0] + (predictions == labels).sum(), train_accuracy[1] + labels.shape[0]]

        if i%100 == 0:
            print(f"{i}: {round(float(output),5)}, {round(int(train_accuracy[0])/int(train_accuracy[1]),5)}")
        i+=1
        
    dev_epoch_loss = 0
    dev_accuracy = [0,0]

    for review_d, labels_d, mask_d in dev_data_loader:
        y_pred_d = model(review_d, mask_d)
        predictions_d = torch.argmax(y_pred_d, dim=1)
        output_d = loss(y_pred_d, labels_d)
        dev_epoch_loss += output_d
        dev_accuracy = [dev_accuracy[0] + (predictions_d == labels_d).sum(), dev_accuracy[1] + labels_d.shape[0]]

    print(f"{epoch}: {round(float(epoch_loss/train_size),5)}, {round(int(train_accuracy[0])/int(train_accuracy[1]),5)}, {round(float(dev_epoch_loss/dev_size),5)}, {round(int(dev_accuracy[0])/int(dev_accuracy[1]),5)}")
