""""
Code for the training of a neuronal network based on two folders with images with one containing tiles with 
a positive cell and the other containing images of negative tiles
""""


import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision.models as models

np.random.seed(0)
torch.manual_seed(0)

%matplotlib inline

#set device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("We're using =>", device)

#location of the root directory
root_dir = "D:/Christof/cnn_database_tests/cd206/"
print("The data lies here =>", root_dir)

#definition of the image transformation function prior to training 
image_transforms = {
    "train": transforms.Compose([
        transforms.Resize((70, 70)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(), 
        
    ]),
    "validation": transforms.Compose([
        transforms.Resize((70, 70)),
        transforms.ToTensor()
    ])
    }

#location of the dataset with the training data       
dataset = datasets.ImageFolder(root = root_dir + "train",
                              transform = image_transforms["train"]
                             )
        
dataset

#allocation of the positive and negative regions to the corresponding classes for training
dataset.class_to_idx

idx2class = {v: k for k, v in dataset.class_to_idx.items()}

def get_class_distribution(dataset_obj):
    count_dict = {k:0 for k,v in dataset_obj.class_to_idx.items()}    
    
    for _, label_id in dataset_obj:
        label = idx2class[label_id]
        count_dict[label] += 1
    return count_dict

#description of the dataset
dataset_size = len(dataset)
print('Size Dataset:' , dataset_size)
dataset_indices = list(range(dataset_size))

np.random.shuffle(dataset_indices)

#splitting of the dataset in training, validation and test set set
val_split_index = int(np.floor(0.2 * dataset_size))

print(val_split_index)
train_idx, val_idx = dataset_indices[val_split_index:], dataset_indices[:val_split_index]

train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)

dataset_test = datasets.ImageFolder(root = root_dir + "test",
                                            transform = image_transforms["validation"]
                                           )
dataset_test

#Dataloader for the Datasets for training, validation and testing
train_loader = DataLoader(dataset= dataset, shuffle=False, batch_size=64, sampler=train_sampler)
val_loader = DataLoader(dataset= dataset, shuffle=False, batch_size=1, sampler=val_sampler)
test_loader = DataLoader(dataset= dataset_test, shuffle=False, batch_size=1)

def get_class_distribution_loaders(dataloader_obj, dataset_obj):
    count_dict = {k:0 for k,v in dataset_obj.class_to_idx.items()}    
    
    if dataloader_obj.batch_size == 1:    
        for _,label_id in dataloader_obj:
            y_idx = label_id.item()
            y_lbl = idx2class[y_idx]
            count_dict[str(y_lbl)] += 1
    else: 
        for _,label_id in dataloader_obj:
            for idx in label_id:
                y_idx = idx.item()
                y_lbl = idx2class[y_idx]
                count_dict[str(y_lbl)] += 1
    return count_dict


single_batch = next(iter(train_loader))

single_batch[0].shape


#definition of the model, loading a preexisting resnet18 and sending it to device
model =  torchvision.models.resnet18(pretrained = False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = nn.DataParallel(model, device_ids=[0, 1])
model.to(device)

#setting the learning rate
learning_rate = 0.000005
#definition of the loss criterion and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr= learning_rate)

#definition of the metrics used for the validation of the trained net
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_tag, dim = 1)    
    correct_results_sum = (y_pred_tags == y_test).sum().float()    
    
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)    
    
    return acc

accuracy_stats = {
    'train': [],
    "val": []
}
loss_stats = {
    'train': [],
    "val": []
}

print("Begin training.")

#Training and Validation of the net

#e: numboer of training iterations
for e in tqdm(range(1, 50)):    
    print('learning Rate:' , learning_rate)
    train_epoch_loss = 0
    train_epoch_acc = 0    
    model.train()
    for X_train_batch, y_train_batch in train_loader:
        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
        optimizer.zero_grad()        

        y_train_pred = model(X_train_batch).squeeze()        

        train_loss = criterion(y_train_pred, y_train_batch)
        train_acc = binary_acc(y_train_pred, y_train_batch)        

        train_loss.backward()
        optimizer.step()        

        train_epoch_loss += train_loss.item()
        train_epoch_acc += train_acc.item()
    # VALIDATION
    with torch.no_grad():
        model.eval()
        val_epoch_loss = 0
        val_epoch_acc = 0
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)            

            y_val_pred = model(X_val_batch).squeeze()
            y_val_pred = torch.unsqueeze(y_val_pred, 0)            

            val_loss = criterion(y_val_pred, y_val_batch)
            val_acc = binary_acc(y_val_pred, y_val_batch)            

            val_epoch_loss += train_loss.item()
            val_epoch_acc += train_acc.item()    
    loss_stats['train'].append(train_epoch_loss/len(train_loader))
    loss_stats['val'].append(val_epoch_loss/len(val_loader))
    accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
    accuracy_stats['val'].append(val_epoch_acc/len(val_loader))
    print(f'Epoch {e+0:02}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | \
    Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}')
    
#save the trained model    
torch.save(model.state_dict(), 'D:/Christof/trained_networks/211001_resnet50_trained_cd206.pth')
