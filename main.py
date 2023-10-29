#!/usr/bin/env python
# coding: utf-8

# # Description

# Performing sentimal analysis on IMDB dataset using RNNs.

# In[1]:


import re
from collections import Counter, OrderedDict

import spacy
import torch
import torch.nn as nn
from bs4 import BeautifulSoup
from torch.utils.data.dataset import random_split
from torchtext.datasets import IMDB
from torchtext.vocab import vocab
import pandas as pd

torch.manual_seed(1)


# # Preprocessing

# ## Step 1: Create datasets

# We will use 20,000 examples for training and 5000 for validation.

# In[2]:


train_dataset = IMDB(split="train")


# In[3]:


train_dataset, valid_dataset = random_split(list(train_dataset), [0.8, 0.2])
print(len(train_dataset), len(valid_dataset))
# train_dataset = torch.utils.data.Subset(train_dataset, [1,2,3,4,5])
print(train_dataset[1])  # print one example


# ## Step 2: Find unique tokens
# 
# Text processing and finding unique tokens to build Vocab

# In[4]:


# nlp = spacy.load("en_core_web_sm")
emoji_pattern = re.compile("(?::|;|=)(?:-)?(?:\)|\(|D|P)")


def tokenizer(text):
    """
    Our custom tokenizer function:
        1. Remove html markups
        2. Preserve emoticons (remove hyphens)
        3. Remove punctuation and non-letter characters
    """

    # Remove html markups
    # text = re.sub("<[^>]*>", "", text)
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text().lower()

    # Collect emoticons
    emoticons = re.findall(emoji_pattern, text.lower())

    # Remove non-letter characters and add emoticons
    text = re.sub("[\W]+", " ", text.lower()) + " ".join(emoticons).replace("-", "")

    # Process the text with spaCy -> takes time!
    # doc = nlp(text)
    # tokens = [token.lemma_ for token in doc if not token.is_stop]

    tokens = text.split()
    return tokens


# In[5]:


token_counts = Counter()
for label, line in train_dataset:
    tokens = tokenizer(line)
    token_counts.update(tokens)

print(f"Vocab size: {len(token_counts)}")


# ## Step3: Encoding unique token to integers

# Next, we will map each unique word to a unique integer. This can be done manually using a Python dictionary. 
# However, the `torchtext` package already provides a class `vocab` for this. We will also add *padding* and *unknown* tokens.

# In[6]:


sorted_by_freq_tuples = sorted(
    token_counts.items(), key=lambda x: x[1], reverse=True
)
ordered_dict = OrderedDict(sorted_by_freq_tuples)
vocab = vocab(ordered_dict)

# Prepend two special tokens "padding" and "unknown"
vocab.insert_token("<pad>", 0) # 0 is placeholder for padding
vocab.insert_token("<unk>", 1) # unknown words will be assigned 1
vocab.set_default_index(1)
print([vocab[token] for token in ["this", "is", "an", "example"]])


# In[7]:


# * Step 3A: Define the transformation functions

text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]
label_pipeline = lambda x: 1. if x == 2 else 0.


# In[8]:


# * Step 3B: wrap the encode and transformation function

def collate_batch(batch):
    label_list, text_list, lengths = [], [], []
    for _label, _text in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        lengths.append(processed_text.size(0))
    label_list = torch.tensor(label_list)
    lengths = torch.tensor(lengths)
    padded_text_list = nn.utils.rnn.pad_sequence(text_list, batch_first=True)
    return padded_text_list, label_list, lengths


# In[9]:


from torch.utils.data import DataLoader

dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False, collate_fn=collate_batch)


# So far, we have converted sequence of words into sequence of integers, and labels of `pos` and `neg` into 1 or 0. However, there is one issue that we need to resolve i.e. the sequences currently have different lengths. Although, in general, RNNs can handle sequences of variable lengths, we still need to make that all the sequences in a mini-batch have the same length to store them efficiently in a tensor. 

# PyTorch provides an efficient method `pad_sequence` for this, which we already included in our `collate_fn`.

# In[10]:


# Check the first batch
text_batch, label_batch, length_batch = next(iter(dataloader))


# In[11]:


# text_batch


# In[12]:


label_batch


# In[13]:


length_batch


# In[14]:


text_batch.shape


# As we can see, all the examples are padded to match the maximum size in a batch.

# In[15]:


# Create dataloaders

batch_size = 32
train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
valid_dl = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)


# Now we will convert these integers to input features using `nn.Embedding` layer.

# In[16]:


# # Example of using nn.Embedding

# embedding = nn.Embedding(num_embeddings=10, embedding_dim=3 ,padding_idx=0)
# sample_examples = torch.LongTensor([[1,2,3,4], [5,6,7,0]])

# print(embedding(sample_examples))


# # Step 4: Building our RNN model

# In[17]:


# # Test model

# class RNN(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers):
#         super().__init__()
#         # Add RNN layer
#         self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
#         # self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
#         # self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

#         # Add final prediciton head
#         self.fc = nn.Linear(hidden_size, 1)

#     def forward(self, x):
#         _, hidden = self.rnn(x)
#         out = hidden[-1, :, :] # we use the final hidden state from the last hidden layer (num_layers, batch_size, hidden_size)
#         out = self.fc(out)
#         return out

        


# In[18]:


# model = RNN(64, 32, 2)
# model(torch.randn(5,3,64))


# Since we have very long sequences, we are going to use an LSTM layer to account for long range effects. 
# 
# We will also use `pack_padded_sequence`  function to prepare the input sequences for efficient processing by RNNs or other sequence processing modules. It eliminates the padding.
# 
# If you need to obtain the original output sequences, you can "unpack" the sequences using the `pad_packed_sequence` function, which will restore the padding and return sequences with the original lengths.

# In[19]:


class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(rnn_hidden_size, fc_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, lengths):
        out = self.embedding(text)
        out = nn.utils.rnn.pack_padded_sequence(out, lengths.cpu().numpy(), enforce_sorted=False, batch_first=True)
        out, (hidden, cell) = self.rnn(out)
        out = hidden[-1, :, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


# In[20]:


vocab_size = len(vocab)
embed_dim = 20
rnn_hidden_size = 64
fc_hidden_size = 64
torch.manual_seed(1)
model = RNN(vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size, num_layers=1)

model


# In[21]:


device = "mps" if torch.backends.mps.is_available() else "cpu"
print(device)


# In[22]:


model.to(device)


# # Step 5: Training pipeline

# In[23]:


def train(dataloader):
    model.train()
    total_acc, total_loss = 0, 0
    for text_batch, label_batch, lengths in dataloader:
        text_batch, label_batch = text_batch.to(device), label_batch.to(device)
        optimizer.zero_grad()
        pred = model(text_batch, lengths)[:, 0]
        loss = loss_fn(pred, label_batch)
        loss.backward()
        optimizer.step()
        total_acc += ((pred >= 0.5).float() == label_batch).float().sum().item()
        total_loss += loss.item()*label_batch.size(0)
    return total_acc / len(dataloader.dataset), total_loss / len(dataloader.dataset)


# In[24]:


def evaluate(dataloader):
    model.eval()
    total_acc, total_loss = 0, 0
    with torch.no_grad():
        for text_batch, label_batch, lengths in dataloader:
            text_batch, label_batch = text_batch.to(device), label_batch.to(device)
            pred = model(text_batch, lengths)[:, 0]
            loss = loss_fn(pred, label_batch)
            total_acc += ((pred >= 0.5).float() == label_batch).float().sum().item()
            total_loss += loss.item()*label_batch.size(0)
        return total_acc / len(dataloader.dataset), total_loss / len(dataloader.dataset)


# In[25]:


loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# In[26]:


# Train the model

num_epochs = 10
torch.manual_seed(1)
for epoch in range(num_epochs):
    acc_train, loss_train = train(train_dl)
    acc_valid, loss_valid = evaluate(valid_dl)
    print(f"Epoch {epoch+1} accuracy: {acc_train:.4f} val_accuracy: {acc_valid:.4f}")


# # Step 6: Test the model

# In[47]:


test_dataset = IMDB(split="test")
test_dataset = list(test_dataset)


# I got some issue `TypeError: _IterDataPipeSerializationWrapper instance doesn't have valid length`. Converting it into a list solved the `length` issue.

# In[48]:


test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)


# In[49]:


acc_test, _ = evaluate(test_dl)
print(f"Test accuracy: {acc_test}")


# # Testing Bidirectional RNN

# In[50]:


class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(rnn_hidden_size, fc_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, lengths):
        out = self.embedding(text)
        out = nn.utils.rnn.pack_padded_sequence(out, lengths.cpu().numpy(), enforce_sorted=False, batch_first=True)
        out, (hidden, cell) = self.rnn(out)
        out = torch.cat((hidden[-2, :, :],
                        hidden[-1, :, :]), dim=1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


# In[51]:


model = BiRNN(vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size, num_layers=1)
model


# In[ ]:


# Train the model

model.to(device)

num_epochs = 10
torch.manual_seed(1)
for epoch in range(num_epochs):
    acc_train, loss_train = train(train_dl)
    acc_valid, loss_valid = evaluate(valid_dl)
    print(f"Epoch {epoch+1} accuracy: {acc_train:.4f} val_accuracy: {acc_valid:.4f}")


# In[ ]:





# In[ ]:




