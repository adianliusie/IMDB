import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.querries = nn.Linear(embed_size, embed_size, bias=False)
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask):
        keys = self.keys(x)
        querries = self.querries(x)
        values = self.values(x)

        scores = torch.bmm(querries, keys.permute(0,2,1))
        scores += mask.unsqueeze(1)
        alphas = self.softmax(scores)
        output = torch.bmm(alphas, values)
        return output

class Querry_Attention(nn.Module):
    def __init__(self, embed_size):
        super(Querry_Attention, self).__init__()
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.querry = torch.randn(embed_size, dtype=torch.float)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask):
        keys = self.keys(x)
        values = self.values(x)

        scores = torch.matmul(keys, self.querry)
        scores += mask
        alphas = self.softmax(scores)
        output = torch.matmul(torch.unsqueeze(alphas, 1), values)
        output = output.squeeze(1)
        return output

class Classifier(nn.Module):
    def __init__(self, vocab_size, word_dimension):
        super(Classifier, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, word_dimension)
        self.attention = SelfAttention(word_dimension)
        self.summarisation = Querry_Attention(word_dimension)
        self.fc1 = nn.Linear(word_dimension, 2)

    def forward(self, input_batch, mask):
        embeddings = self.embedding(input_batch)
        context_vectors = self.attention(embeddings, mask)
        sentence_vectors = self.summarisation(context_vectors, mask)
        output = self.fc1(sentence_vectors)
        return output
