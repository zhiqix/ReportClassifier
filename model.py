import torch.nn as nn
import torch.nn.functional as F
import torch
from config import *
from transformers import AutoModel


class TextCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.bge = AutoModel.from_pretrained(BGE_MODEL)

        for name, param in self.bge.named_parameters():
            param.requires_grad = True

        self.conv1 = nn.Conv2d(1, EMBEDDING_DIM, (2, EMBEDDING_DIM))
        self.conv2 = nn.Conv2d(1, EMBEDDING_DIM, (3, EMBEDDING_DIM))
        self.conv3 = nn.Conv2d(1, EMBEDDING_DIM, (4, EMBEDDING_DIM))
        self.linear = nn.Linear(EMBEDDING_DIM * 3, NUM_CLASSES)

    def conv_and_pool(self, conv, input):
        out = conv(input)
        out = F.relu(out)
        # Prevent squeezing the dimension 0 when batch_size is 1.
        return F.max_pool2d(out, (out.shape[2], out.shape[3])).squeeze(2).squeeze(2)

    def forward(self, input, mask):
        out = self.bge(input, mask)[0].unsqueeze(1)

        out1 = self.conv_and_pool(self.conv1, out)
        out2 = self.conv_and_pool(self.conv2, out)
        out3 = self.conv_and_pool(self.conv3, out)
        out = torch.cat([out1, out2, out3], dim=1)
        return self.linear(out)


class BGE_CLASSIFIER(nn.Module):
    def __init__(self):
        super().__init__()
        self.bge = AutoModel.from_pretrained(BGE_MODEL)

        for name, param in self.bge.named_parameters():
            param.requires_grad = True

        self.linear1 = nn.Linear(EMBEDDING_DIM, int(EMBEDDING_DIM / 2))
        self.linear2 = nn.Linear(int(EMBEDDING_DIM / 2), NUM_CLASSES)

    def forward(self, input, mask):
        sentence_embeddings = self.bge(input, mask)[0][:, 0]
        out = self.linear1(sentence_embeddings)
        return self.linear2(out)
