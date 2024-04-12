import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.dropout1 = nn.Dropout(0.5)
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.dropout2 = nn.Dropout(0.5)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out
    
# class NeuralNet(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(NeuralNet, self).__init__()
#         self.l1 = nn.Linear(input_size, hidden_size)
#         self.bn1 = nn.BatchNorm1d(hidden_size)
#         self.dropout1 = nn.Dropout(0.5)
#         self.l2 = nn.Linear(hidden_size, hidden_size)
#         self.bn2 = nn.BatchNorm1d(hidden_size)
#         self.dropout2 = nn.Dropout(0.5)
#         self.l3 = nn.Linear(hidden_size, num_classes)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         out = self.l1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.dropout1(out)
#         out = self.l2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#         out = self.dropout2(out)
#         out = self.l3(out)
#         return out