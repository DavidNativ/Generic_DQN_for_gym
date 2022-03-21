import numpy as np
import torch
import torch.nn as nn
import torch.functional as F



# notre estimateur de Q value ; nn
class SimpleDQN(nn.Module):
    def __init__(self, input_size, hidden_size_in, hidden_size_out, output_size, drop_prob=0.2):
        super(SimpleDQN, self).__init__()
        self.input_size = input_size
        self.hidden_size_in = hidden_size_in
        self.hidden_size_out = hidden_size_out
        self.output_size = output_size

        # reseau de neurones
        self.fc1 = nn.Linear(self.input_size, self.hidden_size_in)
        self.fc2 = nn.Linear(self.hidden_size_in, self.hidden_size_out)
        self.output = nn.Linear(self.hidden_size_out, self.output_size)

        self.drop_prob = drop_prob
        self.dropout = nn.Dropout(self.drop_prob)
        self.relu = nn.ReLU()

    # forward pass
    def forward(self, input_data):
        x = self.fc1(input_data)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        return self.output(x)


    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
