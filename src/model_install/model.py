import torch
import torch.nn as nn

class LeNet(nn.Module):
    
    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(True),
            nn.Linear(120, 84),
            nn.ReLU(True),
            nn.Linear(84, num_classes),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.classifier(x)
        return x
    
class LSTMModel(nn.Module):
    def __init__(self, feat_size, embed_size, hidden_size, n_layers, num_classes):
        super(LSTMModel, self).__init__()

        self.feat_size = feat_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(feat_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, n_layers, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        embedded_feats = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded_feats, hidden)
        lstm_out = self.dropout(lstm_out)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_size)
        fc_out = self.fc(lstm_out)
        sigmoid_out = self.sigmoid(fc_out)
        sigmoid_out = sigmoid_out.view(x.shape[0], -1)
        sigmoid_last = sigmoid_out[:, -1]

        return sigmoid_last, hidden

    def init_hidden(self, x):
        weight = next(self.parameters()).data
        h = (weight.new(self.n_layers, x.shape[0], self.hidden_size).zero_(),
             weight.new(self.n_layers, x.shape[0], self.hidden_size).zero_())
        return h

    def get_embeddings(self, x):
        return self.embedding(x)

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out
