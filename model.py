import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.embed = nn.Embedding(vocab_size,embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=1, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.hidden = (torch.zeros(1, 1, hidden_size),torch.zeros(1, 1, hidden_size))
    
    def forward(self, features, captions):
        embeds = self.embed(captions[:, :-1])
        features = features.unsqueeze(1)
        embeds = torch.cat((features, embeds), 1)
        lstm_output, self.hidden = self.lstm(embeds)
        output = self.linear(lstm_output)
        return output

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        result = []

        for i in range(max_len) :
            output, states = self.lstm(inputs, states)
            output = self.linear(output.squeeze(1))
            out_max = output.max(1)[1]
            predicted_item = out_max.item()
            result.append(predicted_item)
            inputs = self.embed(out_max).unsqueeze(1)

        return result