import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from utils import text_to_tensor, visualize

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.i2h = nn.ModuleList([nn.Linear(input_size + hidden_size, hidden_size) for _ in range(num_layers)])
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss_history = []

    def forward(self, input_tensor, hidden_tensors):
        combined = torch.cat((input_tensor, hidden_tensors[0]), 1)
        new_hidden_tensors = []
        
        for i in range(self.num_layers):
            hidden = self.i2h[i](combined)
            new_hidden_tensors.append(hidden)
            combined = torch.cat((input_tensor, hidden), 1)
        
        output = self.i2o(combined)
        output = self.softmax(output)
        
        return output, new_hidden_tensors
    
    def init_hidden(self):
        return [torch.zeros(1, self.hidden_size) for _ in range(self.num_layers)]

    def train_model(self, data, num_epochs=10, lr=0.01):
        self.loss_history = []
        criterion = nn.NLLLoss()
        optimizer = optim.SGD(self.parameters(), lr=lr)
        sheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        
        for epoch in range(num_epochs):
            total_loss = 0
            hidden_tensors = self.init_hidden()
            
            for i in range(data.size(0) - 1):
                input_tensor = data[i].unsqueeze(0)
                target_tensor = torch.tensor([torch.argmax(data[i + 1])], dtype=torch.long)
                output, hidden_tensors = self(input_tensor, [h.detach() for h in hidden_tensors])
                loss = criterion(output, target_tensor)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / (data.size(0) - 1)}')
            self.loss_history.append(total_loss)
            sheduler.step(total_loss / (data.size(0) - 1))

def main():
    file_path = '../input/input.txt'
    input_tensor, _, word_to_index, _ = text_to_tensor(file_path, 'rnn')

    input_size = output_size = len(word_to_index)
    hidden_size = 128
    num_layers = 3
    num_epochs = 300
    learning_rate = 0.1  

    rnn = RNN(input_size, hidden_size, output_size, num_layers)

    rnn.train_model(input_tensor, num_epochs=num_epochs, lr=learning_rate)
    visualize(rnn)
    torch.save(rnn, '../output/rnn/rnn.pth')

if __name__ == '__main__':
    main()
