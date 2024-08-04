import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from utils import text_to_tensor, visualize

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.i2h = nn.ModuleList([nn.Linear(input_size + hidden_size if i == 0 else 2 * hidden_size, hidden_size) for i in range(num_layers)])
        self.f2h = nn.ModuleList([nn.Linear(input_size + hidden_size if i == 0 else 2 * hidden_size, hidden_size) for i in range(num_layers)])
        self.c2h = nn.ModuleList([nn.Linear(input_size + hidden_size if i == 0 else 2 * hidden_size, hidden_size) for i in range(num_layers)])
        self.o2h = nn.ModuleList([nn.Linear(input_size + hidden_size if i == 0 else 2 * hidden_size, hidden_size) for i in range(num_layers)])
        self.i2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss_history = []

    def forward(self, input_tensor, hidden_tensors, cell_tensors):
        new_hidden_tensors = []
        new_cell_tensors = []

        for i in range(self.num_layers):
            hidden_expanded = hidden_tensors[i].unsqueeze(0)
            cell_expanded = cell_tensors[i].unsqueeze(0)
            combined = torch.cat((input_tensor, hidden_expanded), 2)
            input_gate = torch.sigmoid(self.i2h[i](combined))
            forget_gate = torch.sigmoid(self.f2h[i](combined))
            cell_gate = torch.tanh(self.c2h[i](combined))
            output_gate = torch.sigmoid(self.o2h[i](combined))

            cell = forget_gate * cell_expanded + input_gate * cell_gate
            hidden = output_gate * torch.tanh(cell)

            new_hidden_tensors.append(hidden.squeeze(0))
            new_cell_tensors.append(cell.squeeze(0))

            input_tensor = hidden

        output = self.i2o(input_tensor.squeeze(0))
        output = self.softmax(output)
        return output, new_hidden_tensors, new_cell_tensors

    def init_hidden(self):
        return [torch.zeros(1, self.hidden_size) for _ in range(self.num_layers)], [torch.zeros(1, self.hidden_size) for _ in range(self.num_layers)]

    def train_model(self, data, num_epochs=10, lr=0.01):
        self.loss_history = []
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

        for epoch in range(num_epochs):
            total_loss = 0
            hidden_tensors, cell_tensors = self.init_hidden()

            for i in range(data.size(0) - 1):
                input_tensor = data[i].unsqueeze(0)
                target_tensor = torch.tensor([torch.argmax(data[i + 1])], dtype=torch.long)

                output, hidden_tensors, cell_tensors = self(input_tensor, [h.detach() for h in hidden_tensors], [c.detach() for c in cell_tensors])
                loss = criterion(output, target_tensor)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()

            average_loss = total_loss / (data.size(0) - 1)
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss}')
            self.loss_history.append(total_loss)
            scheduler.step(average_loss)

def main():
    file_path = '../input/input.txt'
    input_tensor, vocab, word_to_index, _ = text_to_tensor(file_path, 'lstm')
    input_size = len(vocab)
    output_size = len(vocab)
    hidden_size = 128
    num_layers = 3 
    num_epochs = 300
    learning_rate = 0.1

    lstm = LSTM(input_size=input_size, 
                hidden_size=hidden_size, 
                output_size=output_size, 
                num_layers=num_layers)

    lstm.train_model(input_tensor, 
                     num_epochs=num_epochs,
                     lr=learning_rate)
    visualize(lstm)
    torch.save(lstm, '../output/lstm/lstm.model')
    
if __name__ == '__main__':
    main()
