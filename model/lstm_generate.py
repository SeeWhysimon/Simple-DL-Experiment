import torch

from utils import text_to_tensor, words_to_tensor, one_hot_encode, save_text
from lstm import LSTM

def generate_text(model, start_words, word_to_index, index_to_word, num_generate=100):
    model.eval()
    words = start_words.split()
    input_tensor = words_to_tensor(words, word_to_index, len(word_to_index), 'lstm')
    input_tensor = input_tensor.to(torch.device('cpu'))
    hidden, cell = model.init_hidden()

    for i in range(len(words) - 1):
        input_tensor_i = input_tensor[i].unsqueeze(0)
        _, hidden, cell = model(input_tensor_i, hidden, cell)
    
    input_tensor = input_tensor[-1].unsqueeze(0)
    generated_words = words.copy()

    for _ in range(num_generate):
        output, hidden, cell = model(input_tensor, hidden, cell)
        topv, topi = output.data.topk(1)
        word_index = topi.item()
        word = index_to_word[word_index]
        generated_words.append(word)
        input_tensor = one_hot_encode(word, 
                                      word_to_index, 
                                      len(word_to_index)).unsqueeze(0).unsqueeze(0)

    return ' '.join(generated_words)

if __name__ == '__main__':
    model = torch.load('../output/lstm/with scheduler/lstm.pth', map_location=torch.device('cpu'))
    model.device = 'cpu'

    _, vocab, word_to_index, index_to_word = text_to_tensor('../input/input.txt', 'lstm')
    initial_text = "china"
    generated_text = generate_text(model, 
                                   initial_text, 
                                   word_to_index, 
                                   index_to_word, 
                                   num_generate=100)
    print(generated_text)

    save_text(generated_text, '../output/lstm/with scheduler/output.txt')
