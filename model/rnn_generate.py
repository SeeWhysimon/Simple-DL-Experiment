import torch
import torch.nn as nn

from utils import text_to_tensor, save_text
from rnn import RNN

def generate_text(model, start_word, word_to_index, index_to_word, max_length=100):
    input_size = len(word_to_index)
    hidden_tensors = model.init_hidden()
    input_tensor = torch.zeros(1, input_size)
    input_tensor[0][word_to_index[start_word]] = 1

    generated_words = [start_word]

    for _ in range(max_length):
        output, hidden_tensors = model(input_tensor, hidden_tensors)
        topv, topi = output.topk(1)
        next_word_index = topi[0][0].item()
        next_word = index_to_word[next_word_index]

        generated_words.append(next_word)

        input_tensor = torch.zeros(1, input_size)
        input_tensor[0][next_word_index] = 1

        if next_word == '<EOS>':
            break

    return ' '.join(generated_words)

def main():
    file_path = '../input/input.txt'
    model_path = '../output/rnn/no scheduler/rnn.pth'
    _, _, word_to_index, index_to_word = text_to_tensor(file_path, 'rnn')
    rnn = torch.load(model_path).eval()
    
    start_word = 'china'
    generated_text = generate_text(rnn, start_word, word_to_index, index_to_word)
    print(generated_text)
    save_text(generated_text, '../output/rnn/no scheduler/output.txt')
    

if __name__ == '__main__':
    main()
