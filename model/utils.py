import torch
import string
import matplotlib.pyplot as plt

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    return words

def create_vocabulary(words):
    vocab = sorted(set(words))
    word_to_index = {word: idx for idx, word in enumerate(vocab)}
    index_to_word = {idx: word for idx, word in enumerate(vocab)}
    return vocab, word_to_index, index_to_word

def one_hot_encode(word, word_to_index, vocab_size):
    vector = torch.zeros(vocab_size)
    index = word_to_index[word]
    vector[index] = 1
    return vector

def words_to_tensor(words, word_to_index, vocab_size, mode):
    tensors = [one_hot_encode(word, word_to_index, vocab_size) for word in words]
    if mode == 'rnn':
        return torch.stack(tensors)
    elif mode == 'lstm':
        return torch.stack(tensors).unsqueeze(1)

def text_to_tensor(file_path, mode):
    text = read_file(file_path)
    words = preprocess_text(text)
    vocab, word_to_index, index_to_word = create_vocabulary(words)
    vocab_size = len(vocab)
    input_tensor = words_to_tensor(words, word_to_index, vocab_size, mode)
    return input_tensor, vocab, word_to_index, index_to_word

def visualize(model):
    plt.figure(figsize=(12, 8))
    plt.plot(model.loss_history)
    plt.xlabel('Training Time')
    plt.ylabel('Loss')
    plt.title('Loss History')
    plt.tight_layout()
    plt.show()

    
    for i, i2h_layer in enumerate(model.i2h):
        plt.figure(figsize=(12, 8))
        plt.imshow(i2h_layer.weight.data.numpy(), cmap='hot', aspect='auto')
        plt.colorbar()
        plt.title(f'i2h Weight Matrix for Layer {i + 1}')
        plt.xlabel('Weight Index')
        plt.ylabel('Neuron Index')
        plt.tight_layout()
        plt.show()

def save_text(text, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)        

def main():
    file_path = '../input/input.txt'
    input_tensor, vocab, word_to_index, index_to_word = text_to_tensor(file_path)

    print(word_to_index)

if __name__ == '__main__':
    main()
