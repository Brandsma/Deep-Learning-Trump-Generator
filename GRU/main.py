import pandas as pd
import string
from logger import setup_logger

from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

log = setup_logger(__name__)


def clean_text(input_text):
    text = "".join(
        word for word in input_text if word not in string.punctuation).lower()
    text = text.encode("utf8").decode("ascii", "ignore")
    return text


def load_data(filename):
    df = pd.read_csv(filename)
    return df


def preprocessing(df):
    corpus = pd.DataFrame()
    for idx, elem in df.iterrows():
        corpus = corpus.append(
            {"text": clean_text(elem["text"])}, ignore_index=True)

    vocab = []
    for line in corpus["text"]:
        words = line.split()
        for word in words:
            vocab.append(word)

    vocab = set(vocab)

    # TODO: Maybe limit the number of words we get, instead of just using all the words
    tokenizer = Tokenizer(len(vocab))
    tokenizer.fit_on_texts(corpus["text"])
    word2index = tokenizer.word_index
    log.info("Indexed " + str(len(word2index)) +
             " unique words with the tokenizer")
    # print(word2index)

    dictionary = {}
    rev_dictionary = {}
    for word, idx in word2index.items():
        dictionary[word] = idx
        rev_dictionary[idx] = word

    input_sequences = tokenizer.texts_to_sequences(corpus["text"])
    print(input_sequences)

    input_data = []
    target = []
    for line in input_sequences:
        for i in range(1, len(line)-1):
            input_data.append(line[:i])
            target.append(line[i+1])

    MAX_LEN = 0
    for seq in input_data:
        if len(seq) > MAX_LEN:
            MAX_LEN = len(seq)
    log.info("Longest word has " + str(MAX_LEN) + "characters")

    input_data = pad_sequences()

    total_words = len(vocab)
    target = to_categorical(target, num_classes=total_words)
    input_data = pad_sequences(
        maxlen=MAX_LEN, padding="post", truncating="post")

    log.info("Input data has a shape of " + str(input_data.shape))
    log.info("Target data has a shape of " + str(input_data.shape))

    return


def main(input_file: str):
    log.info("Loading data...")
    df = load_data(input_file)
    log.info("Preprocessing...")
    corpus = preprocessing(df)
    log.info("Printing examples")
    print(corpus)


if __name__ == "__main__":
    filename = "../tiny.csv"
    main(filename)
