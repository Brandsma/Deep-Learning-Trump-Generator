import markovify
import pandas as pd


def load_data(filename):
    df = pd.read_csv(filename)
    return df


def preprocessing(df):
    pass


def main(input_file: str, word_count: int = 5):
    df = load_data(input_file)
    cur_text = ""
    for idx in range(len(df["text"])):
        cur_text += df["text"][idx].replace("\n", " ")
    data_model = markovify.Text(cur_text)

    for idx in range(word_count):
        print(idx)
        print(data_model.make_short_sentence(280))
        print('\n')


if __name__ == "__main__":
    filename = "../tweets_01-08-2021.csv"
    word_count = 5
    main(filename, word_count)

# Markov chain very basic model
# Read all tweets, make a transition table
# Which word follows which word, count, and normalize to chance
# Keep in mind when it should stop (140 / 240 characters)
# Make stop also

# Preprocessing
# Tokenize
# Filter undesirable (perhaps regex)
#
