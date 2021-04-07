import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("../all.csv")

trump_tweet_lengths = [len(x) for x in df["text"]]

plt.hist(trump_tweet_lengths)
