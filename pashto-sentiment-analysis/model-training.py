import pickle
import random
import re

import pandas as pd
from nltk import word_tokenize, NaiveBayesClassifier, classify
from nltk.corpus import stopwords
import seaborn as sns
import matplotlib.pyplot as plt


def get_tweets_for_model(cleaned_tweet_list):
    for tweets in cleaned_tweet_list:
        yield dict([token, True] for token in tweets)


def remove_noise(tweets, stop_words=()):
    cleaned_tweets = []
    # print(tweets)
    for token in tweets:
        token = re.sub(pattern='[A-Za-z]+', repl='', string=token)

        # if token not in stop_words:
        cleaned_tweets.append(token)

    return cleaned_tweets


def custom_input():
    # custom = "زه تاسره تا نکووم."
    custom = input("Enter custom text: ")
    custom_tokenized = word_tokenize(text=custom)
    print("Custom tweet is: ", custom)
    custom_data = dict([token, True] for token in custom_tokenized)
    print("The prediction is => ", model.classify(custom_data))


if __name__ == "__main__":
    print("Hello, World")

    # Load the Pashto stopwords
    stop_words = stopwords.words('pashto')

    # Load the Positive Tweets file
    positive = pd.read_csv("tweets/positive-translated.csv", skipinitialspace=True, error_bad_lines=False,
                           encoding='utf-8', dtype='object')

    # Load the Negative Tweets file
    negative = pd.read_csv("tweets/negative-translated.csv", skipinitialspace=True, error_bad_lines=False,
                           encoding='utf-8')

    # load the sample of data to predict after training
    sample = pd.read_csv("tweets/sample-translated.csv", skipinitialspace=True, error_bad_lines=False, encoding='utf-8')

    # Create data frame for the predicted tweets
    prediction = pd.DataFrame()
    sample_list = []

    print("====================== Simple Positive Tweet ======================")
    print(positive[:1])

    # print("====================== Simple Negative Tweet ======================")
    # print(negative[:1])
    positive_cleaned = []
    negative_cleaned = []
    sample_cleaned = []
    i = 1

    # Remove unnecessary code from tweets
    for tweet in sample['sample']:
        sample_list.append(tweet)
        sample_cleaned.append(remove_noise(word_tokenize(tweet), stop_words))
        # print(i)
        # print(tweet)
        # print(word_tokenize(text=tweet))

        # positive_tokenized.append(word_tokenize(text=str(tweet)))

        # i  = i + 1

    # assign tweet to the data frame which is above created
    prediction['tweets'] = sample_list
    # print(prediction)
    # exit()

    # print(sample_cleaned[0])
    # exit()
    # Remove unnecessary code from tweets
    for tweet in positive['positive']:
        positive_cleaned.append(remove_noise(word_tokenize(tweet), stop_words))
        # print(i)
        # print(tweet)
        # print(word_tokenize(text=tweet))

        # positive_tokenized.append(word_tokenize(text=str(tweet)))

        # i  = i + 1

    # Remove unnecessary code from tweets
    for tweet in negative['negative']:
        negative_cleaned.append(remove_noise(word_tokenize(tweet), stop_words))

    print("====================== Tokenized and Cleaned Positive tweet ==================")
    print(positive_cleaned[0])

    # print("====================== Tokenized and Cleaned Negative tweet ==================")
    # print(negative_cleaned[0])

    print("======================== Data for model ========================")
    positive_tweet_for_model = get_tweets_for_model(positive_cleaned)
    print(next(positive_tweet_for_model))

    # Build data for predictive model
    negative_data_for_model = get_tweets_for_model(negative_cleaned)
    sample_data_for_model = get_tweets_for_model(sample_cleaned)

    positive_dataset = [(tweet_token, 'Positive')
                        for tweet_token in positive_tweet_for_model]

    negative_dataset = [(tweet_token, 'Negative')
                        for tweet_token in negative_data_for_model]

    print("====================== Positive Dataset ============================")
    print(positive_dataset)

    # Create datasets
    dataset = positive_dataset + negative_dataset
    random.shuffle(dataset)

    print("\n\n================== Classification Result =======================")
    train = dataset[:7000]
    test = dataset[7000:]

    # Create object of NaiveBayes Classifier
    model = NaiveBayesClassifier.train(train)
    print("The Accuracy is: ", classify.accuracy(classifier=model, gold=test))

    # Save the file for next use (permanently)
    file = open('pashto_text_classifier.sav', mode='wb')
    pickle.dump(obj=model, file=file)

    print("\n\n================== Let's predict 1000 random tweets =====================")
    result = []
    for token in sample_data_for_model:
        result.append(model.classify(featureset=token))

    # assign the sentiment result ot the data frame
    prediction['sentiment'] = result

    print(prediction[:20])

    # show the graphical representation
    sns.countplot(x='sentiment', data=prediction, hue='sentiment')
    plt.show()

    # Test on custom user inputs
    print("\n\n============================ Custom input analysis ===================")
    custom_input()

    try_again = input("\nWould you like to continue? (هو/نه)")

    while try_again == 'هو':
        # custom = input("Enter custom text: ")
        # custom_tokenized = word_tokenize(text=custom)
        # print("Custom tweet is: ", custom)
        # custom_data = dict([token, True] for token in custom_tokenized)
        # print("The prediction is => ", model.classify(custom_data))
        #
        custom_input()
        try_again = input("\nWould you like to continue? (هو/نه)")
