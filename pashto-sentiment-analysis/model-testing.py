import pickle

from nltk import word_tokenize


def custom_input():
    # custom = "زه تاسره تا نکووم."
    custom = input("Enter custom text: ")
    custom_tokenized = word_tokenize(text=custom)
    # print("Custom tweet is: ", custom)
    custom_data = dict([token, True] for token in custom_tokenized)
    print("The prediction is => ", model.classify(custom_data))



if __name__ == "__main__":
    print("\n\n============================ Custom text analysis ===================")

    # the model is loaded, which is already trained and stored
    model = pickle.load(file=open('pashto_text_classifier.sav', mode='rb'))

    # call the custom input method
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
