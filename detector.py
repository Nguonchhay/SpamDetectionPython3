import os
import pickle
from collections import Counter


def load_dataset(clf_file):
    return pickle.load(open(clf_file, "rb"))


def make_dict():
    direc = "emails/"
    files = os.listdir(direc)
    emails = [direc + email for email in files]
    words = []

    for email in emails:
        f = open(email)
        if f is not None:
            content = f.read()
            words += content.split(' ')

    for i in range(len(words)):
        if not words[i].isalpha():
            words[i] = ''

    dictionary = Counter(words)
    del dictionary['']
    return dictionary.most_common(3000)


clf = load_dataset("spam_detection.classifier")
d = make_dict()

# Use input from command line
while True:
    features = []
    inp = input("Enter test email (subject + content): ").split()
    if inp[0] == "exit":
        break
    for word in d:
        features.append(inp.count(word[0]))
    res = clf.predict([features])
    print(["Not Spam", "Spam!"][res[0]])

