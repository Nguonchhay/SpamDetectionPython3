import os
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
import pickle


def save(clf, name):
    pickle.dump(clf, open(name, "wb")


def make_dict():
    email_directory = "emails/"
    files = os.listdir(email_directory)
    emails = [email_directory + email for email in files]
    words = []
    for email in emails:
        f = open(email)
        if f is not None:
            content = f.read()
            words += content.split(' ')

    for i in range(len(words)):
        if not words[i].isalpha():
            words[i] = ''

    dict = Counter(words)
    del dict['']
    return dict.most_common(3000)


def make_dataset(dictionary):
    email_directory = "emails/"
    files = os.listdir(email_directory)
    emails = [email_directory + email for email in files]
    feature_set = []
    feature_labels = []
    num = 1
    for email in emails:
        data = []
        f = open(email)
        if f is not None:
            print(str(num) + '- Load data from: ' + email)
            words = f.read().split(' ')
            for entry in dictionary:
                data.append(words.count(entry[0]))
            feature_set.append(data)

            if "ham" in email:
                feature_labels.append(0)
            if "spam" in email:
                feature_labels.append(1)
            num = num + 1
    return feature_set, feature_labels


dictionary = make_dict()
features, labels = make_dataset(dictionary)

x_train, x_test, y_train, y_test = tts(features, labels)

clf = MultinomialNB()
clf.fit(x_train, y_train)

preds = clf.predict(x_test)
print(accuracy_score(y_test, preds))
save(clf, "spam_detection.classifier")
print("saved")
