import nltk
import random
from nltk.corpus import names
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
import string

male_names = [(name, "male") for name in names.words("male.txt")]
female_names = [(name, "female") for name in names.words("female.txt")]
total_names = male_names + female_names
random.shuffle(total_names)


def gender_feature(name):
    # return {"all_name": name}
    return {'last_two_letter': name[-2:], "first_two_letter": name[0:2]}


feature_set = [(gender_feature(n), g) for (n, g) in total_names]
train_set_size = int(len(feature_set) * 0.6)

train_set = feature_set[:train_set_size]
test_set = feature_set[train_set_size:]

classifier = nltk.NaiveBayesClassifier.train(train_set)
print(classifier.classify(gender_feature('Neo')))
print(nltk.classify.accuracy(classifier, test_set))
classifier.show_most_informative_features()

ent = nltk.MaxentClassifier.train(train_set)
print(ent.classify(gender_feature('Neo')))
print(nltk.classify.accuracy(ent, test_set))
ent.show_most_informative_features()


documents = [(list(movie_reviews.words(fileid)), category) for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]  # 转化为词列表的影评，与标签，组成二元组
random.shuffle(documents)  # 为组成训练集和测试集准备

word_fd = nltk.FreqDist(w.lower() for w in movie_reviews.words()).most_common(100)
feature_words = [w for (w, _) in word_fd if w not in stopwords.words("english") and w not in string.punctuation]


def get_movie_feature(document):
    document_words = set(document)
    feature = {}
    for word in feature_words:
        feature['contains({})'.format(word)] = (word in document_words)
    return feature


feature_set = [(get_movie_feature(n), g) for (n, g) in documents]
train_set_size = int(len(feature_set) * 0.6)

train_set = feature_set[:train_set_size]
test_set = feature_set[train_set_size:]

classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))
classifier.show_most_informative_features()

ent = nltk.MaxentClassifier.train(train_set)
print(nltk.classify.accuracy(ent, test_set))
ent.show_most_informative_features()
