import pandas as pd
from bag_of_words_model import BagOfWordsModel
from naive_bayes import NaiveBayesClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from joblib import dump
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# nltk.download('punkt_tab')
# nltk.download('stopwords')
# nltk.download('wordnet')

def preprocess_text(text):
    
    #initially I had added some custome stop words that prove to be common but after adding the n-gram model this proved to be unncessary

    text = re.sub(r'[^a-zA-Z\s]', '', text)  
    words = word_tokenize(text.lower())  
    stop_words = set(stopwords.words('english')).union({'hotel', 'room', 'stay', 'place'})
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  
    return ' '.join(words)


#using both models by determining importance through liklihood of negative and positive sentiment to vocab
def top_words(bowModel, NB):
    vocab = bowModel.vocab
    word_importance = NB.likelihoods['Negative'] - NB.likelihoods['Positive']
    sorted_words = sorted(zip(vocab, word_importance), key=lambda x: x[1], reverse=True)
    print("\nTop words for Negative sentiment: ")
    for i in range (1,10):
        print(sorted_words[i][0])

    word_importance = NB.likelihoods['Positive'] - NB.likelihoods['Negative'] 
    sorted_words = sorted(zip(vocab, word_importance), key=lambda x: x[1], reverse=True)
    print("\nTop words for Positive sentiment:")
    for i in range (1,10):
        print(sorted_words[i][0])

    # print(sorted_words[:10])

def map_sentiment(rating):
    if rating >= 4:
        return 'Positive'
    else:
        return 'Negative'
    
def map_sentiment2(rating):
    if rating >= 6:
        return 'Positive'
    else:
        return 'Negative'


dataset = pd.read_csv("dataset_tripadvisor-reviews_2024-11-01_19-51-41-777.csv")  
recent_feedback = pd.read_csv('Recent Feedback (2).csv')

recent_feedback['Merged_Responses'] = recent_feedback[recent_feedback.columns[2:13]].apply(
    lambda row: ' '.join(row.dropna().astype(str)), axis=1
)

recent_feedback['cleaned_responses'] = recent_feedback['Merged_Responses'].apply(preprocess_text)
dataset['cleaned_text'] = dataset['text'].apply(preprocess_text)

# dataset.info()
dataset.head()
# recent_feedback.info(), recent_feedback.head()
# print(recent_feedback['cleaned_responses'])

dataset['sentiment'] = dataset['rating'].apply(map_sentiment)
recent_feedback['sentiment'] = recent_feedback['Likelihood to Recommend'].apply(map_sentiment2)
recent_feedback = recent_feedback[['cleaned_responses','sentiment']]

recent_feedback['cleaned_responses']= recent_feedback['cleaned_responses'].replace('', np.nan)
recent_feedback.dropna(subset=['cleaned_responses'], inplace=True)

positive_text = " ".join(dataset[dataset['sentiment'] == 'Positive']['cleaned_text'])
WordCloud().generate(positive_text).to_image().save("pos.png")

negative_text = " ".join(dataset[dataset['sentiment'] == 'Negative']['cleaned_text'])
WordCloud().generate(negative_text).to_image().save("neg.png")

positive_text1 = " ".join(recent_feedback[recent_feedback['sentiment'] == 'Positive']['cleaned_responses'])
WordCloud().generate(positive_text1).to_image().save("pos1.png")

negative_text1 = " ".join(recent_feedback[recent_feedback['sentiment'] == 'Negative']['cleaned_responses'])
WordCloud().generate(negative_text1).to_image().save("neg1.png")

X_train_text, X_test_text, y_train, y_test = train_test_split(
    dataset['cleaned_text'], dataset['sentiment'], test_size=0.2, random_state=42
)

print("Training Bag of Words model on...")
bow_model = BagOfWordsModel(X_train_text, 2)

X_train = bow_model.transform(X_train_text)
X_test = bow_model.transform(X_test_text)

print("Training Naive Bayes classifier...")
nb_model = NaiveBayesClassifier()
nb_model.fit(X_train, np.array(y_train))

print("Evaluating the model...")
y_pred = nb_model.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

top_words(bow_model,nb_model)

X_train_text, X_test_text, y_train, y_test = train_test_split(
    recent_feedback['cleaned_responses'], recent_feedback['sentiment'], test_size=0.2, random_state=42
)

print("Training Bag of Words model...")
bow_model = BagOfWordsModel(X_train_text,2)


X_train = bow_model.transform(X_train_text)
X_test = bow_model.transform(X_test_text)


print("Training Naive Bayes classifier...")
nb_model = NaiveBayesClassifier()
nb_model.fit(X_train, np.array(y_train))

print("Evaluating the model...")
y_pred = nb_model.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

top_words(bow_model, nb_model)

print("Saving models...")
dump(bow_model, "bow_model.joblib")
dump(nb_model, "naive_bayes_model.joblib")

X_train, X_test, y_train, y_test = train_test_split(
    dataset['cleaned_text'], 
    dataset['sentiment'], 
    test_size=0.2, 
    random_state=42
)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)
print("Classification report on datset 1")
print(classification_report(y_test, y_pred))


X_train, X_test, y_train, y_test = train_test_split(
    recent_feedback['cleaned_responses'], 
    recent_feedback['sentiment'], 
    test_size=0.2, 
    random_state=42
)

vectorizer = TfidfVectorizer(ngram_range=(2,3))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)
print("classification report on data set 2")
print(classification_report(y_test, y_pred))


dataset['sentiment'].value_counts().plot(kind='bar')
plt.title("Sentiment Distribution For First Data Set")
plt.savefig("tripadvisor.png")

recent_feedback['sentiment'].value_counts().plot(kind='bar')
plt.title("Sentiment Distribution For Second Data Set")
plt.savefig("Hysat.png")

# review = input("please input a review")
# # processed_review = preprocess_text(review)
# # tf_idf_vector = bow_model.transform([processed_review]) 
# # prediction = nb_model.predict(tf_idf_vector)

# print(review, prediction)


