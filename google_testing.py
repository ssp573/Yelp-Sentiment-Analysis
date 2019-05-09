import pandas as pd
from google.cloud import language_v1
from google.cloud.language_v1 import enums
import six


def label_from_text(content):

    client = language_v1.LanguageServiceClient()

    # content = 'Your text to analyze, e.g. Hello, world!'

    if isinstance(content, six.binary_type):
        content = content.decode('utf-8')

    type_ = enums.Document.Type.PLAIN_TEXT
    document = {'type': type_, 'content': content}


    response = client.analyze_sentiment(document)
    sentiment = response.document_sentiment
    print('Score: {}'.format(sentiment.score))
    print('Magnitude: {}'.format(sentiment.magnitude))

    if sentiment.score >= 0:
        label = 'pos'
    else:
        label = 'neg'

    return label

test_tokenized= pd.read_pickle('./data/comparison_test_set.pkl')[:1000]

correct = 0
total = 0
for token_text, label in test_tokenized.values.tolist():

    if len(token_text) < 5:
        continue

    text = ''
    for t in token_text:
        text += ' ' + t
    pred_label = label_from_text(text)
    if pred_label =='pos' and label == 'pos':
        correct += 1
    if pred_label == 'neg' and label == 'neg':
        correct +=1
    total +=1
    print(text)
    print(correct,total)
    print()

print("Google Predicted {} out of {} or {} correctly".format(correct, total, str(correct / total) + '%'))
