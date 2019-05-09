import json

import pandas as pd
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 \
    import Features, EntitiesOptions, KeywordsOptions

natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2018-11-16',
    iam_apikey='GdEaXIeY4NdlkA_QXjacbxlXhpjp2S0H0L_Djddt061U',
    url='https://gateway.watsonplatform.net/natural-language-understanding/api'
)

def label_from_text(text):
    response = natural_language_understanding.analyze(
        text=text,
        features=Features(
            entities=EntitiesOptions(emotion=False, sentiment=True, limit=2),
            keywords=KeywordsOptions(emotion=False, sentiment=True,
                                     limit=2))).get_result()

    return response['keywords'][0]['sentiment']['label']


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
    if pred_label =='positive' and label == 'pos':
        correct += 1
    if pred_label == 'negative' and label == 'neg':
        correct +=1
    total +=1
    print(text)
    print(correct,total)
    print()

print("IBM NLU Predicted {} out of {} or {} correctly".format(correct, total, str(correct / total) + '%'))
