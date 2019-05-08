import pandas as pd
import numpy as np
results=pd.read_csv('./plots/AmazonComprehendTestDataFullSet.csv')
results['Sentiment'] = np.where(results['Positive%'] >= results['Negative%'], 'POSITIVE', 'NEGATIVE')
results['Accurate'] = np.where(results['Sentiment'] == results['Label'], 1, 0)
result = np.sum(results['Accurate']) / len(results)
print("Amazon Comprehend Accuracy is: {}".format(result))
# import pdb; pdb.set_trace()