import pandas as pd
import numpy as np


#df = pd.DataFrame.from_csv('data/training_set.tsv', sep='\t', encoding='utf-8')
df = pd.read_table('data/training_set.tsv', encoding='latin_1')
print(df.info)
#df = df[:1]
#train = df.as_matrix(columns=['essay', 'rater1_domain1'])
train = df.values
print(train.shape)
#print(train[0])
