# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np 

df = pd.read_csv('Data.csv', encoding = 'ISO-8859-1')

#dividing dataset into train and test data
df_train = df[df['Date']<'20150101']
df_test = df[df['Date']>'20141231']

#removing puntuations 
data = df_train.iloc[:,2:27]
data.replace('[^a-zA-Z]'," ",regex = True ,inplace=True) #this reges expression says that apart from 
#a-z and A-Z removee everything and replace it by " "

#renaming columns for ease of access
list1 = [i for i in range(25)]
new_index = [str(i) for i in list1]
data.columns = new_index


#lowering the uppercase to lower case
for i in new_index:
    data[i] = data[i].str.lower()
    

#now we are going to combine these headlines and convert them into vectors
# for example the first row
' '.join(str(i) for i in data.iloc[1,0:25])

#now combinig all sentences
headlines = []
for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))
    
    

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


tfidfvector = TfidfVectorizer(ngram_range=(2,2))
traindataset = tfidfvector.fit_transform(headlines)



naive = MultinomialNB()



naive.fit(traindataset,df_train['Label'])

test_transform = []
for row in range(0,len(df_test.index)):
    test_transform.append(' '.join(str(x) for x in df_test.iloc[row,2:27]))
test_dataset = tfidfvector.transform(test_transform)
predictions = naive.predict(test_dataset)


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


cm = confusion_matrix(df_test['Label'],predictions)

score = accuracy_score(df_test['Label'],predictions)

report = classification_report(df_test['Label'],predictions)