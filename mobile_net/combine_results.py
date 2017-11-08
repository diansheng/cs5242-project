import pandas as pd
import os

frames = []

for file_name in os.listdir('results_50'):
    if file_name == '.DS_Store':
      continue
    fn = pd.read_csv('results_50/'+file_name)
    frames.append(fn)
    
data = pd.concat(frames)
print(data)

data['indexNumber'] = [int(i.split('.')[0]) for i in data['image_name']]
# Perform sort of the rows
data.sort_values(by=['indexNumber'], ascending = [True], inplace = True)
# Deletion of the added column
data.drop('indexNumber', 1, inplace = True)
print(data)

data.to_csv('results.csv', index=None, header=['image_name','category'])