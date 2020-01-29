import pandas as pd
import requests
import time
import json

verbose = True
url = 'https://translate.yandex.net/api/v1.5/tr.json/translate'
api_key = 'trnsl.1.1.20200128T225610Z.1390dcabff739d56.7019337b0d16d745e66d6072efb81ccec6f37080'
filename = 'database1.txt'

with open(filename, 'w') as f:
   pass

df = pd.read_csv('tasks.csv', sep='\t')
texts = []

for i, (_, row) in enumerate(df.iterrows()):
    text = '\n'.join([str(row.link), str(row.description)])

    connection = False
    while not connection:
        try:
            answer = requests.get(
                url,
                params={
                    'key': api_key,
                    'text': text,
                    'lang': 'ru-en'
                }
            )
            connection = True
        except:
            time.sleep(1)

    try:
        answer = answer.json()
    except:
        print(i, 'json not available')
        continue

    if answer['code'] == 200:
        with open(filename, 'a') as f:
            f.write(json.dumps({'body': answer['text'][0]}) + '\n')
    else:
        print(i, 'code is not 200')

    if verbose and (i + 1) % 100 == 0:
        print(i + 1, 'completed')
