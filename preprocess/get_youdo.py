from bs4 import BeautifulSoup
import certifi
import urllib3
import re
import time


verbose = True
result_file = 'tasks.csv'

http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED',
                           ca_certs=certifi.where())

attrs_props = {
    'price': {'name': 'span', 'class': 'b-tasks__item__price open'},
    'description': {'name': 'p', 'class': 'b-tasks__item__description'},
    'link': {'name': 'a', 'class': 'b-tasks__item__link'},
    'author': {'name': 'span', 'class': 'b-tasks__item__author'},
    'address': {'name': 'span', 'class': 'b-tasks__item__address'}
}
attrs_keys = list(attrs_props.keys())

with open(result_file, 'w') as f:
    f.write('\t'.join(attrs_keys) + '\n')

for i in range(1, 91):
    url = 'https://photo.youdo.com/stasks-{}/'.format(i)
    response = http.request('GET', url)
    soup = BeautifulSoup(response.data, 'lxml')

    for task in soup.findAll('li', attrs={'class': 'b-tasks__item'}):
        new_task = {}
        for key, item in attrs_props.items():
            line = task.find(item['name'], attrs={'class': item['class']})
            line = line.text.strip()
            line = re.sub(r'\s+', ' ', line)
            new_task[key] = line
        new_task = '\t'.join([new_task[key] for key in attrs_keys]) + '\n'
        with open(result_file, 'a') as f:
            f.write(new_task)
    if verbose:
        print(i, 'completed')
    time.sleep(1)
