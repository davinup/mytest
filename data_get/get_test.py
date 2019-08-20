import numpy as np
import pandas as pd
from requests_html import HTMLSession
import re



#循环获取本页面所有书本信息
def get_text_link_from_sel(sel):
    linshi=[]
    try:
        results = r.html.find(sel)
        for result in results:
            mytext = result.text
            lines=mytext.split('\n')
            # mylink = list(result.absolute_links)[0]
            # mytext.append(mylink)
            # mylist.append(mytext)
            linshi.append(lines)
        return linshi
    except:
        return None

# fb=open('books.csv','w',encoding='utf-8')
# df = pd.DataFrame(columns = ['name','author','publisher','date','price','score','abstract','url'])
session_one = HTMLSession()
url = 'https://www.17k.com/top/refactor/top100/10_bookshelf/10_bookshelf_top_100_pc.html'
r = session_one.get(url)
    # print(r.html.absolute_links)
    #获取书本信息所在标记
sel = 'body > div.Main.Top100 > div.content.TABBOX > div:nth-child(2) > div:nth-child(2) > table'
    #获取对应书的信息
results = r.html.find(sel)
    #循环获取所有书本信息
# get_text_link_from_sel(sel)
result_list=get_text_link_from_sel(sel)
colmuns_book=['rank','category','bookname','section','date','author','status','amount']
linshi=result_list[0][8:]
print(result_list)
book_data=[]
j=0
while j<len(linshi):
    book_data.append(linshi[j:j+8])
    j+=8
print(book_data)
df = pd.DataFrame(book_data)
df.columns = colmuns_book     #给数据加列名
# df.to_csv('artical.csv', encoding='utf-8', index=False)      #存到对应的csv文件中
