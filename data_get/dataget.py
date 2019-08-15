import numpy as np
import pandas as pd
from requests_html import HTMLSession

def get_text_link_from_sel(sel):
    mylist = []
    try:
        results = r.html.find(sel)
        for result in results:
            mytext = result.text
            mylink = list(result.absolute_links)[0]
            mylist.append((mytext, mylink))
        return mylist
    except:
        return None


session_one = HTMLSession()
# url = 'https://www.jianshu.com/p/85f4624485b9'

url='https://www.17k.com/top/refactor/top100/10_bookshelf/10_bookshelf_top_100_pc.html'
r = session_one.get(url)
# print(r.html.text)
# print(r.html.absolute_links)
sel = 'body > div.Main.Top100 > div.content.TABBOX > div:nth-child(2) > div:nth-child(3) > table > tbody > tr'
results = r.html.find(sel)
print(results)
# mylist=(results[0].text,list(results[0].absolute_links)[0])
# print(mylist)
# df = pd.DataFrame(get_text_link_from_sel(sel))
# print(get_text_link_from_sel(sel))
# df.columns = ['text', 'link']
# df.to_csv('output.csv', encoding='utf-8', index=False)
# print(df)
