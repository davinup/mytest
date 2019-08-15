import requests
import re

url='https://www.17k.com/top/refactor/top100/10_bookshelf/10_bookshelf_top_100_pc.html'
#发送请求
response=requests.get(url)
#编码格式
response.encoding='utf-8'

html=response.text
dl=re.findall(r'<table width="100%" cellspacing="0" cellpadding="10">.*?</table>',html,re.S)[0]
data_colmuns=re.findall(r'<tr><th>(.*?)</th><th>(.*?)</th><th>(.*?)</th><th>(.*?)</th><th>(.*?)</th><th>(.*?)</th><th>(.*?)</th><th>(.*?)</th></tr>',dl)
title_list=re.findall(r'<td width="30">(.*?)</td><td width="60"><a href="(.*?)" target="_blank">\[(.*?)\]</a></td><td><a class="red" href="(.*?)" title="(.*?)" target="_blank">(.*?)</a></td><td><a href="(.*?)" title="(.*?)" target="_blank">(.*?)</a></td><td>(.*?)</td><td><a href="(.*?)" title="(.*?)" target="_blank">(.*?)</a></td>"',dl)

print(title_list)