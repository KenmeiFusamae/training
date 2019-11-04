from flickrapi import FlickrAPI
from urllib.request import urlretrieve
from pprint import pprint
import os, time, sys

#APIキーの情報
key = "c29432270b740afc22e065d7927332d5"
secret = "2652e603499632a5"
wait_time = 1

#保存先フォルダ
animalname = sys.argv[1]
savedir = "./" + animalname

flickr = FlickrAPI(key, secret, format="parsed-json" )
result = flickr.photos.search(
    text = animalname,
    per_page =400,
    media = "photos",
    sort = 'relevance',
    safe_search =1,
    extras = "url_q, licence"
)

photos = result["photos"]
#pprint(photos)

for i, photo in enumerate(photos['photo']):
    url_q = photo['url_q']
    filepath = savedir +'/'+ photo['id']+'.jpg'
    if os.path.exists(filepath):continue
    urlretrieve(url_q,filepath)
    time.sleep(wait_time)
