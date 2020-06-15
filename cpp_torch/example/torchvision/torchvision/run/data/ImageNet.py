import sys
import os
from urllib import request
from PIL import Image

def download(url, decode=False):
    response = request.urlopen(url)
    if response.geturl() == "https://s.yimg.com/pw/images/en-us/photo_unavailable.png":
        # Flickr :This photo is no longer available iamge.
        raise Exception("This photo is no longer available iamge.")

    body = response.read()
    if decode == True:
        body = body.decode()
    return body

def write(path, img):
    file = open(path, 'wb')
    file.write(img)
    file.close()

# see http://image-net.org/archive/words.txt
classes = {"apple":"n07739125", "banana":"n07753592", "orange":"n07747607"}

offset = 0
max = 1000
for dir, id in classes.items():
    print(dir)
    os.makedirs(dir, exist_ok=True)
    urls = download("http://www.image-net.org/api/text/imagenet.synset.geturls?wnid="+id, decode=True).split()
    print(len(urls))
    i = 0
    for url in urls:
        if i < offset:
            continue
        if i > max:
            break

        try:
            file = os.path.split(url)[1]
            path = dir + "/" + file
            write(path, download(url))
            print("done:" + str(i) + ":" + file)
        except:
            print("Unexpected error:", sys.exc_info()[0])
            print("error:" + str(i) + ":" + file)
        i = i + 1

print("end")