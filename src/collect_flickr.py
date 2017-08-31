import json
import urllib.request
from multiprocessing import Process, Queue
from collections import defaultdict
from flickrapi import FlickrAPI

FLICKR_PUBLIC = "63ab54fd7c13188c8d6214aece7a6c26"
FLICKR_SECRET = "240ac1408ceb3b48"

TAGS = ["animal", "cat", "city", "dog", "flower", "forest", 
        "landmark", "landscape", "mountain", "nature", "park", "sea", "wild", "street"]

def collect():
    flickr = FlickrAPI(FLICKR_PUBLIC, FLICKR_SECRET, format="parsed-json")
    
    im_infos = defaultdict(lambda: list())
    for tag in TAGS:
        photos = flickr.photos.search(text=tag, per_page=500, sort="relevance",
                                      license="1,2,3,4,5",extras="url_o")["photos"]["photo"]
        
        i = 0
        for photo in photos:
            try:
                height, width = int(photo["height_o"]), int(photo["width_o"])
                url = photo["url_o"]
            except:
                continue
            if height > 500 and height < 2500 and width > 500 and width < 2500:
                print(tag, i, height, width, url)
                im_infos[tag].append({"height": height, "width": width, "url":url})
                i += 1

    return im_infos


def works(tag, todo):
    for i, info in enumerate(todo):
        try:
            url = info["url"]
            urllib.request.urlretrieve(url, "images/{}_{}.jpg".format(tag, i))
        except:
            print("Download {} failed".format(url))


def download(infos):
    counter = defaultdict(lambda:0)

    proc_pool = list()
    for i, tag in enumerate(infos.keys()):
        todo = infos[tag]
        proc = Process(target=works, args=(tag, todo))
        proc.start()
        proc_pool.append(proc)
    
    for proc in proc_pool:
        proc.join()


def main():
    im_infos = collect()
    download(im_infos)


if __name__ == "__main__":
    main()
