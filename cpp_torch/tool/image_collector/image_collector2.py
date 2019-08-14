#pip install icrawler

from icrawler.builtin import GoogleImageCrawler
import sys
import os

argv = sys.argv

if not os.path.isdir(argv[1]):
    os.makedirs(argv[1])


crawler = GoogleImageCrawler(storage = {"root_dir" : argv[1]})
crawler.crawl(keyword = argv[2], max_num = 1000)
#