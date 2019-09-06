#pip install icrawler

from icrawler.builtin import GoogleImageCrawler
from icrawler.builtin import BingImageCrawler
from icrawler.builtin import BaiduImageCrawler
from icrawler.builtin import FlickrImageCrawler
import sys
import os

argv = sys.argv

if not os.path.isdir(argv[1]):
    os.makedirs(argv[1])


#crawler = GoogleImageCrawler(storage = {"root_dir" : argv[1]})
crawler = GoogleImageCrawler(storage={'root_dir': f'{argv[1]}/google'})
crawler.crawl(keyword = argv[2], max_num = 10000,  min_size=(200,200), max_size=None)

#bing_crawler = BingImageCrawler(storage = {"root_dir" : argv[1]})
bing_crawler = BingImageCrawler(storage={'root_dir': f'{argv[1]}/bing'})
bing_crawler.crawl(keyword=argv[2], max_num = 10000,  min_size=(200,200), max_size=None)

#baidu_crawler = BaiduImageCrawler(storage = {"root_dir" : argv[1]})
baidu_crawler = BaiduImageCrawler(storage={'root_dir': f'{argv[1]}/baidu'})
baidu_crawler.crawl(keyword=argv[2], max_num = 10000,  min_size=(200,200), max_size=None)

flickr_crawler = FlickrImageCrawler(storage={'root_dir': f'{argv[1]}/flickr'})
flickr_crawler.crawl(keyword=argv[2], max_num = 10000,  min_size=(200,200), max_size=None)

