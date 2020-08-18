import time
import requests
import threading
from concurrent import futures
import concurrent

"""极客并发编程"""
def download(url):
    # url = 'http://127.0.0.1:8090/index'
    time.sleep(1)
    # resp = requests.get(url)
    # print(resp.raise_for_status())
    print(url + '---')
    return url + '='

def furtue_test(urls):
    with futures.ThreadPoolExecutor(max_workers=3) as work:
        to_do = []
        for url in urls:
            res = work.submit(download, url)
            to_do.append(res)
        for resu in concurrent.futures.as_completed(to_do):
            print(resu.result())

def furtue(urls):
    '''异步并发'''
    with futures.ThreadPoolExecutor(max_workers=3) as work:
        try:
            # work.map(download, urls)
            work.map(download, (i for i in "abcdefghijk"))
            print(threading.enumerate())
        except concurrent.futures.TimeoutError as e:
            print(e)

if __name__ == '__main__':
    strt = time.perf_counter()

    furtue_test("abcdefghijk")
    end = time.perf_counter()
    print(end - strt)
