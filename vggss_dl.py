from __future__ import unicode_literals
from concurrent.futures import thread
import csv
import os

from time import time
from time import strftime
from time import gmtime

import json, random, threading, queue
import youtube_dl

import subprocess

import logging
from inspect import currentframe, getframeinfo  # for debugging and logging
SID = "VGGSS_DL"
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    filename='LOG.log',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def get_length(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)

PATH_TO_VGGSS = './vggss.json'
NUM = 5158 # max 5158
MAIN_DIR = 'data'


class VideoItem():
    def __init__(self, id, start, duration, file_name, raw) -> None:
        self.id = id
        self.start = start
        self.duration = duration
        self.file_name = file_name
        self.raw = raw
    
    def __repr__(self) -> str:
        return "VideoItem {"+f"id: {self.id}, start: {self.start}"+"}"
    
    def start_str(self) -> str:
        return strftime("%H:%M:%S", gmtime(self.start))

    def duation_str(self) -> str:
        return strftime("%H:%M:%S", gmtime(self.duration))


def get_files_from_json(json_path:str):
    assert len(json_path.strip()) > 0

    videos = []
    with open(json_path, 'r') as f:
        jf = json.load(f)
        for e in jf:
            videos.append(VideoItem(e['file'][:11],int(e['file'][12:]),10,e['file'], e))
    return videos

class DLThread(threading.Thread):
    def __init__(self, video:VideoItem) -> None:
        threading.Thread.__init__(self)
        self.video = video
    def run(self):
        download_video(self.video)

def download_video(vid:VideoItem):
    logging.info(f'{SID}[{getframeinfo(currentframe()).lineno}] Downloading \'{vid.id}\'...')
    t0 = time()
    try:
        ytdl_logger = logging.getLogger("ytdl-ignore")
        ytdl_logger.disabled = True
        ydl_opts = {
            "logger": ytdl_logger,
            'format': 'best',
            'outtmpl': f'./{MAIN_DIR}/{vid.file_name}.%(ext)s',
            'simulate': False,
            'quiet': True,
            'prefer_ffmpeg': True,
            'external_downloader': 'ffmpeg',
            'external_downloader_args': ['-ss',f'{vid.start_str()}', '-t', f'{vid.duation_str()}'],
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            }],
            }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={vid.id}"])
        vfname = os.path.join(MAIN_DIR,f'{vid.file_name}.mp4')
        if os.path.isfile(vfname): # if exists (sanity check)
            extract_elements_from_video(vid, vfname)
    except youtube_dl.utils.DownloadError as e:
        logging.error(f'{SID}[{getframeinfo(currentframe()).lineno}] Downloading error for {vid.id} : {e}.')
    except Exception as e:
        logging.error(f'{SID}[{getframeinfo(currentframe()).lineno}] Unknown error for {vid.id} : {e}.')
    logging.info(f'{SID}[{getframeinfo(currentframe()).lineno}] Finished \'{vid.id}\', time: {time()-t0} seconds.')

threads = []

def check_missing(vid:VideoItem, force=True):
    vfname = os.path.join(MAIN_DIR,f'{vid.file_name}.mp4')
    if not os.path.isfile(vfname):
        return 1
    if force:
        return extract_elements_from_video(vid, vfname)
    return 0

def extract_elements_from_video(vid, vfname):
    ifname = os.path.join(MAIN_DIR,f'{vid.file_name}.jpg')
    afname = os.path.join(MAIN_DIR,f'{vid.file_name}.wav')
    try:
        vlength = get_length(vfname)
        if vlength < 5:
            logging.warning(f'{SID}[{getframeinfo(currentframe()).lineno}] Removing \'{vid.id}\', duration: {vlength} ({vfname})...')
            os.remove(vfname)
            return 1

        ret = 0
        if not os.path.isfile(ifname):
            logging.info(f'{SID}[{getframeinfo(currentframe()).lineno}] Extracting IMAGE element ({vfname})...')
            ret = os.system(f"ffmpeg -hide_banner -loglevel error -i {vfname} -ss {strftime('%H:%M:%S', gmtime(vlength/2))} -vf scale=512:512 -frames:v 1 -q:v 2 {ifname}")
        elif os.path.getsize(ifname) < 1:
            os.remove(ifname)
            ret = -1
        if ret != 0:
            logging.warning(f'{SID}[{getframeinfo(currentframe()).lineno}] FFMPEG issue for IMAGE ({vid.id}), aborting.')
            os.remove(vfname)
            return 1

        if not os.path.isfile(afname):
            logging.info(f'{SID}[{getframeinfo(currentframe()).lineno}] Extracting AUDIO element ({vfname})...')
            ret = os.system(f"ffmpeg -hide_banner -loglevel error -i {vfname} -ss {strftime('%H:%M:%S', gmtime(vlength/2 - 1))} -t 00:00:03 -f wav -bitexact -acodec pcm_s16le -ar 22050 -ac 1 {afname}")
        elif os.path.getsize(afname) < 1:
            os.remove(afname)
            ret = -1
        if ret != 0:
            logging.warning(f'{SID}[{getframeinfo(currentframe()).lineno}] FFMPEG issue for AUDIO ({vid.id}), aborting.')
            os.remove(vfname)
            os.remove(ifname)
            return 1
    except Exception as e:
        logging.error(f'{SID}[{getframeinfo(currentframe()).lineno}] Unknown error for {vid.id} : {e}.')
    logging.info(f'{SID}[{getframeinfo(currentframe()).lineno}] Data prep. successful ({vid.id}).')
    return 0

if __name__ == "__main__":
    videos = get_files_from_json(PATH_TO_VGGSS)
    print(f'Preparing to download {NUM} videos.')
    logging.info(f'{SID}[{getframeinfo(currentframe()).lineno}] Attempting to download {NUM}/{len(videos)} videos.')
    tt0 = time()
    sampling = random.sample(videos,NUM)
    printProgressBar(0, NUM*2+3, prefix = 'Progress:', suffix = 'Complete', length = 50)
    for i,vid in enumerate(sampling):
        if check_missing(vid) == 1:
            try:
                threads.append(DLThread(vid))
                threads[-1].start()
            except Exception as e:
                logging.error(f'{SID}[{getframeinfo(currentframe()).lineno}] Unknown error for {vid.id} : {e}.')
            printProgressBar(i + 1, NUM*2, prefix = 'Progress:', suffix = 'Complete', length = 50)
    for i,t in enumerate(threads):
        t.join()
        printProgressBar(NUM + i + 1, NUM*2+3, prefix = 'Progress:', suffix = 'Complete', length = 50)
    
    missing = 0
    data_items = []
    for vid in sampling:
        if check_missing(vid, force=False) == 1:
            missing += 1
        else:
            data_items.append(vid.raw)
    
    printProgressBar(NUM*2 + 1, NUM*2+3, prefix = 'Progress:', suffix = 'Complete', length = 50)
    
    with open('./vggss_mini.json', 'w') as f:
        json.dump(data_items, f)
    printProgressBar(NUM*2 + 2, NUM*2+3, prefix = 'Progress:', suffix = 'Complete', length = 50)
    with open('./vggss_mini.csv', 'w') as f:
        csv_w = csv.writer(f)
        for _d in data_items:
            csv_w.writerow([str(_d['file']).strip()])
    printProgressBar(NUM*2 + 3, NUM*2+3, prefix = 'Progress:', suffix = 'Complete', length = 50)

    print(f'Finished. Missing: {missing}/{len(sampling)} Total time: {strftime("%H:%M:%S", gmtime(time()-tt0))}')
    logging.info(f'{SID}[{getframeinfo(currentframe()).lineno}] Finished. Total time: {strftime("%H:%M:%S", gmtime(time()-tt0))}.')