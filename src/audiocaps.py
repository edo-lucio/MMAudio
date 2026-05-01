from audiocaps_download import Downloader

d = Downloader(root_path='data/audiocaps/', n_jobs=4)
d.download(format = 'wav') # it will cross-check the files with the csv files in the original repository