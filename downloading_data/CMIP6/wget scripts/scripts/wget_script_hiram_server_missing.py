import subprocess, os, time

# loop over the years, downloading each file with wget
for year in range(1970,2015):
    url_ua = f"https://accc.rcec.sinica.edu.tw/lama/datashare/HiRAM-SIT-LR/highresSST-present/day/ua_day_HiRAM-SIT-LR_highresSST-present_r1i1p1f1_gn_{year}0101-{year}1231.nc"
    url_va = url_ua.replace('ua', 'va')
    subprocess.run(['wget', url_ua])
    subprocess.run(['wget', url_va])