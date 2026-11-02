import subprocess, os, time

# loop over the years, downloading each file with wget
for year in range(1970,2015):
    url_zg = f"https://accc.rcec.sinica.edu.tw/lama/datashare/HiRAM-SIT-LR/highresSST-present/day/zg_day_HiRAM-SIT-LR_highresSST-present_r1i1p1f1_gn_{year}0101-{year}1231.nc"
    subprocess.run(['wget', url_zg])
