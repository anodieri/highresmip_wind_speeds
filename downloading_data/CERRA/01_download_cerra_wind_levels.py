# environment: era5

# script to download a whole series of CERRA files, year by year
# files will be downloaded in the folder where this script is executed
# so copy the script to the desired location
# and run:
# conda activate era5
# nohup python 01_download_cerra_wind_levels.py >download_status.out &

import cdsapi
import time

def download(years, variable: str = "wind_speed",  freq = '3h'):
    """ Function to activate the donwloading of CERRA from Copernicus
    years: the years to download, as a string or list of strings, e.g. ["2015", "2016", "2017",]
    variable: "wind_speed" or "wind_direction"
    freq: which timestamps should be downloaded: 3h or 6h
    """

    # set the hours depending on the frequency
    if freq == '3h':
        time = [
            "00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"
        ]
    elif freq == '6h':
        time = [
            "00:00", "06:00", "12:00","18:00"
        ]

    dataset = "reanalysis-cerra-height-levels"
    request = {
        "variable": [variable], #"wind_direction"
        "height_level": ["100_m"],
        "data_type": ["reanalysis"],
        "product_type": ["analysis"],
        "year": years,
        "month": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12"
        ],
        "day": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12",
            "13", "14", "15",
            "16", "17", "18",
            "19", "20", "21",
            "22", "23", "24",
            "25", "26", "27",
            "28", "29", "30",
            "31"
        ],
        "time": time,
        "data_format": "netcdf",
    }

    client = cdsapi.Client()
    client.retrieve(dataset, request).download()


years = [#"1984", "1985", "1986",
        #"1987", "1988", "1989",
        #"1990", "1991", "1992",
        #"1993", "1994", "1995",
        #"1996", "1997", "1998",
        #"1999", "2000", "2001",
        #"2002", "2003", "2004",
        #"2005", "2006", "2007",
        #"2008", "2009", "2010",
        #"2011", "2012", "2013",
        #"2014", "2015", "2016", 
        "2017", "2018", "2019", 
        "2020", "2021", "2022",
        "2023", "2024"]

# actually download the given years
for year in years:
    print(year)
    #if an error occurs, keep on trying, except for the keyboard interruption error
    while True:
        try:
            download(year, freq = '3h')
        except KeyboardInterrupt: 
            raise
        except:
            time.sleep(10) #wait 10 seconds
            continue
        break #so the while true is only broken when the download succeeds, otherwise it keeps on trying for the same year