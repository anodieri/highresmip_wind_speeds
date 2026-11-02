# script to download a whole series of ERA5 files
# files will be downloaded in the folder where this script is executed

# environment: era5
# Use
# in terminal with right kernel activated
# nohup python 01_download_era5_wind.py &

import cdsapi
import time

def download(years, variable: str,  freq: str):
    """ Function to activate the downloading of ERA5 from Copernicus for surface level data
    years: the years to download, as a string or list of strings, e.g. ["2015", "2016", "2017",]
    variable: #"10m_u_component_of_wind",
        #"10m_v_component_of_wind",
        #"100m_u_component_of_wind",
        #"100m_v_component_of_wind"
    freq: which timestamps should be downloaded: 1h, 3h or 6h
    """

    # set the hours depending on the frequency
    if freq == '1h':
        time = [
            "00:00", "01:00", "02:00", "03:00", "04:00", "05:00", "06:00", "07:00", "08:00", "09:00"
            "10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00", "17:00", "18:00", "19:00", 
            "20:00", "21:00", "22:00", "23:00"
        ]
    if freq == '3h':
        time = [
            "00:00", "03:00", "06:00", "09:00", "12:00", "15:00"
            "18:00", "21:00"
        ]
    elif freq == '6h':
        time = [
            "00:00", "06:00", "12:00",
            "18:00"
        ]
    dataset = "reanalysis-era5-single-levels"
    request = {
        "variable": [variable],
        "product_type": ["reanalysis"],
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
        "download_format": "unarchived",
        "area": [62, -14, 43, 18]
    }

    client = cdsapi.Client()
    client.retrieve(dataset, request).download()

# prepare all of the years to be downloaded
years = ["1984", "1985", "1986",
        "1987", "1988", "1989",
        "1990", "1991", "1992",
        "1993", "1994", "1995",
        "1996", "1997", "1998",
        "1999", "2000", "2001",
        "2002", "2003", "2004",
        "2005", "2006", "2007",
        "2008", "2009", "2010",
        "2011", "2012", "2013",
        "2014", "2015", "2016", 
        "2017", "2018", "2019", 
        "2020", "2021", "2022",
        "2023", "2024"]

# and let's also do all of the variables at once
variables = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind"
]

# actually download the given years and variables
for year in years:
    print(year)

    for var in variables:
        print(var)
        #if an error occurs, keep on trying, except for the keyboard interruption error
        while True:
            try:
                download(year, var, freq = '1h')
            except KeyboardInterrupt: 
                raise
            except:
                time.sleep(10) #wait 10 seconds
                continue
            break #so the while true is only broken when the download succeeds, otherwise it keeps on trying for the same year
