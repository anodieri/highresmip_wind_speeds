# environment: postprocess (for example)
# edit the dates at the bottom of the file to perform the downloads
# used to download a lot of timesteps at once (but not full dataset)
# use
# python 02_update_dataset_functions.py

# necessary imports
import logging
import sys
from datetime import datetime
from datetime import timezone
from datetime import timedelta
import requests
import time
import os

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel("INFO")

# copied from example script
class OpenDataAPI:
    def __init__(self, api_token: str):
        self.base_url = "https://api.dataplatform.knmi.nl/open-data/v1"
        self.headers = {"Authorization": api_token}

    def __get_data(self, url, params=None):
        return requests.get(url, headers=self.headers, params=params).json()

    def list_files(self, dataset_name: str, dataset_version: str, params: dict):
        return self.__get_data(
            f"{self.base_url}/datasets/{dataset_name}/versions/{dataset_version}/files",
            params=params,
        )

    def get_file_url(self, dataset_name: str, dataset_version: str, file_name: str):
        return self.__get_data(
            f"{self.base_url}/datasets/{dataset_name}/versions/{dataset_version}/files/{file_name}/url"
        )

def download_file_from_temporary_download_url(download_url, filename):
    try:
        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            with open(filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception:
        logger.exception("Unable to download file using download URL")
        sys.exit(1)

    logger.info(f"Successfully downloaded dataset file to {filename}")

# own creation, specifically for knmi lidar wind speed 10 minutes data (stored in files of one day)
def download_hour(key: str, start: datetime):
    """ Donwload the KNMI lidar data for a specified from the start timestep, until 1 hour later
    key: str indicating if personal, public_1 or public_2 api key should be used
    start: datetime(year = start_year, month = start_month, day = start_day, hour= start_hour) example start datetime
    """

    # different api keys from the website, edit if needed
    # TODO EDIT THESE KEYS 
    if key == 'personal': api_key = "" # personal key, from email
    elif key == 'public_1': api_key = "" # public key, from https://developer.dataplatform.knmi.nl/open-data-api#token
    api = OpenDataAPI(api_token=api_key)

    # specifications of the dataset, can be found on knmi website https://dataplatform.knmi.nl/
    dataset_name = "windlidar_nz_wp_platform_10min"
    dataset_version = "1"

    # set the start and stop timestamps
    timestamp_start = start
    timestamp_stop = timestamp_start + timedelta(minutes = 60) # maximally downloads ten files, so 1 hour is a safe step (must be more than 10 minutes)

    # order the files by creation date and begin listing after the specified timestamp, up to the stop timesteps
    params = {"orderBy": "created", "begin": timestamp_start, "end": timestamp_stop}
    response = api.list_files(dataset_name, dataset_version, params)

    # report errors
    if "error" in response:
        logger.error(f"Unable to retrieve list of files: {response['error']}")
        sys.exit(1)

    # Download the files
    for file in response["files"]:
        file_name = file.get("filename")
        # fetch the download url and download the file
        response = api.get_file_url(dataset_name, dataset_version, file_name)
        download_file_from_temporary_download_url(response["temporaryDownloadUrl"], file_name)

def download_multiple_hours(start: datetime, stop: datetime):
    """ Download multiple hours of KNMI lidar data in a loop, from the start until the stop date
    start, stop: in shape datetime(year = start_year, month = start_month, day = start_day, hour= start_hour)
    """
    # prepare the download keys
    keys = ['personal', 'public_1'] # public 2 does not exist anymore
    key_index = 0
    while start < stop:
        while True:
            try:
                print(start, stop)
                download_hour(keys[key_index], start)
            except KeyboardInterrupt: 
                raise
            except:
                time.sleep(10) #wait 10 seconds
                key_index = (key_index + 1)%len(keys) # switch between the keys while trying
                continue
            break #so the while true is only broken when the download succeeds, otherwise it keeps on trying for the same year
        # files are for 1 day, advance one day when file was succesfully downloaded
        start = start + timedelta(days = 1) 

def check_number_of_files(path, lidar_name: str):
    """ Check the number of downloaded files to see if any files are missing, print missing files
    Caution: missing data are just missing files, so this is normal
    """
        
    # Get a list of all files in that folder and sort them
    file_list = os.listdir(path)
    file_list = [f for f in file_list if 'windlidar' in f] # to avoid also using full file/downloadscripts/... in the folder
    file_list_lidar =  [f  for f in file_list if lidar_name in f] # select files for the correct lidar
    file_list_lidar.sort()
    print(len(file_list_lidar))

    # Get the strings with the start dates
    date_strings = [f.split('_')[4] for f in file_list_lidar]
    
    # Convert strings to datetime objects
    dates = [datetime.strptime(s, "%Y%m%d%H%M") for s in date_strings]

    # Construct expected range: every day from first to last date
    expected = []
    current = dates[0]
    while current <= dates[-1]:
        expected.append(current)
        current += timedelta(days=1)

    # Find missing dates
    missing = sorted(set(expected) - set(dates))
    print(*missing, sep = "\n")
    return missing
  
# TODO
# do the actual downloading: use this line in the location the files are supposed to be downloaded
# edit start and stop here
#download_multiple_hours(start = datetime(year = 2026, month = 1, day = 1, hour= 0), stop = datetime(year = 2026, month = 8, day = 31, hour= 0))
