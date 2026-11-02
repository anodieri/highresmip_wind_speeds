# Downloading data from KNMI platform

## Downloading full dataset
1. To download a full dataset, request an dataset bulk API key by sending an email: see https://developer.dataplatform.knmi.nl/open-data-api#example-ten, under `Obtaining an API token`, section `Requesting a dataset bulk key`. You need an account for this, if need be register first on this website. There is also the registered key, which is sent in an email (not the hash in the account on the website!).
2. Edit the example script to download a full dataset on the same page, under `Python example: Download a full dataset`.
3. Modify this script: adapt the dataset name and version (can be found in the data catalogue of the KNMI https://dataplatform.knmi.nl/), the download location, and the API key (sent through email). See `01_download_full_dataset.py`
4. Run the script. Careful: the dataset can only be downloaded once!
5. If some files were not downloaded, just rerun the script in the same folder. It will recognize which files have already been downloaded.

## Update a dataset
When updating a previously downloaded dataset, the KNMI is not willing to send a new bulk API key. Use the script/the functions from `02_update_dataset_functions.py` to update the dataset, looping over the timesteps and trying the public and personal API keys until a download succeeds.

**Important note: the height value in the file is relative to the instrument, located on the roof deck of the platform. For height above mean sea level one has to add the following values: 45m (BSA/BSB/HKZA/HKZB), 42m (HKN), 43m (HKWA/HKWB).**
