# Handling CMIP6 data

## Instructions for downloading with wget and processing

1. Generate the wget script on the esgf download website e.g. https://esgf-metagrid.cloud.dkrz.de/search: select data you want and add it to the cart. There are different nodes which have different datasets: see https://esgf.github.io/nodes.html. Scripts that were used to download the HighResMIP data used here can be found in `./scripts`
2. Navigate to the cart on this website, scroll down to **Download your cart**  and select **wget**, and click the download symbol to download the wget script.
3. Edit the script: it is only possible to remove the files you don't need from the list of files (namely unnecessary timesteps). This can easily done using the function defined in `00_script_editor.ipynb`. This also splits the script into multiple scripts if needed, as only 1000 files can be downloaded by each script at once. Pay attention: sometimes the downloading of the wget script automatically cuts off at 1000 files, without raising any warning.
4. Navigate to the location of the script in a terminal window (the data will be downloaded in this folder)
5. Run the script: `bash wget_script_name.sh -H`. Depending on the node, `-H` might need to be omitted. When prompted, enter your openID which can be found on https://esg-dn1.nsc.liu.se/user/detail/9911/ after creating an account and the password. If you don't have an account yet, create one on e.g. https://esg-dn1.nsc.liu.se/projects/esgf-liu/ 
 To run in the background, use `nohup scriptname -H openid password >outputfile.out &` or for some nodes just `nohup scriptname >outputfile.out &` and optionally use `>/dev/null` to not write any output which is handy because for many files the output can be large.
7. Use the `02_clipper.py` script to limit the files to the geographical area and vertical levels of interest. This can be run in parallel with the downloading script, and is advised to limit storage demands. However, files that are currently being downloaded cannot be clipped and are renamed FILENAME  + `PROBLEM`. These can be renamed to their original filename again once fully downloaded: this can be handled by running `02.5_problem_solver.py` in parallel. This is mainly important for downloads of large files, for small files the problematic files can easily be renamed after the full downloading is completed.
8. Wind speed files have to be concatenated into the usual one big file per experiment using the `03_concatenate_files.py` script.

Notes:
- The wget script is supposed to be smart: when a file is already downloaded in the folder, it detects this and does not redownload it. So when the script is stopped for some reason, it can just be run again to continue downloading the files.
- Each of the scripts mentioned has more detailed instructions at the top.


## More information:
https://esgf.github.io/esgf-user-support/user_guide.html#wget-scripting

## Common problems:
https://esgf.github.io/esgf-user-support/faq.html#esgf-general