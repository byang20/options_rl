# options_rl

## Downloading Data
The data is downloaded from CBOE.  To download the data, create a json file called `login.json`.  This will contain your credentials.  \
Fill in the file as such
```
{
    "SFTP_HOSTNAME": "sftp.datashop.livevol.com",
    "SFTP_USERNAME" : [username],
    "SFTP_PASSWORD" : [password],
    "PATH_TO_ORDER_FILES" : [order id]
}
```
Then run `python import_data.py` to download and extract the data.  Zip files will be in a folder called 'data/' and extracted csv files will be in a folder called 'extracted/'.

## Running Simulation 

## Calculating the Volatilities
