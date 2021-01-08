# options_rl

## Downloading Data
The data is downloaded from ...  to download the data, create a json file called `<login.json>`.  This will contain your credentials.  \
Fill in the file as such
```
{
    "SFTP_HOSTNAME": "sftp.datashop.livevol.com",
    "SFTP_USERNAME" : [username],
    "SFTP_PASSWORD" : [password],
    "PATH_TO_ORDER_FILES" : [order id]
}
```
In your directory, create two folders, `<data/>` and `<extracted/>`.\
Then run `<import_data.py>` to download and extract the data.  

## Running Simulation 

## Calculating the Volatilities
