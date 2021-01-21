import os
import pysftp
import json
import zipfile
from os import listdir
from os.path import isfile, join

with open('login.json','r') as f:
      config = json.load(f)

print(config)

DATA_PATH = 'data'
EXTRACT_PATH = 'extracted'
KNOWN_HOSTS_PATH = 'path/to/known_hosts'
SFTP_HOSTNAME = config['SFTP_HOSTNAME']
SFTP_USERNAME = config['SFTP_USERNAME']
SFTP_PASSWORD = config['SFTP_PASSWORD']
PATH_TO_ORDER_FILES = config['PATH_TO_ORDER_FILES']
LOCAL_DIR_PATH = os.getcwd()


def get_zips(directory, extension):
    return [f for f in listdir(directory) if f.endswith(extension) and isfile(join(directory, f))]


def main():
    os.mkdir(os.path.join(LOCAL_DIR_PATH,DATA_PATH))

    cnopts = pysftp.CnOpts(knownhosts=KNOWN_HOSTS_PATH)
    cnopts.hostkeys = None
    with pysftp.Connection(SFTP_HOSTNAME, username=SFTP_USERNAME, password=SFTP_PASSWORD, cnopts=cnopts) as sftp:
        with sftp.cd(PATH_TO_ORDER_FILES):  # temporarily change directory
            for file_name in sftp.listdir():  # list all files in directory
                sftp.get(file_name, localpath=os.path.join(LOCAL_DIR_PATH, DATA_PATH, file_name))  # get a remote file
                print('File {} downloaded.'.format(file_name))
    
    
    os.mkdir(os.path.join(LOCAL_DIR_PATH,EXTRACT_PATH))

    path_to_zip_file = os.path.join(LOCAL_DIR_PATH, DATA_PATH)
    directory_to_extract = os.path.join(LOCAL_DIR_PATH, EXTRACT_PATH)

    onlyfiles = [f for f in listdir(path_to_zip_file) if isfile(join(path_to_zip_file, f))]
    files = get_zips(path_to_zip_file, '.zip')
    files.sort()

    for file in files:
        file = os.path.join(path_to_zip_file, file)
        with zipfile.ZipFile(file, 'r') as zip_ref:
            zip_ref.extractall(directory_to_extract)

if __name__ == '__main__':
    main()
    


