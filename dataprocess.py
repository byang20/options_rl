import pandas as pd
import numpy as np
import math
import csv
import os.path
import pickle

#global variables
NUM_COLS = ['strike', 'open','high','low','close','trade_volume','bid_size_1545','bid_1545','ask_size_1545','ask_1545','underlying_bid_1545','underlying_ask_1545','bid_size_eod','bid_eod','ask_size_eod','ask_eod','underlying_bid_eod','underlying_ask_eod','vwap','open_interest']
DATE_COLS = ['quote_date','expiration']
STR_COLS = ['underlying_symbol', 'option_type']


'''
Function that returns the data frame 
'''
def get_df(filename):

    cwd = os.getcwd()
    pkl_name = os.path.join( cwd,  filename[:-4] + ".pkl")

    if os.path.exists(pkl_name):
        data = pd.read_pickle(pkl_name)
        return data
    
    else:
        data = pd.read_csv(filename, delimiter=',')
        data = data.drop(['delivery_code'], 1)
        data[NUM_COLS]  = data[NUM_COLS].apply(pd.to_numeric)
        data[DATE_COLS] = data[DATE_COLS].apply(pd.to_datetime)
        data.to_pickle(pkl_name)
        return data

def get_stats(df):

def calc_vol:
    #need T-t, S_t, K, r, sigma
    #ttf sigma 
    #Option price: average bid and ask
    #T-t using date difference 
    #rf rate
    #Strike price: given
    #rf: use tbond data
    


def main(filename='optiondata.csv'):
    data_frame = get_df(filename)
    print(data_frame.head())
    '''
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimeter = ',')
        csv_reader.readline() #skip first row
        for row in csv_reader:
            symbol,date,expiration,strike,cp,open,high,low,close,volume = row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9]
            bid_size345,bid1545,ask_size345,ask1545,under_bid345,under_ask345 = row[10],row[11],row[12],row[13],row[14],row[15]
            bid_sizeeod,bideod,ask_sizeeod,askeod,under_bideod,under_askeod,vwap,open_int,deliv_code=
    '''

if __name__ == "__main__":
    main()