import pandas as pd
import numpy as np
import math
import csv
import os.path
import pickle
from options_price_sim import black_scholes_form, get_greeks

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
        data[NUM_COLS]  = data[NUM_COLS].apply(pd.to_numeric)
        data[DATE_COLS] = data[DATE_COLS].apply(pd.to_datetime)
        data['time_to_mat'] = data['expiration'] - data['quote_date']
        data['stock_price'] = (data['underlying_bid_1545'] + data['underlying_ask_1545'])/2
        data['option_price'] = (data['bid_1545']+data['ask_1545'])/2
        vix = data.loc[data['underlying_symbol'] == 'VIXY', 'stock_price']
        if (len(vix)== 0):
            data['sigma_start'] = '.15'
        else:
            data['sigma_start'] = vix.iloc[0] 
        data.to_pickle(pkl_name)
        return data


'''
Function that calculates implied volatility 
'''
def calc_sig(type, price, k, rf, s_t, q, t, start_sig = .15):
    price_guess = 0
    sig_guess = start_sig
    max_itr_cnt = 100
    
    while ( abs(price_guess - price) > .0001 and max_itr_cnt > 0) :
        vega = 0
        c, p, d1, d2 = black_scholes_form(t, k, s_t, rf, sig_guess, q)
        price_guess = c if type == 'C' else p
        vega = get_greeks(t, k, rf, sig_guess, d1, d2, s_t, q, only_vega=True)
        sig_guess += (price - price_guess)/vega 
        max_itr_cnt -= 1
        print('round ', 100-max_itr_cnt, ': ', sig_guess)
    print(price_guess)
    return sig_guess

    
def add_sig_to_df(df):

    df['imp_sig'] = df.apply(lambda  row : calc_sig(row['option_type'], row['option_price'], row['strike'], rf, row['stock_price'], q, row['time_to_mat'], start_sig = row['']))


def main(filename='optiondata.csv'):
    data_frame = get_df(filename)
    print(data_frame.head())
    #print(data_frame.head())

    price = 3.23
    k = 50
    rf = .03
    s_t = 50
    q = 0
    t = 1
    sig_true = .2

    sig = calc_sig(data_frame, 'P', price, k, rf, s_t, q, t)
    #print(sig, sig_true)


if __name__ == "__main__":
    main()