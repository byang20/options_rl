import pandas as pd
import numpy as np
import math
import csv
import os.path
import pickle
from options_price_sim import black_scholes_form, get_greeks

#global variables
NUM_COLS = ['strike', 'trade_volume','bid_1545','ask_1545','underlying_bid_1545','underlying_ask_1545']
DATE_COLS = ['quote_date','expiration']
STR_COLS = ['underlying_symbol', 'option_type']
DAYS_IN_YR = 260
TO_DROP= ['open', 'high', 'low', 'close', 'bid_size_1545', 'ask_size_1545', 'bid_size_eod', 'ask_size_eod', 'bid_eod', 'ask_eod', 'underlying_bid_eod', 'underlying_ask_eod', 'vwap', 'open_interest', 'delivery_code']


'''
Function that returns the data frame 
'''

def clean_data(filename):
    cwd = os.getcwd()
    pkl_name = os.path.join( cwd,  filename[:-4] + ".pkl")

    data = pd.read_csv(filename, delimiter=',')
    #data = data.drop(columns= TO_DROP)
    data = data.drop(data[data['trade_volume']==0].index)
    data = data.drop(data[data['expiration']==data['quote_date']].index)
    file_new = 'cleaned_' + filename
    data.to_csv(file_new, index=False)

def get_df(filename):

    cwd = os.getcwd()
    pkl_name = os.path.join( cwd,  filename[:-4] + ".pkl")

    if os.path.exists(pkl_name):
        data = pd.read_pickle(pkl_name)
        return data
    
    else:
        data = pd.read_csv(filename, delimiter=',')
        data = data.drop(columns= TO_DROP)
        data = data.drop(data[data['trade_volume']==0].index)
        data = data.drop(data[data['expiration']==data['quote_date']].index)
        data[NUM_COLS]  = data[NUM_COLS].apply(pd.to_numeric)
        data[DATE_COLS] = data[DATE_COLS].apply(pd.to_datetime)
        data['time_to_mat'] = (data['expiration'] - data['quote_date']).astype('timedelta64[D]')/DAYS_IN_YR
        data['stock_price'] = (data['underlying_bid_1545'] + data['underlying_ask_1545'])/2
        data['option_price'] = (data['bid_1545']+data['ask_1545'])/2
        vix = data.loc[data['underlying_symbol'] == 'VIXY', 'stock_price']
        if (len(vix)== 0):
            data['sigma_start'] = .15
        else:
            data['sigma_start'] = vix.iloc[0]/100 
        data.to_pickle(pkl_name)
        return data


'''
Function that calculates implied volatility 
'''
def calc_sig(otype, oprice, k, rf, s_t, q, t, delta, start_sig = .15):
    price_guess = 0
    sig_guess = start_sig
    max_itr_cnt = 70
    
    while ( abs(.5-abs(delta)) < .45 and abs(price_guess - oprice) > .0001 and max_itr_cnt > 0) :
        vega = 0
        c, p, d1, d2 = black_scholes_form(t, k, s_t, rf, sig_guess, q)
        price_guess = c if otype == 'C' else p
        vega = get_greeks(t, k, rf, sig_guess, d1, d2, s_t, q, only_vega=True)
        sig_guess += (oprice - price_guess)*(.5)/vega
        max_itr_cnt -= 1
        #print('round ', 70-max_itr_cnt, ': ', sig_guess, ", ", price_guess)
    #print(price_guess)
    return sig_guess

def calc_delta(otype,k, rf,s_t, q, t, sig_guess):
    rf = rf/260
    try:
        c, p, d1, d2 = black_scholes_form(t, k, s_t, rf, sig_guess, q)
    except ValueError:
        print(t, k, s_t, rf, sig_guess, q)
        exit()
    delta = get_greeks(t, k, rf, sig_guess, d1, d2, s_t, q, only_delta=True)[0] if otype == 'C' else get_greeks(t, k, rf, sig_guess, d1, d2, s_t, q, only_delta=True)[1]
    return delta
    

def add_sig_to_df(df):
    q=0 #dividend
    rf = .0016

    df['delta'] = df.apply(lambda  row : calc_delta(row['option_type'], row['strike'], rf, row['stock_price'], q, row['time_to_mat'], row['sigma_start']),axis=1)
    df['imp_sig'] = df.apply(lambda  row : calc_sig(row['option_type'], row['option_price'], row['strike'], rf, row['stock_price'], q, row['time_to_mat'], row['delta'], start_sig = row['sigma_start']),axis=1)
    return df


def main(filename='small_data.csv'):
    #clean_data('UnderlyingOptionsEODQuotes_2020-07-24.csv')

    df = get_df(filename)

    df = add_sig_to_df(df)
    df.to_csv('imp_vol.csv',index=False)
    


if __name__ == "__main__":
    main()