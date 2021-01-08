import pandas as pd
import numpy as np
import math
import csv
import os.path
import pickle
from options_price_sim import black_scholes_form, get_greeks
import matplotlib.pyplot as plt
from matplotlib import cm

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

        #drop rows and cols
        data = data.drop(columns= TO_DROP)
        data = data.drop(data[data['trade_volume']==0].index)
        data = data.drop(data[data['option_type']=='P'].index)
        data = data.drop(data[data['expiration']==data['quote_date']].index)

        #adjusting type of col values
        data[NUM_COLS]  = data[NUM_COLS].apply(pd.to_numeric)
        data[DATE_COLS] = data[DATE_COLS].apply(pd.to_datetime)

        #creating cols that are specific to this application
        data['time_to_mat'] = (data['expiration'] - data['quote_date']).astype('timedelta64[D]')/DAYS_IN_YR
        data['stock_price'] = (data['underlying_bid_1545'] + data['underlying_ask_1545'])/2
        data['option_price'] = (data['bid_1545']+data['ask_1545'])/2
        vix = data.loc[data['underlying_symbol'] == 'VIXY', 'stock_price']
        data['dv_dk'] = 0
        data['d2v_dk2'] = 0
        data['dv_dt'] = 0
        
        if (len(vix)== 0):
            data['sigma_start'] = .15
        else:
            data['sigma_start'] = vix.iloc[0]/100 
        data.to_pickle(pkl_name)

        print(data.index)
        return data


'''
Function that calculates local volatility 
'''
def calc_loc_sig(otype, oprice, k, rf, s_t, q, t, delta, start_sig = .15):
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


def calc_imp_vol(df, rf):

    #sort by stock
    df.sort_values(by='underlying_symbol', axis=0)
    unique_stocks = df['underlying_symbol'].unique().tolist()
    for stock in unique_stocks:
        stock_df = df[df['underlying_symbol'] == stock]
        print(stock_df)

        #dv/dks
        unique_ttm = stock_df['time_to_mat'].unique().tolist()
        for time in unique_ttm:
            time_df = stock_df[stock_df['time_to_mat'] == time]
            time_df.sort_values(by='strike', axis=0)
            index = time_df.index
            if(len(index)>2):
            
                prices = time_df['option_price'].to_numpy()
                strikes = time_df['strike'].to_numpy()

                dv = prices[:-1] - prices[1:]
                dk = strikes[:-1] - strikes[1:]
                slopes = dv/dk
                dv_dk = (slopes[:-1]+slopes[1:])/2
                k_avg = (strikes[:-1] + strikes[1:])/2
                d_k_avg = k_avg[:-1]-k_avg[1:]
                d2v_dk2 = (slopes[:-1]-slopes[1:])/d_k_avg

                #padding zeros
                padded_dv_dk   = np.pad(dv_dk, (1,1), 'constant', constant_values=(0,0))
                padded_d2v_dk2 = np.pad(d2v_dk2, (1,1), 'constant', constant_values=(0,0))

                df.loc[index, 'dv_dk']   =  padded_dv_dk
                df.loc[index, 'd2v_dk2'] = padded_d2v_dk2

        #dv/dt
        unique_strikes = df['strike'].unique().tolist()
        for strike in unique_strikes:
            strike_df = stock_df[stock_df['strike'] == strike]
            strike_df.sort_values(by='time_to_mat', axis=0)
            index = strike_df.index
            if (len(index)>2):

                prices = strike_df['option_price'].to_numpy()
                times = strike_df['time_to_mat'].to_numpy()

                dv = prices[:-2] - prices[2:]
                dt = times[:-2] - times[2:]
                dv_dt = dv/dt

                #padding zeros
                padded_dv_dt = np.pad(dv_dt, (1,1), 'constant', constant_values=(0,0))

                df.loc[index, 'dv_dt'] = padded_dv_dt

    #calculate imp vol
    df = df.drop(df[df['dv_dk']==0].index)
    df = df.drop(df[df['d2v_dk2']==0].index)
    df = df.drop(df[df['dv_dt']==0].index)
    #deal with dv_dt = 0 
    df['imp_sigma'] = ( (df['dv_dt'] + rf*df['strike']*df['dv_dk'])*2/(df['strike']**2*df['d2v_dk2'])) **.5


    return df
    
def calc_delta(otype,k, rf,s_t, q, t, sig_guess):
    rf = rf/260
    try:
        c, p, d1, d2 = black_scholes_form(t, k, s_t, rf, sig_guess, q)
    except ValueError:
        print(t, k, s_t, rf, sig_guess, q)
        exit()
    delta = get_greeks(t, k, rf, sig_guess, d1, d2, s_t, q, only_delta=True)[0] if otype == 'C' else get_greeks(t, k, rf, sig_guess, d1, d2, s_t, q, only_delta=True)[1]
    return delta


def add_sigs_to_df(df):
    q=0 #dividend
    rf = .0016
    df = calc_imp_vol(df, rf)

    df['delta'] = df.apply(lambda  row : calc_delta(row['option_type'], row['strike'], rf, row['stock_price'], q, row['time_to_mat'], row['imp_sigma']),axis=1)
    df['loc_sigma'] = df.apply(lambda  row : calc_loc_sig(row['option_type'], row['option_price'], row['strike'], rf, row['stock_price'], q, row['time_to_mat'], row['delta'], start_sig = row['imp_sigma']),axis=1)
    
    return df

def graph(filename):
    df = pd.read_csv(filename, delimiter=',')
    stock = 'SPY'
    df = df.loc[df['underlying_symbol'] == stock] #only get certain stock
    df = df.loc[(df['time_to_mat'] - 0.0461538461538461) < 0.001] #only get certain expiration
    #print(df)

    df = df.drop(df[df['sigma_start']==df['imp_sigma']].index) #drop rows where sigma didn't move
    time = df['time_to_mat'].to_numpy()
    strike = df['strike'].to_numpy()
    sig = df['imp_sigma'].to_numpy()
    #time, strike = np.meshgrid(time,strike)
   
    #data = np.stack((time,strike,sig), axis=0)
    #np.savetxt("graph_data.csv", data, delimiter=",")
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    graph = ax.scatter3D(time, strike, sig, cmap=cm.coolwarm)
    ax.set_zlim(0, .2)
    ax.set_xlabel('Time to Maturity')
    ax.set_ylabel('Strike')
    ax.set_zlabel('Sigma')
    plt.show()
    # plt.savefig('graph.png')
    #ValueError: Argument Z must be 2-dimensional.


    

def main(filename='small_data.csv'):
    #clean_data('UnderlyingOptionsEODQuotes_2020-07-24.csv')

    #df = get_df(filename)
    #df = add_sigs_to_df(df)
    #df.to_csv('imp_vol.csv',index=False)

    graph('imp_vol.csv')
    


if __name__ == "__main__":
    main()