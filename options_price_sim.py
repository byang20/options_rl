import os
import numpy as np
import math
import argparse
from collections import Counter
from scipy.stats import norm


def passed_arguments():
	parser = argparse.ArgumentParser(description="Script to....")
	parser.add_argument("--sim_cnt",
											type=int,
											required=True,
											help="Number of rounds to simulate")

	parser.add_argument("--rf",
											type=float,
											required=True,
											help="Risk free rate")

	
	parser.add_argument("--mat",
											type=int,
											required=True,
											help="Maturity, T in years")

	parser.add_argument("--spot_price",
											type=float,
											required=True,
											help="spot price, St")

	parser.add_argument("--strike_price",
											type=float,
											required=True,
											help="strike price, K")
	parser.add_argument("--sigma",
											type=float,
											required=True,
											help="sigma")

   

	args = parser.parse_args()
	return args



'''
Function that simulates one realized path
t - time to maturity
r - stock return drift
v - volatility
start - start price
'''
def round(t, r, sigma, start,days):
    mu = 0
    s = 1
    sigma_daily = sigma / math.sqrt(days)
    rands = np.random.normal(mu, s, t*days)
    st_s = np.zeros(t*days+1)
    st_s[0] = start
    for i in range (days*t):
            st_s[i+1] = st_s[i] + st_s[i] * r / days + st_s[i] * sigma_daily * rands[i]
    return st_s

'''
Function that simulates specified number of paths, returns call and put values
'''
def run_sim(t, runs, r, sigma, start,days, K):
    prices = np.zeros((runs,t*days+1))
    for i in range(runs):
        prices[i,:] = round(t, r, sigma, start, days)


    value_c = np.sum(np.clip(prices[:,-1]-K, a_min=0, a_max=None))/runs
    value_p = np.sum(np.clip(K-prices[:,-1], a_min=0, a_max=None))/runs
    
    return value_c, value_p


def get_pv_sim(value, r, t ):
    return value/((1+r)**(t))


def n_prime(x):
    return 1/math.sqrt(2*math.pi) * math.exp((-x^2)/2)

def get_greeks(t, k, prices, r, sigma, d1, d2, s):
    n_prime_d1 = n_prime(d1)
    n_prime_d2 = n_prime(d2)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)
    neg_cdf_d2 = norm.cdf(-1*d2)
    time = np.arange(t,0,-1)	

    #delta calculations
    delta_c = cdf_d1
    delta_p = delta_c-1

    #gamma calculations
    gamma = n_prime_d1 / (s * sigma * math.sqrt(t))

    #vega calculations
    vega = s * n_prime_d1 * math.sqrt(t)

    #theta calculations
    theta_c = -1*((prices*n_prime_d1) * sigma)/(2*np.sqrt(time)) - r*k*np.exp(-r * time) * cdf_d2
    theta_p = -1*((prices*n_prime_d1) * sigma)/(2*np.sqrt(time)) + r*k*np.exp(-r * time) * neg_cdf_d2

    return [(delta_c, gamma, vega, theta_c),(delta_p, gamma, vega, theta_p)]

'''
Function that returns the black scholes calculation of option price
t - maturity time , int as years
k - strice price
s_t = spot price
r - risk free rate
sigma - volatilities over t
'''
def black_scholes_form(t, k, s_t, r, sigma, days):

    d1 = 1/(sigma*np.sqrt(t)) * (np.log(s_t/k) + (r + np.square(sigma)/2) * t)

    d2 = d1 - sigma*(np.sqrt(t))

    pv = k*np.exp(-r*t)

    c = norm.cdf(d1) * s_t - norm.cdf(d2) * pv

    p = -1 * norm.cdf(-d1) * s_t  + norm.cdf(-d2) * pv
    
    return c, p
    


def main(sim_cnt, r, T, s_t, k, sigma):
    days = 260
    v_c, v_p = run_sim(T, sim_cnt, r, sigma , s_t, days, k) 
    pv_c= get_pv_sim(v_c ,r, T)
    pv_p= get_pv_sim(v_p ,r, T)
    #get_vol(sim_cnt, )

    c, p = black_scholes_form(T, k, s_t, r, sigma, days)
    print('                call             put')
    print('black-scholes: ', c,  '; ', p)
    print('          sim: ', pv_c,' ;', pv_p)


    	

if __name__ == '__main__':
    args = passed_arguments()
    sim_cnt = args.sim_cnt
    rf = args.rf
    mat = args.mat
    spot_price = args.spot_price
    strike_price = args.strike_price
    sigma = args.sigma
    main(sim_cnt, rf, mat,spot_price,strike_price, sigma)
    