import os
import numpy as np
import math
import argparse
from collections import Counter
from scipy.stats import norm


def passed_arguments():
    parser = argparse.ArgumentParser(description="Script to....")
    parser.add_argument("--sim_cnt", type=int, required=True, help="Number of rounds to simulate")
    parser.add_argument("--r", type=float, required=True, help="Stock drift rate")
    parser.add_argument("--rf", type = float, required = True, help = "Risk free rate")
    parser.add_argument("--mat", type=int, required=True, help="Maturity, T in years")
    parser.add_argument("--spot_price", type=float, required=True,help="spot price, St")
    parser.add_argument("--strike_price", type=float, required=True, help="strike price, K")
    parser.add_argument("--sigma", type=float, required=True, help="sigma")
    args = parser.parse_args()
    return args


def n_prime(x):
    return 1/math.sqrt(2*math.pi) * math.exp((-x**2)/2)

'''
Function that calculates 'the greeks'
If only_delta=True returns: (delta_c), (delta_p)
Else returns: (delta_c, gamma, vega, theta_c),(delta_p, gamma, vega, theta_p)
'''
def get_greeks(t, k, r, sigma, d1, d2, s, only_delta=False):
    cdf_d1 = norm.cdf(d1)

    if only_delta:
        delta_c = cdf_d1
        delta_p = delta_c-1
        return np.array([delta_c, delta_p])

    else:
        n_prime_d1 = n_prime(d1)
        n_prime_d2 = n_prime(d2)
        cdf_d2 = norm.cdf(d2)
        neg_cdf_d2 = norm.cdf(-1*d2)
            
        #delta calculations
        delta_c = cdf_d1
        delta_p = delta_c-1

        #gamma calculations
        gamma = n_prime_d1 / (s * sigma * math.sqrt(t))

        #vega calculations
        vega = s * n_prime_d1 * math.sqrt(t)

        #theta calculations
        theta_c = -1*((s*n_prime_d1) * sigma)/(2*(t**.5)) - r*k*math.exp(-r * t) * cdf_d2
        theta_p = -1*((s*n_prime_d1) * sigma)/(2*(t**.5)) + r*k*math.exp(-r * t) * neg_cdf_d2

        return (delta_c, gamma, vega, theta_c),(delta_p, gamma, vega, theta_p)


'''
Function that simulates one realized path
t - time to maturity
r - stock return drift
v - volatility
start - start price
'''
def sim(t, r, rf, sigma, start,days, k):

    mu = 0
    s = 1
    sigma_daily = sigma / math.sqrt(days)
    rands = np.random.normal(mu, s, t*days)
    st_s = np.zeros(t*days)
    st_s[0] = start
    deltas = np.zeros((t*days,2))

    d1 = 1/(sigma_daily*math.sqrt(t)) * (math.log(start/k) + (r/days + (sigma_daily**2)/2) * t)
    d2 = d1 - sigma_daily*(math.sqrt(t))
    deltas[0] = get_greeks(t, k, r/days, sigma_daily, d1, d2, start, only_delta=True)
    cf = np.zeros((t*days,2))
    pnl = np.zeros(2)
    cash = np.zeros(2)
    net = np.zeros(2)


    for i in range(1,days*t):
        st_s[i] = st_s[i-1] + st_s[i-1] * r / days + st_s[i-1] * sigma_daily * rands[i-1]
        term_t = t-i/days
        d1 = 1/(sigma_daily*math.sqrt(term_t)) * (math.log(st_s[i]/k) + (r/days + (sigma_daily**2)/2) * term_t)
        d2 = d1 - sigma_daily*(math.sqrt(term_t))

        deltas[i] = get_greeks(term_t, k, r/days, sigma_daily, d1, d2, st_s[i], only_delta=True)

    #cash flows
    c, p, _, _ = black_scholes_form(t, k, st_s[0], rf, sigma, days)
    d_deltas = deltas[1:,:]-deltas[:-1,:]
    cf[0] = np.array([c, p])
    cf[0][0] += -1 * deltas[0][0] * start 
    cf[0][1] += -1 * deltas[0][1] * start 
    cf[1:,0] = -1 * st_s[1:]*d_deltas[:,0]
    cf[1:,1] = -1 * st_s[1:]*d_deltas[:,1]
    #cf[-1] += k * deltas[-1]

    #pnl
    d_s = (st_s[1:] - st_s[:-1])
    pnl[0] = -1 * np.sum(d_s*deltas[1:,0])
    pnl[1] = -1 * np.sum(d_s*deltas[1:,1])

    #cash
    total_cf = np.zeros((len(cf),2))
    total_cf[0] = cf[0] 
    for i in range(1,len(cf)):
        total_cf[i] = total_cf[i-1]*math.exp(rf/days) + cf[i]

    cash = total_cf[-1]

    #net
    #print('inial option price: ', np.array([c, p]))
    
    #print('call option net comp: ', cash[0], deltas[-1][0]*st_s[-1] , -1 * max(st_s[-1] - k, 0)  )
    #print('put option net comp:  ', cash[1], deltas[-1][1]*st_s[-1] , -1 * max(k-st_s[-1] , 0)  )
    net[0] = cash[0] +  deltas[-1][0]*st_s[-1] + -1 * max(st_s[-1] - k, 0) 
    net[1] = cash[1] + deltas[-1][1]*st_s[-1]  + -1 * max(k-st_s[-1] , 0)  
    #print(deltas[-1])
    #print('net: ', net)

    stat = np.array([round(st_s[-1],2), round(net[0],2), round(net[1],2), round(c,2) ,round(p,2)]) 
    
    #string = '{:>7.2f}{:>7.2f}{:>7.2f}{:>7.2f}{:>7.2f}'.format(round(st_s[-1],2), round(net[0],2), round(net[1],2), round(c,2) ,round(p,2) )
    #print('\n',string )

    return st_s, np.sum(cf,axis=0), pnl, deltas[-1, :], cash, net, stat

def run_sim(t, runs, r, rf, sigma, start,days, K):
    prices = np.zeros((runs,t*days))
    cf = np.zeros((runs,2))
    pnl = np.zeros((runs,2))
    shares = np.zeros((runs,2))
    cash = np.zeros((runs,2))
    net = np.zeros((runs, 2))
    stats = np.zeros((runs,5))

    
    for i in range(runs):
        prices[i,:],cf[i,:], pnl[i,:], shares[i,:], cash[i:], net, stats[i,:]= sim(t, r, rf, sigma, start, days, K)

    header = '{:>7s}{:>7s}{:>7s}{:>7s}{:>7s}'.format('s_T', 'net_c','net_p', 'bs_c', 'bs_p')
    cp = os.path.abspath(os.getcwd())
    np.savetxt(os.path.join(cp, 'data.csv'), stats, delimiter=',' )
    print(header)
    #for i in range(runs):
        #string = '{:>7.2f}{:>7.2f}{:>7.2f}{:>7.2f}{:>7.2f}'.format(stats[i][0],stats[i][1],stats[i][2],stats[i][3],stats[i][4]) 
        #print('\n',string )

    value_c = np.sum(np.clip(prices[:,-1]-K, a_min=0, a_max=None))/runs
    value_p = np.sum(np.clip(K-prices[:,-1], a_min=0, a_max=None))/runs
    cf_total = np.sum(cf, axis = 0)/runs
    pnl_total = np.sum(pnl, axis = 0)/ runs
    cash_avg = np.sum(cash, axis=0)/runs
    un_cost = np.dstack((K-prices[:,-1], prices[:,-1]-K))
    
    op_price =  pnl_total + cf_total + np.sum(un_cost, axis=1)/runs
    #print(np.sum(un_cost, axis=1)/runs)

    po_c = -1*np.clip(prices[:,-1]-K, a_min=0, a_max=None)
    po_p = -1*np.clip(K-prices[:,-1], a_min=0, a_max=None)

    return value_c, value_p, cf_total, pnl_total, po_c, po_p, shares, prices[:,-1], op_price, cash_avg
    
'''
Function that returns the black scholes calculation of option price
t - maturity time , int as years
k - strice price
s_t = spot price
r - risk free rate
sigma - volatilities over t
'''
def black_scholes_form(t, k, s_t, r, sigma, days):

    d1 = 1/(sigma*math.sqrt(t)) * (math.log(s_t/k) + (r + (sigma**2)/2) * t)
    d2 = d1 - sigma*(math.sqrt(t))
    pv = k*np.exp(-r*t)

    c = norm.cdf(d1) * s_t - norm.cdf(d2) * pv
    p = -1 * norm.cdf(-d1) * s_t  + norm.cdf(-d2) * pv

    #g_c, g_p = get_greeks(t, k, r, sigma, d1, d2, s_t)
    
    return c, p, d1, d2

def get_pv_sim(value, r, t ):
    return value/((1+r)**(t))

def main(sim_cnt, r, rf, T, s_t, k, sigma):
    days = 260
    
    v_c, v_p, cf_total, pnl_total, po_c, po_p, shares, s_T, op_price, cash_avg = run_sim(T, sim_cnt, r, rf, sigma , s_t, days, k) 
    pv_c= get_pv_sim(v_c ,rf, T)
    pv_p= get_pv_sim(v_p ,rf, T)

    '''
    print('op: ', op_price)
    print('cf:  ', cf_total)
    print('cash avg: ', cash_avg)
    print('pnl: ', pnl_total)
    '''
    #print('Call Sum: ', shares[0][0]*s_T + po_c + pnl_total[0]+cf_total[0] + pv_c)
    #print('Put Sum:  ', shares[0][1]*s_T + po_p + pnl_total[1]+cf_total[1] + pv_p )

    c, p, d1, d2 = black_scholes_form(T, k, s_t, rf, sigma, days)
    
    print('                call                 put')
    print('black-scholes: ', c,  ';', p)
    print('          sim: ', pv_c,';', pv_p)
    #print('          pnl: ', pnl_total[0],';', pnl_total[1])


if __name__ == '__main__':
    args = passed_arguments()
    sim_cnt = args.sim_cnt
    r = args.r
    rf = args.rf
    mat = args.mat
    spot_price = args.spot_price
    strike_price = args.strike_price
    sigma = args.sigma
    main(sim_cnt, r, rf, mat,spot_price,strike_price, sigma)
    