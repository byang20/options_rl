import os 
import torch
import numpy as np
import math
import pickle
from options_price_sim import sim, get_greeks, black_scholes_form
import random

##Activation Functions and Their Derivatives
def leaky_relu(inputs, leaky_relu_param):
    return np.maximum(inputs*leaky_relu_param,inputs)

def relu_grad(inputs, leaky_relu_param):
    grads = inputs
    grads[grads<0] = leaky_relu_param
    grads[grads>=0] = 1
    return grads

def sigmoid(inputs):
    return 1/(1+np.exp(-inputs))

def sigmoid_grad(inputs):
    return sigmoid(inputs)*(1-sigmoid(inputs))



'''
RNN class
    Params: alpha_1 - float
            alpha_2 - float
            beta - float
    Functions: init
               forward - does a foward pass
               backwad - does a backward pass
               load_model - loads in model params from file
               save_model - saves model params into file
'''
class RNN:

    alpha_1 = 1
    alpha_2 = 1
    beta = 0.01
    leaky_relu_param = 0
    def __init__(self, input_dim=2, output_size=1, leaky_relu_param=.1):
        #init params to random number between 0 and 1
        alpha_1 = random.random()
        alpha_2 = random.random()
        beta = random.random()
        if(alpha_1==0): alpha_1=0.5
        if(alpha_2==0): alpha_2=0.5
        if(beta==0): beta=0.5
        leaky_relu_param = self.leaky_relu_param

    #inputs will be an array (a x b) representing the b S_t values for the a different paths
    '''
        inputs: stock return values; each row is a path and each col is a time step
        k: strike price
        s_t: starting pricing of the stock
        rf: risk free rate
        sigma: stock volatility
        q: dividend yield

        k, s_t, rf, sigma, and q are used to calculate the black scholes delta, which is the detla 
        used for the first rnn forward pass. 

        Returns: 
            delta_ts: the deltas calculated at each time step is of size (a,b+1) where the a is the 
            number of simulations (batchsize) and b is the number of time steps + 1 because of the 
            initial delta (bs delta)
            hidden: the ouput from (a1*delta + a2*st + b) at each time step
    '''
    def forward(self, inputs, k, s_t, rf, sigma, q = 0.0):
        a, b  = inputs.shape

        delta_ts = np.zeros((a, b+1))

        #first delta, delta_{0-1} is found using the black scholes delta
        c, _, d1, d2 = black_scholes_form(b/260, k, s_t, rf, sigma, q)
        deltas = get_greeks(b/260, k, rf, sigma, d1, d2, s_t, q, only_delta=True)
        delta_ts[:,0] = deltas[0]

        hidden = np.zeros((a,b))

        #forward passes through the rnn for all time steps
        for i in range (b):
            delta_prev = delta_ts[:,i]
            st_s = inputs[:,i]
            transform = delta_prev*self.alpha_1 + st_s*self.alpha_2 + self.beta
            activation = sigmoid(transform)
            delta_ts[:, i+1] = activation
            hidden[:,i] = transform
        return delta_ts, hidden

    '''
        inputs: stock return values; each row is a path and each col is a time step
        hidden: the output of delta_t = alpha1 * delta_{t-1} + alpha2 * S_t + beta
        deltas: goes from the init delta(bs delta) to the final delta(RNN output); are T+1 deltas 
                for each path
        sigma: either constant or matrix the size of inputs
        rv: random variable that represents X_t in S_{t+1} = S_t * drift + S_t * sigma * X_t
        lr: learning rate 
    '''
    def backward(self, stock_returns, stock_prices, hidden, deltas, sigma, rv, lr):
        a,b = stock_returns.shape
        pi_wrt_delta = stock_prices * sigma * rv #going from delta 1, ... delta T
        rels_wrt_h = sigmoid_grad(hidden)
        deltas_wrt_alpha_1 = np.zeros((a,b+1))
        pis_wrt_alpha_1 = np.zeros((a,b))

        #alpha_1 start backwards
        for i in range(0, b):
            deltas_wrt_alpha_1[:,i+1] = (deltas[:,i] + self.alpha_1*deltas_wrt_alpha_1[:,i])*rels_wrt_h[:,i]
            pis_wrt_alpha_1[:,i] = pi_wrt_delta[:, i] * rels_wrt_h[:, i] * deltas_wrt_alpha_1[:, i]

        #alpha_2 
        pis_wrt_alpha_2 = pi_wrt_delta * rels_wrt_h  * stock_returns
        
        #beta
        pis_wrt_beta =  pi_wrt_delta * rels_wrt_h  

        #print('Grad Sums: ', np.mean(pis_wrt_alpha_1), np.mean(pis_wrt_alpha_2), np.sum(pis_wrt_beta))

        #adjust params
        self.alpha_1 += lr * np.mean(pis_wrt_alpha_1)
        self.alpha_2 += lr * np.mean(pis_wrt_alpha_2)
        self.beta    += lr * np.mean(pis_wrt_beta)

        return pis_wrt_alpha_1, pis_wrt_alpha_2, pis_wrt_beta


    def load_model(self, save_path):
        params = np.load('rnn_model_params.npy')
        self.alpha_1 = params[0]
        self.alpha_2 = params[1]
        self.beta    = params[2]

    def save_model(self, save_path):
        params = np.zeros(3) 
        params[0] = self.alpha_1
        params[1] = self.alpha_2
        params[2] = self.beta
        np.save('rnn_model_params.npy',params)
		

############## Model Related ###################

'''
Function that trains one epoch
    stock_returns: daily % change in stock prices
    stock_prices: stock price values; each row is a path and each col is a time step
    batch_size: represents the number of samples to use per forward pass
    sigma: volatility of the stock
    rv: random variable, the random steps that the stock takes each day
    k: strike price
    s_t: current stock price
    rf: risk free rate
    lr: learning rate
    
    returns
        deltas: deltas at each time step 
'''
def train_epoch(model, stock_returns, stock_prices, sigma, rv, k,s_t,rf, lr =.1, batch_size=20, ):
    
    a, b = stock_prices.shape
    deltas = np.zeros((a,b+1))

    for i in range(0,a//batch_size):
        if(i*batch_size>=a): break
        end_index = (i+1)*batch_size if (i+1)*batch_size < a else a
        
        curr_delta_ts, curr_hidden = model.forward(stock_returns[i*batch_size:end_index],k,s_t,rf,sigma)
        deltas[i*batch_size:end_index, :] = curr_delta_ts

        #run backward
        model.backward(stock_returns[i*batch_size:end_index], stock_prices[i*batch_size:end_index], curr_hidden, curr_delta_ts, sigma, rv[i*batch_size:end_index], lr )
    return deltas

'''
This function runs a forward pass through the network without making adjustments

    sigma: volatility of the stock
    stock_returns: daily % change in stock prices    
    stock_prices: stock price values; each row is a path and each col is a time step
    rv: random variable, the random steps that the stock takes each day
    k: strike price
    s_t: current stock price
    rf: risk free rate
'''
def evaluation(model, sigma, stock_prices, stock_returns, rv, k, s_t, rf):
    print('\n---Evaluating the Model---')
    a, b = stock_returns.shape
    print('Time Steps:', b, "\nSim Count: ", a )
    print('Model Param Values (a1,a2,b): ', model.alpha_1, model.alpha_2, model.beta)
    #ASSIGN VARIABELS
    delta_ts, _ = model.forward(stock_returns, k, s_t, rf, sigma)

    hedge_pnl = delta_ts[:,1:]*stock_prices * rv * sigma
    pnl_at_end = np.sum(hedge_pnl, axis = 1)
    
    print('Average PnL')
    print(np.mean(pnl_at_end))
    
    variance = np.var(delta_ts[:,1:], axis=0)

    print('\nVariance Per Time Step:')
    print(variance)
    pass

'''
    number_of_epochs- int
    model- your RNN model
    prices- stock prices
    training_data / eval_data: data to use in training and in eval, should be the stock returns
    sigma: stock volatility
    rv: random variavles used in the stock price generation
    k: strike price
    s_t: starting stock price
    rf: risk free rate
    batch_size: batch size to use during training
'''
def train_and_evaluate(number_of_epochs, model, prices, training_data, eval_data, sigma, rv, k, s_t, rf, batch_size=100):
    a,b = training_data.shape
    c,d = eval_data.shape
    lr = 1
    
    print('---Training the Model---')
    for i in range (number_of_epochs):
        
        #calculating lr decay
        lr_t = lr*(1-.1)**i
        
        deltas_training = train_epoch(model, training_data, prices, sigma, rv, k, s_t, rf, lr_t, batch_size=batch_size)
        
        #calculating hedge pnl
        h_pnl_t = deltas_training[:,1:]*prices * rv * sigma
        h_pnl = np.sum(h_pnl_t, axis=1)
        h_pnl_avg = np.mean(h_pnl)
        
        #printing stats about last update
        stats_string =  '\tAverage PnL: ' + np.array2string(h_pnl_avg)
        lr_string = '\tlr: ' + str(lr_t)
        print('Epoch ' + str(i+1) + '/' + str(number_of_epochs) + stats_string  + lr_string )
    
    #evaluate the model
    evaluation(model, sigma, prices, eval_data, rv, k, s_t, rf)

    pass

'''
This function generates a simualated stock movement according to the given parameters

    sim_count: number of simulations(paths)
    start_price: starting price of stock
    steps: number of time steps to take in each path
    strike: strike price of option
    sigma: volatility of stock
    drift: drift of stock
    rf: risk-free rate
    q: dividend yield

    returns the simulated stock prices, random variable for each day, daily % returns, and price of the option
'''
def generate_data_sim(sim_count, start_price, steps, strike, sigma, drift, rf, days_per_year = 260, q = 0):

    stock_prices = np.zeros((sim_count, int(steps*days_per_year)))
    stock_returns = np.zeros((sim_count, int(steps*days_per_year)))
    rv = np.zeros((sim_count, int(steps*days_per_year)))
    pnl_true = np.zeros(sim_count)
    call_price = 0

    for i in range(sim_count):
        st_s, _, pnl, _, _, _, s, rands = sim(steps, drift, rf, sigma, start_price, days_per_year, strike, q)
        stock_prices[i,:] = st_s
        pnl_true[i] = pnl[0]
        rv[i] = rands
        call_price = s[3]

    stock_returns[:,1:] = (stock_prices[:,1:]-stock_prices[:,:-1])/stock_prices[:,:-1]

    #TODO save the simulation values
    #np.save(STOCK_RETURNS, stock_returns)
    #np.save(STOCK_PRICES , stock_prices)
    #np.save(RV, rv)

    return stock_prices, rv, pnl_true, stock_returns, call_price


def main():

    sim_count = 1000
    start_price = 100
    steps = .1
    strike = 100
    drift = .1
    rf = .01
    sigma = .1

    rnn = RNN()
    epochs = 50

    stock_prices, rv, pnl, stock_returns, call_price = generate_data_sim(sim_count, start_price, steps, strike, sigma, drift, rf )

    train_and_evaluate(epochs, rnn, stock_prices, stock_returns, stock_returns, sigma, rv, strike, start_price, rf, batch_size=200)

    print('Call Price: ', call_price)


if __name__ == "__main__":
    main()

    