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


class RNN:

    alpha_1 = 1
    alpha_2 = .001
    beta = 0.001
    leaky_relu_param = 0
    def __init__(self, input_dim=2, output_size=1, leaky_relu_param=.1):
        #init params to random number between 0 and 1
        alpha_1 = 1
        alpha_2 = .001
        beta = .001
        if(alpha_1==0): alpha_1=0.5
        if(alpha_2==0): alpha_2=0.5
        if(beta==0): beta=0.5
        leaky_relu_param = self.leaky_relu_param

    #inputs will be an array (a x b) representing the b S_t values for the a different paths
    def forward(self, inputs, k, s_t, rf, sigma, q = 0.0):
        a, b  = inputs.shape
        #first delta col holds nothing, 0, remove at the end
        delta_ts = np.zeros((a, b+1))
        c, p, d1, d2 = black_scholes_form(b/260, k, s_t, rf, sigma, q)
        deltas = get_greeks(b/260, k, rf, sigma, d1, d2, s_t, q, only_delta=True)
        delta_ts[:,0] = deltas[0]
        hidden = np.zeros((a,b))
        for i in range (b):
            delta_prev = delta_ts[:,i]
            st_s = inputs[:,i]
            transform = delta_prev*self.alpha_1 + st_s*self.alpha_2 + self.beta
            activation = sigmoid(transform)
            delta_ts[:, i+1] = activation
            hidden[:,i] = transform
        return delta_ts, hidden
    

    '''
        inputs: stock price values; each row is a path and each col is a time step
        hidden: the output of delta_t = alpha1 * delta_{t-1} + alpha2 * S_t + beta
        deltas: goes from the init delta(defined in class) to the final delta(RNN output);
                are T+1 deltas for each path
        sigma: either constant or matrix the size of inputs
        rv: random variable that represents X_t in S_{t+1} = S_t * drift + S_t * sigma * X_t
        lr: learning rate 
        
    '''
    def backward(self, inputs, hidden, deltas, sigma, rv, lr=.001, ):
        a,b = inputs.shape
        pi_wrt_delta = inputs * sigma * rv #going from delta 1, ... delta T
        rels_wrt_h = sigmoid_grad(hidden)
        deltas_wrt_alpha_1 = np.zeros((a,b+1))
        pis_wrt_alpha_1 = np.zeros((a,b))

        #alpha_1 start backwards
        for i in range(0, b):
            #fill
            deltas_wrt_alpha_1[:,i+1] = (deltas[:,i] + self.alpha_1*deltas_wrt_alpha_1[:,i])*rels_wrt_h[:,i]
            pis_wrt_alpha_1[:,i] = pi_wrt_delta[:, i] * rels_wrt_h[:, i] * deltas_wrt_alpha_1[:, i]

        #alpha_2 
        pis_wrt_alpha_2 = pi_wrt_delta * rels_wrt_h  * inputs
        
        #beta
        pis_wrt_beta = np.ones((a,b))


        print('Grad Sums: ', np.mean(pis_wrt_alpha_1), np.mean(pis_wrt_alpha_2), np.sum(pis_wrt_beta))
        #print(pis_wrt_alpha_1)

        #adjust params
        self.alpha_1 += lr * np.mean(pis_wrt_alpha_1)
        self.alpha_2 += lr * np.mean(pis_wrt_alpha_2)
        #self.beta    = self.

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
    stock_prices: stock price values; each row is a path and each col is a time step
    batch_size: represents the number of samples to use per forward pass
'''
def train_epoch(model, stock_prices, sigma, rv, k,s_t,rf, batch_size=100):
    
    a, b = stock_prices.shape
    deltas = np.zeros((a,b+1))

    for i in range(0,a//batch_size):
        if(i*batch_size>=a): break
        end_index = (i+1)*batch_size if (i+1)*batch_size < a else a
        
        #ASSIGN ALL VARIABLES

        curr_delta_ts, curr_hidden = model.forward(stock_prices[i*batch_size:end_index],k,s_t,rf,sigma)
        deltas[i*batch_size:end_index, :] = curr_delta_ts

        #run backward
        model.backward(stock_prices[i*batch_size:end_index], curr_hidden, curr_delta_ts, sigma, rv[i*batch_size:end_index], lr=.2, )
    return deltas


def evaluation(model, sigma, stock_prices, rv, k, s_t, rf):
    print('\n---Evaluating Model---')
    a, b = stock_prices.shape
    print('Time Steps:', b, "\nSim Count: ", a )
    print(model.alpha_1, model.alpha_2, model.beta)
    #ASSIGN VARIABELS
    delta_ts, _ = model.forward(stock_prices, k, s_t, rf, sigma)

    port_values = delta_ts[:,1:]*stock_prices * rv * sigma
    pnl_at_end = np.sum(port_values[:,1:] - port_values[:, :-1], axis = 1)
    print('\nAverage Final Values:')
    print(np.mean(port_values[:,-1]))
    print('Average Final PnL')
    print(np.mean(pnl_at_end))
    
    variance = np.var(delta_ts[:,1:], axis=0)

    print('Variance Per Time Step:')
    print(variance)
    pass

def train_and_evaluate(number_of_epochs, model, training_data, eval_data, sigma, rv, k, s_t, rf, batch_size=100):
    a,b = training_data.shape
    c,d = eval_data.shape

    print('---Beginning Training---')
    for i in range (number_of_epochs):
        deltas_training = train_epoch(model, training_data, sigma, rv, k, s_t, rf, batch_size=batch_size)
        port_values = deltas_training[:,1:]*training_data * rv * sigma
        pnl = np.sum(port_values[:,1:] - port_values[:,:-1], axis=1)
        pnl = np.mean(pnl)
        stats_string =  ':: Average PnL: ' + np.array2string(pnl)
        print('Epoch ' + str(i+1) + '/' + str(number_of_epochs) + stats_string )
    
    #evaluate the model
    evaluation(model, sigma, eval_data, rv, k, s_t, rf)

    pass


def generate_data_sim(sim_count, start_price, steps, strike, sigma, drift, rf, days_per_year = 260, q = 0):

    stock_prices = np.zeros((sim_count, int(steps*days_per_year)))
    rv = np.zeros((sim_count, int(steps*days_per_year)))
    pnl_true = np.zeros(sim_count)

    for i in range(sim_count):
        st_s, _, pnl, _, _, _, _, rands = sim(steps, drift, rf, sigma, start_price, days_per_year, strike, q)
        stock_prices[i,:] = st_s
        pnl_true[i] = pnl[0]
        rv[i] = rands
  
    
    return stock_prices, rv, pnl_true


def main():


    sim_count = 1000
    start_price = 100
    steps = .1
    strike = 100
    drift = .1
    rf = .01
    sigma = .1

    rnn = RNN()
    epochs = 20

    stock_prices, rv, pnl = generate_data_sim(sim_count, start_price, steps, strike, sigma, drift, rf )
    
    train_and_evaluate(epochs, rnn, stock_prices, stock_prices, sigma, rv, strike, start_price, rf, batch_size=20)

    print('Average True PnL:\n', np.mean(pnl))

    

if __name__ == "__main__":
    main()

    