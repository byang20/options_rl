import scipy
from scipy import stats
from scipy import optimize
import numpy as np 
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
import datetime as dt
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.api import acf, pacf, graphics

'''
Function that returns the slope, y-intercept, and errors
x is a numpy array with time values
y is a numpy array with stock prices
plot is True if you would like to display a 
     graph of the points and the line of best fit

'''
def lin_reg(x,y, plot=False):
    m, b, r, _, _ = stats.linregress(x,y)
    errors = y-(m*x+b)
    if (plot):
        plt.plot(x,y)
        xline = np.array(x)  
        yline = x*m+b
        plt.scatter(x, y)  
        plt.plot(xline,yline)
        plt.show()

    return m, b, errors

'''
Function that solves for the alphas in 
error_t = alpha_1 *  + alpha_2 * + e
And shows the plot of the predicted values verses the true values
'''
def solve_alphas(errors, errors_test, lag=2, r=1):
    model_fitted = AutoReg(errors, lag).fit()
    print('The coefficients of the model are:\n %s' % model_fitted.params)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    const = model_fitted.params[0]
    alpha_1 = model_fitted.params[1]
    alpha_2 = model_fitted.params[2]

    epsilons_1 = errors_test[:-2]
    epsilons_2 =  errors_test[1:-1]

    preds = const + alpha_1*epsilons_1 + alpha_2 * epsilons_2

    diff = errors_test[lag:] - preds
    t = np.arange(0,len(errors_test), 1)
    ax1.scatter(t[lag:], errors_test[lag:], s=10, c='b', marker="s", label='diff')
    ax1.scatter(t[lag:], preds, s=10, c='r', marker="s", label='pred value')
    plt.show()

    return model_fitted


'''
Function that returns the dates and prices from a csv
'''
def sp_testdata(filename='sp500sample.csv'):
    data = pd.read_csv(filename, delimiter=',')
    data['days_from_start'] = np.busday_count(data['Date'][0],data['Date'])
    prices = data['Close'].apply(pd.to_numeric).to_numpy()
    diffs = data['days_from_start'].to_numpy()

    return diffs, prices

'''
Function to train auto regression with rolling window
'''
def rolling_win_train(data, window_size=50, lag = 2):
    mse = 0
    alphas = np.zeros(lag+1)
    predictions = np.zeros(len(data)-window_size)
    for i in range (len(data)-window_size):
        data_sub = data[i:i+window_size]
        model_fitted = AutoReg(data_sub, lag, old_names=False).fit()
        pred = model_fitted.predict(i+window_size, i+window_size)
        mse += (pred - data[i+window_size])**2
        predictions[i] = pred
        alphas += model_fitted.params
    alphas = alphas/len(predictions)

    print('Const:   ', alphas[0])
    for i in range(1,lag+1):
        print('Alpha_' + str(i) + ': ', alphas[i] )
    print('MSE: ', mse/len(predictions))
    print('MSE AVG: ', (mse/len(predictions))**.5)
    
    fig, (ax1, ax2) = plt.subplots(2)
    t = np.arange(0,len(data), 1)

    ax1.set_title('Actual vs Predicted Data')
    ax1.scatter(t[:], data[:], s=10, c='cornflowerblue', marker="s", label='actual')
    ax1.scatter(t[window_size:], predictions, s=10, c='mediumpurple', marker="s", label='pred value')
    plt.legend(loc="upper left")
    
    errors = np.append(np.zeros(window_size), data[window_size:]-predictions)
    ax2.set_title('Predicted Data Error')
    ax2.scatter(t, errors, s=10, c='mediumpurple', marker="s", label='pred value')
    plt.show()
        
    return model_fitted

def main():
    x,y = sp_testdata()
    split_idx = int(.8 * len(x))
    xtrain = x[:split_idx]
    ytrain = y[:split_idx]
    xtest = x[split_idx:]
    ytest = y[split_idx:]

    m, b, errors = lin_reg(x, y, plot=False)
    errors = errors * 1.
    model = rolling_win_train(errors,window_size = 60, lag=3)


if __name__ == "__main__":
    main()

    