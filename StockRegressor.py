import numpy as np
np.random.seed(0)

import time
import datetime
from calendar import monthrange
import pandas as pd
import requests
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

import scipy.optimize as spo
import glob
from numpy import newaxis

import sys

from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared
from sklearn.kernel_ridge import KernelRidge

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import warnings

from IPython.display import display 
from IPython.display import clear_output


sb.set()
warnings.filterwarnings("ignore")
yf.pdr_override()




def preprocess_data(dfvar):
    dfvar.fillna(method='ffill', inplace=True)
    dfvar.fillna(method='bfill', inplace=True)
    return dfvar
    
    
def get_daily_returns(dfvar, tickers=None):
    
    if tickers is None:
        tickers = dfvar.columns
    
    return preprocess_data(dfvar[tickers].pct_change(periods=1)*100)



   
    
class StockRegressor(object):
    
    def __init__(self, ticker, dates = ['2015-01-01', '2017-01-01'], 
                                 n_days_to_read_ahead = 150, n_days_to_predict = 50, verbose = True):
        
        #'2015-01-01', '2017-07-28'
        
        self.ticker = ticker
        self.n_days_to_read_ahead = n_days_to_read_ahead

        self.training_start_date =  dates[0] 
        self.training_end_date = dates[1]        
        
        # n_days_to_read_ahead roughly corresponds to 6 months of data after the training window, which includes
        # the testing window. The extra data beyond the testing window is only needed for plotting purposes,
        # and is not part of any calculations, testing or training; graphs look nicer this way .. 
        d = datetime.datetime.strptime(dates[1], "%Y-%m-%d") + datetime.timedelta(days=n_days_to_read_ahead)
        
        cutoff = datetime.datetime.today() - datetime.timedelta(2)
        
        if d >  cutoff:
            d = cutoff
        
        if datetime.datetime.strptime(self.training_end_date, "%Y-%m-%d") + \
                            datetime.timedelta(n_days_to_predict) > cutoff:
            
            self.training_end_date = (cutoff - datetime.timedelta(2 * n_days_to_predict)).strftime("%Y-%m-%d")
        
        self.final_end_date = d.strftime("%Y-%m-%d")
        
        if verbose == True:
            print("Getting pricing information for {} for the period {} to {}".format(ticker, 
                                                        self.training_start_date, self.final_end_date))
        
        csv_file_path = 'Stock-{}-{}-{}.csv'.format(self.ticker, self.training_start_date, self.final_end_date)
        
        files_present = glob.glob(csv_file_path)
        

        if not files_present:
            existing_filename = ""
            file_names = [item for item in sorted(glob.glob("Stock-{}-*.csv".format(ticker)))]

            if len(file_names) > 0:
                for name in file_names:
                    f_name = name.replace("Stock-{}-".format(ticker), "").replace(".csv", "")
                    start = f_name[:10]
                    end = f_name[11:]

                    start_dt = datetime.datetime.strptime(start, "%Y-%m-%d") 
                    end_dt = datetime.datetime.strptime(end, "%Y-%m-%d")

                    req_start_dt = datetime.datetime.strptime(self.training_start_date, "%Y-%m-%d") 
                    req_end_dt = datetime.datetime.strptime(self.final_end_date, "%Y-%m-%d") 
                    
                    if req_start_dt >= start_dt and req_end_dt <= end_dt:
                            existing_filename = name

            if existing_filename != "":
                if verbose == True:
                    print("Found a pricing file with wide range of dates, reading ... {} ".format(existing_filename))
                self.pricing_info = pd.read_csv(existing_filename, index_col=0)   
                self.pricing_info = self.pricing_info[self.training_start_date:self.final_end_date]
            else:            
                if verbose == True:
                    print('Pricing file for stock doesnt exist. Downloading prices.')
                t1 = time.time()

                for i in range(30):
                    
                    self.pricing_info = pdr.get_data_yahoo(ticker, start=self.training_start_date, 
                                                                end=self.final_end_date, as_panel = False)    

                    #self.pricing_info = pdr.DataReader(ticker, 'yahoo', self.training_start_date, 
                    #                                                               self.final_end_date)

                    if len(self.pricing_info.columns) > 0:
                        if verbose == True:
                            print("\nYahoo Finance web service didnt return any data.")
                            print("Please wait. Retrying ....")
                        break
                    time.sleep(5)


                if len(self.pricing_info.columns) == 0:
                    if verbose == True:
                        print("\nSomething went wrong with the Yahoo web service, please try again in a few minutes.")
                        print("Pricing info NOT downloaded!!")
                    assert(False)
                    return 

                preprocess_data(self.pricing_info)
                t2 = time.time()  
                
                if verbose == True:
                    print("Took {:.2f} seconds to load.".format(t2-t1))

                self.pricing_info.to_csv(csv_file_path)
        else:
            if verbose == True:
                print('Pricing file for chosen stocks already exists!')
            self.pricing_info = pd.read_csv(csv_file_path, index_col=0)   
        
        #print(self.pricing_info[0:2])
        
        self.pricing_info['dates'] = pd.to_datetime(self.pricing_info.index)
        self.pricing_info['timeline'] = [x for x in range(self.pricing_info.shape[0])]
        self.pricing_info.index = self.pricing_info['timeline']
        self.sample_size = self.pricing_info.shape[0]
        
        self.build_learning_data_frame(n_days_to_predict = n_days_to_predict)
        
    
    def plot_learning_data_frame(self, start_date=None, end_date=None):  
        
        start_date = self.get_best_date(start_date, side='start')
        end_date  = self.get_best_date( end_date, side='end')
        
        self.plotting_learning_df = self.learning_df[start_date:end_date].copy()
        self.plotting_learning_df.index = self.plotting_learning_df['dates']
        
        if 'Volume' in self.plotting_learning_df.columns:
            self.plotting_learning_df = self.plotting_learning_df.drop(labels=['Volume'], axis=1)
        
        self.plotting_learning_df = self.plotting_learning_df.drop(labels=['timeline', 'Rolling Mean-60'], 
                                                                   axis=1)
        
        
        plt.rcParams["figure.figsize"] = (15,8)
        self.plotting_learning_df.plot()
        plt.axvline(self.training_end_date, color='r', linestyle='dashed', linewidth=2)

        plt.show()

        
    def plot_bollinger_bands(self, window=20, start_date=None, end_date=None):

        start_date = self.get_best_date(start_date, side='start')
        end_date  = self.get_best_date(end_date, side='end')

        self.plotting_boll_df = self.adj_close_price[start_date:end_date].copy()
        self.plotting_boll_df = self.plotting_boll_df.drop(labels=['timeline'], axis=1)
        self.plotting_boll_df.index = self.plotting_boll_df['dates']
        
        self.plotting_boll_df['Mean'] = self.plotting_boll_df['Adj Close'].rolling(center=False, 
                                                                                   window=window).mean()    
        self.plotting_boll_df['Upper'] = self.plotting_boll_df['Mean'] \
                    + self.plotting_boll_df['Adj Close'].rolling(center=False, window=window).std()*2
            
        self.plotting_boll_df['Lower'] = self.plotting_boll_df['Mean'] \
                    - self.plotting_boll_df['Adj Close'].rolling(center=False, window=window).std()*2
            
        preprocess_data(self.plotting_boll_df)
        self.plotting_boll_df.plot()
        plt.show()
        
    
    def find_timeline_for_date(self, date):
        return self.get_best_date(date)
    '''
        date_arr = np.where(self.learning_df["dates"]==date)[0]
        
        if date_arr.size == 0:
            return self.learning_df["dates"][0]
        else:
            return date_arr[0]
    '''
    
    def find_date_for_timeline(self, timeline):
        return self.pricing_info.index[timeline]
    
    
    def get_best_date(self, date_val, side='start'):
    
        if date_val is None and side == 'start':
            date_val = self.adj_close_price.index[0]
        elif date_val is None and side == 'end':        
            date_val = self.adj_close_price.index[-1]
        else:
            i = self.adj_close_price["dates"].searchsorted(\
                                            datetime.datetime.strptime(date_val, "%Y-%m-%d"))[0]
            
            if i >= self.adj_close_price.index.shape[0]:
                i = -1
            date_val = self.adj_close_price.index[i]

        return date_val

    
    def build_learning_data_frame(self, training_start_index = None, training_end_index=None, 
                                  n_days_to_predict = 50, keep_indexes = False, verbose = False):
        
        if not hasattr(self, 'n_days_to_predict'):
            self.n_days_to_predict = n_days_to_predict
        
        self.X_train_reg_fitted = {}
        self.X_test_reg_fitted = {}
        self.regression_params = {}
        self.fft_y_train = []
        self.reg_pred = {}
        self.reg_models = {}
        
        if  keep_indexes == False:
            self.training_end_index = -1
            self.training_start_index = -1
        
        if training_start_index is None and training_end_index is None:
            self.adj_close_price = self.pricing_info.copy()
        elif training_start_index is None:
            self.training_end_index = training_end_index
            self.adj_close_price = self.pricing_info[:self.training_end_index].copy()
        elif training_end_index is None:
            self.training_start_index = training_start_index
            self.adj_close_price = self.pricing_info[self.training_start_index:].copy()
        else:
            self.training_end_index = training_end_index
            self.training_start_index = training_start_index
            self.adj_close_price = self.pricing_info[self.training_start_index: \
                                                     self.training_end_index + self.n_days_to_predict].copy()
            
            self.training_end_index = self.training_end_index - self.training_start_index
            self.val_end_index = self.training_end_index + n_days_to_predict

        self.adj_close_price = self.adj_close_price.drop(labels=['Open', 'High', 'Low', 'Close', 'Volume'], 
                                                         axis=1)
        
        self.learning_df = self.adj_close_price.copy()                
        self.learning_df['timeline'] = [x for x in range(self.learning_df.shape[0])]
        
        self.learning_df.index = self.learning_df['timeline']
        preprocess_data(self.learning_df)
     
        
        if self.training_end_index == -1:
            self.training_end_index = self.find_timeline_for_date(self.training_end_date)       

        self.testing_end_index = self.training_end_index + self.n_days_to_predict
        self.testing_end_date = self.find_date_for_timeline(self.testing_end_index)
        
        if False:
            print("self.training_start_date {}".format(self.training_start_date))
            print("self.training_start_index {}".format(self.training_start_index))
            print("self.final_end_date {}".format(self.final_end_date))
            print("self.training_end_index {}".format(self.training_end_index))
            print("self.training_end_date {}".format(self.training_end_date))
            print("self.testing_end_index {}".format(self.testing_end_index))
            print("self.testing_end_date {}".format(self.testing_end_date))
        
        
        if verbose == True:
            print("Training end date is {}, corresponding to the {}th sample".format(self.training_end_date, \
                                                                                     self.training_end_index))
                  
            print("The data has {} training samples and {} testing samples with a total of {} samples" \
                  .format(self.training_end_index, self.learning_df.shape[0] - self.training_end_index, 
                                                                          self.learning_df.shape[0]))
        
        self.learning_df['Rolling Mean-60'] = \
                        self.learning_df['Adj Close'][:self.training_end_index] \
                                                .rolling(center=False, window=60).mean()
                
        self.learning_df['Rolling Mean-60'].fillna(method='bfill', inplace=True)
        
        self.X_train = self.learning_df['timeline'][:self.training_end_index]
        self.y_train = self.learning_df['Adj Close'][:self.training_end_index]

        self.X_test = self.learning_df['timeline'][self.training_end_index:self.testing_end_index]
        self.y_test = self.learning_df['Adj Close'][self.training_end_index:self.testing_end_index]

        if (verbose == True):
            print("Training set has {} samples.".format(self.X_train.shape[0]))
            print("Testing set has {} samples.".format(self.X_test.shape[0]))
        
        
    def trainRegression(self, poly_degree = 1, verbose = True):
        
        reg_col_name = 'Linear Regression Order {}'.format(poly_degree)        
        p_ind = poly_degree
        
        if not reg_col_name in self.learning_df.columns:

            self.regression_model = linear_model.LinearRegression()
            
            
            self.regression_poly = PolynomialFeatures(degree = poly_degree)
            self.X_train_reg_fitted[p_ind] = self.regression_poly.\
                        fit_transform(np.array(self.X_train).reshape(-1, 1))
            self.X_test_reg_fitted[p_ind] = self.regression_poly.fit_transform(np.array(self.X_test)\
                                                                                   .reshape(-1, 1))

            self.regression_model.fit(self.X_train_reg_fitted[p_ind], self.y_train)
            
            if verbose == True:
                print("Regression Model Coefficients of Poly degree {}: {}".format(poly_degree, 
                                                                           self.regression_model.coef_))
                print("Regression Model Intercept of Poly degree {}: {}".format(poly_degree, 
                                                                    self.regression_model.intercept_))

            self.regression_params[p_ind] = (self.regression_model.coef_, self.regression_model.intercept_)
            
            self.X_reg_fitted = np.concatenate((self.X_train_reg_fitted[p_ind], self.X_test_reg_fitted[p_ind]), 
                                                                       axis=0)

            self.learning_df[reg_col_name] = np.nan
            self.learning_df[reg_col_name][:self.testing_end_index] = \
                                                        self.regression_model.predict(self.X_reg_fitted)
                
            self.reg_models[poly_degree] = self.regression_model

        self.reg_pred[poly_degree] = self.learning_df[reg_col_name][:self.testing_end_index]
        
        
        
    def train(self, training_start_index = None, training_end_index=None, 
                             poly_degree = [1, 2, 3], num_harmonics = 4, underlying_trend_poly = 3,
                             days_for_regression = 15, n_days_to_predict = 50, 
                             momentum_split = 0.25, keep_indexes = False, no_FFT = False, verbose = True):
        
        
        self.n_days_to_predict = n_days_to_predict
        
        self.build_learning_data_frame(training_start_index = training_start_index, 
                                       training_end_index = training_end_index, 
                                       n_days_to_predict = n_days_to_predict, 
                                       keep_indexes = keep_indexes,
                                       verbose=verbose)
         
        for i in poly_degree:
            self.trainRegression(poly_degree = i, verbose=verbose)
        
        self.trainMomentum(days_for_regression=days_for_regression, verbose = verbose )
        self.trainAverageMomentum(momentum_split=momentum_split, verbose = verbose )        
        
        if no_FFT == False:
            self.trainFFT(num_harmonics=num_harmonics, underlying_trend_poly=underlying_trend_poly, verbose=verbose)
        
        

    def plotPrediction(self):
        print("Model Coefficients: {}".format(self.model.coef_))
        print("Model Intercept: {}".format(self.model.intercept_))

        print("Next Day Price: {} and Prediction: {}".format(self.y_test.iloc[0], \
                                                             self.model.predict([self.X_test_fitted[0]])) )
        print("Accuracy: {}".format(float(self.y_test.iloc[0])/ \
                                    float(self.model.predict([self.X_test_fitted[0]]))))

        self.train_predictions = self.y_train[-20:].copy()
        self.train_predictions['Predictions'] = self.predictions_train_tot
        plt.rcParams["figure.figsize"] = (15,8)
        self.train_predictions.plot()
        plt.show()



    def get_fft_residual_pricing(self, underlying_trend_poly = 3):
        
        self.fft_rolling_mean60 = preprocess_data(self.learning_df['Rolling Mean-60'] \
                                                                  [:self.training_end_index].copy())
        
        #self.fft_y_train = np.array(self.y_train - self.fft_rolling_mean60).flatten()
        self.fft_y_train = np.array(self.y_train - self.reg_pred[underlying_trend_poly]\
                                                            [:self.training_end_index]).flatten()
        self.fft_x_train = np.array(self.X_train).flatten()
               
        self.fft_polyfit = np.polyfit(self.fft_x_train, self.fft_y_train, 1)        
        self.fft_residual_pricing = self.fft_y_train - self.fft_polyfit[0] * self.fft_x_train
        
    
    
    def trainMomentum(self, days_for_regression, verbose ):
       
        self.days_for_regression = days_for_regression
        momentum_polyfit = np.polyfit(self.X_train[-self.days_for_regression:], 
                                                      self.y_train[-self.days_for_regression:], 1)
        momentum_trend_x = np.concatenate((self.X_train[self.training_end_index-self.days_for_regression:],
                                                                                      self.X_test), axis=0)
        
        self.momentum_linear_reg_o1 = momentum_polyfit[1] + momentum_polyfit[0] * momentum_trend_x
        
        self.learning_df['Momentum'] = np.nan
        self.learning_df['Momentum'][self.training_end_index-self.days_for_regression: \
                                                        self.testing_end_index] = self.momentum_linear_reg_o1

        
    def trainAverageMomentum(self, momentum_split, verbose ):
        
        polys = len(self.reg_pred.keys())
        self.reg_average = np.zeros(self.testing_end_index)
        
        for poly, reg in self.reg_pred.items():
            self.reg_average += reg[:self.testing_end_index] / polys
            
        self.reg_mom_trend = np.concatenate((self.reg_average[:self.training_end_index-self.days_for_regression], 
                                             self.reg_average[self.training_end_index-self.days_for_regression: \
                                                             self.testing_end_index] * (1-momentum_split) + \
                                                             momentum_split * self.momentum_linear_reg_o1)
                                             , axis=0)
        

        self.learning_df['Prediction Reg/Momentum'] = np.nan
        self.learning_df['Prediction Reg/Momentum'][:self.testing_end_index] = self.reg_mom_trend
        
        
    
    def trainFFT(self, num_harmonics, underlying_trend_poly, verbose):
        
        self.fft_underlying_trend_poly = underlying_trend_poly
        
        if len(self.fft_y_train) == 0:
            self.get_fft_residual_pricing(underlying_trend_poly)
            
        self.fft_pricing_in_freq_domain = np.fft.fft(self.fft_y_train) 
        
        self.fft_frequencies = np.fft.fftfreq(self.training_end_index)
        
        highest_freq_amp_indexes = list(range(self.training_end_index))
        highest_freq_amp_indexes.sort(key = lambda x: np.absolute(self.fft_pricing_in_freq_domain[x]), 
                                                                              reverse=True)
        fft_x = np.arange(0, self.testing_end_index)
        
        self.fft_df = self.adj_close_price.copy()
        cols = self.fft_df.columns
        
        self.learning_df['Prediction w/FFT'] = np.nan
        self.fft_df['FFT Waveform'] = np.nan        
        
        self.learning_df['Prediction w/FFT'][:self.testing_end_index] = np.zeros(self.testing_end_index)
        self.fft_df['FFT Waveform'][:self.testing_end_index] = np.zeros(self.testing_end_index)
        
        self.fft_df = self.fft_df.drop(cols, axis=1)
        
        for freq_i in highest_freq_amp_indexes[:1 + num_harmonics * 2]:
            fft_power = np.absolute(self.fft_pricing_in_freq_domain[freq_i]) / self.training_end_index  
            fft_phase = np.angle(self.fft_pricing_in_freq_domain[freq_i])          
            self.fft_df['FFT Waveform'][:self.testing_end_index] += \
                        fft_power *  np.cos(2 * np.pi * self.fft_frequencies[freq_i] * fft_x + fft_phase)
        
        self.fft_trend = self.reg_mom_trend
        self.learning_df['Prediction w/FFT'][:self.testing_end_index] += self.fft_trend + \
                                                        self.fft_df['FFT Waveform'][:self.testing_end_index]
            
        
        self.fft_pred = self.learning_df['Prediction w/FFT'][self.training_end_index:self.testing_end_index]
        
    
    def score_regression(self, training_only=True, verbose=True): 
        
        self.reg_score_train = {}
        self.reg_score_test = {}
        
        for poly, reg in self.reg_pred.items():
            self.reg_score_train[poly] = r2_score(self.y_train, reg[:self.training_end_index])
            self.reg_score_test[poly] = r2_score(self.y_test, reg[self.training_end_index:])
            
            if verbose == True:
                print("R^2 Score of Linear Regression of Poly order {} Training: {:.2f}".format(poly,
                                                                                self.reg_score_train[poly]))
                print("R^2 Score of Linear Regression of Poly order {} Testing: {:.2f}".format(poly, 
                                                                               self.reg_score_test[poly]))
                
        return (self.reg_score_train, self.reg_score_test)
        
    def score_fft(self, verbose=True):    
        
        self.score_fft_train = r2_score(self.y_train, 
                           self.learning_df['Prediction w/FFT'][:self.training_end_index])
        self.score_fft_test = r2_score(self.y_test,
                   self.learning_df['Prediction w/FFT'][self.training_end_index:self.testing_end_index])
        
        self.score_regmom_train = r2_score(self.y_train, 
                           self.learning_df['Prediction Reg/Momentum'][:self.training_end_index])
        self.score_regmom_test = r2_score(self.y_test,
                   self.learning_df['Prediction Reg/Momentum'][self.training_end_index:self.testing_end_index])
        
        if verbose == True:
            print("R^2 Score of Reg/Momentum Training: {:.2f}".format(self.score_regmom_train))
            print("R^2 Score of Reg/Momentum Testing: {:.2f}".format(self.score_regmom_test))
            print("R^2 Score of FFT Training: {:.2f}".format(self.score_fft_train))
            print("R^2 Score of FFT Testing: {:.2f}".format(self.score_fft_test))
        
        return (self.score_fft_train, self.score_fft_test, self.score_regmom_train, self.score_regmom_test)

    
    def score_verbose(self): 
        self.score_regression(verbose = True)
        self.score_fft(verbose = True)
        
    def score(self, verbose = False): 
        if verbose == True:
            print("\n--------------------------------------------------------------")
        
        reg = self.score_regression(verbose = verbose)
        fft = self.score_fft(verbose = verbose)
        
        if verbose == True:
            print("--------------------------------------------------------------\n")
            
        return (reg, fft)


    def prepare_RNN_frame(self, start_index, end_index, nn_input_window_length):
        
        nn_window_frame = pd.DataFrame()
        
        stock_pricing_series = self.adj_close_price['Adj Close'][start_index:end_index].copy()
        preprocess_data(stock_pricing_series)
        
        for i in range(len(stock_pricing_series) - nn_input_window_length):
            data_window = stock_pricing_series[:nn_input_window_length].transpose()
            data_window /= data_window.iloc[0] 
            data_window -= 1
            nn_window_frame = nn_window_frame.append(data_window)
            stock_pricing_series = stock_pricing_series.shift(-1)
        
        nn_window_array = np.array(nn_window_frame)
        y_array = nn_window_array[:, -1]
        nn_window_expanded = np.expand_dims(nn_window_frame, axis=2)
        
        return (nn_window_expanded, y_array)
    


     
    def predictRNN(self, start_date = None, end_date = None, plot_prediction = False, 
                   training = False, this_or_next_window = False):
        
        if start_date is None:
            start_date = self.training_end_index - self.n_days_to_predict
        
        self.y_RNN = pd.DataFrame(self.adj_close_price[start_date:end_date])
        self.y_RNN = self.y_RNN.drop(labels =['timeline'], axis=1)
        self.y_RNN['RNN Prediction'] = np.nan

        pred_windows = -2 + round(len(self.y_RNN) / (self.n_days_to_predict))
        
        if training == True:
            self.window_array_expanded = self.nn_input_window_expanded
        else:
            self.window_array_expanded = self.nn_testing_window_expanded
        
        for window_counter in range(pred_windows):
            predicted_window = []
            rnn_data_window = np.array(self.window_array_expanded[window_counter*self.n_days_to_predict:1 + \
                                                              window_counter*self.n_days_to_predict, :, :])[0]

            for counter in range(self.n_days_to_predict):
                predicted_window.append(self.dl_model.predict(rnn_data_window[newaxis,:,:])[0,0])
                rnn_data_window = rnn_data_window[1:]
                rnn_data_window = np.insert(rnn_data_window, self.n_days_to_predict-1, 
                                                                    predicted_window[-1], axis=0)
            if this_or_next_window == True:
                padding = 0
            else:
                padding = self.n_days_to_predict
                
            self.y_RNN['RNN Prediction'][window_counter*self.n_days_to_predict + padding: \
                                              window_counter*self.n_days_to_predict+self.n_days_to_predict \
                                                    + padding]\
                                                = np.array((rnn_data_window + 1)*self.y_RNN['Adj Close']\
                                                   .iloc[window_counter*self.n_days_to_predict]).flatten()
        
        if plot_prediction == True:
            plt.rcParams["figure.figsize"] = (15,8)
            self.y_RNN.index = self.y_RNN['dates']
            self.y_RNN.plot()
            plt.axvline(self.training_end_date, color='r', linestyle='dashed', linewidth=2)
            plt.show()
        else:
        
            return  self.y_RNN['RNN Prediction'].copy(), padding, \
                                                    self.n_days_to_predict*pred_windows + this_or_next_window
            
            
    def score_RNN(self, verbose=True):    
        
        train_pred, start, end = self.predictRNN(start_date = 0, end_date = self.training_end_index, 
                                                                                             training = True)
        self.score_RNN_train = r2_score(self.y_train[start:end], train_pred[start:end])
        
        test_pred, start, end = self.predictRNN(start_date = self.training_end_index)
        self.score_RNN_test = r2_score(self.y_test, test_pred[start:start + self.n_days_to_predict])

        if verbose == True:
            print("\n--------------------------------------------------------------------")
            print("R^2 Score of RNN Training: {}".format(self.score_RNN_train))
            print("R^2 Score of RNN Testing: {}\n".format(self.score_RNN_test))
        
        return (self.score_RNN_train, self.score_RNN_test)
    
    
    
    # Reference: https://github.com/vsmolyakov/experiments_with_python/blob/master/chp04/keras_lstm_series.ipynb
    # Reference: https://github.com/llSourcell/How-to-Predict-Stock-Prices-Easily-Demo/blob/master/stockdemo.ipynb
    def trainRNN(self):
        
        self.build_learning_data_frame()
        self.nn_input_window_expanded, self.nn_y_train \
                        = self.prepare_RNN_frame(0, self.training_end_index, self.n_days_to_predict)
      
        self.nn_testing_window_expanded, self.nn_y_test \
            = self.prepare_RNN_frame(self.training_end_index, self.learning_df.shape[0], self.n_days_to_predict)
            
   
        self.dl_model = Sequential()
        self.dl_model.add(LSTM(input_dim=1, output_dim=50, return_sequences=True))
        self.dl_model.add(Dropout(0.2))
        
        self.dl_model.add(LSTM(return_sequences=False, units=100))
        self.dl_model.add(Dropout(0.2))
        
        self.dl_model.add(Dense(units=1))
        self.dl_model.add(Activation('linear'))

        self.dl_model.compile(loss='mse', optimizer='rmsprop')

        '''
        self.X_train_nn, self.X_test_nn, self.y_train_nn, self.y_test_nn = train_test_split(self.norm_prices, 
                                                                                            self.adj_close_price, 
                                                                                            test_size = 0.2, 
                                                                                            shuffle = False,
                                                                                            random_state = 0)
        '''
        
        self.dl_model.summary()
        
        self.dl_model.fit(self.nn_input_window_expanded,
                 np.array(self.nn_y_train ) , batch_size = 512, epochs=5, validation_split=0.1)

    def get_best_trading_day(self, date_str, delta):
        day = datetime.datetime.strptime(date_str, "%Y-%m-%d") + datetime.timedelta(days=delta)
        day = day.strftime("%Y-%m-%d")
        return self.pricing_info["dates"].iloc[self.get_best_date(day)]

    
    
    def predict(self, verbose = True):

        reg_col_pr_list = []
        reg_col_pct_list = ['Reg/Mom Pct Var %', 'FFT Pct Var %']
        
        for key, reg in self.reg_pred.items():
            reg_value = 'Reg{} Pred'.format(key)
            reg_pct_var = 'Reg{} Pct Var %'.format(key)
            reg_col_pr_list.append(reg_value)
            reg_col_pct_list.append(reg_pct_var)
            
        pred_columns = ['Day', 'Index', 'Date', 'Adj Close', 'Reg/Mom Pred', 'FFT Prediction']
        
        pred_columns.extend(reg_col_pr_list)
        pred_columns.extend(reg_col_pct_list)
        
        self.prediction_frame = pd.DataFrame(columns=pred_columns)
        pd.options.display.float_format = '{:,.2f}'.format
        
        days_to_pred = np.array([1, 8, 15, 22, 29, 36, 43, 50], int)

        for i, day in np.ndenumerate(days_to_pred):
            p_date = self.get_best_trading_day(self.training_end_date, int(day))
            p_index = self.get_best_date(p_date.strftime("%Y-%m-%d"))
            p_real_price = self.learning_df['Adj Close'].iloc[p_index]
            
            p_rm_price = self.learning_df['Prediction Reg/Momentum'].iloc[p_index]            
            p_rm_pct_var = (p_rm_price / p_real_price - 1) * 100

            p_fft_price = self.learning_df['Prediction w/FFT'].iloc[p_index]            
            p_fft_pct_var = (p_fft_price / p_real_price - 1) * 100
            
            vals = [day, p_index, p_date.strftime("%Y-%m-%d"), p_real_price, p_rm_price, p_fft_price]
            
            reg_pr_list = []
            reg_pct_list = [p_rm_pct_var, p_fft_pct_var]
        
            for key, reg in self.reg_pred.items():
                p_reg_price = reg[p_index]
                reg_pr_list.append(p_reg_price)
                p_reg_pct_var = (p_reg_price / p_real_price - 1) * 100
                reg_pct_list.append(p_reg_pct_var)                
            
            vals.extend(reg_pr_list)
            vals.extend(reg_pct_list)
            
            self.prediction_frame.loc[i] = vals

            if verbose == True and i == (0,):
                print("\nTraining End date: {}".format(self.training_end_date))
                print("First Day of Prediction: {}".format(p_date.strftime("%Y-%m-%d")))
        
        
        self.prediction_frame.style.apply(lambda x: ['background: lightblue' if  i == 'Prediction Reg/Momentum'\
                                                         else '' for i,_ in x.iteritems()], axis = 1)

        if verbose == True:
            display(self.prediction_frame)

            print("\nMean Regression/Momentum Prediction Percent Variation: +/- {:.2f}%".format(\
                                            np.mean(np.absolute(self.prediction_frame['Reg/Mom Pct Var %']))))

            print("Mean FFT Prediction Percent Variation: +/- {:.2f}%".format(\
                                            np.mean(np.absolute(self.prediction_frame['FFT Pct Var %']))))
        reg_pct_means = []
        
        for key, reg in self.reg_pred.items():
            reg_pct_means.append(np.mean(np.absolute(self.prediction_frame['Reg{} Pct Var %'.format(key)])))
            if verbose == True:
                print("Mean Regression Order {} Prediction Percent Variation: +/- {:.2f}%".format(key, \
                                np.mean(np.absolute(self.prediction_frame['Reg{} Pct Var %'.format(key)]))))  
            


        
class StockGridSearch(object):
    
    def __init__(self, ticker = 'GOOG', dates = ['2015-01-01', '2016-01-30'], training_delta_months = 24):

        self.ticker = ticker
        self.dates = dates

        self.date0 = datetime.datetime.strptime(dates[0], "%Y-%m-%d") 
        self.date1 = datetime.datetime.strptime(dates[1], "%Y-%m-%d") 
        
        self.requested_training_delta = self.date1 - self.date0
        
        self.training_delta_months = training_delta_months
        self.training_delta = datetime.timedelta(self.training_delta_months*30)
        
        # begin training roughly 24 months earlier to find the best hyper-parameters
        self.moving_window_start = self.date0 - self.training_delta
        self.moving_window_end = self.moving_window_start + self.requested_training_delta 
        
        self.moving_window_start_str = self.moving_window_start.strftime("%Y-%m-%d")
        self.moving_window_end_str = self.moving_window_end.strftime("%Y-%m-%d")
        
        self.stock = StockRegressor(ticker, ['1999-01-01', '2017-09-03'])
        
        
        self.best_r2_score = -1000
        self.best_r2_combination = (0 , 0 , 0)
        
    
    def get_moving_window_dates(self, delta_days):
        self.strt = (self.moving_window_start + datetime.timedelta(delta_days)).strftime("%Y-%m-%d")
        self.end = (self.moving_window_end + datetime.timedelta(delta_days)).strftime("%Y-%m-%d")
        return [self.strt, self.end]
        
    
    def calculate_index_windows(self, training_window_size = None, n_days_to_predict = 50):
        total_samples = self.stock.sample_size
        
        if training_window_size is None:
            self.training_window =  self.stock.training_end_index - n_days_to_predict
            self.validation_window = n_days_to_predict
            self.testing_window = n_days_to_predict
            self.total_iterations = 1
        
        else:
            # roughly corresponds to one year of historical data + 50 days testing window
            if self.stock.training_end_index - training_window_size  < 0: 
                print("\n--------------------------------------------------------------")
                print("Error: Training period is too short. Training dates are too close. Small sample size!!")
                print("It's less than the requested {} samples for training, and {} days of forecasting" \
                                                      .format(training_window_size, n_days_to_predict)) 
                print("It's recommended to have at least 1 year of historical data")
                assert False, "Training dates are too close. Small sample size!!"
         

        print("\nTotal Iterations {}".format(self.total_iterations))
    
    
    
    def train(self, training_window_size = None, 
                    n_days_to_predict = 50, 
                    num_harmonics = [3, 6, 15, 40],
                    days_for_regression = [5, 15, 25, 60],
                    poly_degree = [1, 2, 3, 4],
                    underlying_trend_poly = [2, 3], 
                    momentum_split = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]):
        
        
        self.t_training_window_size = training_window_size
        self.t_n_days_to_predict = n_days_to_predict
        self.t_num_harmonics = num_harmonics
        self.t_days_for_regression = days_for_regression
        self.t_poly_degree  = poly_degree 
        self.t_underlying_trend_poly = underlying_trend_poly
        self.t_momentum_split = momentum_split
        
        self.max_combinations = len(num_harmonics) * len(days_for_regression) \
                            * len(underlying_trend_poly) * len(momentum_split) * self.training_delta_months
            
        self.combination = 0
        self.fft_scores = {}
        self.reg_mom_scores = {}
        self.reg_scores = {}
        self.combination_scores = []  
                
        t1 = time.time()  
        
        for i in range(self.training_delta_months):
            
            self.stock = StockRegressor(self.ticker, self.get_moving_window_dates(i * 30))
            self.training_month = i

            self.calculate_index_windows(training_window_size, n_days_to_predict)

            for hrmnc in num_harmonics:
                for days_reg in days_for_regression:
                    for u_trend in underlying_trend_poly:
                        for mom_sp in momentum_split:                  
                            self.train_stock( num_harmonics = hrmnc,
                                              days_for_regression = days_reg,
                                              poly_degree = poly_degree,
                                              underlying_trend_poly = u_trend, 
                                              momentum_split = mom_sp)

                            self.combination += 1 

        t2 = time.time()     
        
        r2_means = {}
        r2_means['regmom'] = np.mean(list(self.reg_mom_scores.values()))
        r2_means['fft'] = np.mean(list(self.fft_scores.values()))
        
        print("\n\nModel took {:.2f} seconds to train.".format(t2-t1))        
        print("\nAll mean R^2 score results are:")
        print("Regression with Momentum: {:.2f}".format(r2_means['regmom']))
        print("FFT: {:.2f}".format(r2_means['fft']))
        
        '''
        for poly in self.t_poly_degree:
            r2_means['reg{}'.format(poly)] = np.mean(list(self.reg_scores[poly].values()))
            print("Regression of Order {}: {:.2f}".format(poly, r2_means['reg{}'.format(poly)]))
        '''
        
        best_class = max(r2_means, key=r2_means.get)        
        
        print("\n--------------------------------------------------------------------")
        
        if best_class == 'fft':            
            best_key = max(self.fft_scores, key=self.fft_scores.get)       
            print("Best Method of Estimation is Fast Fourier transform of {} harmonics with underlying \
Regression of Order".format(best_key[0], best_key[2]))
            print("\nBest Hyper-Parameters Combination: {}".format(best_key ))
        else:
            best_key = max(self.reg_mom_scores, key=self.reg_mom_scores.get) 
            
            print("Best Method of Estimation is a combination of Regression of multiple \
orders and momentum regression with {} days before forecast period, and with {} split with momentum"\
                          .format(best_key[1], best_key[3]))
            
        '''    
        else:
            poly = int(best_class[3:])
            best_key = max(self.reg_scores[poly], key=self.reg_scores[poly].get)   
            print("Best Method of Estimation is Regression of order {}".format(poly))
        '''
        
        
        # display best results after grid search
        b_num_harmonics, b_days_for_regression, b_underlying_trend_poly, b_momentum_split = best_key 
        
        '''
        print("\nFor this combination, the mean R^2 score results are:")
        print("Regression with Momentum: {:.2f}".format(np.mean(list(self.reg_mom_scores[best_key]))))
        print("FFT: {:.2f}".format(np.mean(self.fft_scores[best_key])))
        
        for poly in self.t_poly_degree:
            print("Regression of Order {}: {:.2f}".format(poly, np.mean(self.reg_scores[poly][best_key])))
        '''
        
        
        print("\n--------------------------------------------------------------------")
        print("Now training new StockRegressor instance with optimal hyper-parameters.")
        self.stock = StockRegressor(self.ticker, self.dates)
        
        self.stock.train(num_harmonics = b_num_harmonics, 
                         days_for_regression = b_days_for_regression, 
                         underlying_trend_poly = b_underlying_trend_poly,
                         n_days_to_predict = self.testing_window,
                         momentum_split = b_momentum_split,
                         verbose=True)
        
        self.stock.score(verbose=True)
        self.stock.predict()
        self.stock.plot_learning_data_frame()
        
        
        
    def train_stock(self, num_harmonics,
                          days_for_regression,
                          poly_degree ,
                          underlying_trend_poly, 
                          momentum_split):  
        
        reg1_train =[]
        reg1_val = []
        reg2_train =[]
        reg2_val = []
        reg3_train =[]
        reg3_val = []
        fft_train =[]
        fft_val = []


        self.stock.train(training_start_index = 0, 
                         training_end_index = self.training_window, 
                         n_days_to_predict = self.validation_window,
                         poly_degree = poly_degree, 
                         num_harmonics = num_harmonics, 
                         underlying_trend_poly = underlying_trend_poly,
                         days_for_regression = days_for_regression,
                         momentum_split = momentum_split,
                         verbose=False)

        reg, fft =  self.stock.score(verbose= False) 
        

        curr_iter = self.combination

        if self.stock.training_end_index < 250:  
            print("Warning: Training period is too short. Training dates are too close. Small sample size!!")
            print("It's recommended to have at least 1 year of historical data")
        
        if False:
            print("self.training_start_date {}".format(self.stock.training_start_date))
            print("self.training_start_index {}".format(self.stock.training_start_index))
            print("self.final_end_date {}".format(self.stock.final_end_date))
            print("self.training_end_index {}".format(self.stock.training_end_index))
            print("self.training_end_date {}".format(self.stock.training_end_date))
            print("self.testing_end_index {}".format(self.stock.testing_end_index))
            print("self.testing_end_date {}".format(self.stock.testing_end_date))

        print("Total Sample Size: {} samples".format(self.stock.sample_size))
        print("Training Sample Size: {} samples".format(self.total_iterations + self.training_window))
        print("Training Window: {} samples".format(self.training_window))
        print("Validation Window: {} samples".format(self.validation_window))
        print("Testing Sample Size: {} samples".format(self.testing_window))
        print("Training End Date is {} corresponding to the {}th sample".format(self.stock.training_end_date,
                                                             self.stock.training_end_index))

        print("Validation End Index {} over range {} - {} with validation window {}".\
               format(self.stock.val_end_index, 0, self.training_window,self.validation_window))

        print("There are {} combinations with {} iterations each: total iterations is {}".format(\
                self.max_combinations, self.total_iterations, self.max_combinations * self.total_iterations))


        print("\nProgress:")            
        print("Iteration Progress: {} / {}".format(curr_iter + 1, self.max_combinations * self.total_iterations))
        #print("Current Iteration out of {} iterations (combination # {}): {} "\
        #                          .format(self.total_iterations, self.combination, i-self.training_window))

        print("\nHyper-Parameters:")
        print("Harmonics Hyperparamter: {}".format(num_harmonics))
        print("Days Used for Momentum Regression Hyperparamter: {}".format(days_for_regression))
        print("Regression Order for Underlying Trend for FFT Hyperparamter: {}".format(underlying_trend_poly))
        print("Momentum Split for Underlying Trend for FFT Hyperparamter: {}".format(momentum_split))


        print("\nMean R^2 Scores:")
        
        for poly in self.t_poly_degree:
            print("Regression of Order {}: Training {:.2f} | Validation {:.2f}".format(poly, 
                                                                            reg[0][poly], reg[1][poly]))

        print("Regression of Order {} with Momentum: Training {:.2f} | Validation {:.2f}" \
                        .format(underlying_trend_poly, fft[2], fft[3]))
        print("FFT with Underlying Trend of Regression of Order {}: Training {:.2f} | Validation {:.2f}" \
                        .format(underlying_trend_poly, fft[0], fft[1]))
        
        if curr_iter < self.max_combinations * self.total_iterations - 1:
            clear_output(wait=True) 

        
        key = ( num_harmonics, days_for_regression, underlying_trend_poly, momentum_split)
        
        if not key in self.fft_scores:
            self.fft_scores[key] = []
        
        self.fft_scores[key].append(fft[1])
        
        if not key in self.reg_mom_scores:
            self.reg_mom_scores[key] = []
        
        self.reg_mom_scores[key].append(fft[3])
        
            
        for poly in self.t_poly_degree:
            
            if not poly in self.reg_scores:
                self.reg_scores[poly] = {}
            
            if not key in self.reg_scores[poly]:
                self.reg_scores[poly][key] = []
                
            self.reg_scores[poly][key].append(reg[1][poly])
            
        
        
        


        
        
        
        
        