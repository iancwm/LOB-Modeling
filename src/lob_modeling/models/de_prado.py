import math
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import scipy.io as sio
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AutoReg
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys

np.set_printoptions(threshold=sys.maxsize)


class DePrado2014:
    """
    Description: This is a collection of programs/code relatd
    to the work of de Prado et al 2012 - "Optimal Execution Horizon"
    """
    def __init__(self, S_0=10, MU=0.7, EPSILON=0.3, ALPHA=0.5, S_low=5, S_high=100, DELTA=0.3, lob_path='data/example-data/AMZN_20141103.mat', tick_data_path='data/Trades/14081.csv', plot=False, n=500):
        """
        :param S_0: initial price of stock
        :param MU: rate of informed trades
        :param EPSILON: rate of uninformed trades
        :param ALPHA: probability event/news arrives
        :param S_low: price of event after bad news
        :param S_high: price of event after good news
        :param DELTA: probability of bad news
        :param lob_path: there are 3/4 sample mat files in the repo
        :param plot: true -> plot microstructure stuff
        :param n: number of volume blocks for BVC and VPIN calculations
        """
        self.S_0 = float(S_0)
        self.MU = float(MU)
        self.EPSILON = float(EPSILON)
        self.ALPHA = float(ALPHA)
        self.S_low = float(S_low)
        self.S_high = float(S_high)
        self.DELTA = float(DELTA)
        self.n = int(n)
        
        # Use relative paths to preserve privacy
        self.lob_path = Path(lob_path)
        self.tick_data_path = Path(tick_data_path)

        if self.lob_path.exists():
            self.lobster_raw = sio.loadmat(self.lob_path)['LOB']
            self.lobster_data = self.get_lobster_data(self.lobster_raw)
        else:
            print(f"Warning: LOB file not found at {self.lob_path}")
            self.lobster_data = None

        if self.tick_data_path.exists():
            self.tick_data_raw = pd.read_csv(self.tick_data_path)
            self.tick_data = self.clean_data()
        else:
            print(f"Warning: Tick data file not found at {self.tick_data_path}")
            self.tick_data = None

        self.plot_enabled = plot
        if self.plot_enabled and self.lobster_data:
            self.data_vis()
        
        if self.lobster_data is not None:
            self.AR_order_imbalance(plot_regress=False)
        
        if self.tick_data is not None:
            buy_buckets, sell_buckets = self.bvc()
            self.VPIN(buy_buckets, sell_buckets)

    def get_yfinance_data(self, ticker, period, interval):
        """
        :param ticker: string i.e. 'SPY'
        :param period: string i.e. '1d',...,'10y'
        :param interval: string i.e. '1m',...,'3mo'
        :return: yfinance data as dataframe
        """
        return yf.Ticker(ticker).history(period=period, interval=interval)

    def get_lobster_data(self, stock):
        """
        :param stock: path to matlab file of price info
        :return: time (t), change in time (dt), bid, bidvol, ask, askvol, market_orders, midprice, microprice, spread in a class
        """
        t = (np.array((stock['EventTime'][0][0][:, 0])) - 34200000) * 1e-3
        # minute_incremented_t = t[::600] # Unused
        # hour_incremented_t = t[::36000] # Unused
        bid = np.array(stock['BuyPrice'][0][0] * 1e-4)
        bidvol = np.array(stock['BuyVolume'][0][0] * 1.0)
        ask = np.array(stock['SellPrice'][0][0] * 1e-4)
        askvol = np.array(stock['SellVolume'][0][0] * 1.0)
        market_order = np.array(stock['MO'][0][0] * 1.0)
        dt = t[1] - t[0]
        midprice = 0.5 * (bid[:, 0] + ask[:, 0])
        microprice = (bid[:, 0] * askvol[:, 0] + ask[:, 0] * bidvol[:, 0]) / (
                    bidvol[:, 0] + askvol[:, 0])
        spread = ask[:, 0] - bid[:, 0]
        order_imbalance = np.array((bidvol[:, 0] - askvol[:, 0]) /
                                   (bidvol[:, 0] + askvol[:, 0]), ndmin=2).T
        return {'t': t, 'bid': bid, 'bidvol': bidvol, 'ask': ask,
                'askvol': askvol, 'market_order': market_order,
                'dt': dt, 'midprice': midprice, 'microprice': microprice,
                'spread': spread, 'order_imbalance': order_imbalance}

    def get_TICKDATA(self, xlsx_path):
        pass

    def data_vis(self):
        # 11 is number of percentiles to color - (11-1)/2=5 is the bands shown
        percentiles = np.linspace(0, 100, 11)
        spread = np.zeros((int(self.lobster_data['t'][-1]), 11))
        for i in range(11):
            for time in range(int(self.lobster_data['t'][-1])):
                spread[time, i] = np.percentile(self.lobster_data['spread'], percentiles[i])

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Mid - Micro', 'Order Imbalance',
                           'Spread with IQR', 'Cumulative Buy & Sell MOs'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        # Plot 1: Mid - Micro
        fig.add_trace(
            go.Scatter(
                x=self.lobster_data['t'],
                y=self.lobster_data['midprice'] - self.lobster_data['microprice'],
                mode='lines',
                name='Mid - Micro',
                line=dict(color='red'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Plot 2: Order Imbalance
        fig.add_trace(
            go.Scatter(
                x=self.lobster_data['t'],
                y=self.lobster_data['order_imbalance'],
                mode='lines',
                name='Order Imbalance',
                line=dict(color='blue'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Plot 3: Spread with IQR
        fig.add_trace(
            go.Scatter(
                x=np.arange(0, int(self.lobster_data['t'][-1]), 1),
                y=spread[:, 5],
                mode='lines',
                name='Median Spread',
                line=dict(color='black'),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Add confidence bands for spread
        for i in range(5):
            fig.add_trace(
                go.Scatter(
                    x=np.arange(0, int(self.lobster_data['t'][-1]), 1),
                    y=spread[:, i],
                    fill=None,
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=np.arange(0, int(self.lobster_data['t'][-1]), 1),
                    y=spread[:, -(i + 1)],
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    fillcolor=f'rgba(0, 50, 200, {0.1 + i * 0.05})',
                    name=f'Band {i+1}',
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=2, col=1
            )
        
        # Plot 4: Cumulative Buy & Sell MOs
        buy_orders = self.lobster_data['market_order'][:, 7].clip(-1, 0)
        sell_orders = self.lobster_data['market_order'][:, 7].clip(0, 1)
        
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(buy_orders)),
                y=np.cumsum(buy_orders),
                mode='lines',
                name='Cumulative Buy MOs',
                line=dict(color='green'),
                showlegend=False
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(sell_orders)),
                y=np.cumsum(sell_orders),
                mode='lines',
                name='Cumulative Sell MOs',
                line=dict(color='blue'),
                showlegend=False
            ),
            row=2, col=2
        )
        
        fig.update_xaxes(title_text='Time', row=1, col=1)
        fig.update_yaxes(title_text='Price Difference', row=1, col=1)
        
        fig.update_xaxes(title_text='Time', row=1, col=2)
        fig.update_yaxes(title_text='Order Imbalance', row=1, col=2)
        
        fig.update_xaxes(title_text='Time', row=2, col=1)
        fig.update_yaxes(title_text='Spread', row=2, col=1)
        
        fig.update_xaxes(title_text='Time', row=2, col=2)
        fig.update_yaxes(title_text='Cumulative Volume', row=2, col=2)
        
        fig.update_layout(height=800, hovermode='x unified', template='plotly_white')
        fig.show()

    def AR_order_imbalance(self, plot_regress):
        # separating buy & sell MO data
        MO_buy_vol = np.array((self.lobster_data['market_order'][:, 0] / 3.6e6,
                               self.lobster_data['market_order'][:, 6],
                               self.lobster_data['market_order'][:, 7])).T
        MO_buy_vol[:, 1] = np.where(MO_buy_vol[:, 2] < 0, 0, MO_buy_vol[:, 1])
        MO_buy_vol = MO_buy_vol[:, 0:2]
        MO_sell_vol = np.array((self.lobster_data['market_order'][:, 0] / 3.6e6,
                                self.lobster_data['market_order'][:, 6],
                                self.lobster_data['market_order'][:, 7])).T
        MO_sell_vol[:, 1] = np.where(MO_sell_vol[:, 2] > 0, 0,
                                     MO_sell_vol[:, 1])
        MO_sell_vol = MO_sell_vol[:, 0:2]
        
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=MO_buy_vol[:, 0],
                y=MO_buy_vol[:, 1],
                mode='markers',
                name='Market Order Buy Volumes',
                marker=dict(
                    color=np.random.rand(len(MO_buy_vol)),
                    colorscale='Viridis',
                    size=6
                )
            )
        )
        fig.update_layout(
            title='Market Order Buy Volumes Against Time',
            xaxis_title='Time Since Midnight (Hours)',
            yaxis_title='Volume',
            hovermode='closest',
            template='plotly_white',
            height=600
        )
        fig.show()
        
        # Regressing Bid Volume - might change this to something more autocorrelated
        train = self.lobster_data['bidvol'][
                0:int(len(self.lobster_data['bidvol'][:, 0]) / 3), 0]
        test = self.lobster_data['bidvol'][
               int(len(self.lobster_data['bidvol'][:, 0]) / 3):len(
                   self.lobster_data['bidvol'][:, 0]) - 1, 0]
        model = AutoReg(train, lags=5).fit()
        coef = model.params

        def predict(params, history):
            Y = params[0]
            for i in range(1, len(params)):
                Y += params[i] * history[-i]
            return Y

        history = [train[i] for i in range(len(train))]
        pred = []
        for t in range(len(test)):
            Y = predict(coef, history)
            observ = test[t]
            pred.append(observ)
            history.append(observ)

        rmse = math.sqrt(mean_squared_error(test, pred))
        print(f'RMSE: {rmse}')
        if plot_regress:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(test))),
                    y=test,
                    mode='lines',
                    name='Actual',
                    line=dict(color='blue')
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(pred))),
                    y=pred,
                    mode='lines',
                    name='Predicted',
                    line=dict(color='orange')
                )
            )
            fig.update_layout(
                title='AR Order Imbalance Prediction',
                xaxis_title='Time',
                yaxis_title='Bid Volume',
                hovermode='x unified',
                template='plotly_white',
                height=600
            )
            fig.show()

    def init_PIN(self):
        # probability of informed order conditional on sell
        prob_sell = (self.ALPHA * self.DELTA * self.MU) / (self.EPSILON + (self.ALPHA * self.DELTA * self.MU))
        # probability of informed order conditional on buy
        prob_buy = (self.ALPHA * (1 - self.DELTA) * self.MU) / (self.EPSILON + (self.ALPHA * (1 - self.DELTA) * self.MU))
        PIN = prob_sell + prob_buy
        return PIN

    def init_trade_range(self):
        # range MMs are willing to provide liquidity at time = 0
        return self.init_PIN() * (self.S_high - self.S_low)

    def PIN_estimate(self):
        # TODO - use MLE to estimate/update ALPHA, MU, DELTA, EPSILON over time
        pass

    def prob_buy_sell(self, X, Y, t):
        # returns probability of X buy & Y sell orders at time t
        prob_good_news = (self.ALPHA * (1 - self.DELTA) * math.exp(-(self.MU + 2 * self.EPSILON)) *
               ((self.MU + self.EPSILON) ** X) * (self.EPSILON ** Y)) / \
               (math.factorial(X) * math.factorial(Y))
        prob_bad_news = (self.ALPHA * self.DELTA * math.exp(-(self.MU + 2 * self.EPSILON)) *
                         ((self.MU + self.EPSILON) ** Y) * (self.EPSILON ** X)) / \
                        (math.factorial(X) * math.factorial(Y))
        prob_no_news = ((1 - self.ALPHA) * math.exp(-2 * self.EPSILON) * (self.EPSILON ** (X + Y))) / \
                       (math.factorial(X) * math.factorial(Y))
        prob = prob_good_news + prob_bad_news + prob_no_news
        return prob

    def clean_data(self):
        df = self.tick_data_raw.drop(columns=['8:36:37', 'D', 'TB', '0', '1957', 'N', 'C', 'T', 'X', 'Unnamed: 12'])
        df.columns = ['Time', 'Price', 'Volume']
        return df

    def bvc(self):
        """
        :return: volume buckets of LOB data (bulk volume classification)
        """
        total_volume = self.tick_data['Volume'].sum()
        volume_bucket_size = total_volume / self.n
        price_change = np.diff(self.tick_data['Price'], prepend=self.tick_data['Price'][0]) # noqa: F841
        price_deviation = np.std(price_change) # noqa: F841
        buy_volume_buckets = []
        sell_volume_buckets = []
        total_volume_list = []
        price_changes_list = []
        v_i = []
        P_i = []
        # VPIN = [] # Unused
        for tick in self.tick_data.itertuples():
            if sum(v_i) + tick[3] < volume_bucket_size:
                v_i.append(tick[3])
                P_i.append(tick[2])
                continue

            else:
                price_change_val = P_i[-1] - P_i[0]
                price_changes_list.append(price_change_val)
                # Note: stats.norm.cdf is likely what was intended here but stats was not imported in original
                buy_volume = sum(v_i[0:-1]) * stats.norm.cdf(price_change_val / price_deviation) # type: ignore
                sell_volume = sum(v_i[0:-1]) - buy_volume
                total_volume_list.append(sum(v_i[0:-1]))
                buy_volume_buckets.append(buy_volume)
                sell_volume_buckets.append(sell_volume)
                v_i = [v_i[-1]]
                P_i = [P_i[-1]]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(total_volume_list)),
                y=buy_volume_buckets,
                mode='markers',
                name='Buy Volume',
                marker=dict(color='blue', size=6)
            )
        )
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(total_volume_list)),
                y=sell_volume_buckets,
                mode='markers',
                name='Sell Volume',
                marker=dict(color='black', size=6)
            )
        )
        fig.update_layout(
            title='Buy & Sell Volumes using BVC',
            xaxis_title='Volume Bucket',
            yaxis_title='Volume',
            hovermode='closest',
            template='plotly_white',
            height=600
        )
        fig.show()
        return buy_volume_buckets, sell_volume_buckets

    def VPIN(self, buy_buckets, sell_buckets):
        """
        :param buy_buckets: estimated classified buys from BVC
        :param sell_buckets:"" sells ""
        :return: VPIN metric - volume-synchronized prob of informed trade
        """
        oi = [abs(buy_buckets[i] - sell_buckets[i]) for i in range(len(buy_buckets))]
        cumulative_oi = np.array(np.cumsum(oi))
        cumulative_volume = np.cumsum([buy_buckets[i] + sell_buckets[i] for i in range(len(buy_buckets))])
        weighted_vol = np.array([i * cumulative_volume[i] for i in range(len(cumulative_volume))])
        VPIN = cumulative_oi[1:] / weighted_vol[1:]
        
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(VPIN)),
                y=VPIN,
                mode='lines',
                name='VPIN',
                line=dict(color='red', width=2)
            )
        )
        fig.update_layout(
            title='Evolution of VPIN over Day',
            xaxis_title='Volume Bucket',
            yaxis_title='VPIN',
            hovermode='x unified',
            template='plotly_white',
            height=600
        )
        fig.show()
        return VPIN
