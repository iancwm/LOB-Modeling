"""De Prado et al. (2012) - Optimal execution horizon and market microstructure.

This module implements the De Prado et al. (2012) model for optimal execution
horizon, including volume-synchronized probability of informed trading (VPIN)
and bulk volume classification (BVC).
"""

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AutoReg

import scipy.io as sio

np.set_printoptions(threshold=sys.maxsize)


class DePrado2014:
    """De Prado model for optimal execution horizon and market microstructure.

    This class implements the De Prado et al. (2012) framework for analyzing
    market microstructure, including VPIN (Volume-synchronized Probability of
    Informed Trading) and BVC (Bulk Volume Classification).

    Attributes:
        S_0: Initial stock price.
        MU: Rate of informed trades.
        EPSILON: Rate of uninformed trades.
        ALPHA: Probability that news/event arrives.
        S_low: Price after bad news.
        S_high: Price after good news.
        DELTA: Probability of bad news.
        n: Number of volume buckets for BVC and VPIN calculations.
        lob_path: Path to LOBSTER data file.
        tick_data_path: Path to tick data file.
        plot_enabled: Whether plotting is enabled.
        lobster_data: Processed LOBSTER data dictionary.
        tick_data: Cleaned tick data DataFrame.
    """

    def __init__(
        self,
        S_0: float = 10,
        MU: float = 0.7,
        EPSILON: float = 0.3,
        ALPHA: float = 0.5,
        S_low: float = 5,
        S_high: float = 100,
        DELTA: float = 0.3,
        lob_path: str = "data/example-data/AMZN_20141103.mat",
        tick_data_path: str = "data/Trades/14081.csv",
        plot: bool = False,
        n: int = 500,
    ) -> None:
        """Initialize the De Prado model with specified parameters.

        Args:
            S_0: Initial stock price. Defaults to 10.
            MU: Rate of informed trades. Defaults to 0.7.
            EPSILON: Rate of uninformed trades. Defaults to 0.3.
            ALPHA: Probability of news arrival. Defaults to 0.5.
            S_low: Price after bad news. Defaults to 5.
            S_high: Price after good news. Defaults to 100.
            DELTA: Probability of bad news. Defaults to 0.3.
            lob_path: Path to LOBSTER MAT file. Defaults to
                'data/example-data/AMZN_20141103.mat'.
            tick_data_path: Path to tick data CSV. Defaults to
                'data/Trades/14081.csv'.
            plot: If True, plot microstructure visualizations. Defaults to False.
            n: Number of volume buckets for BVC/VPIN. Defaults to 500.
        """
        self.S_0 = float(S_0)
        self.MU = float(MU)
        self.EPSILON = float(EPSILON)
        self.ALPHA = float(ALPHA)
        self.S_low = float(S_low)
        self.S_high = float(S_high)
        self.DELTA = float(DELTA)
        self.n = int(n)

        self.lob_path = Path(lob_path)
        self.tick_data_path = Path(tick_data_path)

        if self.lob_path.exists():
            self.lobster_raw = sio.loadmat(self.lob_path)["LOB"]
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

    def get_yfinance_data(
        self, ticker: str, period: str, interval: str
    ) -> pd.DataFrame:
        """Fetch historical market data from Yahoo Finance.

        Args:
            ticker: Stock ticker symbol (e.g., 'SPY').
            period: Data period (e.g., '1d', '10y').
            interval: Data interval (e.g., '1m', '3mo').

        Returns:
            DataFrame with historical price data.
        """
        return yf.Ticker(ticker).history(period=period, interval=interval)

    def get_lobster_data(self, stock: Any) -> Dict[str, np.ndarray]:
        """Parse LOBSTER MAT file into structured data.

        Args:
            stock: Path to MATLAB file with price information.

        Returns:
            Dictionary containing:
                - 't': Time array in seconds.
                - 'bid': Bid prices.
                - 'bidvol': Bid volumes.
                - 'ask': Ask prices.
                - 'askvol': Ask volumes.
                - 'market_order': Market order data.
                - 'dt': Time increment.
                - 'midprice': Midprice series.
                - 'microprice': Microprice series.
                - 'spread': Bid-ask spread.
                - 'order_imbalance': Order imbalance series.
        """
        t = (np.array((stock["EventTime"][0][0][:, 0])) - 34200000) * 1e-3
        bid = np.array(stock["BuyPrice"][0][0] * 1e-4)
        bidvol = np.array(stock["BuyVolume"][0][0] * 1.0)
        ask = np.array(stock["SellPrice"][0][0] * 1e-4)
        askvol = np.array(stock["SellVolume"][0][0] * 1.0)
        market_order = np.array(stock["MO"][0][0] * 1.0)
        dt = t[1] - t[0]
        midprice = 0.5 * (bid[:, 0] + ask[:, 0])
        microprice = (bid[:, 0] * askvol[:, 0] + ask[:, 0] * bidvol[:, 0]) / (
            bidvol[:, 0] + askvol[:, 0]
        )
        spread = ask[:, 0] - bid[:, 0]
        order_imbalance = np.array(
            (bidvol[:, 0] - askvol[:, 0]) / (bidvol[:, 0] + askvol[:, 0]), ndmin=2
        ).T
        return {
            "t": t,
            "bid": bid,
            "bidvol": bidvol,
            "ask": ask,
            "askvol": askvol,
            "market_order": market_order,
            "dt": dt,
            "midprice": midprice,
            "microprice": microprice,
            "spread": spread,
            "order_imbalance": order_imbalance,
        }

    def get_TICKDATA(self, xlsx_path: str) -> None:
        """Placeholder for Excel tick data loading.

        Args:
            xlsx_path: Path to Excel file with tick data.
        """
        pass

    def data_vis(self) -> None:
        """Create interactive visualization of market microstructure data.

        Generates a 2x2 subplot figure showing:
            1. Mid price minus microprice over time
            2. Order imbalance over time
            3. Spread with interquartile range bands
            4. Cumulative buy and sell market orders
        """
        percentiles = np.linspace(0, 100, 11)
        spread = np.zeros((int(self.lobster_data["t"][-1]), 11))
        for i in range(11):
            for time in range(int(self.lobster_data["t"][-1])):
                spread[time, i] = np.percentile(
                    self.lobster_data["spread"], percentiles[i]
                )

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Mid - Micro",
                "Order Imbalance",
                "Spread with IQR",
                "Cumulative Buy & Sell MOs",
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}],
            ],
        )

        fig.add_trace(
            go.Scatter(
                x=self.lobster_data["t"],
                y=self.lobster_data["midprice"] - self.lobster_data["microprice"],
                mode="lines",
                name="Mid - Micro",
                line=dict(color="red"),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=self.lobster_data["t"],
                y=self.lobster_data["order_imbalance"],
                mode="lines",
                name="Order Imbalance",
                line=dict(color="blue"),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=np.arange(0, int(self.lobster_data["t"][-1]), 1),
                y=spread[:, 5],
                mode="lines",
                name="Median Spread",
                line=dict(color="black"),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        for i in range(5):
            fig.add_trace(
                go.Scatter(
                    x=np.arange(0, int(self.lobster_data["t"][-1]), 1),
                    y=spread[:, i],
                    fill=None,
                    mode="lines",
                    line_color="rgba(0,0,0,0)",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=np.arange(0, int(self.lobster_data["t"][-1]), 1),
                    y=spread[:, -(i + 1)],
                    fill="tonexty",
                    mode="lines",
                    line_color="rgba(0,0,0,0)",
                    fillcolor=f"rgba(0, 50, 200, {0.1 + i * 0.05})",
                    name=f"Band {i+1}",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=2,
                col=1,
            )

        buy_orders = self.lobster_data["market_order"][:, 7].clip(-1, 0)
        sell_orders = self.lobster_data["market_order"][:, 7].clip(0, 1)

        fig.add_trace(
            go.Scatter(
                x=np.arange(len(buy_orders)),
                y=np.cumsum(buy_orders),
                mode="lines",
                name="Cumulative Buy MOs",
                line=dict(color="green"),
                showlegend=False,
            ),
            row=2,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=np.arange(len(sell_orders)),
                y=np.cumsum(sell_orders),
                mode="lines",
                name="Cumulative Sell MOs",
                line=dict(color="blue"),
                showlegend=False,
            ),
            row=2,
            col=2,
        )

        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_yaxes(title_text="Price Difference", row=1, col=1)
        fig.update_xaxes(title_text="Time", row=1, col=2)
        fig.update_yaxes(title_text="Order Imbalance", row=1, col=2)
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Spread", row=2, col=1)
        fig.update_xaxes(title_text="Time", row=2, col=2)
        fig.update_yaxes(title_text="Cumulative Volume", row=2, col=2)

        fig.update_layout(height=800, hovermode="x unified", template="plotly_white")
        fig.show()

    def AR_order_imbalance(self, plot_regress: bool) -> None:
        """Analyze and predict order imbalance using autoregressive model.

        Separates buy and sell market order volumes and fits an AR model to
        predict bid volume.

        Args:
            plot_regress: If True, creates a plot comparing actual vs predicted
                bid volumes.
        """
        MO_buy_vol = np.array(
            (
                self.lobster_data["market_order"][:, 0] / 3.6e6,
                self.lobster_data["market_order"][:, 6],
                self.lobster_data["market_order"][:, 7],
            )
        ).T
        MO_buy_vol[:, 1] = np.where(MO_buy_vol[:, 2] < 0, 0, MO_buy_vol[:, 1])
        MO_buy_vol = MO_buy_vol[:, 0:2]
        MO_sell_vol = np.array(
            (
                self.lobster_data["market_order"][:, 0] / 3.6e6,
                self.lobster_data["market_order"][:, 6],
                self.lobster_data["market_order"][:, 7],
            )
        ).T
        MO_sell_vol[:, 1] = np.where(MO_sell_vol[:, 2] > 0, 0, MO_sell_vol[:, 1])
        MO_sell_vol = MO_sell_vol[:, 0:2]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=MO_buy_vol[:, 0],
                y=MO_buy_vol[:, 1],
                mode="markers",
                name="Market Order Buy Volumes",
                marker=dict(
                    color=np.random.rand(len(MO_buy_vol)),
                    colorscale="Viridis",
                    size=6,
                ),
            )
        )
        fig.update_layout(
            title="Market Order Buy Volumes Against Time",
            xaxis_title="Time Since Midnight (Hours)",
            yaxis_title="Volume",
            hovermode="closest",
            template="plotly_white",
            height=600,
        )
        fig.show()

        train = self.lobster_data["bidvol"][
            0 : int(len(self.lobster_data["bidvol"][:, 0]) / 3), 0
        ]
        test = self.lobster_data["bidvol"][
            int(len(self.lobster_data["bidvol"][:, 0]) / 3) : len(
                self.lobster_data["bidvol"][:, 0]
            )
            - 1,
            0,
        ]
        model = AutoReg(train, lags=5).fit()
        coef = model.params

        def predict(params: np.ndarray, history: List[float]) -> float:
            """Predict next value using AR coefficients.

            Args:
                params: AR model coefficients.
                history: Historical values for prediction.

            Returns:
                Predicted value.
            """
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
        print(f"RMSE: {rmse}")
        if plot_regress:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(test))),
                    y=test,
                    mode="lines",
                    name="Actual",
                    line=dict(color="blue"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(pred))),
                    y=pred,
                    mode="lines",
                    name="Predicted",
                    line=dict(color="orange"),
                )
            )
            fig.update_layout(
                title="AR Order Imbalance Prediction",
                xaxis_title="Time",
                yaxis_title="Bid Volume",
                hovermode="x unified",
                template="plotly_white",
                height=600,
            )
            fig.show()

    def init_PIN(self) -> float:
        """Calculate initial Probability of Informed Trading (PIN).

        Returns:
            Initial PIN estimate based on model parameters.
        """
        prob_sell = (self.ALPHA * self.DELTA * self.MU) / (
            self.EPSILON + (self.ALPHA * self.DELTA * self.MU)
        )
        prob_buy = (self.ALPHA * (1 - self.DELTA) * self.MU) / (
            self.EPSILON + (self.ALPHA * (1 - self.DELTA) * self.MU)
        )
        PIN = prob_sell + prob_buy
        return PIN

    def init_trade_range(self) -> float:
        """Calculate initial trading range for market makers.

        Returns:
            Range where market makers are willing to provide liquidity at t=0.
        """
        return self.init_PIN() * (self.S_high - self.S_low)

    def PIN_estimate(self) -> None:
        """Estimate PIN parameters using MLE.

        TODO: Implement maximum likelihood estimation to update ALPHA, MU,
        DELTA, and EPSILON over time.
        """
        pass

    def prob_buy_sell(self, X: int, Y: int, t: float) -> float:
        """Calculate probability of X buy and Y sell orders at time t.

        Args:
            X: Number of buy orders.
            Y: Number of sell orders.
            t: Time parameter.

        Returns:
            Joint probability of observing X buys and Y sells.
        """
        prob_good_news = (
            self.ALPHA
            * (1 - self.DELTA)
            * math.exp(-(self.MU + 2 * self.EPSILON))
            * ((self.MU + self.EPSILON) ** X)
            * (self.EPSILON**Y)
        ) / (math.factorial(X) * math.factorial(Y))
        prob_bad_news = (
            self.ALPHA
            * self.DELTA
            * math.exp(-(self.MU + 2 * self.EPSILON))
            * ((self.MU + self.EPSILON) ** Y)
            * (self.EPSILON**X)
        ) / (math.factorial(X) * math.factorial(Y))
        prob_no_news = (
            (1 - self.ALPHA)
            * math.exp(-2 * self.EPSILON)
            * (self.EPSILON ** (X + Y))
        ) / (math.factorial(X) * math.factorial(Y))
        prob = prob_good_news + prob_bad_news + prob_no_news
        return prob

    def clean_data(self) -> pd.DataFrame:
        """Clean and format tick data.

        Returns:
            DataFrame with columns: Time, Price, Volume.
        """
        df = self.tick_data_raw.drop(
            columns=[
                "8:36:37",
                "D",
                "TB",
                "0",
                "1957",
                "N",
                "C",
                "T",
                "X",
                "Unnamed: 12",
            ]
        )
        df.columns = ["Time", "Price", "Volume"]
        return df

    def bvc(self) -> Tuple[List[float], List[float]]:
        """Perform Bulk Volume Classification (BVC).

        Classifies trade volume into buy and sell volume buckets based on
        price changes and volume.

        Returns:
            Tuple of (buy_volume_buckets, sell_volume_buckets).
        """
        total_volume = self.tick_data["Volume"].sum()
        volume_bucket_size = total_volume / self.n
        price_change = np.diff(self.tick_data["Price"], prepend=self.tick_data["Price"][0])
        price_deviation = np.std(price_change)
        buy_volume_buckets = []
        sell_volume_buckets = []
        total_volume_list = []
        price_changes_list = []
        v_i = []
        P_i = []

        for tick in self.tick_data.itertuples():
            if sum(v_i) + tick[3] < volume_bucket_size:
                v_i.append(tick[3])
                P_i.append(tick[2])
                continue
            else:
                price_change_val = P_i[-1] - P_i[0]
                price_changes_list.append(price_change_val)
                buy_volume = (
                    sum(v_i[0:-1])
                    * stats.norm.cdf(price_change_val / price_deviation)  # type: ignore
                )
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
                mode="markers",
                name="Buy Volume",
                marker=dict(color="blue", size=6),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(total_volume_list)),
                y=sell_volume_buckets,
                mode="markers",
                name="Sell Volume",
                marker=dict(color="black", size=6),
            )
        )
        fig.update_layout(
            title="Buy & Sell Volumes using BVC",
            xaxis_title="Volume Bucket",
            yaxis_title="Volume",
            hovermode="closest",
            template="plotly_white",
            height=600,
        )
        fig.show()
        return buy_volume_buckets, sell_volume_buckets

    def VPIN(
        self, buy_buckets: List[float], sell_buckets: List[float]
    ) -> np.ndarray:
        """Calculate Volume-synchronized Probability of Informed Trading (VPIN).

        Args:
            buy_buckets: Estimated buy volumes from BVC.
            sell_buckets: Estimated sell volumes from BVC.

        Returns:
            VPIN series over volume buckets.
        """
        oi = [abs(buy_buckets[i] - sell_buckets[i]) for i in range(len(buy_buckets))]
        cumulative_oi = np.array(np.cumsum(oi))
        cumulative_volume = np.cumsum(
            [buy_buckets[i] + sell_buckets[i] for i in range(len(buy_buckets))]
        )
        weighted_vol = np.array([i * cumulative_volume[i] for i in range(len(cumulative_volume))])
        VPIN = cumulative_oi[1:] / weighted_vol[1:]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(VPIN)),
                y=VPIN,
                mode="lines",
                name="VPIN",
                line=dict(color="red", width=2),
            )
        )
        fig.update_layout(
            title="Evolution of VPIN over Day",
            xaxis_title="Volume Bucket",
            yaxis_title="VPIN",
            hovermode="x unified",
            template="plotly_white",
            height=600,
        )
        fig.show()
        return VPIN
