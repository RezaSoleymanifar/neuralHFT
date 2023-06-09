"""
base.py

Description:
------------
    Defines the base class for a trader that can execute orders using a
    trading client and a model to produce actions. This trader requires
    a trading client to connect to a trading environment, a data client
    to stream the live data and an agent to perform the decision making.
    The agent has a model to generate actions, a data pipe to modify
    base environment, and metadata for the dataset used to train the
    agent. Metadata will be used to create aggregated data stream
    matching the training data.

License:
--------
    MIT License. See LICENSE.md file.

Author(s):
-------
    Reza Soleymanifar, Email: Reza@Soleymanifar.com

Class(es):
----------
    AbstractTrader:
        Abstract base class for defining a trader that can execute
        orders using a trading client and a model to produce actions.
"""
from abc import ABC
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd
from torch import nn

from neural.client.alpaca import AbstractTradeClient, AbstractDataClient
from neural.common.log import logger
from neural.data.alpaca import AlpacaAsset
from neural.data.base import AsyncDataFeeder
from neural.env.base import TradeMarketEnv
from neural.meta.agent import Agent
from neural.wrapper.base import AbstractMarketEnvMetadataWrapper


class AbstractTrader(ABC):
    """
    Abstract base class for defining a trader that can execute orders
    using a trading client and a model to produce actions. This trader
    requires a trading client to connect to a trading environment, a
    data client to stream the live data and an agent to perform the
    decision making. The agent has a model to generate actions, a data
    pipe to modify base environment, and metadata for the dataset used
    to train the agent. Metadata will be used to create aggregated data
    stream matching the training data. 
    
    TODO: support or multiple data clients for streaming from multiple
    data stream sources.

    Attributes:
    ----------
        trade_client (AbstractTradeClient):
            An instance of the trading client to connect to the trading
            platform.
        data_client (AbstractDataClient):
            An instance of the data client to stream data.
        agent (Agent):
            An instance of the agent to perform decision making.
        _data_feeder (AsyncDataFeeder):
            An instance of the asynchronous data feeder to stream data.
        _trade_market_env (TradeMarketEnv):
            An instance of the trading environment.
        _model (nn.Module):
            The PyTorch neural network model used by the agent.
        handle_non_trade_time (bool):
            A flag to indicate if the trader should handle non-trade
            time. Non-trade time is defined as the time outside of the
            trading schedule. If the flag is set to True, then the
            trader will cancel all open orders and wait for the market
            to open.
    
    Properties:
    -----------
        data_feeder (AsyncDataFeeder):
            The data feeder used to stream data from the data client.
        trade_market_env (TradeMarketEnv):
            The trading environment used to execute orders.
        market_metadata_wrapper (AbstractMarketEnvMetadataWrapper):
            The metadata wrapper used by the trading environment.
        schedule (pd.DataFrame):
            The schedule of the trading environment. The schedule is a
            pandas DataFrame with two columns, start and end and date as
            the index.
        assets (List[AlpacaAsset]):
            A numpy array of assets held by the trader.
        cash (float):
            Cash available in the trading account. Cash can be positive
            or negative depending on the amount of cash borrowed from
            the broker.
        asset_quantities (np.ndarray[float]):
            A numpy array of current quantity of each asset held by the
            trader. Asset quantities can be positive or negative.
            Negative quantities indicate that the trader has shorted the
            asset, namely the trader owes the asset to the broker.
        asset_prices (np.ndarray[float]):
            A numpy array of current price of each asset held by the
            trader.
        model (nn.Module):
            Returns the model used by the agent to generate actions.
    
    Methods:
    --------
        _check_time(self) -> bool:
            A method to check if the current time is within the trading
            schedule. If the current time is not within the trading
            schedule, then all open orders are cancelled.
        trade (self) -> None:
            Starts the trading process by creating a trading environment
            and executing actions from the model.
        place_orders(self, actions: np.ndarray, *args, **kwargs) ->
        None:
            Abstract method for placing an order for a single asset. The
            restrictions of the API should be enforced in this method.
    """
    def __init__(self, trade_client: AbstractTradeClient,
                 data_client: AbstractDataClient, agent: Agent) -> None:
        """
        Initializes an AbstractTrader object.

        Args:
        ------
            trade_client (AbstractTradeClient):
                An instance of the trading client to connect to the
                trading platform.
            data_client (AbstractDataClient):
                An instance of the data client to stream data.
            agent (Agent):
                An instance of the agent to perform decision making.
        """
        self.trade_client = trade_client
        self.data_client = data_client
        self.agent = agent

        self._data_feeder = None
        self._trade_market_env = None
        self._model = None
        self.handle_non_trade_time = False

        return None

    @property
    def data_feeder(self) -> AsyncDataFeeder:
        """
        The data feeder used to stream data from the data client.

        Returns:
        --------
            data_feeder (AsyncDataFeeder):
                An instance of the data feeder to stream data.
        """
        if self._data_feeder is None:
            stream_metadata = self.agent.dataset_metadata.stream
            self._data_feeder = AsyncDataFeeder(stream_metadata,
                                                self.data_client)
        return self._data_feeder

    @property
    def trade_market_env(self) -> TradeMarketEnv:
        """
        The trading environment used to execute orders.

        Returns:
        --------
            trade_market_env (TradeMarketEnv):
                An instance of the trading environment.
        """
        if self._trade_market_env is None:
            env = TradeMarketEnv(trader=self)
            self._trade_market_env = self.agent.pipe.pipe(env)
        return self._trade_market_env

    @property
    def market_metadata_wrapper(self) -> AbstractMarketEnvMetadataWrapper:
        """
        The metadata wrapper used by the trading environment.

        Returns:
        --------
            market_metadata_wrapper (AbstractMarketEnvMetadataWrapper):
                An instance of the metadata wrapper used by the
                trading environment.
        """
        market_metadata_wrapper = self.agent.pipe.get_market_metadata_wrapper(
            self.trade_market_env)
        return market_metadata_wrapper

    @property
    def schedule(self) -> pd.DataFrame:
        """
        The schedule of the trading environment. The schedule is a list
        of datetime objects representing the times at which the
        environment is reset.

        Returns:
        --------
            schedule (List[datetime]):
                The schedule of the trading environment.
        """
        return self.market_metadata_wrapper.schedule

    @property
    def assets(self) -> List[AlpacaAsset]:
        """
        A numpy array of assets held by the trader.

        Returns:
        --------
            assets (List[AlpacaAsset]):
                The assets held by the trader.
        """
        return self.market_metadata_wrapper.assets

    @property
    def cash(self) -> float:
        """
        Cash available in the trading account. Cash can be positive or
        negative depending on the amount of cash borrowed from the
        broker.

        Returns:
        --------
            cash (float):
                Cash available in the trading account.
        """
        return self.trade_client.cash

    @property
    def asset_quantities(self) -> np.ndarray[float]:
        """
        A numpy array of current quantity of each asset held by the
        trader. Asset quantities can be positive or negative. Negative
        quantities indicate that the trader has shorted the asset,
        namely the trader owes the asset to the broker.

        Returns:
        --------
            asset_quantities (np.ndarray[float]):
                The current quantity of each asset held by the trader.
        """
        return self.trade_client.get_asset_quantities(assets=self.assets)

    @property
    def asset_prices(self) -> np.ndarray[float]:
        """
        A numpy array of current price of each asset held by the
        trader.

        Returns:
        --------
            asset_prices (np.ndarray[float]):
                The current price of each asset held by the trader.
        """
        return self.market_metadata_wrapper.asset_prices

    @property
    def model(self) -> nn.Module:
        """
        Returns the model used by the agent to generate actions.

        Returns:
        --------
            model (nn.Module):
                The model used by the agent to generate actions.
        """
        return self.agent.model

    def _check_time(self) -> bool:
        """
        A method to check if the current time is within the trading
        schedule. If the current time is not within the trading
        schedule, then all open orders are cancelled.

        Returns:
        --------
            bool:
                True if the current time is within the trading schedule,
        """
        current_time = datetime.utcnow().date()
        current_day = current_time.date()
        start, end = self.schedule.loc[current_day].values()

        if not start <= current_time <= end:
            if self.handle_non_trade_time:
                self.trade_client.cancel_all_orders()
                if current_time < start:
                    logger.log(f'Waiting for market to open at {start}.')
                elif current_time > end:
                    next_day = current_day + timedelta(days=1)
                    next_start = self.schedule.loc[next_day]['start']
                    logger.log(f'Waiting for market to open at {next_start}.')
                self.handle_non_trade_time = False
            return False
        self.handle_non_trade_time = True
        return True

    def trade(self) -> None:
        """
        Starts the trading process by creating a trading environment and
        executing actions from the model.
        """
        logger.info('Initiating trading process.')
        self.trade_client.check_connection()
        model = self.model

        while True:
            if self._check_time():
                break

        observation = self.trade_market_env.reset()
        while True:
            if self._check_time():
                action = model(observation)
                observation, reward, done, info = (
                    self.trade_market_env.step(action))
                if done:
                    break
        return None

    def place_orders(self, actions: np.ndarray, *args, **kwargs):
        """
        Places orders based on the actions generated by the model. Child
        classes should implement this method to place orders.

        Args:
        ------
            actions (np.ndarray):
                The actions generated by the model. The actions are the
                notional value of trades for each asset. i.e. acttion =
                +100 means buy $100 worth of the corresponding asset if
                base currency is USD.
            *args:
                Variable length argument list.
            **kwargs:
                Arbitrary keyword arguments.
        """
        raise NotImplementedError
