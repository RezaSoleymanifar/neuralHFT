from abc import ABC
from typing import List
import pickle
import tarfile

import torch
from torch import nn

from neural.client.alpaca import AbstractTradeClient, AbstractDataClient
from neural.data.enums import DatasetMetadata
from neural.data.base import AsyncDataFeeder
from neural.env.base import TradeMarketEnv
from neural.wrapper.pipe import AbstractPipe



class Agent:

    def __init__(
        self, 
        dataset_metadata: DatasetMetadata, 
        pipe: AbstractPipe, 
        model: nn.Module):
        
        self.dataset_metadata = dataset_metadata
        self.pipe = pipe
        self.model = model

    def save(self, filename):

        with tarfile.open(filename, 'w') as tar:
            # Save dataset metadata
            dataset_metadata_bytes = pickle.dumps(self.dataset_metadata)
            dataset_metadata_file = tarfile.TarInfo(name='dataset_metadata')
            dataset_metadata_file.size = len(dataset_metadata_bytes)
            tar.addfile(dataset_metadata_file,
                        fileobj=pickle.BytesIO(dataset_metadata_bytes))

            # Save pipe
            pipe_bytes = pickle.dumps(self.pipe)
            pipe_file = tarfile.TarInfo(name='pipe')
            pipe_file.size = len(pipe_bytes)
            tar.addfile(pipe_file, fileobj=pickle.BytesIO(pipe_bytes))

            # Save model
            with tarfile.open(mode='w', fileobj=tar) as inner_tar:
                inner_tar.addfile(
                    'model', fileobj=torch.save(self.model, 'model'))

    @staticmethod
    def load(filename):
        with tarfile.open(filename, 'r') as tar:
            # Load dataset metadata
            dataset_metadata_file = tar.getmember('dataset_metadata')
            dataset_metadata_bytes = tar.extractfile(
                dataset_metadata_file).read()
            dataset_metadata = pickle.loads(dataset_metadata_bytes)

            # Load pipe
            pipe_file = tar.getmember('pipe')
            pipe_bytes = tar.extractfile(pipe_file).read()
            pipe = pickle.loads(pipe_bytes)

            # Load model
            with tarfile.open(mode='r', fileobj=tar) as inner_tar:
                model_file = inner_tar.getmember('model')
                model_bytes = inner_tar.extractfile(model_file).read()
                model = torch.load(model_bytes)

            return Agent(dataset_metadata, pipe, model)



class AbstractTrader(ABC):

    """
    Abstract base class for defining a trader that can execute orders based on model actions.
    This trader requires a client to connect to a trading environment, a model to generate
    actions, a data pipe to feed data to the model, and metadata for the dataset being used
    to create aggregated data stream matching the training data.
    """

    def __init__(self,
        trade_client: AbstractTradeClient,
        data_clients: List[AbstractDataClient],
        agent: Agent):

        """
        Initializes an AbstractTrader object.

        Args:
            client (AbstractClient): An instance of the client to connect to the trading environment.
            model (nn.Module): A PyTorch model used to generate actions for the trader.
            pipe (AbstractPipe): An instance of the data pipe used to feed data to the model.
            dataset_metadata (DatasetMetadata): Metadata for the dataset being used for training and validation.
        """

        self.trade_client = trade_client
        self.data_clients = data_clients
        self.agent = agent

        self.stream_metadata = self.agent.dataset_metadata.stream
        self.datafeeder = AsyncDataFeeder(self.stream_metadata, self.data_clients)

        return None


    def apply_rules(self, *args, **kwargs):

        """
        Applies trading rules to the trades. Override this method to apply custom rules
        before placing orders. This allows rule based trading to complement the model based
        trading. For example, a rule could be to only buy a stock if it has a positive
        sentiment score. Or execute a techinical analysis strategy whenever a condition is met
        to override the normal behavior of the model.

        Raises:
            NotImplementedError: This method must be implemented by a subclass.
        """

        raise NotImplementedError
    

    def trade(self):

        """
        Starts the trading process by creating a trading environment and executing
        actions from the model.

        Raises:
            NotImplementedError: This method must be implemented by a subclass.
        """

        self.trade_market_env = TradeMarketEnv(trader=self)

        piped_trade_env = self.pipe(self.trade_market_env)
        observation = piped_trade_env.reset()

        while True:

            action = self.model(observation)
            observation, *_ = piped_trade_env.step(action)


    def place_orders(self, actions, *args, **kwargs):
        """
        Takes actions from the model and places relevant orders.

        Args:
            actions (np.ndarray): A 2D numpy array of actions generated by the model.

        Raises:
            NotImplementedError: This method must be implemented by a subclass.
        """
        # Get the list of symbols from the dataset metadata
        symbols = self.dataset_metadata.symbols

        # Loop over the symbols and actions and place orders for each symbol
        for symbol, action in zip(symbols, actions):
            self.trade_client.place_order(symbol, action, *args, **kwargs)
