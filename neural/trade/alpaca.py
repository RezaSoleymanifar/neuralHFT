from torch import nn
from typing import Callable

from neural.client.alpaca import AlpacaTradeClient, AbstractDataClient
from neural.data.enums import DatasetMetadata
from neural.trade.base import AbstractTrader
from neural.wrapper.pipe import AbstractPipe



class AlpacaTraderTemplate(AbstractTrader):

    """
    A template implementation of the AbstractTrader class for trading with the Alpaca API.
    """


    def __init__(self,
        trade_client: AlpacaTradeClient,
        model: nn.Module,
        pipe: AbstractPipe,
        dataset_metadata: DatasetMetadata:

        """
        Initializes an AlpacaTraderTemplate object.

        Args:
            client (AlpacaMetaClient): An instance of the Alpaca client to connect to the trading environment.
            model (nn.Module): A PyTorch model used to generate actions for the trader.
            pipe (AbstractPipe): An instance of the data pipe used to feed data to the model.
            dataset_metadata (DatasetMetadata): Metadata for the dataset being used for training and validation.
        """

        super().__init__(
            trade_client,
            model,
            pipe,
            dataset_metadata)
    

    def place_orders(self, action, *args, **kwargs):
        
        """
        Places orders based on the given actions.

        Args:
            action (torch.Tensor): The tensor of actions generated by the model.

        Raises:
            NotImplementedError: This method must be implemented by a subclass.
        """

        raise NotImplementedError



class CustomAlpacaTrader(AlpacaTraderTemplate):

    """
    A custom implementation of the AlpacaTraderTemplate that allows for custom order placing and rule application.

    Args:
        client (AlpacaMetaClient): An instance of the Alpaca client.
        model (nn.Module): The PyTorch model to use for trading.
        pipe (AbstractPipe): The data pipe to use for feeding the model with data.
        dataset_metadata (DatasetMetadata): The metadata for the dataset used for training the model.

    Attributes:
        client (AlpacaMetaClient): An instance of the Alpaca client.
        model (nn.Module): The PyTorch model to use for trading.
        pipe (AbstractPipe): The data pipe to use for feeding the model with data.
        dataset_metadata (DatasetMetadata): The metadata for the dataset used for training the model.

    """

    def __init__(self, 
        client: AlpacaTradeClient, 
        model : nn.Module, 
        pipe: AbstractPipe, 
        dataset_metadata: DatasetMetadata):

        super().__init__(
            client,
            model,
            pipe,
            dataset_metadata)


    def constraints(self, place_orders_func: Callable):


        def customized_place_order(action):

            self.check_trade_constraints()

            place_orders_func(action)

        return customized_place_order
    

    def rules(self, place_orders_func: Callable):

        """
        Decorator factory that returns a new function `custom_place_order`.
        The purpose of `custom_place_order` is to wrap around a given `place_order_func` function and enforce certain 
        constraints and rules before calling it.
        
        The `check_trade_constraints` method is used to check if any trade constraints are violated, and the `apply_rules` 
        method is used to apply additional rules. Once these constraints and rules have been checked and applied, 
        `place_order_func` is called with the `action` argument.
        
        The `custom` method is designed to allow users to customize the `place_orders` method while still enforcing the 
        necessary constraints and rules. It can be used by defining a custom `place_orders` function and then decorating 
        it with `custom` to ensure that the necessary checks are performed before the orders are placed.
        """

        def customized_place_order(action):

            self.check_trade_constraints()

            try:
                self.apply_rules()

            except NotImplementedError:
                pass

            place_orders_func(action)

        return customized_place_order
    

    @rules
    def place_orders(self, action, *args, **kwargs):

        return super().place_orders(action, *args, **kwargs)