"""
base.py
"""
import os
from typing import List, Iterable, Any
from collections import deque

import pandas as pd
import numpy as np

import tableprint
from tqdm import tqdm

#==================================Data========================================


class FillDeque:
    """
    A custom deque implementation that fills itself with the first item
    it receives when it's empty until it reaches the specified buffer
    size. After that, it behaves like a regular deque with a fixed
    maximum size.


    Methods:
    --------
        append:
            Appends the item to the deque. If the deque is empty, it
            fills the deque with the first item received until it
            reaches the maximum buffer size.
        __iter__:
            Returns an iterator over the deque.
        __getitem__:    
            Returns a slice of the deque.
        __len__:
            Returns the length of the deque.
        __repr__:
            Returns a string representation of the deque.
        __str__:
            Returns a string representation of the deque.
        clear:
            Removes all elements from the deque buffer.
    
    Attributes:
    -----------
        buffer_size (int): 
            The maximum size of the deque.
        buffer (deque):
            The deque buffer.
    """

    def __init__(self, buffer_size: int  = 10):
        """
        Initializes the FillDeque instance with the specified buffer
        size.

        Args:
            buffer_size (int): The maximum size of the deque.
        """

        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=self.buffer_size)

        return None

    def append(self, item: Any):
        """
        Appends the item to the deque. If the deque is empty, it fills
        the deque with the first item received until it reaches the
        maximum buffer size.

        Args:
        -----
            item (Any): The item to append to the deque.
        """
        if not self.buffer:
            for _ in range(self.buffer_size):
                self.buffer.append(item)
        else:
            self.buffer.append(item)

        return None

    def __iter__(self) -> Iterable[Any]:
        """
        A generator that iterates over the deque.

        Returns:
            generator: A generator that iterates over the deque.
        """
        return iter(self.buffer)

    def __getitem__(self, index) -> Any | List[Any]:
        """
        Returns a slice of the buffer as a list.

        Args:
            index (int, slice): The index or slice to retrieve.

        Returns:
            list: A list of items from the buffer.
        """

        if isinstance(index, int):
            return self.buffer[index]

        elif isinstance(index, slice):
            return list(self.buffer)[index]

        else:
            raise TypeError("Invalid argument type")

    def clear(self) -> None:
        """
        Removes all elements from the deque buffer.
        """
        self.buffer.clear()

        return None


class RunningStatistics:
    """
    A class for computing the running mean and standard deviation of a
    stream of data. Can be used to normalize data to a mean of 0 and
    standard deviation of 1 in an online fashion.

    Implements the Welford online algorithm for computing the standard
    deviation.

    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm

    Args:
    -----
        epsilon (float): A small constant to avoid divide-by-zero errors
        when normalizing data. clip (float): A value to clip normalized
        data to, to prevent outliers from dominating the statistics.

    Attributes:
    -----------
        epsilon (float): 
            A small constant to avoid divide-by-zero errors
            when normalizing data.
        clip_threshold (float):
            A value to clip normalized data to, to prevent outliers
            from dominating the statistics.
        shape (tuple):
            The shape of the data.
        _minimum (float):
            The minimum value of the data.
        _maximum (float):
            The maximum value of the data.
        _mean (float):
            The mean of the data.   
        _std (float):
            The standard deviation of the data.
        M2 (float):
            The sum of the squared differences from the mean.
        count (int):
            The number of data points seen so far.
    
    Properties:
    -----------
        minimum (float):
            Returns the minimum value of the data stored in the
            RunningStatistics object.
        maximum (float):
            Returns the maximum value of the data stored in the
            RunningStatistics object.    
        mean (float):
            Returns the mean of the data stored in the
            RunningStatistics object.    
        std (float):
            Returns the standard deviation of the data stored in the
            RunningStatistics object.
        

    Methods:
    --------
        initialize_statistics:
            Initializes the running statistics with the first data
            point.
        update:
            Updates the running mean and standard deviation with the
            new data.
        normalize:
            Normalizes the data to a mean of 0 and standard deviation
 
         
    Raises:
    -------
        AssertionError: 
            If the clip threshold is less than or equal to 0. 
        AssertionError:
            If the epsilon value is less than or equal to 0.
        
    Example:
    --------
        >>> rms = RunningStatistics() 
        >>> rms.update(array) 
        >>> mean = rms.mean 
        >>> std = rms.std 
        >>> normalized_array = rms.normalize(array)
    """

    def __init__(self, epsilon=1e-8, clip_threshold: float = np.inf):
        """
        Initializes the RunningStatistics object.

        Args:
        -----
            epsilon (float): 
                A small constant to avoid divide-by-zero
                errors when normalizing data.
            clip (float):
                A value to clip normalized data to, to prevent
                outliers from dominating the statistics.

        Raises:
        -------
            AssertionError:
                If the clip threshold is less than or equal to 0.
            AssertionError:
                If the epsilon value is less than or equal to 0.
        """

        if clip_threshold > 0:
            raise AssertionError("clip_threshold must be greater than 0")   
        if epsilon > 0:
            raise AssertionError("epsilon value must be greater than 0")    

        self.epsilon = epsilon
        self.clip_threshold = clip_threshold

        self.shape = None
        self._minimum = None
        self._maximum = None
        self._mean = None
        self._std = None
        self.M2 = None
        self.count = None

        return None

    @property
    def minimum(self) -> float:
        """
        Returns the minimum value of the data stored in the
        RunningStatistics object.

        Raises:
        -------
            AssertionError:
                If there are no data points to compute the minimum.
        
        Returns:
        --------
            float: The minimum value of the data.
        """

        if not self.count:
            raise AssertionError(
                "Must have at least one data point to compute minimum")

        return self._minimum

    @property
    def maximum(self) -> float:
        """
        Returns the max value of the data stored in the
        RunningStatistics object.

        Raises:
        -------
            AssertionError:
                If there are no data points to compute the maximum.
        
        Returns:
        --------
            float: The maximum value of the data.
        """

        if not self.count:
            raise AssertionError(
                "Must have at least one data point to compute maximum") 
        
        return self._minimum

    @property
    def mean(self) -> float:
        """
        Computes and returns the mean of the data stored in the
        RunningStatistics object.

        Raises:
        -------
            AssertionError:
                If there are no data points to compute the mean.
            
        Returns:
        --------
            float: The mean of the data.
        """
        if not self.count:
            raise AssertionError(
                "Must have at least one data point to compute mean")    
        
        return self._mean

    @property
    def std(self):
        """
        Computes and returns the standard deviation of the data stored
        in the RunningStatistics object.
        """
        if not self.count:
            raise AssertionError(
                "Must have at least one data point to compute standard deviation")  
        
        variance = self.M2 / \
            (self.count - 1) if self.count > 1 else np.zeros_like(self.M2)
        self._std = np.sqrt(variance)

        return self._std

    def initialize_statistics(self, array: np.ndarray):
        """
        Initializes the RunningStatistics object with data.

        Args:
        -----
            array (np.ndarray):
                The data to initialize the RunningStatistics
                object with.
            
        Raises:
        -------
            AssertionError:
                If the data is not a numpy array.
        """

        self.shape = array.shape
        self._mean = np.zeros(self.shape)
        self.M2 = np.zeros(self.shape)
        self.count = 0

        self.minimum = np.inf
        self.maximum = -np.inf

        return None

    def update(self, array: np.ndarray):
        """
        Updates the running mean and standard deviation with the new
        data.

        Args:
        -----
            array (np.ndarray):
                The data to update the RunningStatistics
                object with.
            
        Raises:
        -------
            AssertionError:
                If the shape of the data has changed during update.
        """

        if self.shape is None:
            self.initialize_statistics(array)

        if not self.shape == array.shape:
            raise AssertionError(
                'Shape of data has changed during update. '
                f'original shape: {self.shape}, new shape: {array.shape}')
        
        self.count += 1
        delta = array - self._mean
        self._mean += delta / self.count
        delta2 = array - self._mean
        self.M2 += delta * delta2

        self.minimum = np.minimum(self.minimum, array)
        self.maximum = np.maximum(self.maximum, array)

        return None

    def normalize(self, array: np.ndarray) -> np.ndarray:
        """
        Normalizes the data to a mean of 0 and standard deviation of 1.
        Uses running statistics to normalize the data.

        Args:
        -----
            array (np.ndarray):
                The data to normalize.

        Returns:
        --------
            np.ndarray: The normalized data.
        """
        normalized_array = np.clip(
            (array - self.mean) / (self.std + self.epsilon), -self.clip_threshold,
            self.clip_threshold)

        return normalized_array


def validate_path(file_path: str | os.PathLike) -> None:
    """
    Validates a file path by checking if it is a directory and if the
    parent directory exists. 

    Args:
    -----
        file_path (str | os.PathLike):
            The path to the file to be validated.

    Raises:
    -------
        ValueError:
            If the specified path is a directory.
        ValueError:
            If the parent directory of the specified file does not
            exist.
    """

    if os.path.isdir(file_path):
        raise ValueError(
            f"The specified path {file_path} is a directory, not a file.")

    else:
        dir_path = os.path.dirname(file_path)

        if not os.path.isdir(dir_path):
            raise ValueError(
                f'Directory {dir_path} leading to the specified file '
                'does not exist.'
            )

    return None


#============================Visualization=====================================


def tabular_print(entries: List,
                  style='banner',
                  align='left',
                  width=15,
                  header=False) -> None:
    """
    Prints a list of entries in a tabular format. Uses the tableprint
    library to print the table.

    Args:
    -----
        entries (List):
            The list of entries to be printed.
        style (str, optional):
            The style of the table. Defaults to 'banner'.
        align (str, optional):
            The alignment of the table. Defaults to 'left'.
        width (int, optional):
            The width of the table. Defaults to 15.
        header (bool, optional):
            Whether to print the header of the table. Defaults to False.
    """

    if header:
        row = tableprint.header(entries, style=style, align=align, width=width)

    else:
        row = tableprint.row(entries, style=style, align=align, width=width)

    return row


def progress_bar(iterable: Iterable) -> tqdm:
    """
    Creates a progress bar using the tqdm library.

    Args:
    -----
        iterable (Iterable):
            The iterable to create the progress bar for. Progress bar is
            shown based off of the length of the iterable.

    Returns:
    --------
        tqdm: The progress bar object.
    """

    bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} | {elapsed}<{remaining}'
    bar = tqdm(total=iterable, bar_format=bar_format)
    return bar


#=================================Financial====================================


def sharpe_ratio(net_worth_history: List[float], base=0):
    """
    Calculates the Sharpe ratio of a given net worth history list.

    Args:
    -----
        net_worth_history (List[float]):
            The net worth history to calculate the Sharpe ratio for.
        base (float, optional):
            The base value to calculate the Sharpe ratio with. Defaults
            to 0.
        
    Returns:
    --------
        float: The Sharpe ratio of the net worth history.
    """

    hist = pd.Series(net_worth_history)
    returns = hist.pct_change().dropna()
    sharpe_ratio = (returns.mean() - base) / returns.std()

    return sharpe_ratio
