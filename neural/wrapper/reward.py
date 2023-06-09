"""
reward.py
"""
from typing import Optional
from gym import Env
from abc import ABC, abstractmethod
from typing import Dict
from gym.core import Env
import numpy as np
from gym import (Env, RewardWrapper)

from neural.utils.base import RunningStatistics
from neural.wrapper.base import metadata


@metadata
class RewardGeneratorWrapper(RewardWrapper):
    """
    A wrapper that generates rewards for the environment. By default the
    market env returns None as reward. This wrapper combined with the
    metadata wrapper provide the reward signal which is the change in
    equity from the previous step. Equity is defined as the net value
    owned by the agent in cash and assets. This is sum of all cash and
    assets owned by the agent minus cash and asset debt. E = L + C - S
    where E is equity, L total value of longs, C cash, S total value of
    shorts. Note that cash can be negative if the agent has borrowed
    cash.

    Attributes:
    ----------
        env (gym.Env): 
            The environment to wrap. equity_history
        (list[float]): 
            A list of equity values for each step in the episode.

    Methods:
    -------
        reward(reward: float) -> float:
            Generates the reward signal.
    """

    def __init__(self, env: Env) -> None:
        """
        Initializes the reward generator wrapper.

        Args:
        -------
            env (gym.Env): 
                The environment to wrap.
        """
        super().__init__(env)
        self.equity_history = self.market_metadata_wrapper.equity_history

    def reward(self, reward: float) -> float:
        reward = self.equity_history[-1] - self.equity_history[-2]
        return reward


class RewardNormalizerWrapper(RewardWrapper):
    """
    This wrapper will normalize immediate rewards. This should typically
    be the last wrapper in the reward wrapper stack. This wrapper
    normalizes immediate rewards so that rewards have mean 0 and
    standard deviation 1.

    Usage:
    -------
        env = NormalizeReward(env, epsilon=1e-8, clip_threshold=10)


    Methods:
    -------
        reward(reward: float) -> float
            Normalize the reward.
    """

    def __init__(
        self,
        env: Env,
        epsilon: float = 1e-8,
        clip_threshold: float = np.inf,
    ) -> None:
        """
        This wrapper normalizes immediate rewards so that rewards have
        mean 0 and standard deviation 1.

        Args:
        -------
            env (gym.Env):
                The environment to wrap.
            epsilon (float):
                A small value to avoid division by zero.
            clip_threshold (float):
                The maximum value to clip the normalized reward. This is
                useful to prevent the agent from receiving very large
                rewards.

        Example
        -------
        >>> from neural.meta.env.base import TrainMarketEnv
        >>> from neural.meta.env.wrapper.reward import RewardNormalizerWrapper
        >>> env = TrainMarketEnv(...)
        >>> env = RewardNormalizerWrapper(env)
        """

        super().__init__(env)

        self.epsilon = epsilon
        self.clip_threshold = clip_threshold
        self.reward_statistics = RunningStatistics(
            epsilon=self.epsilon, clip_threshold=self.clip_threshold)

        return None

    def reward(self, reward: float) -> float:
        """
        Normalize the reward. This method should be the last wrapper in
        the reward wrapper stack.

        Args:
        --------
            reward (float): 
                The immediate reward to normalize.

        Returns:
        --------
            float: 
                The normalized reward.
        """
        self.reward_statistics.update(reward)
        normalized_reward = self.reward_statistics.normalize(reward)

        return normalized_reward


@metadata
class LiabilityInterstRewardWrapper(RewardWrapper):
    """
    This wrapper charges an interest rate at the end of the day on the
    liabilities of the agent. The liabilities include borrowed cash, or
    borrowed assets. The interest rate is calculated as a percentage of
    the liability. Apply this wrapper prior to normalization of rewards
    as this substracts the notional value of interest from the reward.
    Applies the interest rate on a daily basis.

    Args:
    -------
        env (gym.Env):
            The environment to wrap.
        interest_rate (float):
            The interest rate to charge on liabilities. This is
            expressed as a percentage of the liability. The interest
            rate is applied daily. The default value is 8% per annum.

    Attributes:
    -----------
        interest_rate (float):
            The interest rate to charge on liabilities. This is
            expressed as a percentage of the liability. The interest
            rate is applied daily. The default value is 8% per annum.

        previous_day (datetime.date):
            The day of the previous step. This is used to determine if
            the day has changed and interest should be charged.

    Methods:
    --------
        reset() -> np.ndarray:
            Reset the environment.
        reward(reward: float) -> float:
            Compute the reward.
        compute_interest() -> float:
            Compute the interest to charge on liabilities.
    """

    def __init__(self, env, interest_rate=0.08):
        super().__init__(env)
        self.interest_rate = interest_rate
        self.previous_day = None

    @property
    def daily_interest_rate(self):
        """
        Compute the daily interest rate.

        Returns:
        --------
            float:
                The daily interest rate.
        """
        return self.interest_rate / 360

    def reset(self) -> np.ndarray[float] | Dict[str, np.ndarray[float]]:
        """
        Reset the environment.

        Returns:
        --------
            observation (np.ndarray):
                The initial observation.
        """
        observation = self.env.reset()
        self.previous_day = self.market_metadata_wrapper.day
        return observation

    def reward(self, reward: float) -> float:
        """
        Generate the reward signal.

        Args:
        --------
            reward (float):
                The reward to modify.
    
        Returns:
        --------
            float:
                The modified reward. Subtracts the interest from the
                reward.
        """
        current_day = self.market_metadata_wrapper.day
        if current_day != self.previous_day:
            interest = self.compute_interest()
            reward -= interest
            self.previous_day = current_day

        return reward

    def compute_interest(self) -> float:
        """
        Compute the interest to charge on liabilities. Liabilities
        include borrowed cash, or borrowed assets. The interest rate is
        applied on a daily basis.

        Returns:
        --------
            float:
                The interest to charge on liabilities.
        """
        cash_debt = abs(min(self.market_metadata_wrapper.cash, 0))

        positions = self.market_metadata_wrapper.positions
        asset_quantitties = self.market_metadata_wrapper.asset_quantities
        asset_debt = sum(
            position for position, quantity in zip(positions, asset_quantitties)
            if quantity < 0)

        debt_interest = (cash_debt + asset_debt) * self.daily_interest_rate

        return debt_interest


class AbstractRewardShaper(RewardWrapper, ABC):
    """
    Fixed reward shaper wrapper. Fixed because the reward shaping uses a fixed
    scale value. If a condition is met then applies a reward shaping
    based on the reward statistics. This is a blueprint class for fixed reward
    shaping wrappers. This class is designed to be subclassed for creating
    custom fixed reward shaping wrappers for market environments.

    Attributes:
    ----------  
        env (gym.Env):
            The environment to wrap.
        use_std (bool or None, optional):
            Whether to use the standard deviation of the rewards. Defaults to
            None.
        use_min (bool or None, optional):
            Whether to use the min/max reward statistics. Defaults to None. if
            use_min = Flase, then with default scale = -1 the shaped reward
            will be -1 * max meaning if reward condition is met the shaped
            reward will be the negative maximum reward.
        reward_statistics (RunningStatistics):
            A running statistics object for tracking reward statistics.

    Properties:
    ----------
        scale (float):
            Shaped reward is either mean + scale * std or scale * min/max.
        
    Args:
    -----
        env (Env): 
            The environment to wrap. 
        use_std (bool or None, optional): 
            Whether to use the standard deviation of the rewards. Defaults to
            None.
        use_min (bool or None, optional): 
            Whether to use the min/max reward statistics. Defaults to None. if
            use_min = Flase, then with default scale = -1 the shaped reward
            will be -1 * max meaning if reward condition is met the shaped
            reward will be the negative maximum reward. 
        scale (float, optional): The scaling factor for the
            shaped reward. Defaults to -1.0 meaning if for example reward
            shaping condition is met and use_std is True, the shaped reward
            will be the mean minus the standard deviation.
        reward_statistics (RunningStatistics, optional):
            A running statistics object for tracking reward statistics.


    Methods:
    -------
        check_condition(*args, **kwargs) -> bool:
            An abstract method for checking whether to apply reward shaping.
        reward(reward: float, *args, **kwargs) -> float:    
            An abstract method for shaping the reward signal.
        shape_reward(reward: float) -> float:
            Calculate the shaped reward based on the input parameters.
        step(action) -> tuple:
            Advances the environment by one step and updates the reward signal,
            if condition is met.
    
    Raises:
    ------
        AssertionError:
            If both `use_min` and `use_std` parameters are set to a non-None
            value, or if both are set to None.
        
    Notes:
    ------
        Reward shaping wrappers are used to modify the reward signal obtained
        by an agent in order to encourage or discourage certain behaviours
        during training. highly useful for pretraining an agent with some
        degrees of freedom in actions. Apply relevant reward shaping wrappers
        to define and restrict unwanted actions. Start with a pipe of wrappers
        that enforce the desired behaviour and later remove the influencing
        wrappers to allow the agent to learn the desired behaviour. if desired
        behavior is a starting point, then in a final step remove the reward
        shaping wrapper and the agent may learn to improve on it.
    """

    def __init__(
        self,
        env: Env,
        use_std: Optional[bool] = None,
        use_min: Optional[bool] = None,
    ) -> None:
        """
        Initializes the abstract fixed reward shaper wrapper.

        Args:
        -----
            env (Env): 
                The environment to wrap. 
            use_std : bool, optional
                A boolean indicating whether to use the reward's
                standard deviation in shaping the reward.
            use_min : bool, optional
                A boolean indicating whether to use the minimum reward
                value in shaping the reward. If False, the maximum
                reward value is used.
        
        Raises:
        ------
            AssertionError:
                If both `use_min` and `use_std` parameters are set to a
        """
        super().__init__(env)
        if use_min is not None and use_std is not None:
            raise AssertionError(
                "Cannot set both use_min and use_std parameters at the same time."
            )

        if use_min is None and use_std is None:
            raise AssertionError(
                "Either use_min or use_std parameter must be set.")

        self.use_std = use_std
        self.use_min = use_min
        self.reward_statistics = RunningStatistics()

    @property
    @abstractmethod
    def scale(self) -> float:
        """
        The scaling factor for the shaped reward. This will determine the
        direction and intensity of the reward shaping. It should be implemented
        by the user. It can be a fixed value or a dynamic value that changes
        based on deviation from desired behavior.

        Returns:
        --------
            float:
                The scaling factor for the shaped reward.
        """
        return self._scale

    @abstractmethod
    def check_condition(self) -> bool:
        """
        An abstract method for checking whether to apply reward shaping. This
        method should be implemented by subclasses to determine whether to
        apply reward shaping to the current step.

        Returns:
        --------
            bool: 
                True if the reward should be shaped, False otherwise.
        """
        raise NotImplementedError

    def shape_reward(self) -> float:
        """
        Calculate the shaped reward based on scale and reward statistics.
        Outpus a scalar that is either mean + scale * std or scale * min/max.

        Returns
        -------
            float
                A float value representing the shaped reward.
        """
        if self.use_min is not None:
            shaped_reward = (self.scale * self.reward_statistics.minimum
                             if self.use_min else self.scale *
                             self.reward_statistics.maximum)

        elif self.use_std is not None:
            shaped_reward = (self.reward_statistics.mean +
                             self.scale * self.reward_statistics.std)

        return shaped_reward

    def reward(self, reward: float) -> float:
        """
        Shapes the reward when check_condition is True.

        Args:
        ------
            reward (float): 
                The original reward.

        Returns:
        --------
            float: 
                The shaped reward.
        """
        if self.check_condition():
            reward = self.shape_reward()
        return reward

    def step(
        self,
        action: np.ndarray[float] | Dict[str, np.ndarray[float]],
    ) -> np.ndarray[float] | Dict[str, np.ndarray[float]]:
        """
        Advances the environment by one step and updates the reward
        signal.

        Args:
        -----
            action (int, Tuple[int], Any): 
                The action taken by the agent.

        Returns:
        --------
            observation (np.ndarray[float] | Dict[str,
            np.ndarray[float]]):
                The observation of the environment.
            reward (float):
                The shaped reward.
            done (bool):
                Whether the episode is done.
            info (dict):
                A dictionary containing additional information about the
                environment.
        """
        observation, reward, done, info = self.env.step(action)
        self.reward_statistics.update(reward)

        if self.check_condition():
            reward = self.reward(reward)

        return observation, reward, done, info


class AbstractFixedRewardShaper(AbstractRewardShaper, ABC):
    """
    Fixed reward shaper wrapper. Fixed because the reward shaping uses a fixed
    scale value provided at constructor. This is a blueprint class for fixed
    reward shaping wrappers. This class is designed to be subclassed for
    creating custom fixed reward shaping wrappers for market environments.
    Output is either mean + scale * std or scale * min/max.

    Args:
    -----
        env (Env): 
            The environment to wrap. 
        use_std (bool or None, optional): 
            Whether to use the standard deviation of the rewards. Defaults to
            None.
        use_min (bool or None, optional): 
            Whether to use the min/max reward statistics. Defaults to None. if
            use_min = Flase, then with default scale = -1 the shaped reward
            will be -1 * max meaning if reward condition is met the shaped
            reward will be the negative maximum reward. 
        scale (float, optional): The scaling factor for the
            shaped reward. Scale is received from the user. Defaults to -1.0
            meaning if for example reward shaping condition is met and use_std
            is True, the shaped reward will be the mean + (-1.0) * std.
        reward_statistics (RunningStatistics, optional):
            A running statistics object for tracking reward statistics.

    Attributes:
    ----------  
        env (gym.Env):
            The environment to wrap.
        use_std (bool or None, optional):
            Whether to use the standard deviation of the rewards. Defaults to
            None.
        use_min (bool or None, optional):
            Whether to use the min/max reward statistics. Defaults to None. if
            use_min = Flase, then with default scale = -1 the shaped reward
            will be -1 * max meaning if reward condition is met the shaped
            reward will be the negative maximum reward.
        _scale (float):
            fixed scale provided by the user. output is either mean + scale *
            std or scale * min/max.
        reward_statistics (RunningStatistics):
            A running statistics object for tracking reward statistics.

    Properties:
    ----------
        scale (float):
            Shaped reward is either mean + scale * std or scale * min/max.


    Methods:
    -------
        check_condition(*args, **kwargs) -> bool:
            An abstract method for checking whether to apply reward shaping.
        shape_reward(reward: float) -> float:
            Calculate the shaped reward based on the input parameters.
        reward(reward: float, *args, **kwargs) -> float:    
            An abstract method for shaping the reward signal.
        step(action) -> tuple:
            Advances the environment by one step and updates the reward signal,
            if condition is met.
    
    Raises:
    ------
        AssertionError:
            If both `use_min` and `use_std` parameters are set to a non-None
            value, or if both are set to None.
        
    Notes:
    ------
        Reward shaping wrappers are used to modify the reward signal obtained
        by an agent in order to encourage or discourage certain behaviours
        during training. highly useful for pretraining an agent with some
        degrees of freedom in actions. Apply relevant reward shaping wrappers
        to define and restrict unwanted actions. Start with a pipe of wrappers
        that enforce the desired behaviour and later remove the influencing
        wrappers to allow the agent to learn the desired behaviour. if desired
        behavior is a starting point, then in a final step remove the reward
        shaping wrapper and the agent may learn to improve on it.
    """

    def __init__(
        self,
        env: Env,
        use_std: Optional[bool] = None,
        use_min: Optional[bool] = None,
        scale: float = -1.0,
    ) -> None:
        """
        Initializes the abstract fixed reward shaper wrapper.

        Args:
        -----
            env (Env): 
                The environment to wrap. 
            use_std : bool, optional
                A boolean indicating whether to use the reward's
                standard deviation in shaping the reward.
            use_min : bool, optional
                A boolean indicating whether to use the minimum reward
                value in shaping the reward. If False, the maximum
                reward value is used.
            scale (float, optional): The scaling factor for the
                shaped reward. Defaults to -1.0. shaped reward is either
                mean + scale * std or scale * min/max.
        
        Raises:
        ------
            AssertionError:
                If both `use_min` and `use_std` parameters are set to a
        """
        super().__init__(env)
        if use_min is not None and use_std is not None:
            raise AssertionError(
                "Cannot set both use_min and use_std parameters at the same time."
            )

        if use_min is None and use_std is None:
            raise AssertionError(
                "Either use_min or use_std parameter must be set.")

        self.use_std = use_std
        self.use_min = use_min
        self._scale = scale
        self.reward_statistics = RunningStatistics()

    @property
    def scale(self) -> float:
        """
        The scaling factor for the shaped reward.

        Returns:
        --------
            float:
                The scaling factor for the shaped reward.
        """
        return self._scale

    @abstractmethod
    def check_condition(self) -> bool:
        """
        An abstract method for checking whether to apply reward shaping. This
        method should be implemented by subclasses to determine whether to
        apply reward shaping to the current step.

        Returns:
        --------
            bool: 
                True if the reward should be shaped, False otherwise.
        """
        raise NotImplementedError


class AbstractDynamicRewardShaper(AbstractRewardShaper, ABC):
    """
    Abstract base class for a dynamic reward shaper wrapper. For positive scale
    and base the shaped reward will be reward = sign(multiplier) * base **
    (deviation_ratio * abs(multiplier)) where deviation ratio is define as
    metric / threshold. If metric > threshold, the deviation ratio will be
    greater than 1. If metric < threshold, the deviation ratio will be zero.
    This allows to dynamically adjust reward based on deviation or its linear
    and/or exponential modification. This class defines the interface for a
    dynamic reward shaper wrapper, which shapes the reward signal of an
    environment based on a dynamically adjusted scale. To create a custom
    dynamic reward shaper, users must inherit from this class and implement the
    abstract methods: `metric`, and `threshold`.

    Attributes: -
        env (Env): 
            The environment to wrap. 
        use_std : bool, optional
            A boolean indicating whether to use the reward's standard deviation
            in shaping the reward. Default is None.
        use_min : bool, optional
            A boolean indicating whether to use the minimum reward value in
            shaping the reward. If False, the maximum reward value is used.
        multiplier : float, optional
            reward = sign(multiplier) * base ** (deviation_ratio *
            abs(multiplier)). Defaults to +1.0. Thus with default multiplier =
            1.0, base = 1.0, the shaped reward will be reward =
            deviation_ratio.
        base (float, optional): 
            The base value used in the exponential scaling adjustment. Defaults
            to 1.0.

    Methods:
    -------
        check_condition() -> bool:
            Abstract method that checks whether the reward should be shaped
            based on the current episode state.
    """

    def __init__(
        self,
        env: Env,
        use_std: bool = None,
        use_min: bool = None,
        multiplier: Optional[float] = 1,
        base: float = 1.0,
    ) -> None:
        """
        Initializes the abstract dynamic reward shaper wrapper.

        Args:
        -----
            env (Env): 
                The environment to wrap. use_std (bool or None,
                optional): Whether to use the standard deviation of the
                rewards. Defaults to None.
            use_min (bool or None, optional): 
                Whether to use the maximum reward. Defaults to None.
            multiplier : float, optional
                reward = sign(multiplier) * base ** (deviation_ratio *
                abs(multiplier)). Defaults to +1.0. Thus with default multiplier =
                1.0, base = 1.0, the shaped reward will be reward =
                deviation_ratio.
            base (float, optional): 
                The base value used in the exponential scaling adjustment. Defaults
                to 1.0.
            
        Raises:
        ------
            AssertionError:
                If base < 1.
        """
        super().__init__(env, use_std=use_std, use_min=use_min)

        if base < 1:
            raise AssertionError("Base must be greater than or equal to 1.")
        self.multiplier = multiplier
        self.base = base

    @property
    @abstractmethod
    def metric(self) -> float:
        """
        Abstract property that defines the metric to measure the deviation from
        expected behavior. deviation_ratio = metric / threshold when metric >
        threshold. deviation_ratio = 0 when metric < threshold.

        Returns:
        --------
            float >= 0: 
                The metric used to adjust the scaling factor.
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def threshold(self) -> float:
        """
        Abstract property that defines the threshold used for shaping
        the reward. If metric exceeds the threshold, the reward will be
        shaped based on deviation of metric above the threshold. If you
        want to shape the reward when metric falls below the threshold,
        use the inverse of metric and threshold.

        Returns:
        --------
            float > 0: 
                The threshold used for shaping the reward.
        """

        raise NotImplementedError

    @property
    def scale(self) -> float:
        """
        Use as a replacement for fixed scale argument. The returned scale from
        this function can be used to adjust the reward signal based on the
        reward statistics. The shaped reward is either mean + scale * std or
        scale * min/max.
        
        scale is computed as sign(multiplier) * base ** (deviation_ratio *
        abs(multiplier)) where deviation_ratio = metric / threshold.

        Returns:
        --------    
            float:
                The scaling factor for the shaped reward.
        """

        scale = (
            np.sign(self.multiplier) *
            np.power(self.base, abs(self.deviation_ratio * self.multiplier)))
        scale = 0 if self.deviation_ratio <= 1 else scale
        return scale

    @property
    def deviation_ratio(self) -> float:
        """
        The ratio of metric to threshold. If metric > threshold, the
        deviation ratio will be greater than 1. If metric < threshold,
        the deviation ratio will be less than 1.

        Returns:
        --------
            float: 
                The ratio of metric to threshold.
        """

        if not self.threshold > 0:
            raise AssertionError(f'Threshold must be greater than 0. '
                                 f'Current value: {self.threshold}')

        if not self.metric >= 0:
            raise AssertionError(f'Metric must be greater than or equal to 0. '
                                 f'Current value: {self.metric}')

        ratio = self.metric / self.threshold
        return ratio if ratio > 1 else 0

    def check_condition(self) -> bool:
        """
        Checks whether the reward should be shaped based on the current
        episode state. By default returns True if metric > threshold. If
        metric < threshold is desired use inverse of metric, and
        threshold.

        Returns:
        --------
            bool:
                True if the reward should be shaped, False otherwise.
        """
        return True if self.deviation_ratio > 1 else False


@metadata
class FixedExcessMarginRatioRewardShaper(AbstractFixedRewardShaper):
    """
    A reward shaping wrapper that penalizes the excess margin ratio
    falling below a given threshold.
    """

    def __init__(
        self,
        env: Env,
        excess_margin_ratio_threshold: float = 0.2,
        use_std: bool = None,
        use_min: bool = None,
        scale: float = -1.0,
    ) -> None:
        super().__init__(env, use_std=use_std, use_min=use_min, scale=scale)

        if excess_margin_ratio_threshold <= 0:
            raise AssertionError(
                'Excess margin ratio threshold must be a positive number.')

        self.excess_margin_ratio_threshold = excess_margin_ratio_threshold

    def check_condition(self) -> bool:
        """
        An abstract method for checking whether to apply reward shaping.
        """
        excess_margin_ratio = self.market_metadata_wrapper.excess_margin_ratio
        if excess_margin_ratio < self.excess_margin_ratio_threshold:
            return True
        else:
            return False


@metadata
class DynamicExcessMarginRewardShaper(AbstractDynamicRewardShaper):
    """
    A reward shaping wrapper that penalizes the excess margin ratio.
    Uses deviation from excess margin ratio threshold to shape the
    reward. If excess margin ratio is smaller than the threshold, the
    reward will be shaped. Incentivizes the agent to maintain excess
    margin ratio above the threshold. Can teach agent to avoid margin
    calls and maintain constant liquidity for unhindered trading.

    Attributes:
    ----------
        env (Env): 
            The environment to wrap. 
        use_std : bool, optional
            A boolean indicating whether to use the reward's standard deviation
            in shaping the reward. Default is None.
        use_min : bool, optional
            A boolean indicating whether to use the minimum reward value in
            shaping the reward. If False, the maximum reward value is used.
        multiplier : float, optional
            reward = sign(multiplier) * base ** (deviation_ratio *
            abs(multiplier)). Defaults to +1.0. Thus with default multiplier =
            1.0, base = 1.0, the shaped reward will be reward =
            deviation_ratio.
        base (float, optional): 
            The base value used in the exponential scaling adjustment. Defaults
            to 1.0.
    """

    def __init__(
        self,
        env: Env,
        use_std: bool = None,
        use_min: bool = None,
        multiplier: Optional[float] = None,
        base: float = 1.0,
    ) -> None:
        super().__init__(env=env,
                         use_std=use_std,
                         use_min=use_min,
                         multiplier=multiplier,
                         base=base)

    @property
    def threshold(self) -> float:
        """
        The threshold for the excess margin ratio. This is inverse
        of the excess margin ratio threshold. If this threshold is
        exceeded by inverse of the excess margin ratio, the reward
        will be shaped.

        Returns:
        --------
            float: 
                The threshold for the excess margin ratio reward
                shaping.
        """
        return 1 / self.excess_margin_ratio_threshold

    @property
    def metric(self) -> float:
        """
        The short ratio.

        Returns:
            float: The short ratio.
        """
        excess_margin_ratio = self.market_metadata_wrapper.excess_margin_ratio
        return 1 / excess_margin_ratio
