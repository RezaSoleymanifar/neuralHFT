from abc import abstractmethod, ABC
from typing import Optional

from neural.wrapper.base import (MarginAccountMetaDataWrapper,
                                 ConsoleTearsheetRenderWrapper)

from neural.wrapper.action import (
    MinTradeSizeActionWrapper, IntegerAssetQuantityActionWrapper,
    PositionCloseActionWrapper, InitialMarginActionWrapper,
    ExcessMarginActionWrapper, ShortingActionWrapper,
    EquityBasedUniformActionInterpreter, ActionClipperWrapper)

from neural.wrapper.observation import (ObservationStackerWrapper,
                                        ObservationBufferWrapper,
                                        FlattenToNUmpyObservationWrapper,
                                        ObservationNormalizerWrapper)

from neural.wrapper.reward import (RewardNormalizerWrapper,
                                   RewardGeneratorWrapper,
                                   LiabilityInterstRewardWrapper)
from neural.utils.base import RunningStatistics


class AbstractPipe(ABC):
    """
    Abstract class for environment pipes, which add extended
    functionality to an existing environment by applying wrappers
    successively. A pipe is a stack of wrappers applied in a
    non-conflicting way. Use wrappers to customize the base market env,
    manipulate actions and observations, impose trading logic, etc.
    according to your specific needs. Wrappers are intantiated every
    time the pipe method is called. If you need to restore state of some
    wrappers, you can make that state a constructor argument of both
    wrapper class and and the pipe and set the argument passed to
    wrapper equal to state of wrapper. If both satate are immutable, the
    values will be synchronized. Pipes can be combined to create more
    complex pipes.
    """

    @abstractmethod
    def pipe(self, env):
        """
        Abstract method for piping an environment. Wrappers
        are added successively akin to layers in PyTorch.
        """

        raise NotImplementedError


class RewardPipe(AbstractPipe):

    def __init__(self,
                 reward_statistics: Optional[RunningStatistics] = None,
                 track_statistics=True) -> None:

        self.reward_statistics = reward_statistics
        self.track_statistics = track_statistics

        self.reward_generator = RewardGeneratorWrapper
        self.reward_normalizer = RewardNormalizerWrapper

    def pipe(self, env):
        env = self.reward_generator(env)
        env = self.reward_normalizer(env,
                                     reward_statistics=self.reward_statistics,
                                     track_statistics=self.track_statistics)


class ObservationPipe(AbstractPipe):

    def __init__(self,
                 buffer_size: int = 10,
                 stack_size: int = None,
                 observation_statistics: Optional[RunningStatistics] = None,
                 track_statistics: bool = True) -> None:

        self.buffer_size = buffer_size
        self.stack_size = stack_size
        self.observation_statistics = observation_statistics
        self.track_statistics = track_statistics

        self.flatten = FlattenToNUmpyObservationWrapper
        self.buffer = ObservationBufferWrapper
        self.stacker = ObservationStackerWrapper
        self.normalize_observation = ObservationNormalizerWrapper

    def pipe(self, env):
        env = self.flatten(env)
        env = self.buffer(env, buffer_size=self.buffer_size)
        env = self.stacker(env, stack_size=self.stack_size)

        env = self.normalize_observation(
            env,
            observation_statistics=self.observation_statistics,
            track_statistics=self.track_statistics)

        return env

class ActionPipe(AbstractPipe):
    def __init__(self) -> None:
        pass
    def pipe(self, env):
        pass
    
class MarginAccountPipe(AbstractPipe):
    """
    A pipe for margin account environments. The pipe adds the trading
    logics of a margin account to the base market environment.
    """

    def __init__(self,
                 trade_equity_ratio: float = 0.02,
                 min_trade: float = 1,
                 integer: bool = False,
                 buffer_size: int = 10,
                 stack_size: int = None,
                 excess_margin_ratio_threshold: float = 0.1,
                 verbosity: int = 20,
                 observation_statistics: Optional[RunningStatistics] = None,
                 reward_statistics: Optional[RunningStatistics] = None,
                 track_statistics: bool = True) -> None:

        self.trade_ratio = trade_equity_ratio

        self.min_trade = min_trade
        self.integer = integer

        self.buffer_size = buffer_size
        self.stack_size = stack_size
        self.observation_statistics = observation_statistics
        self.reward_statistics = reward_statistics
        self.track_statistics = track_statistics

        self.excess_margin_ratio_threshold = excess_margin_ratio_threshold
        self.verbosity = verbosity

        self.metadata_wrapper = MarginAccountMetaDataWrapper
        self.render = ConsoleTearsheetRenderWrapper

        self.min_trade = MinTradeSizeActionWrapper
        self.integer_sizing = IntegerAssetQuantityActionWrapper
        self.position_close = PositionCloseActionWrapper

        self.initial_margin = InitialMarginActionWrapper
        self.excess_margin = ExcessMarginActionWrapper
        self.shorting = ShortingActionWrapper
        self.action_interpreter = EquityBasedUniformActionInterpreter
        self.clip = ActionClipperWrapper

        self.observation_pipe = ObservationPipe

        self.reward_pipe = RewardPipe

        return None

    def pipe(self, env):
        """
        Applies a stack of market wrappers successively to an environment.
        Wrappers are addedd successively akin to layers in PyTorch. state of
        wrappers are seved to an attribute of the Pipe class so that they can
        be restored later.

        Args:
        - env (AbstractMarketEnv): the environment to be wrapped.

        Returns:
        - env (gym.Env): the wrapped environment.
        """

        env = self.metadata_wrapper(env)
        env = self.render(env, verbosity=self.verbosity)

        env = self.observation_pipe(
            buffer_size=self.buffer_size,
            stack_size=self.stack_size,
            observation_statistics=self.observation_statistics,
            track_statistics=self.track_statistics).pipe(env)

        # action wrappers
        env = self.min_trade(env, min_action=self.min_trade)
        env = self.integer_sizing(env, self.integer)
        env = self.position_close(env)

        env = self.initial_margin(env)
        env = self.excess_margin(
            env,
            excess_margin_ratio_threshold=self.excess_margin_ratio_threshold)
        env = self.shorting(env)

        env = self.action_interpreter(env, trade_ratio=self.trade_ratio)

        env = self.reward_pipe(reward_statistics=self.reward_statistics,
                               track_statistics=self.track_statistics).pipe(env)

        return env
