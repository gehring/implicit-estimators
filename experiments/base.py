import dataclasses
import multiprocessing
import operator
import secrets
from typing import Callable, ClassVar, Generator, Optional, Protocol, Tuple, TypeVar

import gin


class Logger(Protocol):

    def append(self, key, value):
        raise NotImplementedError

    def push(self, value):
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


@dataclasses.dataclass(frozen=True)  # type: ignore
class Experiment(Protocol):
    name: str
    entrypoint: Callable
    logger: Logger

    SETUP_ARG_NAME: ClassVar[str] = "setup"
    RESULT_ARG_NAME: ClassVar[str] = "result"

    def setup(self, *args, **kwargs):
        pass

    def cleanup(self, *args, logger, **kwargs):
        pass

    def __call__(self, *args, logger, **kwargs):
        return self.entrypoint(*args, logger=logger, **kwargs)

    def run(self, *args, **kwargs):
        if self.SETUP_ARG_NAME in kwargs:
            raise ValueError(
                (f"The keyword argument '{self.SETUP_ARG_NAME}' is reserved for passing any "
                 "values returned by `self.setup()`.")
            )
        if self.RESULT_ARG_NAME in kwargs:
            raise ValueError(
                (f"The keyword argument '{self.RESULT_ARG_NAME}' is reserved for passing any "
                 "values returned by `self.entrypoint()`.")
            )

        setup_output = self.setup(*args, **kwargs)
        if setup_output is not None:
            kwargs[self.SETUP_ARG_NAME] = setup_output

        with self.logger as logger:
            run_output = self.__call__(*args, logger=logger, **kwargs)
            if run_output is not None:
                kwargs[self.RESULT_ARG_NAME] = run_output

            self.cleanup(*args, logger=logger, **kwargs)


@dataclasses.dataclass(frozen=True)  # type: ignore
class GinExperiment(Experiment):
    seed: int = dataclasses.field(default_factory=lambda: secrets.randbits(128))
    config: str = ""
    overrides: Tuple[str, ...] = dataclasses.field(default_factory=tuple)

    OPERATIVE_CONFIG_NAME: ClassVar[str] = "operative_config"

    def parse_config(self, finalize: bool = True):
        gin.parse_config(self.config)
        gin.parse_config(self.overrides + (f"SEED = {self.seed}",))
        if finalize:
            gin.finalize()

    def setup(self, *args, **kwargs):
        self.parse_config(finalize=kwargs.get("finalize", True))

    def cleanup(self, *args, logger, **kwargs):
        logger.append(self.OPERATIVE_CONFIG_NAME, gin.operative_config_str())


E = TypeVar("E", bound=Experiment, covariant=True)


class ExperimentGroup(Experiment, Protocol[E]):

    def __call__(self, *args, logger, **kwargs):
        results = [exp.run(*args, **kwargs) for exp in self.experiments()]
        if any(r is not None for r in results):
            return results
        return None

    def experiments(self, *args, **kwargs) -> Generator[E, None, None]:
        raise NotImplementedError


class MultiProcessGroup(ExperimentGroup[E]):
    processes: Optional[int]

    def __call__(self, *args, logger, **kwargs):
        with multiprocessing.pool.Pool(processes=self.processes) as pool:
            results = pool.imap(operator.methodcaller("run", *args, **kwargs), self.experiments())
        if any(r is not None for r in results):
            return results
        return None
