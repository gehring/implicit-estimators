import dataclasses
import itertools
from typing import Callable, Dict, Generator, Iterable, Optional, Sequence

from experiments.base import GinExperiment, MultiProcessGroup


@dataclasses.dataclass(frozen=True)  # type: ignore
class GridSearchExperiment(GinExperiment):
    hparams: Dict[str, str] = dataclasses.field(default_factory=dict)

    def __init__(self, name, entrypoint, logger, config: str, overrides: Iterable[str], **hparams):
        hparams_overrides = tuple(f"{k} = {v}" for k, v in hparams.items() if k != "seed")

        super().__init__(
            name=name,
            entrypoint=entrypoint,
            logger=logger,
            config=config,
            overrides=tuple(overrides) + hparams_overrides,
            **hparams,
        )
        object.__setattr__(self, "hparams", hparams)

    def __call__(self, *args, logger, **kwargs):
        logger.append("hparams", self.hparams)
        return super().__call__(*args, logger=logger, **kwargs)


@dataclasses.dataclass(frozen=True)  # type: ignore
class GridSearchExperimentGroup(MultiProcessGroup[GridSearchExperiment]):
    entrypoint: Callable
    config: str
    overrides: Sequence[str]
    hparams: Dict[str, Sequence]
    processes: Optional[int] = None

    def experiments(self, *args, **kwargs) -> Generator[GridSearchExperiment, None, None]:
        names, all_values = zip(*self.hparams.items())

        for values in itertools.product(*all_values):
            hparams = dict(zip(names, values))
            name = self.make_name(hparams)
            logger = self.make_logger(name, hparams)
            yield GridSearchExperiment(
                name, self.entrypoint, logger, self.config, self.overrides, **hparams)

    def make_name(self, hparams: Dict):
        pass

    def make_logger(self, name: str, hparams: Dict):
        pass
