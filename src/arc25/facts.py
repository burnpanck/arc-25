import abc
import typing

import attrs

from .dataset import Challenge


class FactDefinition(abc.ABC):
    @abc.abstractmethod
    def __call__(self, chal: Challenge) -> str | None:
        pass


@attrs.frozen
class PredicateFact(FactDefinition):
    descr: str
    predicate: typing.Callable

    def __call__(self, chal: Challenge) -> str | None:
        p = self.predicate(chal)
        if p:
            return self.descr.format(pred=p)


def single_element_or_none(arg: list | tuple | set):
    if len(arg) == 1:
        (ret,) = arg
        return ret


default_facts = (
    PredicateFact(
        "Inputs of equal shape: {pred}",
        lambda chal: single_element_or_none(
            set(e.input.shape for e in chal.train + chal.test)
        ),
    ),
    PredicateFact(
        "Outputs of equal shape: {pred}",
        lambda chal: single_element_or_none(set(e.output.shape for e in chal.train)),
    ),
    PredicateFact(
        "Output shapes match input shapes",
        lambda chal: all(e.output.shape == e.input.shape for e in chal.train),
    ),
    PredicateFact(
        "Consistent uniform size increase by factor {pred}",
        lambda chal: single_element_or_none(
            set(
                d
                for d, m in (
                    divmod(e.output.shape[i], e.input.shape[i])
                    for i in range(2)
                    for e in chal.train
                )
                if not m and d > 1
            )
        ),
    ),
    PredicateFact(
        "Consistent uniform size decrease by factor {pred}",
        lambda chal: single_element_or_none(
            set(
                d
                for d, m in (
                    divmod(e.input.shape[i], e.output.shape[i])
                    for i in range(2)
                    for e in chal.train
                )
                if not m and d > 1
            )
        ),
    ),
    # TODO: facts about colours
)
