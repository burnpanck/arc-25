import abc
import hashlib
import itertools
import random
import typing
from types import MappingProxyType

import attrs
import cbor2

from ..dataset import Challenge, IOPair, ReasonedSolution
from ..dsl.types import AnyImage
from ..lib.compat import Self


@attrs.frozen
class SynthSpec:
    # --- limits on each individual canvas
    max_cells: int | None = None
    max_side_length: int = 30
    # --- limits on the challenge as a whole
    max_io_pairs: int | None = None
    max_images: int | None = None
    max_tests: int | None = None
    # the total number of cells over all inputs and outputs
    max_total_cells: int | None = None

    def supports(self, **kw: int) -> bool:
        kw.setdefault("side_length", 1)
        kw.setdefault("io_pairs", 1)
        kw.setdefault("tests", 1)
        kw.setdefault("cells", 1)
        kw.setdefault("images", kw["io_pairs"] * 2 + kw["tests"])
        for k, v in kw.items():
            lim = getattr(self, f"max_{k}")
            if lim is not None and lim < v:
                return False
        return True

    def allows(self, chal: Challenge) -> bool:

        def flatten(obj):
            match obj:
                case IOPair():
                    return [obj.input, obj.output]
                case AnyImage():
                    return [obj]
                case tuple() | list():
                    ret = []
                    for o in obj:
                        ret += flatten(o)
                    return ret
            raise TypeError(type(obj).__name__)

        shapes = [i.shape for i in flatten([chal.train, chal.test])]
        kw = dict(
            cells=max(h * w for h, w in shapes),
            side_length=max(max(h, w) for h, w in shapes),
            io_pairs=len(chal.train),
            tests=len(chal.test),
            images=len(chal.train) * 2 + len(chal.test),
            total_cells=sum(h * w for h, w in shapes),
        )
        return self.supports(**kw)


@attrs.frozen
class SynthChallenge:
    challenge: Challenge
    rules: tuple[ReasonedSolution, ...]


@attrs.frozen(kw_only=True)
class ChallengeSynth(abc.ABC):
    # rules and their implementation
    spec: SynthSpec
    rules: tuple[ReasonedSolution, ...]

    @classmethod
    @abc.abstractmethod
    def rule_variations(cls, spec: SynthSpec) -> int:
        """Return the number of different rules this synthesizer can generate
        for a given spec.
        """

    @classmethod
    def make_variation(
        cls,
        spec: SynthSpec,
        idx: int,
        *,
        rgen: random.Random | None = None,
        n_impl_samples: int = 0,
    ) -> Self:
        assert idx < cls.rule_variations(spec)
        return cls._make_variation(spec, idx, rgen=rgen, n_impl_samples=n_impl_samples)

    @classmethod
    @abc.abstractmethod
    def _make_variation(
        cls,
        spec: SynthSpec,
        idx: int,
        *,
        rgen: random.Random | None = None,
        n_impl_samples: int = 0,
    ) -> Self:
        """Generate a specific rule variation, potentially sampling multiple implementations for the rule(s)."""
        pass

    @property
    @abc.abstractmethod
    def inherent_entropy(self) -> float:
        """An estimation of the number of bits of entropy for this specific rule,
        not considering symmetric variations which aren't explicitly mentioned by the rule (i.e. symmetries).
        This is used to scale how many samples are taken from this rule,
        when attempting to sample many sample challenges.
        """

    def sample_challenge(
        self, rgen: random.Random, *, k: int | None = None
    ) -> SynthChallenge | typing.Iterable[SynthChallenge]:
        """If `k` is None, sample a single challenge, returned directly.
        Otherwise, sample `k` samples without replacement.
        """
        if k is not None:
            return itertools.filterfalse(
                lambda s: not self.spec.allows(s.challenge),
                self._sample_challenges(rgen, k=k),
            )
        (ret,) = self._sample_challenges(rgen, k=1)
        return ret

    @abc.abstractmethod
    def _sample_challenges(
        self, rgen: random.Random, *, k: int
    ) -> typing.Iterable[SynthChallenge]:
        """Sample `k` challenges without replacement."""
        pass

    def _hash_key(self) -> typing.Any:
        return None

    def _id_base(self) -> str:
        k = self._hash_key()
        pfx = type(self).__name__
        if k is None:
            return pfx
        h = hashlib.sha256(cbor2.dumps(k)).hexdigest()[:8]
        return f"{pfx}-{h}"

    def _make_chal(self, **kw) -> SynthChallenge:
        id_base = self._id_base()
        ret = Challenge(id=id_base, **kw).canonicalise()
        return SynthChallenge(
            challenge=attrs.evolve(
                ret,
                id=f"{id_base}-{hashlib.sha256(cbor2.dumps(ret._key())).hexdigest()[:8]}",
            ),
            rules=self.rules,
        )
