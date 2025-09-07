import inspect
import random
import typing

from ..sandbox import assert_rule_ok
from ..synth import tiny
from ..synth.base import ChallengeSynth, SynthChallenge


def test_all_synth():
    rgen = random.Random(42)
    for synthmod, spec in [(tiny, tiny.tiny_spec)]:
        for synthcls in vars(synthmod).values():
            if (
                not isinstance(synthcls, type)
                or not issubclass(synthcls, ChallengeSynth)
                or inspect.isabstract(synthcls)
            ):
                continue
            n = synthcls.rule_variations(spec)
            if n <= 10:
                idx_sel = range(n)
            else:
                idx_sel = sorted(set(rgen.choices(range(n), k=5)))
            print(f"Testing {len(idx_sel)}/{n} variations of {synthcls.__name__}")
            for idx in idx_sel:
                synth = synthcls.make_variation(spec, idx)
                for schal in synth.sample_challenge(rgen, k=3):
                    schal = typing.cast(SynthChallenge, schal)
                    print(f"{schal.challenge.id}")
                    for rule in schal.rules:
                        assert_rule_ok(schal.challenge, rule.rule_impl)
