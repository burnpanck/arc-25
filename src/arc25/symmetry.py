import dataclasses
import enum
from types import MappingProxyType, SimpleNamespace

import numpy as np

from .lib.compat import Self, enum_nonmember, enum_property


class SymOpBase(enum.Enum):
    @property
    def inverse(self) -> Self:
        raise NotImplementedError

    def combine(self, rhs: Self) -> Self:
        """Apply `rhs` first, then `self`"""
        raise NotImplementedError


class D4(SymOpBase):
    """Represents the symmetry operations of the rectangular grid;

    Internally encoded as the combinations of three sequential mirror ops.
    The axis-aligned mirror axes commute, but the mirror axis along the main diagonal does not;
    therefore, order matters. The operations chosen (in order) are:
     1. Flip the x coordinates
     2. Flip the y coordinates
     3. Transpose/exchange x and y
    """

    e = 0b000
    x = 0b001  # flip the x-coordinates
    y = 0b010  # flip the y-coordinates
    i = 0b011
    t = 0b100  # exchange coordinates = flip over the main diagonal
    l = 0b101  # noqa
    r = 0b110  # clock-wise rotation when x->right, y->down (vision coordinate system)
    d = 0b111  # flip over the anti-diagonal

    @property
    def inverse(self) -> Self:
        S = type(self)
        return {S.l: S.r, S.r: S.l}.get(self, self)

    def combine(self, rhs: Self) -> Self:
        """Apply `rhs` first, then `self`"""
        v = rhs.value
        s = self.value
        # swap the order of rhs.transpose and lhs.flip_x/flip_y;
        if v & 0b100 and bool(s & 0b001) != bool(s & 0b010):
            # we need to swap flip_x and flip_y
            s ^= 0b011
        # the flips commute with each other, and all ops obviously commute with themselves,
        # so by ordering all flips to the right and all transposes to the left
        # we can now simply apply atom^2 = 1 individually to each atomic op.
        v ^= s
        return type(self)(v)


@dataclasses.dataclass(frozen=True)
class PermRepBasisElement:
    symbol: str
    stabiliser: frozenset[D4] = dataclasses.field(repr=False)
    action: dict[D4, str] = dataclasses.field(repr=False)
    mapping_to: dict[str, frozenset[D4]] = dataclasses.field(repr=False)
    canonical_mapping_to: dict[str, D4] = dataclasses.field(repr=False)


class PermRepMeta(enum.EnumMeta):
    def __call__(
        cls,
        value,
        names=None,
        **kw,
    ):
        if names is not None:
            scls = super().__call__(
                value,
                names,
                **kw,
            )
            scls._symbol_to_name = MappingProxyType({v.value.symbol: v for v in scls})
            return scls
        assert isinstance(value, str)
        return cls._symbol_to_name[value]


class PermRepBase(enum.Enum, metaclass=PermRepMeta):
    def __init__(self, symbol: str, stabiliser: set[D4], action: dict[D4, str]):
        stabiliser = set(stabiliser)
        stab = stabiliser | {D4.e}
        for a in stabiliser:
            stab.add(a.inverse)
            for b in stabiliser:
                stab.add(a.combine(b))
        act = {}
        action[D4.e] = symbol
        mapping = {}
        cmapping = {}
        for k, v in action.items():
            ts = set()
            for a in stab:
                # stabiliser must come first!
                a = k.combine(a)
                act[a] = v
                ts.add(a)
            mapping[v] = frozenset(ts)
            cmapping[v] = k
        assert len(act) == len(D4)
        self._value_ = PermRepBasisElement(
            symbol=symbol,
            stabiliser=frozenset(stab),
            action=MappingProxyType(act),
            mapping_to=MappingProxyType(mapping),
            canonical_mapping_to=MappingProxyType(cmapping),
        )

    @enum_property
    def symbol(self):
        return self._value_.symbol

    @enum_property
    def action(self):
        return self._value_.action

    @enum_property
    def stabiliser(self):
        return self._value_.stabiliser

    def __str__(self):
        return f"{self.value.symbol}"

    def __repr__(self):
        return f"<{type(self).__name__}.{self.name}, symbol={self.value.symbol!r}>"

    def apply(self, tfo: D4) -> Self:
        return type(self)(self.action[tfo])

    def mapping_to(self, other: Self) -> frozenset[D4]:
        """Given `self` and `other` find the set of ops such that `self.apply(op) == other`."""
        return self._value_.mapping_to[other.symbol]

    def canonical_mapping_to(self, other: Self) -> D4:
        """Given `self` and `other` find the set of ops such that `self.apply(op) == other`."""
        return self._value_.canonical_mapping_to[other.symbol]

    @enum_nonmember
    @classmethod
    def fmt_action_table(cls):
        ret = [f"{cls.__name__}:"]
        ret.append("     " + " ".join(v.name for v in D4))
        cmap = {v: f"\x1b[{30+k}m" for k, v in enumerate(cls)}

        def tostr(basis: Self) -> str:
            return f"{cmap[basis]}{basis}\x1b[0m"

        for basis in cls:
            ret.append(
                f"{basis.name:4s} " + " ".join(tostr(basis.apply(v)) for v in D4)
            )
        return ret

    @enum_nonmember
    @classmethod
    def make_repr(cls, name, spec: str) -> type[Self]:
        ops = None
        elements = Ellipsis
        for line in spec.split("\n"):
            if not (line := line.strip()):
                continue
            if ops is None:
                ops = tuple(D4[v] for v in line.split())
                elements = {}
                continue
            tok = line.split()
            key, tok = tok[0], tok[1:]
            action, tok = tok[: len(ops)], tok[len(ops) :]
            (stab,) = tok or [""]
            action = {k: v for k, v in zip(ops, action) if v != "."}
            elements[key] = SimpleNamespace(
                name=key,
                symbol=action[D4.e],
                action=action,
                stabiliser=frozenset(D4[v] for v in stab),
            )
            if len(elements) < len(ops):
                continue
            # consistency-check / completion of action table
            emap = {e.symbol: e for e in elements.values()}
            for e in emap.values():
                for k1, v1 in e.action.items():
                    for k2, v2 in e.action.items():
                        if k2 == k1:
                            continue
                        a = k2.combine(k1.inverse)
                        ov2 = emap[v1].action.setdefault(a, v2)
                        assert ov2 == v2, (
                            f"Inconsistency in {name}! {e.name} sais {k1.name}·{e.symbol}={v1} and {k2.name}·{e.symbol}={v2},"
                            f" but {emap[v1].name} says {a.name}·{v1}={ov2}"
                        )
            ret = cls(
                name,
                [
                    (e.name, (e.symbol, e.stabiliser, e.action))
                    for e in elements.values()
                ],
            )
            ops = Ellipsis
            elements = Ellipsis
        return ret


_box_drawing_rep_symbols = "│─ ╱╲  ╞╡╥╨ ╴╵╶╷ ┌┐└┘      ╒╓╕╖╘╙╛╜"
_arrow_rep_symbols = "↔︎↕︎ ⤡⤢ ↺↻ ⟲⟳ →←↑↓ ↘︎↗︎↖︎↙︎ ⇄⇆⇅⇵ ⦨⦩⦪⦫⦬⦭⦮⦯"

TrivialRep = PermRepBase.make_repr(
    "TrivialRep",
    """
  e
o * xyr
""",
)
AxisRep = PermRepBase.make_repr(
    "AxisRep",
    """
  e r
h ↔︎ ↕︎ xy
v ↕︎ ↔︎ xy
""",
)

DiagRep = PermRepBase.make_repr(
    "DiagRep",
    """
  e r
d ⤡ ⤢ dt
t ⤢ ⤡ dt
""",
)

ChiralityRep = PermRepBase.make_repr(
    "ChiralityRep",
    """
    e x
cw  ↻ ↺ rl
ccw ↺ ↻ rl
""",
)

AxialDirRep = PermRepBase.make_repr(
    "AxialDirRep",
    """
  e r i l
d ↓ ← ↑ → x
u ↑ → ↓ ← x
r → ↓ ← ↑ y
l ← ↑ → ↓ y
""",
)

DiagDirRep = PermRepBase.make_repr(
    "DiagDirRep",
    """
   e r i l
rd ↘︎ ↙︎ ↖︎ ↗︎ t
ld ↙︎ ↖︎ ↗︎ ↘︎ d
lu ↖︎ ↗︎ ↘︎ ↙︎ t
ru ↗︎ ↘︎ ↙︎ ↖︎ d
""",
)

InversionRep = PermRepBase.make_repr(
    "InversionRep",
    """
   e r x t
rl ⇄ ⇅ ⇆ ⇵ i
lr ⇆ . . . i
ud ⇅ . . . i
du ⇵ . . . i
""",
)

FullRep = PermRepBase.make_repr(
    "FullRep",
    """
   e x y i t l r d
lu ⦨ ⦩ ⦪ ⦫ ⦯ ⦭ ⦮ ⦬
ld ⦪ . . . . . . .
ru ⦩ . . . . . . .
rd ⦫ . . . . . . .
ul ⦯ . . . . . . .
ur ⦮ . . . . . . .
dl ⦭ . . . . . . .
dr ⦬ . . . . . . .
""",
)


def transform_rep(s: D4, basis: PermRepBase) -> PermRepBase:
    return basis.apply(s)


def transform_rep_idx(s: D4, rep: type[PermRepBase]) -> np.ndarray:
    backmap = {v: k for k, v in enumerate(rep)}
    return np.array([backmap[basis.apply(s)] for basis in rep], dtype=int)


def transform_image(s: D4, image: np.ndarray) -> np.ndarray:
    """Applies the provided symmetry operation to the first two dimensions."""
    transpose = bool(s.value & 0b100)
    flip_y = bool(s.value & 0b010)
    flip_x = bool(s.value & 0b001)
    img = image
    if flip_y:
        img = img[::-1, :]
    if flip_x:
        img = img[:, ::-1]
    if transpose:
        img = img.transpose(1, 0, *range(2, img.ndim))
    return img


def transform_vector(s: D4, vec: np.ndarray) -> np.ndarray:
    """

    `vec` has shape `(..., 2)`, with the last dimension representing `[row, col] = [y, x]`.
    """
    transpose = bool(s.value & 0b100)
    flip_y = bool(s.value & 0b010)
    flip_x = bool(s.value & 0b001)
    vec = np.where([flip_y, flip_x], -vec, vec)
    if transpose:
        vec = vec[..., ::-1]
    return vec


def transform_coordinate(s: D4, co: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """

    `co` and `shape` have shape `(..., 2)`, with the last dimension representing `[row, col] = [y, x]`.
    """
    transpose = bool(s.value & 0b100)
    flip_y = bool(s.value & 0b010)
    flip_x = bool(s.value & 0b001)
    s = shape - 1
    co = np.where([flip_y, flip_x], shape - 1 - co, co)
    if transpose:
        co = co[..., ::-1]
    return co
