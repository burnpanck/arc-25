from .. import symmetry
from ..symmetry import D4, PermRepBase, PermRepMeta


def test_inverse():
    for op in D4:
        assert op.inverse.inverse == op
        assert op.inverse.combine(op) == D4.e
        assert op.combine(op.inverse) == D4.e


def test_associativity():
    for a in D4:
        for b in D4:
            assert a.combine(b).inverse == b.inverse.combine(a.inverse)
            for c in D4:
                assert a.combine(b).combine(c) == a.combine(b.combine(c))


def test_reprs():
    for rep in vars(symmetry).values():
        if not isinstance(rep, PermRepMeta) or rep is PermRepBase:
            continue
        print(f"{rep.__module__}." + "\n".join(rep.fmt_action_table()))
        for basis in rep:
            image = set()
            assert basis.apply(D4.e) == basis
            for op in D4:
                nxt = basis.apply(op)
                assert op in basis.mapping_to(nxt)
                image.add(nxt)
                for op2 in D4:
                    third = nxt.apply(op2)
                    assert basis.apply(op2.combine(op)) == third
            assert image == set(rep)
