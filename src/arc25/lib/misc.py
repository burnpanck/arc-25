def first_from(*args):
    for arg in args:
        if arg is not None:
            return arg
    return None


def show_dims(dimnames: str, obj) -> str:
    try:
        shape = obj.shape
    except AttributeError:
        shape = obj
    batch = shape[: -len(dimnames)]
    ret = [str(n) for n in batch] + [
        f"{k}={v}" for k, v in zip(dimnames, shape[-len(dimnames) :])
    ]
    ret = ",".join(ret)
    return f"({ret})"
