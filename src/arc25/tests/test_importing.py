import importlib.util
import pkgutil
import platform


def test_importing():
    todo = set(
        f"arc25.{p}"
        for p in """
        emda2
    """.split()
    )
    skip = set()
    done = set()
    failed = dict()
    while todo:
        name = todo.pop()
        done.add(name)
        if name in skip:
            continue
        spec = importlib.util.find_spec(name)
        if spec and spec.submodule_search_locations is not None:
            for module_info in pkgutil.iter_modules(spec.submodule_search_locations):
                if module_info.name in done:
                    continue
                todo.add(f"{name}.{module_info.name}")
        try:
            importlib.import_module(name)
        except Exception as ex:
            failed[name] = ex
    if failed:
        first_ex = None
        for k in sorted(done):
            ex = failed.get(k)
            if first_ex is None:
                first_ex = ex
            if ex is None:
                print(f"{k}: ", "skipped" if k in skip else "OK")
            else:
                print(f"{k}: {ex!r}")
        raise ImportError(f"Could not import all submodules: {failed}") from first_ex
