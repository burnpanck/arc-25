import attrs
import jax


class AttrsModel:
    """Helper to tell JAX how to handle our classes representing groups of parameters."""

    def __init_subclass__(scls):
        ret = attrs.frozen(kw_only=True, slots=False)(scls)
        assert ret is scls
        jax.tree_util.register_pytree_node_class(ret)

    def tree_flatten(self):
        # Return the flattenable components and the static data (if any)
        children = []  # pytree children
        aux_data = []  # static metadata, if any
        for f in attrs.fields(type(self)):
            v = getattr(self, f.name)
            if f.metadata.get("static"):
                aux_data.append(v)
            else:
                children.append(v)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        kw = {}
        aux_data = list(aux_data)
        children = list(children)
        for f in attrs.fields(cls):
            if f.metadata.get("static"):
                v = aux_data.pop(0)
            else:
                v = children.pop(0)
            kw[f.name] = v
        assert not aux_data
        assert not children
        return cls(**kw)
