from .parameters import Parameters

__all__ = []

for attr in dir(Parameters):
    if not attr.startswith("_"):
        value = getattr(Parameters, attr)

        # evitar métodos
        if not callable(value):
            globals()[attr] = value
            __all__.append(attr)

