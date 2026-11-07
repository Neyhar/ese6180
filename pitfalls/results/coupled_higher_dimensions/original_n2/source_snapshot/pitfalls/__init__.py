"""History and action chunking on original and coupled hard examples."""
__all__ = ["challenging_pair"]


def challenging_pair(mu=.2):
    """Lazy compatibility export; the CLI configures NumPy before importing it."""
    from .original.system import challenging_pair as construct
    return construct(mu)
