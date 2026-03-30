from echobench.adapters.base import BaseAdapter, EncoderAdapter


def get_adapter(name, **kwargs):
    """Factory to instantiate a registered adapter by name."""
    if name == "echojepa":
        from echobench.adapters.echojepa import EchoJEPAAdapter

        return EchoJEPAAdapter(**kwargs)
    elif name == "videomae":
        from echobench.adapters.videomae import VideoMAEAdapter

        return VideoMAEAdapter(**kwargs)
    elif name == "echoprime":
        from echobench.adapters.echoprime import EchoPrimeAdapter

        return EchoPrimeAdapter(**kwargs)
    elif name == "panecho":
        from echobench.adapters.panecho import PanEchoAdapter

        return PanEchoAdapter(**kwargs)
    elif name == "echofm":
        from echobench.adapters.echofm import EchoFMAdapter

        return EchoFMAdapter(**kwargs)
    else:
        raise ValueError(
            f"Unknown adapter: {name}. "
            "Available: echojepa, videomae, echoprime, panecho, echofm"
        )


__all__ = ["EncoderAdapter", "BaseAdapter", "get_adapter"]
