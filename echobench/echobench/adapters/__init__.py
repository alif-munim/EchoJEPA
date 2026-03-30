from echobench.adapters.base import BaseAdapter, EncoderAdapter


def get_adapter(name, **kwargs):
    """Factory to instantiate a registered adapter by name."""
    if name == "echojepa":
        from echobench.adapters.echojepa import EchoJEPAAdapter

        return EchoJEPAAdapter(**kwargs)
    elif name == "videomae":
        from echobench.adapters.videomae import VideoMAEAdapter

        return VideoMAEAdapter(**kwargs)
    else:
        raise ValueError(f"Unknown adapter: {name}. Available: echojepa, videomae")


__all__ = ["EncoderAdapter", "BaseAdapter", "get_adapter"]
