from .base import Broker, OrderResult


def get_broker(name: str, state_dir: str = "./state") -> Broker:
    if name == "robinhood":
        from .robinhood import RobinhoodBroker

        return RobinhoodBroker(state_dir=state_dir)
    if name == "paper":
        from .paper import PaperBroker

        return PaperBroker(state_dir=state_dir)
    raise ValueError(f"Unknown broker {name!r}")


__all__ = ["Broker", "OrderResult", "get_broker"]
