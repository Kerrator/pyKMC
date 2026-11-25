"""Event searches: the `EventSearch` facade and the strategies it runs."""

from .strategies import EventSearchStrategy
from .eventsearch import EventSearch, SearchRequest

__all__ = [
    "EventSearchStrategy",
    "EventSearch",
    "SearchRequest",
]
