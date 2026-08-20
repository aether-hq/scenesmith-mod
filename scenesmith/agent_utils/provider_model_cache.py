"""Thread-safe provider-aware lifecycle for heavyweight model instances."""

from __future__ import annotations

import threading

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Generic, Iterator, TypeVar

KeyT = TypeVar("KeyT")
ValueT = TypeVar("ValueT")


@dataclass
class _Entry(Generic[ValueT]):
    value: ValueT
    references: int = 0
    retired: bool = False


class ProviderModelCache(Generic[KeyT, ValueT]):
    """Hold one current provider model and retire it after active users finish."""

    def __init__(
        self,
        *,
        loader: Callable[[KeyT], ValueT],
        releaser: Callable[[KeyT, ValueT], None] | None = None,
    ) -> None:
        self._loader = loader
        self._releaser = releaser or (lambda _key, _value: None)
        self._entries: dict[KeyT, _Entry[ValueT]] = {}
        self._current_key: KeyT | None = None
        self._lock = threading.RLock()

    @property
    def current_key(self) -> KeyT | None:
        with self._lock:
            return self._current_key

    @contextmanager
    def use(self, key: KeyT) -> Iterator[ValueT]:
        release_now: tuple[KeyT, ValueT] | None = None
        with self._lock:
            if key != self._current_key:
                old_key = self._current_key
                if old_key is not None:
                    old_entry = self._entries[old_key]
                    old_entry.retired = True
                    if old_entry.references == 0:
                        release_now = (old_key, old_entry.value)
                        del self._entries[old_key]
                entry = self._entries.get(key)
                if entry is None:
                    entry = _Entry(self._loader(key))
                    self._entries[key] = entry
                entry.retired = False
                self._current_key = key
            entry = self._entries[key]
            entry.references += 1
        if release_now is not None:
            self._releaser(*release_now)
        try:
            yield entry.value
        finally:
            release_after: tuple[KeyT, ValueT] | None = None
            with self._lock:
                entry.references -= 1
                if entry.retired and entry.references == 0:
                    if self._entries.get(key) is entry:
                        del self._entries[key]
                    release_after = (key, entry.value)
            if release_after is not None:
                self._releaser(*release_after)

    def reset(self) -> None:
        """Retire the current model; active leases delay final release."""

        release_now: tuple[KeyT, ValueT] | None = None
        with self._lock:
            key = self._current_key
            self._current_key = None
            if key is not None:
                entry = self._entries[key]
                entry.retired = True
                if entry.references == 0:
                    release_now = (key, entry.value)
                    del self._entries[key]
        if release_now is not None:
            self._releaser(*release_now)
