from ._C import VectorAdder as _NativeVectorAdder  # pyright: ignore[reportMissingModuleSource]


class VectorAdder:
    def __init__(self) -> None:
        self._native = _NativeVectorAdder()

    def add(self, left: list[float], right: list[float]) -> list[float]:
        return self._native.add(left, right)
