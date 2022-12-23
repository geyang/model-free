from collections import deque

import numpy as np


class Reservoir(deque):
    def __repr__(self):
        return f"{self.__class__.__name__}(len={self.maxlen}, count={self.count}, items={self.list})"

    def __init__(self, maxlen, seed):
        from ml_logger import logger

        self.rng = np.random.RandomState(seed)
        self.count = 0
        self.maxlen_ = maxlen
        # This is a fucking hack and needs to be fixed
        super().__init__(maxlen=1500)

    def add(self, item, *args):
        if self.count < self.maxlen:
            super().append([item, *args] if args else item)
        else:
            r = self.rng.randint(0, self.count)
            if r < self.maxlen:
                self[r] = [item, *args] if args else item
        self.count += 1

    @property
    def list(self):
        return list(self)


if __name__ == '__main__':
    r = Reservoir(1000, 42)
    for i in range(100_000):
        r.add(i, f"entry {i}")
    print(r)
