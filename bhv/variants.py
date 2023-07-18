from typing import Generic, TypeVar, Type
from copy import deepcopy
from .abstract import *

HV = TypeVar("HV", bound=BooleanAlgBHV, covariant=True)


class LinearBHVModel(Generic[HV]):
    def __init__(self, bhv: Type[HV]):
        self.bhv = bhv
        self.used = IdSet()
    
    def spawn(self, one: 'HV.ONE', zero: 'HV.ZERO') -> (HV, HV):
        assert one not in self.used and zero not in self.used
        self.used.add(one); self.used.add(zero)
        # I O  ->  P N
        # 1 0  ->  1 0  |  0 1
        r = self.bhv.select_rand(one, zero)
        return (r, ~r)

    def xor(self, x: HV, y: HV) -> (HV, HV, HV):
        assert x not in self.used and y not in self.used
        self.used.add(x); self.used.add(y)
        # L R  ->  X L R
        # 0 0  ->  0 0 0
        # 0 1  ->  1 0 0
        # 1 0  ->  1 0 0
        # 1 1  ->  0 1 1
        b = x & y
        return (x ^ y, b, deepcopy(b))

    def thresholds3(self, x: 'HV', y: 'HV', z: 'HV') -> ('HV', 'HV', 'HV'):
        assert x not in self.used and y not in self.used and z not in self.used
        self.used.add(x); self.used.add(y); self.used.add(z)
        # x y z  ->  + M -
        # x y z  ->  + M -
        # 0 0 0  ->  0 0 0

        # 0 0 1  ->  0 0 1
        # 0 1 0  ->  0 0 1
        # 1 0 0  ->  0 0 1

        # 0 1 1  ->  0 1 1
        # 1 0 1  ->  0 1 1
        # 1 1 0  ->  0 1 1

        # 1 1 1  ->  1 1 1
        return (x & y & z, self.bhv.majority([x, y, z]), x | y | z)

    def and_or(self, x: 'HV', y: 'HV') -> ('HV', 'HV'):
        assert x not in self.used and y not in self.used
        self.used.add(x); self.used.add(y)
        # x y  ->  & |
        # 0 0  ->  0 0
        # 0 1  ->  0 1
        # 1 0  ->  0 1
        # 1 1  ->  1 1
        return (x & y, x | y)

    def switch(self, cond: 'HV', left: 'HV', right: 'HV') -> ('HV', 'HV', 'HV'):
        assert cond not in self.used and left not in self.used and right not in self.used
        self.used.add(cond); self.used.add(left); self.used.add(right)
        # C L R  ->  C P N
        # 0 0 0  ->  0 0 0
        # 0 0 1  ->  0 0 1
        # 0 1 0  ->  0 1 0
        # 0 1 1  ->  0 1 1

        # 1 0 0  ->  1 0 0
        # 1 0 1  ->  1 1 0
        # 1 1 0  ->  1 0 1
        # 1 1 1  ->  1 1 1

        pos = self.bhv.select(cond, left, right)
        neg = self.bhv.select(cond, right, left)
        return (deepcopy(cond), pos, neg)

    def invert(self, x: 'HV', one: 'HV.ONE', zero: 'HV.ZERO') -> ('HV', 'HV', 'HV'):
        # P I O  ->  N A B
        # 0 1 0  ->  1 0 0
        # 1 1 0  ->  0 1 1
        (self_1, self_2, inverted) = self.switch(x, one, zero)
        return (inverted, self_1, self_2)

    def permute(self, x: 'HV', permutation_id: int) -> 'HV':
        assert x not in self.used
        self.used.add(x)
        return self.bhv.permute(permutation_id)


class Tank:
    def __init__(self, capacity: float, fill=None, check=True, record=False):
        self.capacity = capacity
        self.level = capacity if fill is None else fill
        self.check = check
        self.record = record
        self.updates = []

    def update(self, amount: float):
        if self.check:
            if amount < 0:
                assert self.level >= amount, "not enough to drain"
            else:
                assert amount <= self.capacity, "too much to fill"
        if self.record:
            self.updates.append(amount)
        self.level += amount

    def historical_levels(self):
        assert self.record, "history not recorded, please pass record=True"
        level = self.level
        record = [level]
        for update in reversed(self.updates):
            level -= update
            record.append(level)
        record.reverse()
        return record


class AdiabaticBHVModel:
    def __init__(self, tank: Tank, bhv: Type[HV]):
        self.tank = tank
        self.bhv = bhv
        self.used = IdSet()

    def get_one(self) -> 'HV.ONE':
        self.tank.update(-DIMENSION)
        return deepcopy(self.bhv.ONE)

    def get_zero(self) -> 'HV.ZERO':
        self.tank.update(0)
        return deepcopy(self.bhv.ZERO)

    def spawn(self) -> ('HV', 'HV'):
        # Note: maybe there's a more efficient way to generate random vectors in this context
        # I O  ->  P N
        # 1 0  ->  1 0  |  0 1
        r = self.bhv.rand()
        self.tank.update(-DIMENSION)
        return (r, ~r)

    def xor(self, x: 'HV', y: 'HV') -> 'HV':
        assert x not in self.used and y not in self.used
        self.used.add(x); self.used.add(y)
        # L R  ->  X
        # 0 0  ->  0
        # 0 1  ->  1
        # 1 0  ->  1
        # 1 1  ->  0
        b = (x & y).active()
        self.tank.update(2*b)
        return x ^ y

    def majority3(self, x: 'HV', y: 'HV', z: 'HV') -> 'HV':
        assert x not in self.used and y not in self.used and z not in self.used
        self.used.add(x); self.used.add(y); self.used.add(z)
        # x y z  ->  M
        # 0 0 0  ->  0

        # 0 0 1  ->  0
        # 0 1 0  ->  0
        # 1 0 0  ->  0

        # 0 1 1  ->  1
        # 1 0 1  ->  1
        # 1 1 0  ->  1

        # 1 1 1  ->  1
        self.tank.update((x & y & z).active() + (x | y | z).active())
        return self.bhv.majority([x, y, z])

    def and_or(self, x: 'HV', y: 'HV') -> ('HV', 'HV'):
        assert x not in self.used and y not in self.used
        self.used.add(x); self.used.add(y)
        # x y  ->  & |
        # 0 0  ->  0 0
        # 0 1  ->  0 1
        # 1 0  ->  0 1
        # 1 1  ->  1 1
        self.tank.update(0)
        return (x & y, x | y)

    def switch(self, cond: 'HV', left: 'HV', right: 'HV') -> ('HV', 'HV', 'HV'):
        assert cond not in self.used and left not in self.used and right not in self.used
        self.used.add(cond); self.used.add(left); self.used.add(right)
        # C L R  ->  C P N
        # 0 0 0  ->  0 0 0
        # 0 0 1  ->  0 0 1
        # 0 1 0  ->  0 1 0
        # 0 1 1  ->  0 1 1

        # 1 0 0  ->  1 0 0
        # 1 0 1  ->  1 1 0
        # 1 1 0  ->  1 0 1
        # 1 1 1  ->  1 1 1

        pos = cond.select(left, right)
        neg = cond.select(right, left)
        self.tank.update(0)
        return (deepcopy(cond), pos, neg)

    def invert(self, x: 'HV') -> 'HV':
        assert x not in self.used
        self.used.add(x)
        # P I O  ->  N A B
        # 0 1 0  ->  1 0 0
        # 1 1 0  ->  0 1 1
        self.tank.update(2*(x.active() - DIMENSION//2))
        return ~x

    def permute(self, permutation_id: int) -> 'HV':
        self.tank.update(0)
        return self.bhv.permute(permutation_id)

