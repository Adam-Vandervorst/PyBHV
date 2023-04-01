from bhv.abstract import AbstractBHV
from bhv.symbolic import Var


if __name__ == '__main__':
    print(AbstractBHV._majority7_simplified(*[Var(f"X{i}") for i in range(7, 0, -1)]).graphviz())
