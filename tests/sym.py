from bhv.abstract import AbstractBHV
from bhv.symbolic import Var


if __name__ == '__main__':
    print(AbstractBHV._majority9_simplified(*[Var(f"X{i}") for i in range(9, 0, -1)]).graphviz())
