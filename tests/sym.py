from bhv.abstract import AbstractBHV
from bhv.symbolic import Var
from string import ascii_uppercase


def short_name(i: int, letters=ascii_uppercase) -> str:
    n = len(letters)
    return letters[i%n] + str(i//n)


if __name__ == '__main__':
    AbstractBHV._majority_simplified([Var(short_name(i)) for i in range(98, -1, -1)]).graphviz(structural=False)
