from bhv.abstract import AbstractBHV
from bhv.symbolic import Var, SymbolicBHV
from string import ascii_uppercase


def short_name(i: int, letters=ascii_uppercase) -> str:
    n = len(letters)
    return letters[i%n] + str(i//n)


def large_majority_plot():
    all_vars = [Var(short_name(i)) for i in range(98, -1, -1)]
    AbstractBHV.majority(all_vars).graphviz(structural=False, compact_select=True)


if __name__ == '__main__':
    print(SymbolicBHV.random(.25).select((Var("X") & ~Var("Y")), (Var("X") | ~Var("Y")))
          .show(impl="SymbolicBHV.", symbolic_var=True))
