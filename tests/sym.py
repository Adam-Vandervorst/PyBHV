from bhv.abstract import AbstractBHV
from bhv.symbolic import Var, SymbolicBHV, Rand


def logic_majority_plot():
    print("digraph {")
    all_vars = [Var.shortname(i) for i in range(16, -1, -1)]
    SymbolicBHV._logic_majority(all_vars).graphviz(structural=False, compact_select=True)
    print("}")


def logic_window_plot():
    print("digraph {")
    all_vars = [Var.shortname(i) for i in range(16, -1, -1)]
    SymbolicBHV._ite_window(all_vars, 4, 10).simplify().optimal_sharing().graphviz(structural=False, compact_select=True)
    print("}")


def active_fraction():
    rfs = [1/2, 1/4, 3/4, 5/8, 3/8, 9/16, 11/16, 31/32, 61/128, 123/256]

    for rf in rfs:
        res = SymbolicBHV.synth_af(rf)
        print(res.show())
        print(*[r.nodename() for r in res.preorder() if not isinstance(r, Rand)])
        assert rf == res.expected_active_fraction()


def mock(T, **fields):
    return type(f"Mocked{T.__name__}", (T,), fields)


if __name__ == '__main__':
    # print(SymbolicBHV.random(.25).select((Var("X") & ~Var("Y")), (Var("X") | ~Var("Y")))
    #       .show(impl="SymbolicBHV.", symbolic_var=True))

    # print(mock(SymbolicBHV, majority=vars(AbstractBHV)['majority']).majority([Var("X"), Var("Y")])
    #       .show(random_id=True))

    logic_majority_plot()
    # logic_window_plot()
