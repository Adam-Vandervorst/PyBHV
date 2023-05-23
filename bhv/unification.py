from .symbolic import Var, Symbolic
from typing import Optional

Knowledge = dict[str, Symbolic]


# Unify t1 with t2 using/improving the passed knowledge.
def unify(t1: Symbolic, t2: Symbolic, knowledge: Optional[Knowledge] = None) -> Optional[Knowledge]:
    if knowledge is None:
        knowledge = {}

    if isinstance(t1, Var) and isinstance(t2, Var) and t1.name == t2.name:
        return knowledge  # does global variable equality
    elif isinstance(t2, Var):
        return unify_variable(t2.name, t1, knowledge)
    elif isinstance(t1, Var):
        return unify_variable(t1.name, t2, knowledge)
    else:
        if t1.constant() != t2.constant():
            return None
        else:
            cs1 = t1.children()
            cs2 = t2.children()
            if len(cs1) != len(cs2):
                return None
            else:
                k = knowledge

                for l, r in zip(cs1, cs2):
                    k = unify(l, r, k)
                    if k is None:
                        return None
                return k


def unify_variable(v: str, t: Symbolic, knowledge: Knowledge) -> Optional[Knowledge]:
    if v in knowledge:
        return unify(knowledge[v], t, knowledge)
    elif isinstance(t, Var) and t.name in knowledge:
        return unify(Var(v), knowledge[t.name], knowledge)
    elif occurs_check(v, t, knowledge):
        return None
    else:
        return {**knowledge, v: t}


def occurs_check(v: str, t, knowledge: Knowledge):
    if isinstance(t, Var) and t.name == v:
        return True
    elif isinstance(t, Var) and t.name in knowledge:
        return occurs_check(v, knowledge[t.name], knowledge)
    else:
        return any(occurs_check(v, arg, knowledge) for arg in t.children())
