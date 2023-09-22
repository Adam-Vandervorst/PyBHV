from bhv.np import NumPyPacked64BHV
from bhv.symbolic import Var, SymbolicBHV

p, q, r = Var("p"), Var("q"), Var("r")

byte_lookup = [
    SymbolicBHV.ZERO,
    ~(p | q | r),
    r & ~(p | q),
    ~(p | q),
    q & ~(p | r),
    ~(p | r),
    ~p & (q ^ r),
    ~(p | (q & r)),
    q & r & ~p,
    ~(p | (q ^ r)),
    r & ~p,
    ~p & (r | ~q),
    q & ~p,
    ~p & (q | ~r),
    ~p & (q | r),
    ~p,
    p & ~(q | r),
    ~(q | r),
    ~q & (p ^ r),
    ~(q | (p & r)),
    ~r & (p ^ q),
    ~(r | (p & q)),
    (p & q & r) ^ p ^ q ^ r,
    ~((p | q) & (r | (p & q))),
    (p ^ q) & (p ^ r),
    ~((p & q) | (q ^ r)),
    (r | (p & q)) ^ p,
    ~((r & (p ^ q)) ^ q),
    (q | (p & r)) ^ p,
    ~((q & (p ^ r)) ^ r),
    (q | r) ^ p,
    ~(p & (q | r)),
    p & r & ~q,
    ~(q | (p ^ r)),
    r & ~q,
    ~q & (r | ~p),
    (p ^ q) & (q ^ r),
    ~((p & q) | (p ^ r)),
    (r | (p & q)) ^ q,
    ~((r & (p ^ q)) ^ p),
    r & (p ^ q),
    ~((p & q) | (p ^ q ^ r)),
    r & ~(p & q),
    ~(((p ^ q) & (q ^ r)) ^ p),
    (q | r) & (p ^ q),
    (q | ~r) ^ p,
    (q | (p ^ r)) ^ p,
    ~(p & (q | ~r)),
    p & ~q,
    ~q & (p | ~r),
    ~q & (p | r),
    ~q,
    (p | (q & r)) ^ q,
    ~((p & (q ^ r)) ^ r),
    (p | r) ^ q,
    ~(q & (p | r)),
    (p | r) & (p ^ q),
    (p | ~r) ^ q,
    (p | (q ^ r)) ^ q,
    ~(q & (p | ~r)),
    p ^ q,
    ~(p | r) | (p ^ q),
    (r & ~p) | (p ^ q),
    ~(p & q),
    p & q & ~r,
    ~(r | (p ^ q)),
    (p ^ r) & (q ^ r),
    ~((p & r) | (p ^ q)),
    q & ~r,
    ~r & (q | ~p),
    (q | (p & r)) ^ r,
    ~((q & (p ^ r)) ^ p),
    q & (p ^ r),
    ~((p & r) | (p ^ q ^ r)),
    (q | r) & (p ^ r),
    (r | ~q) ^ p,
    q & ~(p & r),
    ~(((p ^ r) & (q ^ r)) ^ p),
    (r | (p ^ q)) ^ p,
    ~(p & (r | ~q)),
    p & ~r,
    ~r & (p | ~q),
    (p | (q & r)) ^ r,
    ~((p & (q ^ r)) ^ q),
    ~r & (p | q),
    ~r,
    (p | q) ^ r,
    ~(r & (p | q)),
    (p | q) & (p ^ r),
    (p | ~q) ^ r,
    p ^ r,
    ~(p | q) | (p ^ r),
    (p | (q ^ r)) ^ r,
    ~(r & (p | ~q)),
    (q & ~p) | (p ^ r),
    ~(p & r),
    p & (q ^ r),
    ~((q & r) | (p ^ q ^ r)),
    (p | r) & (q ^ r),
    (r | ~p) ^ q,
    (p | q) & (q ^ r),
    (q | ~p) ^ r,
    q ^ r,
    ~(p | q) | (q ^ r),
    (p | q | r) ^ p ^ q ^ r,
    ~(p ^ q ^ r),
    (p & q) ^ r,
    ~((p | q) & (p ^ q ^ r)),
    (p & r) ^ q,
    ~((p | r) & (p ^ q ^ r)),
    (q & ~p) | (q ^ r),
    ~p | (q ^ r),
    p & ~(q & r),
    ~(((p ^ q) | (p ^ r)) ^ p),
    (r | (p ^ q)) ^ q,
    ~(q & (r | ~p)),
    (q | (p ^ r)) ^ r,
    ~(r & (q | ~p)),
    (p & ~q) | (q ^ r),
    ~(q & r),
    (q & r) ^ p,
    ~((q | r) & (p ^ q ^ r)),
    (p & ~q) | (p ^ r),
    ~q | (p ^ r),
    (p & ~r) | (p ^ q),
    ~r | (p ^ q),
    (p ^ q) | (p ^ r),
    ~(p & q & r),
    p & q & r,
    ~((p ^ q) | (p ^ r)),
    r & ~(p ^ q),
    ~(p ^ q) & (r | ~p),
    q & ~(p ^ r),
    ~(p ^ r) & (q | ~p),
    (q | r) & (p ^ q ^ r),
    ~((q & r) ^ p),
    q & r,
    (q | r | ~p) ^ q ^ r,
    r & (q | ~p),
    ~((q | (p ^ r)) ^ r),
    q & (r | ~p),
    ~((r | (p ^ q)) ^ q),
    ((p ^ q) | (p ^ r)) ^ p,
    (q & r) | ~p,
    p & ~(q ^ r),
    ~(q ^ r) & (p | ~q),
    (p | r) & (p ^ q ^ r),
    ~((p & r) ^ q),
    (p | q) & (p ^ q ^ r),
    ~((p & q) ^ r),
    p ^ q ^ r,
    ~(p | q) | (p ^ q ^ r),
    (p | q | r) ^ q ^ r,
    ~(q ^ r),
    (p & ~q) ^ r,
    ~((p | q) & (q ^ r)),
    (p & ~r) ^ q,
    ~((p | r) & (q ^ r)),
    (q & r) | (p ^ q ^ r),
    ~(p & (q ^ r)),
    p & r,
    (p | r | ~q) ^ p ^ r,
    r & (p | ~q),
    ~((p | (q ^ r)) ^ r),
    (p | q | r) ^ p ^ r,
    ~(p ^ r),
    (q & ~p) ^ r,
    ~((p | q) & (p ^ r)),
    r & (p | q),
    ~((p | q) ^ r),
    r,
    r | ~(p | q),
    (p & (q ^ r)) ^ q,
    ~((p | (q & r)) ^ r),
    r | (q & ~p),
    r | ~p,
    p & (r | ~q),
    ~((r | (p ^ q)) ^ p),
    ((p ^ r) & (q ^ r)) ^ p,
    (p & r) | ~q,
    (q & ~r) ^ p,
    ~((q | r) & (p ^ r)),
    (p & r) | (p ^ q ^ r),
    ~(q & (p ^ r)),
    (q & (p ^ r)) ^ p,
    ~((q | (p & r)) ^ r),
    r | (p & ~q),
    r | ~q,
    (p & r) | (p ^ q),
    ~((p ^ r) & (q ^ r)),
    r | (p ^ q),
    ~(p & q & ~r),
    p & q,
    (p | q | ~r) ^ p ^ q,
    (p | q | r) ^ p ^ q,
    ~(p ^ q),
    q & (p | ~r),
    ~((p | (q ^ r)) ^ q),
    (r & ~p) ^ q,
    ~((p | r) & (p ^ q)),
    q & (p | r),
    ~((p | r) ^ q),
    (p & (q ^ r)) ^ r,
    ~((p | (q & r)) ^ q),
    q,
    q | ~(p | r),
    q | (r & ~p),
    q | ~p,
    p & (q | ~r),
    ~((q | (p ^ r)) ^ p),
    (r & ~q) ^ p,
    ~((q | r) & (p ^ q)),
    ((p ^ q) & (q ^ r)) ^ p,
    (p & q) | ~r,
    (p & q) | (p ^ q ^ r),
    ~(r & (p ^ q)),
    (r & (p ^ q)) ^ p,
    ~((r | (p & q)) ^ q),
    (p & q) | (p ^ r),
    ~((p ^ q) & (q ^ r)),
    q | (p & ~r),
    q | ~r,
    q | (p ^ r),
    ~(p & r & ~q),
    p & (q | r),
    ~((q | r) ^ p),
    (q & (p ^ r)) ^ r,
    ~((q | (p & r)) ^ p),
    (r & (p ^ q)) ^ q,
    ~((r | (p & q)) ^ p),
    (p & q) | (q ^ r),
    ~((p ^ q) & (p ^ r)),
    (p | q) & (r | (p & q)),
    (p & q) | (~r ^ p ^ q),
    r | (p & q),
    r | ~(p ^ q),
    q | (p & r),
    q | ~(p ^ r),
    q | r,
    q | r | ~p,
    p,
    p | ~(q | r),
    p | (r & ~q),
    p | ~q,
    p | (q & ~r),
    p | ~r,
    p | (q ^ r),
    ~(q & r & ~p),
    p | (q & r),
    p | ~(q ^ r),
    p | r,
    p | r | ~q,
    p | q,
    p | q | ~r,
    p | q | r,
    SymbolicBHV.ONE
]

x, y, z = SymbolicBHV.nrand(3)


i = 0
for op, f in enumerate(byte_lookup):
    if sum(f.truth_assignments([p, q, r])) == 4:
        # print(int('{:08b}'.format(op)[::-1], 2), f.show())
        print(op, end=", ")
        i += 1

# for i, f in enumerate(byte_lookup):
#
#     print("case " + str(i) + ": target[i] = " + f.substitute({'p': Var("x[i]"), 'q': Var("y[i]"), 'r': Var("z[i]")}).show() + "; break;")


quit()
wa = 0
sy = 0
x, y, z = NumPyPacked64BHV.nrand(3)

# for i, rule in enumerate(byte_lookup):
#     # print(rule.size())
#     mask = [b == '1' for b in bin(i)[2:].rjust(8, "0")][::-1]
#     rule_ = SymbolicBHV.synth([p, q, r], mask).simplify()
#     print(rule.show())
#     print(rule_.show())
#     print()
#     assert rule.execute(vars=dict(p=x, q=y, r=z), bhv=NumPyPacked64BHV) == \
#            rule_.execute(vars=dict(p=x, q=y, r=z), bhv=NumPyPacked64BHV)
#     wa += rule.size()
#     sy += rule_.size()
#
# print(wa/256, sy/256)

# i = 42
# j = ~42 & 255
# print(byte_lookup[i].execute(vars=dict(p=x, q=y, r=z), bhv=NumPyPacked64BHV) == ~byte_lookup[j].execute(vars=dict(p=x, q=y, r=z), bhv=NumPyPacked64BHV))

# i = 42
# j = ~42 & 255
# print(byte_lookup[i].execute(vars=dict(p=x, q=y, r=z), bhv=NumPyPacked64BHV) == ~byte_lookup[j].execute(vars=dict(p=x, q=y, r=z), bhv=NumPyPacked64BHV))

def reverse_Bits(n, no_of_bits):
    result = 0
    for i in range(no_of_bits):
        result <<= 1
        result |= n & 1
        n >>= 1
    return result

i = 174
j = reverse_Bits(i, 8) & 255
print(j)
print(byte_lookup[i].show(), byte_lookup[j].show())
print(byte_lookup[i].execute(vars=dict(p=x, q=y, r=z), bhv=NumPyPacked64BHV) == ~byte_lookup[j].execute(vars=dict(p=x, q=y, r=z), bhv=NumPyPacked64BHV))
