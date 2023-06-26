import io
from time import monotonic_ns

from bhv.abstract import AbstractBHV, DIMENSION
from bhv.native import NativePackedBHV
from bhv.np import NumPyBoolBHV, NumPyPacked8BHV, NumPyPacked64BHV
# from bhv.pytorch import TorchBoolBHV, TorchPackedBHV
from bhv.vanilla import VanillaBHV

from bhv.visualization import Image

all_implementations = [VanillaBHV, NumPyBoolBHV, NumPyPacked8BHV, NumPyPacked64BHV, NativePackedBHV]

N = 5


for impl in all_implementations:
    rs = impl.nrand(N)

    print(impl.__name__)
    print(" binary")
    with io.BytesIO() as f:
        t0 = monotonic_ns()
        Image(rs).pbm(f, binary=True)
        print("  serializing", monotonic_ns() - t0)

        contents = f.getvalue()

    with io.BytesIO(contents) as f:
        t0 = monotonic_ns()
        rs_ = Image.load_pbm(f, impl, binary=True).hvs
        print("  deserializing", monotonic_ns() - t0)

    assert len(rs) == len(rs_)
    for r, r_ in zip(rs, rs_):
        assert r == r_

    print(" string")
    with io.StringIO() as f:
        t0 = monotonic_ns()
        Image(rs).pbm(f, binary=False)
        print("  serializing", monotonic_ns() - t0)

        string = f.getvalue()

    with io.StringIO(string) as f:
        t0 = monotonic_ns()
        rs_ = Image.load_pbm(f, impl, binary=False).hvs
        print("  deserializing", monotonic_ns() - t0)

    assert len(rs) == len(rs_)
    for r, r_ in zip(rs, rs_):
        assert r == r_

