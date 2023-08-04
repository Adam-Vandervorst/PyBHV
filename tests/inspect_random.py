from statistics import fmean, stdev

from scipy.stats import kstest, shapiro, anderson, probplot
import matplotlib
matplotlib.use("cairo")
import matplotlib.pyplot as plt

from bhv.native import NativePackedBHV as BHV, DIMENSION


N = 100_000


def test_unbiased_variance():
    rs = BHV.nrand(N)
    afs = [int(r.active()) for r in rs]

    af = fmean(afs)
    sd = stdev(afs)
    mn = min(afs)
    mx = max(afs)
    print(af, sd, mn, mx)
    print(shapiro(afs), kstest(afs, 'norm'), anderson(afs))
    probplot(afs, dist="norm", plot=plt)
    with open(f"rand_{BHV.__name__}.svg", 'wb') as f:
        plt.savefig(f, format="svg")

    with open(f"rand_{BHV.__name__}.bin", "wb") as f:
        for r in rs:
            f.write(r.to_bytes())

def test_unrelated():
    rs = BHV.nrand(int(N**.5))

    afs = [int(r.hamming(r_)) for r in rs for r_ in rs if r is not r_]

    af = fmean(afs)
    sd = stdev(afs)
    mn = min(afs)
    mx = max(afs)
    print(af, sd, mn, mx)
    print(shapiro(afs), kstest(afs, 'norm'), anderson(afs))
    probplot(afs, dist="norm", plot=plt)
    with open(f"hamming_{BHV.__name__}.svg", 'wb') as f:
        plt.savefig(f, format="svg")


test_unbiased_variance()
