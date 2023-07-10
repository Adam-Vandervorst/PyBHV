from .abstract import AbstractBHV, DIMENSION


class DistanceGraph:
    @classmethod
    def from_scope(cls, local_dict):
        return cls(*zip(*[(v, k) for k, v in local_dict.items() if isinstance(v, AbstractBHV)]))

    def __init__(self, hvs: 'list[AbstractBHV]', labels: 'list[str]'):
        self.hvs = hvs
        self.labels = labels
        self.adj = [[round(min(v.std_apart(w, invert=True), v.std_apart(w))) if not v.unrelated(w) else None
                     for v in self.hvs]
                    for w in self.hvs]

    def graphviz(self):
        for i, (r, l) in enumerate(zip(self.adj, self.labels)):
            print(f"{i} [label=\"{l}\"];")
            for j, d in enumerate(r):
                if i < j and d is not None:
                    print(f"{i} -- {j} [label=\"{d}\"];")


class Image:
    @classmethod
    def load_pbm(cls, file: 'IO[Any]', bhv: 'AbstractBHV', binary=False):
        if binary:
            header = file.readline().rstrip()
            assert header == b"P4"
            dimension, n = map(int, file.readline().rstrip().split(b" "))
            assert dimension == DIMENSION
            hvs = [bhv.from_bytes(file.read(DIMENSION//8)) for _ in range(n)]
            return cls(hvs)
        else:
            header = file.readline().rstrip()
            assert header == "P1"
            dimension, n = map(int, file.readline().rstrip().split(" "))
            assert dimension == DIMENSION
            hvs = [bhv.from_bitstring(file.readline().rstrip()) for _ in range(n)]
            return cls(hvs)

    def __init__(self, hvs: 'list[AbstractBHV]'):
        self.hvs = hvs

    def pbm(self, file: 'IO[Any]', binary=False):
        if binary:
            file.write(b"P4\n" + bytes(str(DIMENSION), "ascii") + b" " + bytes(str(len(self.hvs)), "ascii") + b"\n")
            for hv in self.hvs:
                file.write(hv.to_bytes())
        else:
            file.write(f"P1\n{DIMENSION} {len(self.hvs)}\n")
            for hv in self.hvs:
                file.write(hv.bitstring())
                file.write('\n')
