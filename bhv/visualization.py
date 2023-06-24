from .abstract import AbstractBHV, DIMENSION


class DistanceGraph:
    @classmethod
    def from_scope(cls, local_dict):
        return cls(*zip(*[(v, k) for k, v in local_dict.items() if isinstance(v, AbstractBHV)]))

    def __init__(self, hvs: 'list[AbstractBHV]', labels: 'list[str]'):
        self.hvs = hvs
        self.labels = labels
        self.adj = [[round(min(v.std_apart(w, invert=True), v.std_apart(w))) if not v.unrelated(w) else 0
                     for v in self.hvs]
                    for w in self.hvs]

    def graphviz(self):
        for i, (r, l) in enumerate(zip(self.adj, self.labels)):
            print(f"{i} [label=\"{l}\"];")
            for j, d in enumerate(r):
                if d != 0 and i < j:
                    print(f"{i} -- {j} [label=\"{d}\"];")


class Image:
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
                for bit in hv.bits():
                    file.write('1' if bit else '0')
                file.write('\n')
