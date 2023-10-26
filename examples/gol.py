from bhv.visualization import Image
# from bhv.native import NativePackedBHV as BHV
from bhv.np import NumPyPacked64BHV as BHV
from time import sleep, time_ns


init = [
    ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .",
    ". . . . . . . . . . . . . . . . . . . . . . . . . 1 . . . . . . . . . . . . . .",
    ". . . . . . . . . . . . . . . . . . . . . . . 1 . 1 . . . . . . . . . . . . . .",
    ". . . . . . . . . . . . . 1 1 . . . . . . 1 1 . . . . . . . . . . . . 1 1 . . .",
    ". . . . . . . . . . . . 1 . . . 1 . . . . 1 1 . . . . . . . . . . . . 1 1 . . .",
    ". 1 1 . . . . . . . . 1 . . . . . 1 . . . 1 1 . . . . . . . . . . . . . . . . .",
    ". 1 1 . . . . . . . . 1 . . . 1 . 1 1 . . . . 1 . 1 . . . . . . . . . . . . . .",
    ". . . . . . . . . . . 1 . . . . . 1 . . . . . . . 1 . . . . . . . . . . . . . .",
    ". . . . . . . . . . . . 1 . . . 1 . . . . . . . . . . . . . . . . . . . . . . .",
    ". . . . . . . . . . . . . 1 1 . . . . . . . . . . . . . . . . . . . . . . . . .",
    ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .",
    ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .",
    ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .",
    ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .",
    ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .",
    ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ."
]

step1 = [
    ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .",
    ". . . . . . . . . . . . . . . . . . . . . . . . 1 . . . . . . . . . . . . . . .",
    ". . . . . . . . . . . . . . . . . . . . . . 1 . 1 . . . . . . . . . . . . . . .",
    ". . . . . . . . . . . . . 1 . . . . . . . 1 . 1 . . . . . . . . . . . 1 1 . . .",
    ". . . . . . . . . . . . 1 1 . . . . . . 1 . . 1 . . . . . . . . . . . 1 1 . . .",
    ". 1 1 . . . . . . . . 1 1 . . . . 1 1 . . 1 . 1 . . . . . . . . . . . . . . . .",
    ". 1 1 . . . . . . . 1 1 1 . . . . 1 1 . . . 1 . 1 . . . . . . . . . . . . . . .",
    ". . . . . . . . . . . 1 1 . . . . 1 1 . . . . . 1 . . . . . . . . . . . . . . .",
    ". . . . . . . . . . . . 1 1 . . . . . . . . . . . . . . . . . . . . . . . . . .",
    ". . . . . . . . . . . . . 1 . . . . . . . . . . . . . . . . . . . . . . . . . .",
    ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .",
    ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .",
    ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .",
    ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .",
    ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .",
    ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .",
]

step2 = [
    ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .",
    ". . . . . . . . . . . . . . . . . . . . . . . 1 . . . . . . . . . . . . . . . .",
    ". . . . . . . . . . . . . . . . . . . . . . 1 . 1 . . . . . . . . . . . . . . .",
    ". . . . . . . . . . . . 1 1 . . . . . . . 1 . 1 1 . . . . . . . . . . 1 1 . . .",
    ". . . . . . . . . . . 1 . 1 . . . . . . 1 1 . 1 1 . . . . . . . . . . 1 1 . . .",
    ". 1 1 . . . . . . . 1 . . . . . . 1 1 1 . 1 . 1 1 . . . . . . . . . . . . . . .",
    ". 1 1 . . . . . . . 1 . . 1 . . 1 . . 1 . . 1 . 1 . . . . . . . . . . . . . . .",
    ". . . . . . . . . . 1 . . . . . . 1 1 . . . . 1 . . . . . . . . . . . . . . . .",
    ". . . . . . . . . . . 1 . 1 . . . . . . . . . . . . . . . . . . . . . . . . . .",
    ". . . . . . . . . . . . 1 1 . . . . . . . . . . . . . . . . . . . . . . . . . .",
    ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .",
    ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .",
    ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .",
    ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .",
    ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .",
    ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ."
]

step30 = [
    ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .",
    ". . . . . . . . . . . . . . . . . . . . . . . . . 1 . . . . . . . . . . . . . .",
    ". . . . . . . . . . . . . . . . . . . . . . . 1 . 1 . . . . . . . . . . . . . .",
    ". . . . . . . . . . . . . 1 1 . . . . . . 1 1 . . . . . . . . . . . . 1 1 . . .",
    ". . . . . . . . . . . . 1 . . . 1 . . . . 1 1 . . . . . . . . . . . . 1 1 . . .",
    ". 1 1 . . . . . . . . 1 . . . . . 1 . . . 1 1 . . . . . . . . . . . . . . . . .",
    ". 1 1 . . . . . . . . 1 . . . 1 . 1 1 . . . . 1 . 1 . . . . . . . . . . . . . .",
    ". . . . . . . . . . . 1 . . . . . 1 . . . . . . . 1 . . . . . . . . . . . . . .",
    ". . . . . . . . . . . . 1 . . . 1 . . . . . . . . . . . . . . . . . . . . . . .",
    ". . . . . . . . . . . . . 1 1 . . . . . . . . . . . . . . . . . . . . . . . . .",
    ". . . . . . . . . . . . . . . . . . . . . . . . 1 . . . . . . . . . . . . . . .",
    ". . . . . . . . . . . . . . . . . . . . . . . . . 1 1 . . . . . . . . . . . . .",
    ". . . . . . . . . . . . . . . . . . . . . . . . 1 1 . . . . . . . . . . . . . .",
    ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .",
    ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .",
    ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ."
]

W = 64
H = 128


def viz_to_grid(v: list[str]) -> list[list[bool]]:
    return [[c == '1' for c in r.split(' ')] for r in v]


def pad_grid(g: list[list[bool]]) -> list[list[bool]]:
    h = len(g)
    w = len(g[0])
    return [r + [False]*(W - w) for r in g] + [[False]*W for _ in range(H - h)]


def chunk_grid(g: list[list[bool]], w, h, w0=0, h0=0) -> list[list[bool]]:
    return [r[w0:w] for r in g[h0:h]]


def grid_to_viz(g: list[list[bool]]) -> list[str]:
    return [" ".join('1' if b else '.' for b in r) for r in g]


def grid_to_hv(g: list[list[bool]]) -> BHV:
    return BHV.from_bitstream(c for r in g for c in r)


def hv_to_grid(hv: BHV) -> list[list[bool]]:
    bs = list(hv.bits())
    assert len(bs) == H*W
    return [bs[i:i+W] for i in range(0, W*H, W)]


def step(hv: BHV) -> BHV:
    l = hv.roll_word_bits(-1)
    r = hv.roll_word_bits(1)
    b = hv.roll_words(-1)
    t = hv.roll_words(1)

    tl = l.roll_words(1)
    bl = l.roll_words(-1)
    tr = r.roll_words(1)
    br = r.roll_words(-1)

    nbs = [l, r, b, t, tl, bl, tr, br]

    return hv.select(BHV.window(nbs, 2, 3), BHV.window(nbs, 3, 3))


def sanity_check():
    print(len(init), len([c for c in init[0] if c != ' ']))
    for r in grid_to_viz(pad_grid(viz_to_grid(step30))):
        print(r)

    step13_hv = grid_to_hv(pad_grid(viz_to_grid(step30)))
    # print(step13_hv.active())
    assert hv_to_grid(step13_hv) == pad_grid(viz_to_grid(step30))
    print()

    for r in grid_to_viz(hv_to_grid(step13_hv.roll_word_bits(1))):
        print(r)


def run(initial_viz: list[str], generations: int):
    initial = viz_to_grid(initial_viz)
    h = len(initial)
    w = len(initial[0])
    petri_dish_hv = grid_to_hv(pad_grid(initial))
    print(0)
    for r in initial_viz:
        print(r)
    sleep(4)
    for gen in range(1, generations):
        print(gen)
        petri_dish_hv = step(petri_dish_hv)
        for r in grid_to_viz(chunk_grid(hv_to_grid(petri_dish_hv), w, h)):
            print(r)
        sleep(.4)


def export(initial_viz: list[str], generations: int, filename: str):
    init_hv = grid_to_hv(pad_grid(viz_to_grid(initial_viz)))
    petri_dish_history = [init_hv]

    for _ in range(generations):
        petri_dish_history.append(step(petri_dish_history[-1]))

    with open(filename, 'wb') as f:
        Image(petri_dish_history).pbm(f, binary=True)


def benchmark():
    init_hv = grid_to_hv(pad_grid(viz_to_grid(init)))
    step30_hv = grid_to_hv(pad_grid(viz_to_grid(step30)))
    petri_dish_hv = init_hv

    for _ in range(30):
        petri_dish_hv = step(petri_dish_hv)

    print(petri_dish_hv == step30_hv)

    t0 = time_ns()
    for _ in range(1000000):
        petri_dish_hv = step(petri_dish_hv)

    t1 = time_ns()

    print(1000000/((t1 - t0)/1e9))


run(init, 50)
# benchmark()
# export(init, 8191, "../bhv/cnative/gol8192.pbm")
