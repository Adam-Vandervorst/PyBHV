from bhv.native import NativePackedBHV as BHV

W = 4
H = 5

image0 = [
    [3, 0, 0, 3],
    [0, 2, 2, 0],
    [0, 1, 2, 0],
    [0, 2, 2, 0],
    [3, 0, 0, 3],
]

image1 = [
    [3, 3, 3, 3],
    [3, 1, 1, 3],
    [0, 0, 0, 0],
    [3, 1, 1, 3],
    [3, 3, 3, 3],
]

image2 = [
    [2, 3, 3, 2],
    [3, 2, 2, 3],
    [3, 1, 1, 3],
    [3, 2, 2, 3],
    [2, 3, 3, 2],
]


image3 = [  # corruption of image 1
    [3, 3, 3, 3],
    [3, 1, 1, 3],
    [0, 1, 0, 0],
    [2, 1, 2, 3],
    [3, 3, 3, 3],
]

image4 = [  # corruption of image 2
    [2, 3, 3, 1],
    [1, 2, 2, 3],
    [3, 1, 0, 3],
    [3, 3, 2, 2],
    [0, 2, 3, 0],
]

Lo, Hi = BHV.nrand(2)
levels = [Lo.select_random(Hi, 0), Lo.select_random(Hi, 1/3), Lo.select_random(Hi, 2/3), Lo.select_random(Hi, 1)]

def embed(im):
    return BHV.parity([levels[im[i][j]].roll_words(i).roll_word_bits(j) for i in range(H) for j in range(W)])

abc = BHV.nrand(3)
mem = BHV.majority([embed(im) ^ l for im, l in zip([image0, image1, image2], abc)])

print((mem ^ embed(image3)).closest(abc))
print((mem ^ embed(image4)).closest(abc))
