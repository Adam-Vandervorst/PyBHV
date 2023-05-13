# TurnstileExample
# Implementing the Turnstile state machine with Hypervectors
# The state machine: https://en.wikipedia.org/wiki/Finite-state_machine#Example:_coin-operated_turnstile

from bhv.vanilla import VanillaBHV as BHV, VanillaPermutation as Perm


# states
locked = BHV.rand()
unlocked = BHV.rand()
# input symbols
token = BHV.rand()
push = BHV.rand()
# next state permutation
PNext = Perm.random()
# inverse for querying the next state
QNext = ~PNext

transition = BHV.majority([
    (push ^ locked ^ PNext(locked)),
    (token ^ locked ^ PNext(unlocked)),
    (push ^ unlocked ^ PNext(locked)),
    (token ^ unlocked ^ PNext(unlocked))
])

# note this doesn't exactly give the right state
def next_state(state, input):
    return QNext(transition ^ input ^ state)

# so we make a noisy lookup table
table = [locked, unlocked]
def closest(noisy):
    return min(table, key=noisy.hamming)

# and check if the transition system works as expected
assert closest(next_state(locked, push)) == locked
assert closest(next_state(locked, token)) == unlocked
assert closest(next_state(unlocked, push)) == locked
assert closest(next_state(unlocked, token)) == unlocked
