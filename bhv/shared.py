try:
    from typing import Self
except ImportError:
    Self = 'AbstractBHV'

DIMENSION = 8192

from itertools import groupby
from functools import partial
from dataclasses import fields, is_dataclass
from base64 import _urlsafe_encode_translation
import hashlib
import binascii
import sys

sys.setrecursionlimit(75_000)


def stable_hash(d, try_cache=False) -> bytes:
    """
    Python's `hash` is not stable. That's a problem if you want to use it for persistence and inter-run consistency.
    This one is hand rolled, so don't fully trust it. In fact, if you see anything suspicious, please let me know.
    :param d: Something you want to hash
    :return: The 16-byte md5 hash of the bytes of all (nested) items.
    """
    if try_cache and hasattr(d, "__stable_hash"):
        return d.__stable_hash

    stable_hash_try_cache = partial(stable_hash, try_cache=try_cache)

    t = bytes(type(d).__name__, 'utf-8')
    t += bytes([2])
    if isinstance(d, str):
        t += bytes(d, 'utf-8')
    elif isinstance(d, float):
        t += bytes(d.hex(), 'ascii')
    elif isinstance(d, list) or isinstance(d, tuple):
        t += b''.join(map(stable_hash_try_cache, d))
    elif isinstance(d, set) or isinstance(d, frozenset):
        t += b''.join(sorted(map(stable_hash_try_cache, d)))
    elif isinstance(d, dict):
        for k, v in d.items():
            t += stable_hash_try_cache(k)
            t += bytes([1])
            t += stable_hash_try_cache(v)
    elif is_dataclass(d):
        for f in fields(d):
            t += bytes(f.name, 'utf-8')
            t += bytes([1])
            t += stable_hash_try_cache(getattr(d, f.name))
    else:
        t += bytes(d)
    res = hashlib.md5(t, usedforsecurity=False).digest()
    if try_cache:
        try:
            d.__stable_hash = res
        except AttributeError:
            pass
    return res


def stable_hashcode(d, version=0, try_cache=False) -> str:
    """
    A thin wrapper around `stable_hash` which returns a string and incorporates a version.
    :param d: Something you want to hash
    :param version: Increase this number if you want to unconditionally break all hashcodes
    :return: The 24 character base64 (with - and _) string hash
    """
    h = stable_hash(d, try_cache)
    padded = h.rjust(18, version.to_bytes(1, "little"))
    base64 = binascii.b2a_base64(padded, newline=False)
    url_safe = base64.translate(_urlsafe_encode_translation)
    return url_safe.decode()


def nbs(i, w):
    k = 1
    for _ in range(w):
        yield i ^ k
        k <<= 1


def binw(i, w):
    return bin(i)[2:].rjust(w, '0')


def to_bitmask(s):
    return [x == '1' for x in s]


def bin_bitmask(m):
    return ''.join("01"[x] for x in m)


def bitconfigs(n):
    return [to_bitmask(binw(i, n)) for i in range(2**n)]


def unique_by_id(xs):
    return list(reversed({id(x): x for x in reversed(xs)}.values()))


def format_multiple(xs, start="", sep="", end="", indent="", aindent="", newline_threshold=40):
    ss = list(map(str, xs))
    maxlen = max(map(len, ss))
    if maxlen >= newline_threshold:
        return start + sep.rstrip(" ").join("\n" + aindent + indent + s for s in ss) + "\n" + aindent + end
    else:
        return start + sep.join(s for s in ss) + end


def format_list(xs, **kwargs):
    return format_multiple(xs, start="[", sep=", ", end="]", **kwargs)
