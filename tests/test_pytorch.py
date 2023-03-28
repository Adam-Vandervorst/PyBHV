import unittest

import torch

from bhv.pytorch import TorchPackedBHV, TorchBoolBHV


class TestTorchBoolBHV(unittest.TestCase):
    def test_basic(self):
        self.assertTrue(True)


class TestTorchBHVConversion(unittest.TestCase):
    def test_random(self):
        rp = TorchPackedBHV.rand()
        self.assertTrue(torch.equal(rp.data, rp.unpack().pack().data))
        ru = TorchBoolBHV.rand()
        self.assertTrue(torch.equal(ru.data, ru.pack().unpack().data))

    def test_extrema(self):
        self.assertTrue(torch.equal(TorchPackedBHV.ZERO.unpack().data, TorchBoolBHV.ZERO.data))
        self.assertTrue(torch.equal(TorchPackedBHV.ZERO.data, TorchBoolBHV.ZERO.pack().data))
        self.assertTrue(torch.equal(TorchPackedBHV.ONE.unpack().data, TorchBoolBHV.ONE.data))
        self.assertTrue(torch.equal(TorchPackedBHV.ONE.data, TorchBoolBHV.ONE.pack().data))


if __name__ == '__main__':
    unittest.main()
