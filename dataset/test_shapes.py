import os
import unittest
import numpy as np


class TestShapes(unittest.TestCase):
    def setUp(self):
        if not hasattr(self, "savedir"):
            self.savedir = os.getcwd()
        os.chdir(self.savedir)
    def tearDown(self):
        os.chdir(self.savedir)
    def general(self, path):
        os.chdir(path)
        labels = np.load("label.npy")
        product_features = np.load("product_features.npy")
        user_features = np.load("user_features.npy")
        user_product = np.load("user_product.npy")
        nuser = user_features.shape[0]
        nprod = product_features.shape[0]
        nlabels = 0
        self.assertEqual(nuser + nlabels, user_product[:,0].max() + 1)
        self.assertEqual(0 + nlabels, user_product[:,0].min())
        self.assertEqual(nprod + nlabels + nuser, user_product[:,1].max() + 1)
        self.assertEqual(0 + nlabels + nuser, user_product[:,1].min())
    def test_train(self):
        self.general("train")
    # def test_val(self):
    #     self.general("val")
    # def test_test(self):
    #     self.general("test")


if __name__=="__main__":
    unittest.main()
