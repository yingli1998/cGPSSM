"""============================================================================
Abstract dataset attributes.
============================================================================"""
import numpy as np
# import torch
import pdb


# -----------------------------------------------------------------------------
class Dataset:

    def __init__(self, rng, name, is_categorical, Y, t=None, test_split=None, X=None, labels=None, latent_dim=None):

        self.rng = rng
        self.name = name
        self.is_categorical = is_categorical
        self.has_true_X = X is not None
        self.has_t = t is not None
        self.has_labels = labels is not None
        self._latent_dim = latent_dim

        try:
            if is_categorical and labels is None:
                raise ValueError('Labels must be provided for categorical data.')
        except:
            pdb.set_trace()

        self.Y = Y
        self.X = X
        self.t = t
        self.labels = labels

        self.was_split = test_split > 0
        if self.was_split:
            self.Y_ma, self.mask = self.Y_missing(test_split)
        else:
            self.Y_ma = None
            self.F_ma = None
            self.mask = None

    def Y_missing(self, test_split, fill_value=0):
        if not self.was_split:
            raise ValueError('Data has not been split.')
        Y_missing = np.copy(self.Y).astype(float)
        mask = self.rng.binomial(1, test_split, size=self.Y.shape).astype(bool)
        # mask 是 1 的地方应该代表 observation
        Y_missing[~mask] = fill_value
        Y_missing_ = Y_missing
        Y_missing = np.ma.array(Y_missing, mask=mask)
        Y_missing = Y_missing.harden_mask()

        return Y_missing_, mask

    @property
    def Y_mask(self):
        return self.train_mask(self.Y)

    @property
    def Y_without_mask(self):
        return self.test_mask(self.Y)

    def train_mask(self, Y):
        if not self.was_split:
            raise ValueError('Data has not been split.')
        return Y[self.mask]

    def test_mask(self, Y):
        if not self.was_split:
            raise ValueError('Data has not been split.')
        return Y[~self.mask]

    def __str__(self):
        return f"<class 'datasets.Dataset ({self.name})'>"

    def __repr__(self):
        return str(self)

    @property
    def latent_dim(self):
        if self._latent_dim:
            return self._latent_dim
        elif self.has_true_X:
            return self.X.shape[1]
        else:
            return 2