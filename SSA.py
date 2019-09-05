import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd
from time import time

# Modified class from https://www.kaggle.com/jdarcy/introducing-ssa-for-time-series-decomposition
class SSA(object):
    __supported_types = (pd.Series, np.ndarray, list)

    def __init__(self, tseries, L, save_mem=True, verbose=True, presolved=False, components=None, sigmas=None,
                 center_tseries=False, calculate_wcorr=False):
        """
        Decomposes the given time series with a singular-spectrum analysis. Assumes the values of the time series are
        recorded at equal intervals.

        Parameters
        ----------
        tseries : The original time series, in the form of a Pandas Series, NumPy array or list. 
        L : The window length. Must be an integer 2 <= L <= N/2, where N is the length of the time series.
        save_mem : Conserve memory by not retaining the elementary matrices. Recommended for long time series with
            thousands of values. Defaults to True.
        verbose : Some logs if True
        presolved : If True, presolved components and sigmas will be saved instead of computing it from scratch.
        components : presolved components, use it if presolved=True, in the form of a Pandas Series or DataFrame,
            NumPy array or list.
        sigmas : presolved sigmas, use it if presolved=True, in the form of a Pandas Series or DataFrame,
            NumPy array or list.
        center_tseries : If True, time series at self.orig_TS will be centered before applying the method.
        calculate_wcorr : Wcorr will be calculated if True.

        Note: Even if an NumPy array or list is used for the initial time series, all time series returned will be
        in the form of a Pandas Series or DataFrame object.
        """

        # Tedious type-checking for the initial time series
        if not isinstance(tseries, self.__supported_types):
            raise TypeError("Unsupported time series object. Try Pandas Series, NumPy array or list.")

        if presolved and ((components is None) or (sigmas is None)):
            raise ValueError("presolved=True, but components or sigmas are not provided.")

        self.components_importances = None

        # Checks to save us from ourselves
        self.N = len(tseries)
        if not 2 <= L <= self.N / 2:
            raise ValueError("The window length must be in the interval [2, N/2].")

        self.L = L
        if center_tseries:
            self.mean = np.mean(tseries)
            self.orig_TS = pd.Series(tseries - self.mean)
        else:
            self.orig_TS = pd.Series(tseries)
        self.K = self.N - self.L + 1

        if not presolved:
            # Embed the time series in a trajectory matrix
            tic = time()
            self.X = np.array([self.orig_TS.values[i:L + i] for i in range(0, self.K)]).T
            tac = time()

            if verbose:
                print('trajectory matrix - OK')
                print(tac - tic, 'seconds\n')

            # Decompose the trajectory matrix
            tic = time()
            self.d = np.linalg.matrix_rank(self.X)
            tac = time()

            if verbose:
                print('matrix rank - OK:', self.d)
                print(tac - tic, 'seconds\n')

            tic = time()
            self.U, self.Sigma, VT = randomized_svd(self.X, self.d, random_state=42)
            self.calc_components_importances()
            tac = time()

            if verbose:
                print('randomized SVD - OK')
                print(tac - tic, 'seconds\n')

            self.TS_comps = np.zeros((self.N, self.d))

            tic = time()
            if not save_mem:
                # Construct and save all the elementary matrices
                self.X_elem = np.array([self.Sigma[i] * np.outer(self.U[:, i], VT[i, :]) for i in range(self.d)])

                # Diagonally average the elementary matrices, store them as columns in array.
                for i in range(self.d):
                    X_rev = self.X_elem[i, ::-1]
                    self.TS_comps[:, i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0] + 1, X_rev.shape[1])]

                self.V = VT.T
            else:
                # Reconstruct the elementary matrices without storing them
                for i in range(self.d):
                    X_elem = self.Sigma[i] * np.outer(self.U[:, i], VT[i, :])
                    X_rev = X_elem[::-1]
                    self.TS_comps[:, i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0] + 1, X_rev.shape[1])]

                self.X_elem = "Re-run with save_mem=False to retain the elementary matrices."

                # The V array may also be very large under these circumstances, so we won't keep it.
                self.V = "Re-run with save_mem=False to retain the V matrix."
            tac = time()

            if verbose:
                print('components - OK')
                print(tac - tic, 'seconds')
        else:
            self.Sigma = np.squeeze(np.array(sigmas))
            self.TS_comps = np.array(components)
            self.d = self.TS_comps.shape[1]
            self.calc_components_importances()

            if verbose:
                print('presolved results saved')

        # Calculate the w-correlation matrix.
        if calculate_wcorr:
            tic = time()
            self.calc_wcorr()
            tac = time()

            if verbose:
                print('wcorr - OK')
                print(tac - tic, 'seconds\n')


    def components_to_df(self, n=0):
        """
        Returns all the time series components in a single Pandas DataFrame object.
        """
        if n > 0:
            n = min(n, self.d)
        else:
            n = self.d

        # Create list of columns - call them F0, F1, F2, ...
        cols = ["F{}".format(i) for i in range(n)]
        return pd.DataFrame(self.TS_comps[:, :n], columns=cols, index=self.orig_TS.index)

    def reconstruct(self, indices):
        """
        Reconstructs the time series from its elementary components, using the given indices. Returns a Pandas Series
        object with the reconstructed time series.

        Parameters
        ----------
        indices: An integer, list of integers or slice(n,m) object, representing the elementary components to sum.
        """
        if isinstance(indices, int): indices = [indices]

        ts_vals = self.TS_comps[:, indices].sum(axis=1)
        return pd.Series(ts_vals, index=self.orig_TS.index)

    def calc_wcorr(self):
        """
        Calculates the w-correlation matrix for the time series.
        """

        # Calculate the weights
        w = np.array(list(np.arange(self.L) + 1) + [self.L] * (self.K - self.L - 1) + list(np.arange(self.L) + 1)[::-1])

        def w_inner(F_i, F_j):
            return w.dot(F_i * F_j)

        # Calculated weighted norms, ||F_i||_w, then invert.
        F_wnorms = np.array([w_inner(self.TS_comps[:, i], self.TS_comps[:, i]) for i in range(self.d)])
        F_wnorms = F_wnorms ** -0.5

        # Calculate Wcorr.
        self.Wcorr = np.identity(self.d)
        for i in range(self.d):
            for j in range(i + 1, self.d):
                self.Wcorr[i, j] = abs(w_inner(self.TS_comps[:, i], self.TS_comps[:, j]) * F_wnorms[i] * F_wnorms[j])
                self.Wcorr[j, i] = self.Wcorr[i, j]

    def plot_wcorr(self, min=None, max=None):
        """
        Plots the w-correlation matrix for the decomposed time series.
        """
        if min is None:
            min = 0
        if max is None:
            max = self.d

        if self.Wcorr is None:
            self.calc_wcorr()

        ax = plt.imshow(self.Wcorr)
        plt.xlabel(r"$\tilde{F}_i$")
        plt.ylabel(r"$\tilde{F}_j$")
        plt.colorbar(ax.colorbar, fraction=0.045)
        ax.colorbar.set_label("$W_{i,j}$")
        plt.clim(0, 1)

        # For plotting purposes:
        if max == self.d:
            max_rnge = self.d - 1
        else:
            max_rnge = max

        plt.xlim(min - 0.5, max_rnge + 0.5)
        plt.ylim(max_rnge + 0.5, min - 0.5)

    def calc_components_importances(self):
        """
        Calculates importances of components: (singular values)**2 / sum(singular values)**2)
        """
        sqr_sigmas = self.Sigma**2
        self.components_importances = sqr_sigmas / np.sum(sqr_sigmas)
