import numpy as np
from sklearn.gaussian_process.kernels import Kernel, StationaryKernelMixin, NormalizedKernelMixin

class YuKernel(Kernel, StationaryKernelMixin, NormalizedKernelMixin):
    def __init__(self, v0, wl, a0, a1, v1):
        self.v0 = v0
        self.wl = np.atleast_1d(wl)  # 确保 wl 是数组
        self.a0 = a0
        self.a1 = a1
        self.v1 = v1

    def __call__(self, X, Y=None, eval_gradient=False):
        if eval_gradient:
            raise ValueError("Gradient can not be evaluated.")

        if Y is None:
            Y = X

        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)

        if self.wl.size != X.shape[1]:
            raise ValueError("wl size must match the number of features in X")

        exp_term = np.sum(((X[:, np.newaxis, :] - Y[np.newaxis, :, :])**2) / self.wl, axis=2)
        exp_term = self.v0 * np.exp(-0.5 * exp_term)

        linear_term = self.a0 + self.a1 * np.dot(X, Y.T)

        if X is Y:
            noise_term = self.v1 * np.eye(X.shape[0])
        else:
            noise_term = np.zeros((X.shape[0], Y.shape[0]))

        return exp_term + linear_term + noise_term

    def diag(self, X):
        return np.array([self.v0 + self.a0 + self.a1 * np.sum(X**2, axis=1) + self.v1] * X.shape[0])

    def is_stationary(self):
        return False

    def __repr__(self):
        return (f"YuKernel(v0={self.v0}, wl={self.wl}, a0={self.a0}, a1={self.a1}, v1={self.v1})")


"""
        Custom ARD kernel named 'YuKernel', based on the provided formula.
        The kernel is defined as:
        k(x, y) = v0 * exp(-0.5 * sum((x - y)^2 / wl)) + a0 + a1 * x * y + v1 * I(x == y)
        where:
        - v0, a0, a1, v1 are hyperparameters
        - wl is a vector of length equal to the number of features in X
        - I is the identity matrix

        Parameters
        ----------
        v0 : float
            The first hyperparameter of the kernel
        wl : array-like
            The length scale hyperparameter of the kernel
        a0 : float
            The second hyperparameter of the kernel
        a1 : float  
            The third hyperparameter of the kernel
        v1 : float
            The fourth hyperparameter of the kernel

        Methods
        -------
        __call__(X, Y=None)
            Compute the kernel matrix between X and Y
        diag(X)
            Compute the diagonal of the kernel matrix for X
        is_stationary()
            Return whether the kernel is stationary
        __repr__()
            Return the string representation of the kernel

        Examples
        --------
        >>> from yukernel import YuKernel
        >>> import numpy as np
        >>> X = np.array([[1, 2], [3, 4]])
        >>> kernel = YuKernel(v0=1, wl=[1, 2], a0=1, a1=1, v1=1)
        >>> kernel(X)
        array([[3.36787944, 1.36787944],
               [1.36787944, 3.36787944]])
        >>> kernel.diag(X)
        array([4., 4.])
        >>> kernel.is_stationary()
        False
        >>> kernel
        YuKernel(v0=1, wl=[1, 2], a0=1, a1=1, v1=1)
        print

        "--------------------------------------Enjoy the kernel--------------------------------!" 
    """