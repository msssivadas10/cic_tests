#!/usr/bin/python3

import numpy as np
import warnings
from typing import Any, Callable, Tuple, Union
from itertools import product, repeat
from collections import namedtuple
from scipy.interpolate import CubicSpline
from scipy.integrate import quad, tplquad
from scipy.special import gamma
from scipy.optimize import newton, curve_fit
from scipy.stats import binned_statistic

# ====================================================
# Exceptions
# ====================================================

class CICError(Exception):
    """
    Error raised by functions and classes related to CIC computations.
    """
    ...

class CatalogError(CICError):
    """
    Error raised by galaxy catalog objects.
    """
    ...

# =====================================================
# Objects
# ===================================================== 

class CartesianCatalog:
    r"""
    A galaxy catalog in cartesian coordinates. The catalog stores the galaxy positions
    and velocity. It also stores some extra attributes such as the mass, magnitude etc.

    Parameters
    ----------
    objx: array_like
        Position array. An ndarray of float (comoving) positions with 3 columns. Its 
        number of rows is taken as the numbe of objects in the catalog. 
    objv: array_like, optional
        Velocity array. An ndarray of same size as the position array. Velocity means the 
        galaxy peculiar velocity in units if the Hubble parameter at that time.
    z: float, optional
        Redshift.
    space: str, optional
        Which space the position is given - redshift (`s`) or real (`r`, default) space.
    **attrs: key-value pairs, optional
        Additional attributes to the catalog. These attributes are of two types, object 
        attributes and catalog attributes. Object attributes are ndarrays with same shape 
        as the positions. Catalog attributes are properties of the catalog and they are 
        of scalars (i.e., int float and str). These should be specified as keyword 
        arguments.
        
    Examples
    --------
    A catalog of 512 objects with random positions and velocities. Extra attributes used 
    are the redshift and object mass (also random) attributes.

    >>> from numpy import random
    >>> x = random.uniform(0., 500., (512, 3)) # uniform dist. positions in 500 unit box
    >>> v = random.uniform(-10., 10., (512, 3))# uniform dist. velocity in [-10, 10]
    >>> m = random.normal(1.e+5, 5., (512, ))  # normally dist. mass 
    >>> # creating the catalog:
    >>> cat = CartesianCatalog(x, v, mass = m, redshift = 0.)
    >>> cat
    <'CartesianCatalog' of 512 objects>

    """
    __slots__ = 'objx', 'objv', 'objattrs', 'z', 'n', 'attrs', 'space', '_cm', 

    def __init__(self, objx: Any, objv: Any = ..., z: float = ..., space: str = "r", **attrs) -> None:
        self.setPosition(objx)
        if objv is not ... :
            self.setVelocity(objv)
        self.z = ...
        if z is not ... :
            self.setRedshift(z)
        if space not in ("s", "r"):
            raise ValueError("space can be either `s` (redshift) or `r` (real)")
        self.space = space

        self._cm = ...

        # attributes:
        self.objattrs, self.attrs = {}, {}
        for key, value in attrs.items():
            if not np.isscalar(value):
                self.setAttr(key, value)
            else:
                self.attrs[key] = value
    
    def setPosition(self, objx: Any) -> None:
        """ 
        Set object positions. This will clear any velocity data if present.
        """
        objx, _fail = np.asarray(objx), False
        if objx.ndim != 2:
            _fail = True
        elif objx.shape[1] != 3:
            _fail = True
        if _fail:
            raise CatalogError("invalid shape for position array")
        self.objx = objx
        self.objv = ...
        self.n    = objx.shape[0] # number of objects
        return

    def setVelocity(self, objv: Any) -> None:
        """ Set object velocity. """
        objv, _fail = np.asarray(objv), False
        if objv.ndim != 2:
            _fail = True
        elif objv.shape[0] != self.n or objv.shape[1] != 3:
            _fail = True
        if _fail:
            raise CatalogError("invalid shape for velocity array")
        self.objv = objv
        return

    def setRedshift(self, z: float) -> None:
        """ Set redshift. """
        if not isinstance(z, (int, float)):
            raise TypeError("z must be a number ('int' or 'float'")
        self.z = z
        return

    def setAttr(self, key: str, value: Any) -> None:
        """ Set an object attribute. """
        value = np.asarray(value)
        if value.shape[0] != self.n:
            raise CatalogError(f"invalid shape for object attribute, {value.shape}")
        self.objattrs[key] = value
        return

    def __repr__(self) -> str:
        """ Return the canonical string representation of the object. """
        return "<'CartesianCatalog' of {} objects>".format(self.n)

    def __getitem__(self, key: str) -> Any:
        """ Get the object attribute. """
        if key in self.objattrs.keys():
            return self.objattrs[key]
        elif key == 'x': # get position
            return self.objx
        elif key == 'v': # get velocity
            return self.objv
        elif key == "z": # get redshift
            return self.z
        raise CatalogError(f"cannot find object attribute `{key}`")

    def __setitem__(self, key: str, value: Any) -> None:
        """ Set the object attribute. """
        if key in self.objattrs.keys():
            self.setAttr(key, value)
        elif key == 'x': # get position
            self.setPosition(value)
        elif key == 'v': # get velocity
            self.setVelocity(value)
        elif key == "z": # get redshift
            self.setRedshift(value)
        raise CatalogError(f"cannot find object attribute `{key}`")

    def real2redshift(self, ) -> None:
        r"""
        Transform the coordinates from real to redshift space. It is done by a plane 
        parallel approximation along the z axis. i.e., by shifting the z-coordinate 
        by a factor corresponding to the z-velocity (in units of Hubble parameter).

        .. math::
            s_{Z} = x_{Z} + v_{Z} (1 + z)

        Notes
        -----
        To convert a real (cartetian) space galaxy catalog to a redshift space catalog 
        using the plane parallel transformation,

        .. math::
            {\bf s} = {\bf x} + ({\bf v} \cdot \hat{\bf l}) \frac{(z + 1)\hat{\bf l}}{H}

        where, :math:`{\bf s}` and :math:`{\bf x}` are the comoving position of the 
        galaxies in redshift and real spaces, respectively. :math:`{\bf l}` is the 
        line-of-sight vector.

        """ 
        if self.space == "s":
            return # already in redshift space
        if self.objv is ... :
            raise CatalogError("no velocity data is available")
        elif self.z is ... :       
            raise CatalogError("no redshift data is available")
        self.objx[:,2] += self.objv[:,2] * (1. + self.z)
        self.space      = "s"
        return

    def redshift2real(self, ) -> None:
        r"""
        Transform the coordinates from redshift to real space.

        TODO: implement redshift-real transform
        """
        if self.space == "r":
            return # already in real space
        raise NotImplementedError("function not implemented")

    def createCountMatrix(self, subdiv: int, boxsize: Union[float, tuple], offset: Union[float, tuple] = 0.) -> None:
        r"""
        Create the count matrix. This will put the galaxies in this catalog on the 
        region specified and divide it into a given number of cells. The count 
        matrix stores the number of galaxies in each cell.

        Parameters
        ----------
        subdiv: int
            Number of subdivisions of the region. If it is :math:`n`, then there will 
            be :math:`n^3` cells.
        boxsize: float, tuple of 3 floats
            Size of the box bounding the region. If it is a number, then a cubical 
            region is assumed. Otherwise, it should be a 3-tuple of the x, y and z 
            lengths of the box. 
        offset: float, tuple of 3 floats, optional
            Offset position of the region. By default, it is set to 0.


        """
        self._cm = CountMatrix(self.objx, subdiv, boxsize, offset)
        return

    def cicProbability(self, bins: int, merge: bool = False, nlow: int = ..., style: str = "lin") -> tuple:
        r"""
        Estimate the count-in-cells probability distribution from the catalog.

        Parameters
        ----------
        bins: int
            Number of bins to use.
        merge: bool, optional
            Wheather to merge bins with cells, less than a lower threshold. This 
            is disabled by default (`False`).
        nlow: int, optional
            Lowest number of cells in a bin. Needed when merging bins is enabled.
        style: str, optional
            Binning style used - linear (`lin`, default) or logarithmic (`log`).
        
        Returns
        -------
        centers: array_like
            Bin centers.
        prob: array_like
            Probability at the bin centers. Have the same size as `centers`.
        err: array_like
            Error estimate for the probability. Have the same size as `centers`.

        """
        if self._cm is ... :
            raise CatalogError("cic matrix not created")
        return self._cm.countProbability(bins, merge, nlow, style)

class CountMatrix:
    """
    A cell structure storing the number of galaxies in cells. This is used for count-
    in-cell estimation, by catalog objects.

    Parameters
    ----------
    objx: array_like 
        Position array. An ndarray of float (comoving) positions with 3 columns. 
    subdiv: int
        Number of subdivisions of the region. If it is :math:`n`, then there will 
        be :math:`n^3` cells.
    boxsize: float, tuple of 3 floats
        Size of the box bounding the region. If it is a number, then a cubical 
        region is assumed. Otherwise, it should be a 3-tuple of the x, y and z 
        lengths of the box. 
    offset: float, tuple of 3 floats, optional
        Offset position of the region. By default, it is set to 0.
    
    """
    __slots__ = "count", "subdiv", "boxsize", "offset", "cellsize", 

    def __init__(self, objx: Any, subdiv: int, boxsize: Union[float, tuple], offset: Union[float, tuple] = 0.) -> None:
        if not isinstance(subdiv, int):
            raise TypeError("subdiv should be 'int'")

        if isinstance(boxsize, (int, float)):
            boxsize = [float(boxsize), ] * 3
        else:
            boxsize = list(boxsize)
            if len(boxsize) != 3:
                raise TypeError("boxsize must be a 3-tuple or number")

        if isinstance(offset, (int, float)):
            offset = [float(offset), ] * 3
        else:
            offset = list(offset)
            if len(offset) != 3:
                raise TypeError("offset must be a 3-tuple or number")
        
        boxsize, offset = np.asarray(boxsize), np.asarray(offset)
        if any(np.min(objx, axis = 0) < offset) or any(np.max(objx, axis = 0) > offset + boxsize):
            warnings.warn("some objects are outside the specified box", Warning)

        cellsize = boxsize / subdiv

        self.boxsize  = boxsize
        self.cellsize = cellsize
        self.subdiv   = subdiv
        self.offset   = offset
 
        i      = self._pos2index(objx)
        i, cnt = np.unique(i, return_counts = True)
        
        self.count = dict(zip(i, cnt)) 

    def __repr__(self) -> str:
        """ Return the canonical string representation of the object. """
        bs = self.boxsize
        if all(np.abs(bs - bs[0]) < 1e-6):
            bs = bs[0]
        os = self.offset
        if all(np.abs(os - os[0]) < 1e-6):
            os = os[0]    
        return f"<CountMatrix: subdiv = {self.subdiv}, boxsize = {bs}, offset = {os}>"

    def _pos2index(self, objx: Any) -> Any:
        i = ((objx - self.offset) // self.cellsize).astype(int)
        return self.subdiv * (self.subdiv * i[:,0] + i[:,1]) + i[:,2]

    def pos2index(self, objx: Any) -> Any:
        """
        Convert object positions to cell index. 

        Parameters
        ----------
        objx: array_like
            Position array, an ndarray of float positions with 3 columns. 

        Returns
        -------
        i: array_like
            Cell index corresponding to the positions.
        """
        objx = np.asarray(objx)
        if objx.ndim == 1:
            if objx.shape[0] != 3:
                raise ValueError("position must be a 3-vector")
            objx = objx[np.newaxis, :]
        elif objx.ndim == 2:
            if objx.shape[1] != 3:
                raise ValueError("invalid shape for position array")
        else:
            raise ValueError("invalid shape for position array")
        return self._pos2index(objx)
        
    def countof(self, i: int) -> int:
        """ Get the count at i-th cell (flttened). """
        cells = self.subdiv**3
        if i < cells:
            if i in self.count.keys():
                return self.count[i]
            return 0
        raise IndexError(f"invalid index, must be in 0-{cells-1}")    

    def countVector(self, ) -> Any:
        """ Get the counts as a vector. """
        return np.asarray(list(map(self.countof, range(self.subdiv**3))))

    def countProbability(self, bins: int, merge: bool = False, nlow: int = ..., style: str = "lin") -> tuple:
        r"""
        Estimate the count-in-cells probability distribution. Estimate of probability 
        at the center of a bin :math:`B` is given by :math:`\hat{P}(N) = \frac{n}{w}`, 
        where :math:`N` is the bin center, :math:`n` is the number of cells with count 
        falling in the bin and :math:`w` is the bin width.

        Parameters
        ----------
        bins: int
            Number of bins to use.
        merge: bool, optional
            Wheather to merge bins with cells, less than a lower threshold. This 
            is disabled by default (`False`).
        nlow: int, optional
            Lowest number of cells in a bin. Needed when merging bins is enabled.
        style: str, optional
            Binning style used - linear (`lin`, default) or logarithmic (`log`).
        
        Returns
        -------
        centers: array_like
            Bin centers.
        prob: array_like
            Probability at the bin centers. Have the same size as `centers`.
        err: array_like
            Error estimate for the probability. Have the same size as `centers`.

        """
        def _mergeBins(x: Any, y: Any, cutoff: int) -> Tuple[Any, Any]:
            """  For bin merging (TODO: is there a better way?) """
            start, stop = 0, len(y)
            while start < stop-1:
                for i in range(start, stop):
                    yi, start = y[i], i
                    if yi < cutoff:
                        __ix, __iy = (i, i-1) if i == stop-1 else (i+1, i+1)
                        y[__iy] += yi
                        stop    -= 1

                        y, x = np.delete(y, i), np.delete(x, __ix) 
                        break
            return x, y

        n = self.countVector()

        # binning the n value:
        n_min, n_max = min(n). max(n)
        if not n_min:
            warnings.warn("empty cells are present, but lowest bin edge will be 1", Warning)
            n_min = 1

        if style == 'lin':
            _bins = np.linspace(n_min, n_max, bins)
        elif style == 'log':
            _bins = np.logspace(np.log10(n_min), np.log10(n_max), bins)
        else:
            raise ValueError("style can be either `lin` or `log` only")

        count, edges = np.histogram(n, bins = _bins, )
        
        if merge:
            if nlow is ... :
                raise CICError("lower n cut-off should be given if merging enabled")
            elif not isinstance(nlow, int):
                raise ValueError("`nlow` must be 'int'")
            
            # merging bins:
            edges, count = _mergeBins(edges, count, nlow)
        
        # probability distr.:  
        cells = self.subdiv**3  
        width = np.diff(edges)
        prob  = count / cells / width

        # error estimate:
        err = np.sqrt(prob * (1 - prob) / cells / width)

        # bin centers:
        if style == 'log':
            centers = np.sqrt(edges[:-1] * edges[1:]) # geom mean
        else:
            centers = (edges[:-1] + edges[1:]) / 2.   # mean
        return centers, prob, err

class PowerLaw:
    r"""
    A standard power law function, :math:`y = ax^b`.

    Parameters
    ----------
    coef: float, optional
        Value of the coefficient :math:`a`.
    expt: float, optional
        Value of the exponent :math:`b`.

    """
    __slots__ = 'coef', 'expt',

    def __init__(self, coef: float = ..., expt: float = ..., ) -> None:
        self.coef, self.expt = ..., ...

        if not (self.coef is ... or self.expt is ...):
            self.setParam(coef, expt)

    def setParam(self, coef: float, expt: float) -> None:
        if not isinstance(coef, (float, int)):
            raise TypeError("'coef must be a number")
        elif not isinstance(expt, (float, int)):
            raise TypeError("'expt must be a number")

        self.coef, self.expt = coef, expt
        return

    def eval(self, x: Any) -> Any:
        """ 
        Evaluate the function. 
        
        Parameters
        ----------
        x: array_like
            Function indepenedent variable.

        Returns
        -------
        y: array_like
            Function values - will have the same shape as `x`.
        """
        if self.coef is ... or self.expt is ... :
            raise RuntimeError("function not initialised")
        return PowerLaw.f(x, self.coef, self.expt)

    def __call__(self, x: Any) -> Any:
        return self.eval(x)

    @staticmethod
    def f(x: Any, coef: float, expt: float) -> Any:
        """
        Standard power law function. 

        Parameters
        ----------
        x: array_like
            Function independent variable.
        coef: float
            Coefficient. Must be a scalar.
        expt: float
            Exponent. Must be a scalar.

        Returns
        -------
        y: array_like
            Function values. Will have the same shape as `x`.
        """
        return coef * x**expt

class LinearPowerSpectrum:
    """
    An object storing the power spectrum as a table. The linear matter power spectrum 
    at z = 0 is given by, without normalisation, :math:`P(k) = k^n T^2(k)`. Here T is 
    the transfer function and :math:`n` is the spectral index.
    
    The power table should have two columns `lnk`, logarithm of the k ang `lnpk`, 
    logarithm of the P(k). A cubic spline is created using this spline and can be 
    used to get interpolated power spectrum values.

    Parameters
    ----------
    data: array_like
        Power spectrum table as logk vs. logp. Must be an ndarray of floats with 
        shape(n, 2)

    """
    __slots__ = 'data', '_f', '_itype', '_norm'

    def __init__(self, data: Any) -> None:
        # pk table is used to create a (linear) power spectrum spline:
        # pk table should be in log-log format: i.e., log k vs. log pk 
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError("pk table must be a 2D array")
        elif data.shape[1] != 2:
            raise ValueError("pk table ,ust have two columns")
        self.data  = data
        self._norm = 1.

        # create the spline:
        x, y = data.T # x is log k and y is log pk
        self._f = CubicSpline(x, y, )

    def __call__(self, lnk: Any, normalise: bool = True) -> Any:
        return self.power(lnk, normalise)

    def power(self, lnk: Any, normalise: bool = True) -> Any:
        r""" 
        Return the power spectrum as a function of :math:`\log k`.

        Parameters
        ----------
        lnk: array_like
            Input argument, natural logarithm of k.
        normalise: bool, optioinal
            Whether to normalise the power spectrum. Default is true.

        Returns
        -------
        pk: array_like
            Power spectrum (interpolated).

        """
        pk = np.exp(self._f(lnk))
        if normalise:
            return  pk * self._norm
        return pk

    def var(self, r: float, normalise: bool = True) -> float:
        r"""
        Linear matter variance, smoothed with a top-hat smoothing filter.

        Parameters
        ----------
        r: float
            Smooting radius. Must be a scalar.
        normalise: bool, optioinal
            Whether to normalise the power spectrum. Default is true.
        
        Returns
        -------
        var: float
            Value of variance.

        """
        if not np.isscalar(r):
            raise ValueError("r should be a scaler")

        def filt(x: Any) -> Any:
            """ spherical top-hat filter in fourier space. """
            return (np.sin(x) - x * np.cos(x)) * 3. / x**3 

        def varInteg(lnk: Any) -> Any:
            """ variance integrand. """
            k = np.exp(lnk)
            return k**3. * self.power(lnk, normalise = False) * filt(k*r)**2.
        
        retval, err = quad(varInteg, -8., 8., limit = 100)
        
        var = retval / 2. / np.pi**2
        if normalise:
            return var * self._norm
        return var

    def cellvar(self, kn: float) -> float:
        r"""
        Compute the variance from power spectrum. The power is integrated over a sphere 
        in k-space, with no smoothing. If linear power is used, then the computed will 
        be the linear variance.

        .. math::
            \sigma^2 = \int_0^{k_N} \frac{{\rm d}k k^2}{2 \pi^2} P(k) 

        Parameters
        ----------
        kn: float
            Radius of the k-sphere. For the case of cells, this correspond to the 
            Nyquist wavenumber.

        Returns
        -------
        var: float
            Value of cell variance.
        """
        def varInteg(lnk: Any) -> Any:
            """ integrand used to compute linear variance. """
            return np.exp(lnk)**3 * self.power(lnk)

        retval, err = quad(varInteg, -8., np.log(kn), limit = 100)
        return retval / 2. / np.pi**2

    def normalise(self, sigma8: float = ...) -> None:
        r"""
        Normalise the power spectrum using :math:`\sigma_8`. 

        Parameters
        ----------
        sigma8: float, optional
            If given use this value to normalise. If not given, de-normalise the power 
            spectrum by setting the resetting norm factor.
        """
        self._norm = 1.
        if sigma8 is ... :
            return
        self._norm = sigma8**2 / self.var(8., )
        return

class CellPowerSpectrum:
    r"""
    Measured log field power spectrum object. This corresponds to a box region. It 
    uses a functional form for power if the k vector is shorter than nyquist wavenumber 
    and extend this as a power law after it. This functional form can be modified by 
    changing the linear power function. Also, this power is expressed per the log field 
    bias factor. i.e., the actual log field bias factor is found by multiplying this 
    with the bias factor.

    Parameters
    ----------
    f: callable
        Function to call when linear power spectrum is needed. Must be a function of 
        the length of the k vector.
    kn: float
        Nyquist wavenumber.
    quantize: bool, optional
        If set, quantize the power spectrum to create a spline. This will try to expess 
        the power as a function of the k-vector length. It is set true by default. 
        
    """
    __slots__ = '_pfunc', 'kn', '_headfit', '_tailfit', '_qpower'

    def __init__(self, f: Callable, kn: float, quantize: bool = False) -> None:
        self._headfit = PowerLaw()
        self._tailfit = PowerLaw()
        
        self.kn = kn        # nyquist wavenumber
        self.setFunction(f, quantize) # set the function and apply continuation

    def fbound(self, kx: Any, ky: Any, kz: Any) -> Any:
        """
        Compute the measured log field power spectrum in a cell. This definition is 
        valid only when the k vector is smaller than the nyquist k vector. For other 
        vectors, this has to be continued by a power law.

        Parameters
        ----------
        kx, ky, kz: array_like
            k-vector components. For this form to be accurate, length of the vector 
            must be less than :math:`k_N` but it is not checked.

        Notes
        -----
        This uses the cloud-in-cells weight function.

        """

        def _pkterm(kxi: Any, kyi: Any, kzi: Any) -> Any:
            """ to find a term in the sum to get power. """
            k   = np.sqrt(kxi**2 + kyi**2 + kzi**2)
            p2  = 4. # for cloud-in-cell weight function (squared)
            pki = self._pfunc(k)
            wxi = np.sinc(kxi / self.kn / 2.)
            wyi = np.sinc(kyi / self.kn / 2.)
            wzi = np.sinc(kzi / self.kn / 2.)
            return pki * (wxi * wyi * wzi)**p2

        kx, ky, kz = np.asarray(kx), np.asarray(ky), np.asarray(kz) # k vector components

        pk   = 0.
        for nx, ny, nz in product(*repeat(range(3), 3)):
            if nx**2 + ny**2 + nz**2 >= 9.:
                continue # only abs(n) < 3 are needed
            pk += _pkterm(
                            kx + 2. * nx * self.kn, 
                            ky + 2. * ny * self.kn, 
                            kz + 2. * nz * self.kn
                         )
        return pk

    def setFunction(self, f: Callable, quantize: bool = False) -> None:
        """
        Set the linear power spectrum function.

        Parameters
        ----------
        f: callable
            Function or a callable used to call. It accept only one argument, `k`.
        quantize: bool, optional
            If set true, quantize the power in the k-range :math:`[10^{-3}, k_N]` and 
            use interpolated values instead of the exact values. Its default value is 
            false. 

        """         
        if not callable(f):
            raise TypeError("f must be a python callable")
        self._pfunc = f
        
        # find the power law continuation of the function for longer vectors
        self._applyContinuation()

        # quantise the power if told:
        self._qpower  = ... 
        if quantize:
            self._quantizePower()
        return

    def _applyContinuation(self, ) -> None:
        """
        Get the power law continuation of the measured power outside its input range. 
        This interpolates the power spectrum near the limit with a power law and use 
        this to extend the definition.
        """
        def _getContinuation(kx: Any, ky: Any, kz: Any, k: Any) -> tuple:
            """ get the best fitting power law parameters. """
            pk              = self.fbound(kx, ky, kz) 
            (coef, expt), _ = curve_fit(PowerLaw.f, k, pk )
            return coef, expt

        def _headContinuation() -> tuple:
            """ power law fit for the head part. """
            # generate random k vetors of length < 1e-3
            kx   = 10**np.random.uniform(-7, -3, 1_000_000)
            ky   = 10**np.random.uniform(-7, -3, 1_000_000)
            kz   = 10**np.random.uniform(-7, -3, 1_000_000)
            k    = np.sqrt(kx**2 + ky**2 + kz**2)
            mask = np.where(k < 1e-3)[0] 
            return _getContinuation(kx[mask], ky[mask], kz[mask], k[mask])

        def _tailContinuation() -> tuple:
            """ power law fit for the tail part. """ 
            # generate random k vectors of length in 0.9kn - kn range
            kx   = np.random.uniform(0.5*self.kn, self.kn, 1_000_000)
            ky   = np.random.uniform(0.5*self.kn, self.kn, 1_000_000)
            kz   = np.random.uniform(0.5*self.kn, self.kn, 1_000_000)
            k    = np.sqrt(kx**2 + ky**2 + kz**2)
            mask = np.where((k > 0.9*self.kn) & (k <= self.kn))[0] 
            return _getContinuation(kx[mask], ky[mask], kz[mask], k[mask])

        # short-k (head) continuation:
        coef_h, expt_h = _headContinuation()
        self._headfit.setParam(coef_h, expt_h)

        # long-k (tail) continuation:
        coef_t, expt_t = _tailContinuation()
        self._tailfit.setParam(coef_t, expt_t)
        return

    def _quantizePower(self, ) -> None:
        """
        Quantise the power in the range :math:`10^{-3} < k < k_N`. Quantization uses the 
        binned mean power, with 100 bins. May not be accuarate.
        """
        kn = self.kn

        kx = 10**(np.random.uniform(-6, np.log10(kn), 1000_000))
        ky = 10**(np.random.uniform(-6, np.log10(kn), 1000_000))
        kz = 10**(np.random.uniform(-6, np.log10(kn), 1000_000))

        k  = np.sqrt(kx**2 + ky**2 + kz**2)
        pk = self._power(kx, ky, kz, use_spline = False) # power without using the spline

        # find the binned mean powers with 100 bins:
        lnp, lnk, _ = binned_statistic(np.log(k), np.log(pk), statistic = 'mean', bins = 100, )
        lnk         = 0.5 * (lnk[:-1] + lnk[1:]) # bin centers

        # make the spline from the data
        self._qpower = CubicSpline(lnk, lnp)
        return

    def _power_scalarIn(self, kx: float, ky: float, kz: float) -> float:
        """ Get the power for single k-vector input. """
        k = np.sqrt(kx**2 + ky**2 + kz**2) # length of k vector
        if k < 1.e-3:
            return self._headfit(k)
        elif k > self.kn:
            return self._tailfit(k)
        return self.fbound(np.array([kx, ky, kz]))

    def _power(self, kx: Any, ky: Any, kz: Any, use_spline: bool = False) -> Any:
        """ Get the computed power from functional form. """
        kx, ky, kz = np.asarray(kx), np.asarray(ky), np.asarray(kz)
        
        k = np.sqrt(kx**2 + ky**2 + kz**2) # length of k vector

        if np.isscalar(k):
            return self._power_scalarIn(kx, ky, kz) # raise error when not a 3-vector

        pk   = np.empty_like(k)

        # apply continuation for k > kn:
        mask_t     = np.where(k > self.kn, True, False) # long-k mask
        pk[mask_t] = self._tailfit.eval(k[mask_t])

        # apply continuation for k < 1e-3:
        mask_h     = np.where(k < 1.00e-3, True, False) # short-k mask
        pk[mask_h] = self._headfit.eval(k[mask_h])

        # compute for other ks
        mask     = ~(mask_h | mask_t) # rest-of-k mask
        if use_spline:
            # if quantization is not enabled, ignore it
            if self._qpower is not ... : 
                pk[mask] = np.exp(self._qpower(np.log(k[mask])))
                return pk

        pk[mask] = self.fbound(kx[mask], ky[mask], kz[mask])
        return pk

    def powerk(self, k: Any) -> Any:
        """ 
        Get the power as a function of k, length of k-vector. This is a test function 
        and is available only when quantization is enabled.  
        """
        if self._qpower is ... :
            raise ValueError("function available only when quantization enebled")
        k  = np.asarray(k)
        pk = np.empty_like(k)

        # apply continuation for k > kn:
        mask_t     = np.where(k > self.kn, True, False) # long-k mask
        pk[mask_t] = self._tailfit.eval(k[mask_t])

        # apply continuation for k < 1e-3:
        mask_h     = np.where(k < 1.00e-3, True, False) # short-k mask
        pk[mask_h] = self._headfit.eval(k[mask_h])

        # compute for other ks
        mask     = ~(mask_h | mask_t) # rest-of-k mask
        pk[mask] = np.exp(self._qpower(np.log(k[mask])))
        return pk

    def power(self, kx: Any, ky: Any, kz: Any, use_spline: bool = False) -> Any:
        """
        Compute the power spectrum. When the k vector is within the range, use the 
        functional form to get the value. Outside that range, use the computed power 
        law continuation to extend it.

        Parameters
        ----------
        kx, ky, kz: array_like
            k vector components, each must be an ndarray of floats.
        use_spline: bool, optional
            If true, use the quantised power spectrum values to get the power in the 
            range :math:`[10^{-3}, k_N]`. If no quantization enabled, use the original 
            form.

        Returns
        -------
        pk: array_like
            Power spectrum values, has the same size as the largest of `ki`.

        """
        return self._power(kx, ky, kz, use_spline)

    def __call__(self, kx: Any, ky: Any, kz: Any, use_spline: bool = False) -> Any:
        return self.power(kx, ky, kz, use_spline)

class Cosmology:
    r"""
    An object storing a specific Lambda-CDM cosmology model.

    Parameters
    ----------
    Om0: float
        Present value of the normalized matter density, :math:`\Omega_{\rm m}`.
    Ode0: float
        Present value of the normalized dark-energy density, :math:`\Omega_{\rm de}`.
    h: float
        Presnt value of the Hubble parameter in units of 100 km/sec/Mpc.
    ns: float
        Slope of the linear power spectrum.
    pk_tab: array_like
        Power spectrum table. This must be a 2D array with two columns - :math:`\ln k`
        in the first and :math:`\ln P(k)` in the second. The table should have enough 
        resolution to include the details, if present.

    """
    __slots__ = 'Om0', 'Ode0', 'Ok0', 'h', 'ns', 'pk', 'sigma8', 

    def __init__(self, Om0: float, Ode0: float, h: float, ns: float, pk_tab: Any) -> None:
        for o, name in zip([Om0, Ode0, h], ['Om0', 'Ode0', 'h']):
            if not isinstance(o, (int, float)):
                raise TypeError("{} must be a number".foramte(name))
            elif o < 0.:
                raise ValueError("{} must be non-negative".format(name))
        self.Om0  = Om0
        self.Ode0 = Ode0
        self.h    = h

        Ok0      = 1. - Om0 - Ode0
        self.Ok0 = Ok0 if abs(Ok0) > 1.e-8 else 0.

        # ns can be negative also:
        if not isinstance(ns, (int, float)):
            raise TypeError("ns should be a number")
        self.ns = ns
        
        self.sigma8 = ... # sigma8 is set when normalising the power 

        # create the power spectrum table
        self.pk = LinearPowerSpectrum(pk_tab)
    
    def __repr__(self) -> str:
        return f"Cosmology(Om0 = {self.Om0}, Ode0 = {self.Ode0}, h = {self.h}, ns = {self.ns})"

    def Ez(self, z: Any) -> Any:
        r"""
        Evaluate the function :math:`E(z) := H(z) / H_0` as a function of the 
        redshift :math:`z`.

        .. math::
            E(z) = \sqrt{\Omega_{\rm m} (1 + z)^3 + \Omega_{\rm k} (1 + z)^2 + 
                         \Omega_{\rm de}}

        Parameters
        ----------
        z: array_like
            Redshift.

        Returns
        -------
        Ez: array_like
            Value of the function at z.

        """
        zp1 = 1. + np.asarray(z)
        return np.sqrt(self.Om0 * zp1**3 + self.Ok0 * zp1**2 + self.Ode0)

    def Omz(self, z: Any) -> Any:
        r"""
        Evaluate the normalized density of (dark) matter at redshift :math:`z`. 
        
        .. math::
            \Omega_{\rm m}(z) = \frac{\Omega_{\rm m} (z + 1)^3}{E^2(z)}
        
        Parameters
        ----------
        z: array_like
            Redshift

        Returns
        -------
        Omz: array_like
            Normalized matter density at z.


        """
        zp1 = 1 + np.asarray(z)
        Omz = self.Om0 * zp1**3
        return Omz / (Omz + self.Ok0 * zp1**2 + self.Ode0)

    def Dz(self, z: Any) -> float:
        r""" 
        Linear growth factor, computed as the integral

        .. math::
            D_{+}(z) \propto H(z) \int_z^\infty {\rm d}z' \frac{1 + z'}{E^3(z')} 
        
        Parameters
        ----------
        z: array_like
            Redshift, must be greater than -1.
        
        Returns
        -------
        Dz: array_like
            Normalised value of growth factor.
        """
        def growthInteg(a: float):
            """ integrand to find growth """
            a = np.asarray(a)
            return a**1.5 / (self.Om0 + self.Ok0 * a + self.Ode0 * a**3)**1.5

        def _Dz(z: float):
            """ un-normalised growth. """
            retval, err = quad(growthInteg, 0., 1. /(1. + z))
            return retval * self.Ez(z) * self.Om0 * 2.5
        
        if not np.isscalar(z):
            return np.array(list(map(_Dz, z))) / _Dz(0.)
        return _Dz(z) / _Dz(0.)

    def fz(self, z: Any) -> Any:
        r"""
        Linear growth rate, given as :math:`f(z) \approx \Omega_{\rm m}(z)^{0.6}`.

        Parameters
        ----------
        z: array_like
            Redshift

        Returns
        -------
        zz: array_like
            Linear graowth rate at z.

        """
        return self.Omz(z)**0.6
    
    def power(self, k: Any, z: float = 0., normalise: bool = True) -> Any:
        r"""
        Compute the power spectrum by interpolating from the table.

        Parameters
        ----------
        k: array_like
            Wavenumber.
        z: float, optional
            Redshift, default is 0. 
        normalise: bool, optioinal
            Whether to normalise the power spectrum. Default is true.

        Returns
        -------
        pk: array_like
            Power spectrum. Has the same shape as `k`.
        """
        return self.pk(np.log(k), normalise) * self.Dz(z)**2
    
    def var(self, r: float, normalise: bool = True) -> float:
        r"""
        Linear matter variance, smoothed with a top-hat smoothing filter.

        Parameters
        ----------
        r: float
            Smooting radius. Must be a scalar.
        normalise: bool, optioinal
            Whether to normalise the power spectrum. Default is true.
        
        Returns
        -------
        var: float
            Value of variance.

        """
        return self.pk.var(r, normalise)

    def cicvar(self, kmax: float) -> float:
        r"""
        Compute the variance from power spectrum. The power is integrated over a sphere 
        in k-space, with no smoothing. This variance is the variance in the cell.

        Parameters
        ----------
        kmax: float
            Upper limit of integration, radius of the sphere.

        Returns
        -------
        var: float  
            Value of the variance.

        """
        if not np.isscalar(kmax):
            raise TypeError("kmax should be a scalar")
        return self.pk.var(kmax)

    def normalisePower(self, sigma8: float = ... ) -> None:
        r"""
        Normalise the power spectrum using :math:`\sigma_8`.

        Parameters
        ----------
        sigma8: float, optional
            Value to use for normalisation. If not given, set norma factor to 1.
        """
        self.pk.normalise(sigma8)
        self.sigma8 = sigma8
        return

class cicDeltaDistribution:
    r"""
    One point dark-matter distribution. This distribution depends on the redshift and 
    cosmology. This object can be used for the probability distribution function of 
    :math:`\delta` in count-in-cells calculations.

    Parameters
    ----------
    z: float
        Redshift parameter - must be greater than -1.
    pixsize: float
        Size of the cell used in count-in-cell calculations.
    model: :class:`Cosmology`, optional
        Lambda-CDM cosmology model to use. Alternatively, one can give the model parameters 
        as keyword arguments instead of giving a model object. The required keywords are 
        `Om0` (matter density), `Ode0` (dark-energy density), `h` (hubble parameter), `ns` 
        (spectral index) and `pk_tab` (power spectrum table).

    """
    __slots__   = 'pixsize', 'z', 'kn', '_cosmo', '_power', '_params', 

    # a namedtuple to hold distribution parameters: 
    distrParams = namedtuple("distrParams", ['mu', 'sigma', 'xi'], )

    def __init__(self, z: float, pixsize: float, model: Cosmology = ..., **kwargs) -> None:
        raise DeprecationWarning("class is to be re-defined")

        if z < -1.:
            raise ValueError("z cannot be less than -1")
        self.z = z

        if not isinstance(pixsize, (int, float)):
            raise TypeError("pixsize must be a number")
        elif pixsize <= 0.:
            raise ValueError("pixsize must be positive")
        self.pixsize = pixsize
        self.kn      = np.pi / self.pixsize # nyquist wavelength

        # initialise the cosmology model:
        if model is not ... :
            if not isinstance(model, Cosmology):
                raise TypeError("model must be a 'Cosmology' object")
            self._cosmo = model
        else:
            self._cosmo = Cosmology(**kwargs)
        
        self._power  = ...      # store power spectrum object
        self._params = ...      # store distribution parameters (mu, sigma, xi)
        self.preparePower()     # initialise the measured power        

    def linvar(self, ) -> float:
        """ 
        Compute the linear variance in the cell. 
        """
        return self._cosmo.cicvar(self.kn)    

    def linpower(self, k: Any) -> Any:
        """
        Get the linear power spectrum.

        Parameters
        ---------
        k: array_like
            Wavenumber.

        Returns
        -------
        pk: array_like
            Power spectrum.  Has the same shape as `k`.

        """
        return self._cosmo.power(k, self.z)
    
    def measpower(self, k: Any) -> Any:
        """
        Get the log field measured power spectrum per bias factor. Multiplying by the 
        log field bias factor will give the actual power.

        Parameters
        ----------
        k: array_like
            k vectors. Must be an ndarray with 3 columns or unpackable as 3.

        Returns
        -------
        pk: array_like
            Power spectrum values.
        """
        return self._power(k)

    def preparePower(self, ) -> None:
        """
        Prepare the measured log power spectrum object.  This can be used to get the 
        measured log field power spectrum (per the bias factor).
        """
        self._power = CellPowerSpectrum(self.linpower, self.kn)
        return

    def cicvar(self, ) -> float:
        r"""
        Compute the count-in-cell measured variance. This is the integral of the 
        measured log field power over the cube in k space, with side ranging from 
        :math:`-k_N` to :math:`k_N`, excluding the 0. 
        
        This variance is expressed per bias factor. So, actual log field variance 
        will be found as `bias * cicvar`.

        NOTE: this variance is computed using a naive monte-carlo approch, need to 
        find a fast/efficient way.
        """
        kn  = self.kn
        vol = (2. * kn)**3 # k space volume: cube

        def mcIntegral(n: int) -> tuple:
            kvec = np.random.uniform(-kn, kn, (n, 3)) 

            # zero vectors are filtered out
            mask = np.where(np.sqrt(np.sum(kvec**2, axis = -1)) > 1e-8)[0]
            kvec = kvec[mask, :]
            pk   = self.measpower(kvec)
            return np.mean(pk) * vol, np.std(pk) * vol / np.sqrt(len(pk)) # integral and error

        retval, err = mcIntegral(1000_000) # using 1M points in the cube
        return retval / (2. * np.pi)**3

    def _distrParameters(self, ) -> None:
        r"""
        Compute the location, shape and scale parameters of the distribution. These 
        are computed using various fits and solving for the parameters.
        """        
        # variances and bias:
        vlin = self.linvar()               # linear variance

        mu   = 0.73
        vlog = mu * np.log(1. + vlin / mu) # log field variance (fit)

        bias = vlog / vlin                 # square of the log bias

        vcic = self.cicvar() * bias        # count-in-cell variance

        # mean of the log field:
        lamda = 0.65 
        mlog  = -lamda * np.log(1. + vlin / 2. / lamda)

        # skewness of the log field:
        a, b, c, d = -0.70, 1.25, -0.26, 0.06

        nsp3 = self._cosmo.ns + 3.
        slog = (a * nsp3 + b) * vcic**(-d - c * np.log(nsp3)) # T * vcic**-p

        r1 = slog * np.sqrt(vcic)         # pearson moment coeff.

        # solve for shape parameter:
        def shapeEqn(xi: float) -> float:
            """ relation between shape parameter and skewness. """
            g1mx, g1m2x = gamma(1. - xi), gamma(1. - 2.*xi)

            num = gamma(1. - 3.*xi) - 3.*g1mx * g1m2x + 2.*g1mx**3
            return r1 + num / (g1m2x - g1mx**2)**1.5

        # TODO: using `r1` as an initial guess - need to find a better one.
        xi   = newton(shapeEqn, r1, )

        # scale parameter:
        g1mx  = gamma(1. - xi)
        sigma = np.sqrt(xi**2 * vcic / (gamma(1. - 2.*xi) - g1mx**2))

        # location parameter:
        mu    = mlog - sigma * (g1mx - 1.) / xi

        self._params = cicDeltaDistribution.distrParams(mu, sigma, xi)
        return

    def oneDistribution(self, x: Any, log: bool = False) -> Any:
        r"""
        Get the distribution for the log or linear field.

        Parameters
        ----------
        x: array_like
            Value of the field argument.
        log: bool, optional
            If true, get the distribution for the log field. Default is false.

        Returns
        -------
        px: array_like
            Value of the distribution function - probability. This has the same shape 
            as the `x` input.

        """
        mu, sigma, xi = self._params

        x = np.asarray(x)

        if log:
            # log field distribution:
            t = (1. + (x - mu) / sigma)**(-1./xi)
            return t**(1 + xi) * np.exp(-t) / sigma
        
        # linear field distribution:
        t = (1. + (np.log(x) - mu) / sigma)**(-1./xi)
        return t**(1 + xi) * np.exp(-t) / sigma / (1. + x)

    def __call__(self, x: Any, log: bool = False) -> Any:
        return self.oneDistribution(x, log)



