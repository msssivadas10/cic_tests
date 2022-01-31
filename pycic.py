#!/usr/bin/python3

import matplotlib
import numpy as np
import warnings
from typing import Any, Tuple, Union
from itertools import product, repeat
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
from scipy.special import gamma
from scipy.optimize import newton

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
            Probability at the bin centers.

        """
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
            raise NotImplementedError()
            pass
        
        # probability distr.:    
        prob = count / self.subdiv**3 / np.diff(edges)

        # bin centers:
        if style == 'log':
            centers = np.sqrt(edges[:-1] * edges[1:]) # geom mean
        else:
            centers = (edges[:-1] + edges[1:]) / 2.   # mean
        return centers, prob

    def countProbabilityError(self, ) -> tuple:
        r"""
        Estimate the error in count-in-cells probability distribution.

        """
        raise NotImplementedError()


class cicDistribution:
    r"""
    Implementation of the theoretical count-in-cells distribution given in Repp and 
    Szapudi (2020). 

    Parameters
    ----------
    pk_table: array_like
        Power spectrum table. This must be a 2D array with two columns - :math:`\ln k`
        in the first and :math:`\ln P(k)` in the second. The table should have enough 
        resolution to include the details, if present.
    z: float
        Redshift value. Must be greater than -1. 
    Om0: float
        Present value of the normalized matter density, :math:`\Omega_{\rm m}`.
    Ode0: float
        Present value of the normalized dark-energy density, :math:`\Omega_{\rm de}`.
    h: float
        Presnt value of the Hubble parameter in units of 100 km/sec/Mpc.
    n: float
        Slope of the linear power spectrum.
    pixsize: float
        Size of a pixel (cell). Th entire space is divided into cells of this size.

    """
    __slots__ = "pk_spline", "z", "Om0", "Ode0", "Ok0", "h", "n", "pixsize", "kn", 

    def __init__(self, pk_table: Any, z: float, Om0: float, Ode0: float, h: float, n: float, pixsize: float) -> None:
        if z < -1.:
            raise CICError("redshift cannot be less than -1")
        self.z = z

        if Om0 < 0.:
            raise CICError("Om0 cannot be negative")
        self.Om0 = Om0

        if Ode0 < 0.: 
            raise CICError("Ode0 cannot be negative")
        self.Ode0 = Ode0
        self.Ok0  = 1. - Om0 - Ode0
        if abs(self.Ok0) < 1.e-08:
            self.Ok0 = 0.

        if h < 0.:
            raise CICError("h cannot be neative")
        self.h = h
        self.n = n

        if not isinstance(pixsize, (float, int)):
            raise CICError("pixsize should be a number")
        self.pixsize = pixsize
        self.kn      = np.pi / self.pixsize # nyquist wavenumber 

        pk_table = np.asarray(pk_table)
        if pk_table.ndim != 2:
            raise CICError("power spectrum should be given as a table")
        elif pk_table.shape[1] != 2:
            raise CICError("power table should have two columns")
        lnk, lnpk      = pk_table.T
        self.pk_spline = CubicSpline(lnk, lnpk)

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

        Examples
        --------
        TODO

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

        Examples
        --------

        """
        zp1 = 1 + np.asarray(z)
        Omz = self.Om0 * zp1**3
        return Omz / (Omz + self.Ok0 * zp1**2 + self.Ode0)

    def powerLin(self, lnk: Any) -> Any:
        r"""
        Get the power spectrum, calculated by interpolating the values in the power
        spectrum table.

        Parameters
        ----------
        lnk: array_like
            Natural logarithm of the wavenumber.

        Returns
        -------
        pk: array_like
            Value of the power spectrum.

        """
        return np.exp(self.pk_spline(lnk))

    def _Pdelta(self, delta: Any, mu: float, sigma: float, xi: float) -> Any:
        r"""
        PDF of :math:`\delta` as given in Repp and Szapudi (2020).

        .. math::
            P(\delta) = \frac{1}{(1 + \delta) \sigma}t^{1 + \xi} e^{-t}

        where, :math:`t \equiv t(\delta)` is given by

        .. math::
            t(\delta) = \left( 1 + \frac{\ln \delta - \mu}{\sigma} \xi \right)^{-1/\xi}

        and :math:`\mu, \sigma, \xi` are the location, scale and shape parameters.

        Parameters
        ----------
        delta: array_like
            :math:`\delta` value - density fluctuation.
        mu: float
            Lccation parameter.
        sigma: float
            Scale parameter.
        xi: float
            Shape parameter.

        Returns
        -------
        Pdelta: array_like
            Probability density for the delta value.

        Examples
        --------
        TODO

        """
        t = (1. + (np.log(delta) - mu) * xi / sigma)**(-1./xi)
        return t**(1 + xi) * np.exp(-t) / (1. + delta) / sigma

    def _var_lin_integrand(self, lnk: Any) -> Any:
        r"""
        Function to be integrated in order to find the linear variance in the cell,
        :math:`\sigma^2_{\rm lin}`, given by :math:`k^3 P_{\rm lin}(k)`, where the 
        integration variable is :math:`\ln k`. The :math:`1/2\pi^2` factor is not 
        included here. 

        Parameters
        ----------
        lnk: array_like
            Integration variable, natural logarithm of wavenumber.

        Returns
        -------
        fk: array_like
            Value of the function.

        """
        return np.exp(lnk)**3 * self.powerLin(lnk)

    def varLin(self, ) -> float:
        r"""
        Compute the value of the linear variance in the cell. It is computed from
        the linear power spectrum as the integral

        .. math::
            \sigma_{\rm lin}^2 = \int_0^{k_N} \frac{{\rm d}k k^2}{2 \pi^2} P_{\rm lin}(k) 

        where :math:`k_N` is the Nyquist wavenumber.

        Returns
        -------
        sigma2_lin: float
            Value of linear variance.
        """
        retval, err = quad(self._var_lin_integrand, -8., np.log(self.kn))
        return retval / 2. / np.pi**2

    def biasA(self, vlin: float) -> float:
        r"""
        Get the A-bias factor. It is given by 

        .. math::
            b_A^2 = \frac{\sigma^2_a (k_N)}{\sigma^2_{\rm lin} (k_N)}

        Parameters
        ----------
        vlin: float
            Linear variance.
        
        Returns
        -------
        b: float
            Bias value.

        """
        return np.sqrt(self.varA(vlin) / vlin)

    def powerA(self, lnk: Any) -> Any:
        r"""
        Power spectrum of the log density. It is related to the linear power spectrum 
        by a bias factor as :math:`P_A(k) = b_A^2 P_{\rm lin}(k)`. The bias factor is 
        the ratio of log to linear variances.

        Parameters
        ----------
        k: array_like
            Natural logarith of wavenumber.

        Returns
        -------
        Pk: array_like
            Value of power spectrum.


        """
        vlin = self.varLin()           # linear variance
        b2   = self.varA(vlin) / vlin # bias squared
        return self.powerLin(lnk) * b2

    def varA(self, vlin: float) -> float:
        r"""
        Get the A-variance in the cell, where :math:`A = \ln (1 + \delta)`. This uses 
        the fit given in Repp & Szapudi (2018).

        .. math::
            \sigma_A^2 = \mu \ln \left(1 + \frac{\sigma_{\rm lin}^2}{\mu} \right)
        
        Both the variances are evaluated at the Nyquist wavenumber :math:`k_N` and the
        value of :math:`\mu` is taken as 0.73 (best-fit value).

        Parameters
        ----------
        vlin: float
            Linear variance.

        Returns
        -------
        sigma: float
            Value of variance.

        """
        mu = 0.73
        return mu * np.log(1. + vlin / mu)

    def meanA(self, vlin: float) -> float:
        r"""
        Return the fitted value of mean of A as a function the linear variance. This 
        uses the fit given in Repp & Szapudi (2018)

        .. math::
            <A> = - \lambda \ln \left(1 + \frac{\sigma_{\rm lin}^2}{2\lambda} \right)

        where the best fitting value is :math:`\lambda = 0.65`.

        Parameters
        ----------
        vlin: float
            Linear variance.

        Returns
        -------
        avA: float
            Mean value of A.

        """
        lamda = 0.65
        return -lamda * np.log(1. + vlin / 2. / lamda)

    def skewA(self, vl: float) -> float:
        r"""
        Return the skewness of the A distribution as a function the count-in-cells 
        variance. The value if calculated by a fit given by Repp & Szapudi (2018),

        .. math::
            T_3 = T(n) [\sigma_A^2(l)]^{-p(n)}

        where :math:`n` is the slope of the power spectrum and, 

        .. math::
            T(n) = a(n + 3) + b \quad {\rm and} \quad p(n) = d + c \ln(n + 3)

        The best fit values are a = -0.70, b = 1.25, c = -0.26 and d = 0.06.

        Parameters
        ----------
        vl: float
            Count in cell variance. This is not the fitted value, but calculated from
            the *measured* A power spectrum.

        Returns
        -------
        T3: float
            Value of skewness.

        """
        a  = -0.70
        b  =  1.25
        c  = -0.26
        d  =  0.06

        np3 = self.n + 3.
        Tn  = a * np3 + b
        pn  = d + c * np.log(np3)
        return Tn * vl**(-pn)

    def getParameters(self, vlin: float, vl: float) -> Tuple[float, float, float]:
        r"""
        Get the location, scale and shape parameters of the probability distribution 
        corrsponding to the linear and count-in-cell variance value.

        Parameters
        ----------
        vlin: float
            Linear Variance.
        vl: float
            Count-in-cells variance.

        Returns
        -------
        mu: float
            Value of location parameter, :math:`\mu`.
        sigma: float
            Value of scale parameter, :math:`\sigma`.
        xi: float
            Value of shape parameter, :math:`\xi`. 

        """    
        # finding for xi:
        r1 = vl * self.skewA(vl) # Pearson’s moment coefficient (\gamma_1)

        def fxi(xi: float) -> float:
            """ function whose roort is xi """
            fn = gamma(1 - 3*xi) - 3*gamma(1 - xi)*gamma(1 - 2*xi) + 2*gamma(1 - xi)**3
            fd = (gamma(1 - 2*xi) - gamma(1 - xi)**2)**1.5
            return r1 + fn / fd

        # TODO: using r1 as an initial guess root. need to find a better one.
        xi = newton(fxi, r1, )

        # finding sigma:
        sigma = xi * vl / np.sqrt(gamma(1 - 2*xi) - gamma(1 - xi)**2)

        # finding mu:
        mu = self.meanA(vlin) - sigma * (gamma(1 - xi) - 1) / xi

        return mu, sigma, xi

    def _powerA_meas(self, kx: Any, ky: Any, kz: Any) -> Any:
        r"""
        Measured A power spectrum. This is valid only for :math:`\bf k` vectors 
        such that :math:`\sqrt{k_x^2 + k_y^2 + k_z^2} \le k_N`. No error will be
        raised otherwise.

        Parameters
        ----------
        kx: array_like
            X component of the k vectors.
        ky: array_like
            Y component of the k vectors.
        kz: array_like
            Z component of the k vectors.

        Returns
        -------
        Pk: array_like
            Power spectrum values.

        """
        # XXX: using the cloud in cell weight function as W(k)
        p    = 2.
        nmax = 3 
        kn   = self.kn


        def _powerA_vec(kx: Any, ky: Any, kz: Any) -> Any:
            """ A power for vector inputs """
            k = np.sqrt(kx**2 + ky**2 + kz**2)
            return self.powerA(np.log(k))

        def _weight(kx: Any, ky: Any, kz: Any) -> Any:
            """ mass assignment function (squared) """
            wx = np.sinc(kx / kn / 2.)
            wy = np.sinc(ky / kn / 2.)
            wz = np.sinc(kz / kn / 2.)
            return (wx * wy * wz)**(2.*p)

        def _power_term(kx: Any, ky: Any, kz: Any) -> Any:
            """ a term in the power sum """
            return _powerA_vec(kx, ky, kz) * _weight(kx, ky, kz)

        Pk = 0.
        for nx, ny, nz in product(*repeat(range(nmax), 3)):
            if nx**2 + ny**2 + nz**2 >= 9.:
                continue # only abs(n) < 3 are needed
            Pk += _power_term(
                                kx + 2. * nx * kn, 
                                ky + 2. * ny * kn, 
                                kz + 2. * nz * kn
                             )
        
        return Pk

    def powerA_meas(self, kx: Any, ky: Any, kz: Any) -> Any:
        r"""
        Measured power spectrum of A. It is computed as the sum

        .. math::
            P_{A, \rm meas} = \sum_{\bf n} P_A ({\bf k} + 2 k_N * {\bf n})
                                W^2 ({\bf k} + 2 k_N * {\bf n})

        where :math:`{\bf n}` is a vector of three integers with length less
        than 3. This is valid for :math:`\vert {\bf k} \vert \le k_N`. After 
        that limit, it is power law continued. 

        Parameters
        ----------
        kx: array_like
            X component of the k vectors.
        ky: array_like
            Y component of the k vectors.
        kz: array_like
            Z component of the k vectors.

        Returns
        -------
        Pk: array_like
            Power spectrum values.

        """
        raise NotImplementedError()



