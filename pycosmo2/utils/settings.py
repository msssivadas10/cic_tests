# value to be used for zero and infinity in integrations
ZERO, INF = 1.0E-08, 1.0E+08

# default tolerance values
ABSTOL, RELTOL = 1.0E-06, 1.0E-06

# default order of gaussian integration
DEFAULT_N = 64

# functions to set values:
def setZeroValue(value: float) -> None:
    global ZERO
    if not isinstance( value, ( float, int ) ):
        raise TypeError("value must be a real number")
    ZERO = value

def setInfinityValue(value: float) -> None:
    global INF
    if not isinstance( value, ( float, int ) ):
        raise TypeError("value must be a real number")
    INF = value
    
def setTolerance(reltol: float = None, abstol: float = None) -> None:
    global RELTOL, ABSTOL

    if reltol is not None:
        if not isinstance( reltol, ( float, int ) ):
            raise TypeError("reltol value must be a real number")
        RELTOL = reltol

    if abstol is not None:
        if not isinstance( abstol, ( float, int ) ):
            raise TypeError("abstol value must be a real number")
        ABSTOL = abstol

def setN(value: int) -> None:
    global DEFAULT_N

    if not isinstance( value, int ):
        raise TypeError("value must be an integer")
    elif value < 2:
        raise ValueError("value must be greater than 1")
    DEFAULT_N = value