from typing import Any 
import numpy as np

class Cube:
    r"""
    An object representing a cube or box in 3D.

    Parameters
    ----------
    size: float
        Size of the cube or the length of the box in x dimension.
    ysize, zsize: float, optional
        If given, size of the box in the x and z dimension. If not given, use the size for x dimension.

    """

    VERTS = np.asfarray([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0], 
                         [0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
    FACES = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [3, 7, 6, 2], [0, 4, 5, 1], [0, 4, 7, 3], [1, 5, 6, 2]])
    EDGES = np.array([[0, 1], [1, 2], [2, 3],[3, 0],[4, 5],[5, 6],[6, 7],[7, 4],[0, 4],[1, 5],[2, 6],[3, 7]]) 

    __slots__ = 'size', 'nfaces', 'nverts', 'nedges'

    def __init__(self, size: float, ysize: float = None, zsize: float = None) -> None:
        ysize = size if ysize is None else ysize
        zsize = size if zsize is None else zsize

        self.size   = np.asfarray([ size, ysize, zsize ])
        self.nverts = 8
        self.nfaces = 6
        self.nedges = 12    

    @property
    def vertices(self) -> Any:
        return self.size * self.VERTS

    @property
    def faces(self) -> Any:
        return np.asfarray( list( map( self.face, range(self.nfaces) ) ) )

    @property
    def edges(self) -> Any:
        return np.asfarray( list( map( self.edge, range(self.nedges) ) ) )

    def face(self, i: int) -> Any:
        i = i % 6
        return self.vertices[ self.FACES[i], : ]

    def edge(self, i: int) -> Any:
        i = i % 12
        return self.vertices[ self.EDGES[i], : ]


class ParticleData:
    r"""
    Stores the information of a particle distribution at redshift :math:`z`.

    Parameters
    ----------
    position: array_like of shape (N, 3)
        Particle positions in some unit.
    velocity: array_like of shape (N, 3)
        Particle velocities in some unit.
    z: float
        Redshift.
    boxsize: float
        Size of the bounding cube of the particle system.
    gridsize: int
        Number of grid points in each dimension of the box.
        
    """

    __slots__ = 'position', 'velocity', 'z', 'boxsize', 'gridsize', 'bbox'

    def __init__(self, position: Any, velocity: Any, z: float, boxsize: float, gridsize: int) -> None:
        self.position = np.asfarray(position)
        self.velocity = np.asfarray(velocity)
        self.z        = z
        self.boxsize  = boxsize
        self.gridsize = gridsize

        self.bbox = Cube( self.boxsize )

    @property
    def boundindgBox(self) -> Cube:
        return self.bbox

    @property
    def nParticles(self) -> int:
        return self.position.shape[0]

    