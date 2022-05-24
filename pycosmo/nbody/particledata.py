from typing import Any 

class ParticleData:

    __slots__ = 'position', 'velocity', 'z', 'boxsize', 'gridsize', 

    def __init__(self, position: Any, velocity: Any, z: float, boxsize: float, gridsize: int) -> None:
        self.position = position
        self.velocity = velocity
        self.z        = z
        self.boxsize  = boxsize
        self.gridsize = gridsize

    