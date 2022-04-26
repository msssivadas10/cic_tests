from pycosmo2.universe import *

m = Matter( 0.3 )
m.addchild( Baryon(0.05) )
m.addchild( MassiveNeutrino( 0.02, 1.0 ) )
m.fill(ColdDarkMatter)
u = Universe()
u.addchild( m )
u.finalize()


print(u, u.reminder)