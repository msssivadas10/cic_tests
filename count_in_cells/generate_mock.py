import numpy as np
import numpy.random as rnd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

n = 1_000_00#0

ra  = rnd.uniform(-1., 20., n)
dec = rnd.uniform(-1, 3, n)



g_mask = np.zeros(n, dtype = 'bool')
for ra0, dec0, radius in [(5., 1., 0.5), (10., 1.2, 1.)]:
    g_mask =  g_mask | ( (ra - ra0)**2 + (dec - dec0)**2 < radius**2 )


df = pd.DataFrame({'ra': ra, 'dec': dec, 'g_mask': g_mask})
df.to_csv("./count_in_cells/random.csv.gz", index = False, compression = 'gzip')

plt.figure()
sbn.scatterplot(df, x = 'ra', y = 'dec', hue = 'g_mask')
plt.show()