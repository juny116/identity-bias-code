import numpy as np
import scipy.stats

x = np.array([60, 53.9, 61.66, 58, 51.66, 56.3, 61, 50.33, 53.33])
y = np.array([76.66, 74.46, 85, 67.66, 66, 66.33, 75, 76.33, 80.33])

r = np.corrcoef(x, y)
r = scipy.stats.pearsonr(x, y)
print(r)
