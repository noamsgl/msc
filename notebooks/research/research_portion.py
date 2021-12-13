import numpy as np

import portion as P

d = P.IntervalDict()
d[P.closedopen(10, 20)] = 'banana'
d[4] = 'apple'
d[5] = np.array([[1, 2, 3], [4, 5, 6]])
