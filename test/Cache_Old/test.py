import numpy as np
from hashlib import sha1

def test(k, *args):
    for i, v, c in k.kronprod(do_index=True, do_change=True):
        if k.changed('a'):
            if 1 == 1:
                curr_args = [k.value('a'), 1, 1]
                seed = int(sha1(str(curr_args).encode('utf-8')).hexdigest(), 16)
                rng = np.random.default_rng(seed)
                rnger = rng
                f = rnger.random()*0.1
        for j in range(len(args)):
            args[j][i] = v['a']+f
        get_ipython().run_line_magic('timeit', '-o lambda x, y: x*y')
    return args
