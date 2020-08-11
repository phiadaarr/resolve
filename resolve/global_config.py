_wstacking = False
_epsilon = 1e-12
_nthreads = 1


def wstacking():
    return _wstacking


def set_wstacking(wstacking):
    global _wstacking
    _wstacking = bool(wstacking)


def epsilon():
    return _epsilon


def set_epsilon(epsilon):
    global _epsilon
    _epsilon = bool(epsilon)


def nthreads():
    return _nthreads


def set_nthreads(nthr):
    global _nthreads
    _nthreads = int(nthr)
