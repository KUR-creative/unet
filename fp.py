from pymonad.Reader import curry
import functools, itertools
def pipe(*functions):
    def pipe2(f, g):
        return lambda x: g(f(x))
    return functools.reduce(pipe2, functions, lambda x: x)

cmap = curry(lambda f,xs: map(f,xs))
cfilter = curry(lambda f,xs: filter(f,xs))
flatten = lambda x: itertools.chain.from_iterable(x)
cflatMap = curry(lambda f,xs: flatten(cmap(f,xs)))

'''
def flip(func):
    'Create a new function from the original with the arguments reversed'
    @functools.wraps(func)
    def newfunc(*args):
        return func(*args[::-1])
    return newfunc
#flip = lambda f: lambda *a: f(*reversed(a))
#crepeat = curry(flip(itertools.repeat))
#def flip(f):
'''
    
flip = lambda f: lambda a,b: f(b,a)
crepeat = curry(flip(itertools.repeat))

import unittest
class Test(unittest.TestCase):
    def test_flip(self):
        m = lambda a,b: a - b
        fm = flip(m)
        self.assertEqual(fm(3,1), 1 - 3)

    def test_crepeat(self):
        get4 = crepeat(4)
        self.assertEqual(list(get4(1)), [1,1,1,1])


if __name__ == '__main__':
    unittest.main()

