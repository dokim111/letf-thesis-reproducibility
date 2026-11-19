# Numerical checks for the math thesis: sharp bounds of g_beta(r) = beta*log(1+r) - log(1+beta*r) on |r| <= c.
import mpmath as mp
mp.mp.dps = 40
def g(b, r): return b*mp.log1p(r) - mp.log1p(b*r)
def h(b, r):
    r = mp.mpf(r)
    if abs(r) < mp.mpf('1e-8'): return mp.mpf(b)*(b-1)/2 - mp.mpf(b)*(b*b-1)/3*r
    return g(b, r)/r**2
c2 = lambda b: mp.mpf(b)*(b-1)/2
c3 = lambda b: -mp.mpf(b)*(b*b-1)/3
c4 = lambda b: mp.mpf(b)*(b**3-1)/4
def rho2(b, r):   # (g - c2 r^2)/r^3
    r = mp.mpf(r); return c3(b) + c4(b)*r if abs(r) < mp.mpf('1e-8') else (g(b, r) - c2(b)*r**2)/r**3
def rho3(b, r):   # (g - c2 r^2 - c3 r^3)/r^4
    r = mp.mpf(r); return c4(b) if abs(r) < mp.mpf('1e-6') else (g(b, r) - c2(b)*r**2 - c3(b)*r**3)/r**4
def extrema(f, c, n=2001):
    c = mp.mpf(c); xs = [-c + 2*c*i/(n-1) for i in range(n)]; vs = [f(x) for x in xs]
    i0 = min(range(n), key=lambda i: vs[i]); i1 = max(range(n), key=lambda i: vs[i])
    return (xs[i0], vs[i0]), (xs[i1], vs[i1])
print('== h = g/r^2: sharp constants m, M on [-c, c] ==')
for c in ['0.30', '0.15']:
    for b in [2, 3, -1, -2, -3]:
        if mp.mpf(c) >= min(1, 1/abs(mp.mpf(b))): continue
        (xm, vm), (xM, vM) = extrema(lambda r: h(b, r), c)
        print('c', c, 'beta', b, 'm', mp.nstr(vm, 8), 'at', mp.nstr(xm, 5), 'M', mp.nstr(vM, 8), 'at', mp.nstr(xM, 5), 'c2', mp.nstr(c2(b), 6))
print('== remainders: rho2 = (g - c2 r^2)/r^3, rho3 = (g - c2 r^2 - c3 r^3)/r^4 on [-c, c] ==')
for c in ['0.30', '0.15']:
    for b in [2, -1, -2]:
        (a, va), (z, vz) = extrema(lambda r: rho2(b, r), c); (a3, va3), (z3, vz3) = extrema(lambda r: rho3(b, r), c)
        print('c', c, 'beta', b, 'rho2 in [', mp.nstr(va, 7), '@', mp.nstr(a, 4), ',', mp.nstr(vz, 7), '@', mp.nstr(z, 4), '] c3', mp.nstr(c3(b), 5), '| rho3 in [', mp.nstr(va3, 7), '@', mp.nstr(a3, 4), ',', mp.nstr(vz3, 7), '@', mp.nstr(z3, 4), '] c4', mp.nstr(c4(b), 5))
print('== minimiser r*(b) of h_{-b} and the threshold b0 ==')
def hprime_num(b, r): return mp.diff(lambda x: h(b, x), r)
for bb in ['1.5', '2', '2.4', '2.491277567', '3']:
    b = mp.mpf(bb); gp = lambda r: (-b)*(-b-1)*r/((1+r)*(1-b*r)); rs = mp.findroot(lambda r: r*gp(r) - 2*g(-b, r), (mp.mpf('-0.6'), mp.mpf('-0.05')), solver='anderson', tol=1e-25)
    print('b', bb, 'r*', mp.nstr(rs, 10), '1/b', mp.nstr(1/b, 10), 'interior transition possible:', bool(abs(rs) < min(1, 1/b)), 'h(r*)', mp.nstr(h(-b, rs), 10))
b0 = mp.findroot(lambda b: (b+1)/(2*(b-1)) + 2*b*mp.log(1-1/b) + 2*mp.log(2), mp.mpf('2.5'))
print('b0', mp.nstr(b0, 15), '1/b0', mp.nstr(1/b0, 15))
# duality h_{-b}(r) = b^3 h_{-1/b}(-b r)
b = mp.mpf('2'); r = mp.mpf('-0.21'); print('duality check', mp.nstr(h(-b, r) - b**3*h(-1/b, -b*r), 5))
# N < 0 certificate at c = 0.3, beta = -2
print('N', mp.nstr(mp.mpf(27)/56 + 2*mp.log(mp.mpf(98)/125), 10), 'bound', mp.nstr(-mp.mpf(27)/12488, 10))
