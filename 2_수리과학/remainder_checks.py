# Integral remainder: (g - T_n)/r^(n+1) = (-1)^n beta * int_0^1 u^n [1/(1+ru) - beta^n/(1+beta r u)] du
import mpmath as mp, sympy as sp
mp.mp.dps = 30
g = lambda b, r: b*mp.log1p(r) - mp.log1p(b*r)
T = lambda b, r, n: sum((-1)**k*(mp.mpf(b)**k - b)/k*r**k for k in range(2, n+1))
I = lambda b, r, n: (-1)**n*b*mp.quad(lambda u: u**n*(1/(1+r*u) - mp.mpf(b)**n/(1+b*r*u)), [0, 1])
worst = 0
for b in [2, 3, 1.5, -1, -2, -0.5, -3]:
    for r in [-0.3, -0.1, 0.05, 0.3]:
        if 1 + b*r <= 0 or 1 + r <= 0: continue
        for n in [1, 2, 3, 4, 5]:
            lhs = (g(b, mp.mpf(r)) - T(b, mp.mpf(r), n))/mp.mpf(r)**(n+1); worst = max(worst, abs(lhs - I(b, mp.mpf(r), n)))
print('max |lhs - integral| over the grid:', mp.nstr(worst, 5))
# monotonicity of the integrand psi_n(x) = [(1-beta^n) + (beta-beta^n) x]/((1+x)(1+beta x)) in x (x = r u)
x = sp.symbols('x', real=True)
for b, dom in [(2, (-sp.Rational(1, 2), sp.oo)), (-1, (-1, 1)), (-2, (-1, sp.Rational(1, 2)))]:
    for n in [1, 2, 3]:
        psi = ((1 - b**n) + (b - b**n)*x)/((1 + x)*(1 + b*x))
        num = sp.factor(sp.numer(sp.together(sp.diff(psi, x))))
        roots = [r for r in sp.real_roots(sp.Poly(num, x)) if dom[0] < r < dom[1]] if num.has(x) else []
        print('beta', b, 'n', n, 'd psi/dx numerator:', num, '| real roots inside the domain:', [sp.N(r, 6) for r in roots])
