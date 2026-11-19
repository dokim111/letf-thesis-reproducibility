"""Deterministic checks for the general theorems in mathematics thesis v1.1.

Run: python3 general_theory_checks.py > general_theory_checks_output.json
No market data or fitted parameters are used. Numerical checks support the
implementation; the proofs in the manuscript establish the general results.
"""
import json
import mpmath as mp

mp.mp.dps = 40
M = mp.mpf
betas = [M('1.1'), M('1.5'), M(2), M(3), M('-.5'), M(-1), M('-1.5'), M(-2), M(-3)]

def rho(beta, n, r):
    return (-1)**n * beta * mp.quad(
        lambda u: u**n * (1/(1+r*u) - beta**n/(1+beta*r*u)), [0, 1])

def rho_derivative(beta, n, j, r):
    return (-1)**(n+j) * beta * mp.factorial(j) * mp.quad(
        lambda u: u**(n+j) * (1/(1+r*u)**(j+1)
                  - beta**(n+j)/(1+beta*r*u)**(j+1)), [0, 1])

def psi(m, b):
    return mp.quad(lambda u: u**(2*m+2) *
        (b**(-2*m)/(b-u)**2 - 1/(1+u)**2), [0, 1])

def bisect_decreasing(fun, lo, hi, steps=70):
    assert fun(lo)>0 and fun(hi)<0
    for _ in range(steps):
        mid=(lo+hi)/2
        if fun(mid)>0: lo=mid
        else: hi=mid
    return (lo+hi)/2

checks = dict(positive_beta_derivative_signs=0, negative_even_monotonicity=0,
    negative_odd_convexity=0, duality_identities=0, uniform_path_bounds=0,
    derivative_formula_checks=0, origin_extensions=0, whole_domain_identities=0)
max_duality_error = M(0)
max_derivative_error = M(0)
for beta in betas:
    cap = min(M(1), 1/abs(beta))
    c = M('.6')*cap
    for n in range(1,8):
        origin = (-1)**(n+1)*(beta**(n+1)-beta)/(n+1)
        assert abs(rho(beta,n,M(0))-origin)<M('1e-32')
        checks['origin_extensions'] += 1
        for r in [-c, M(0), c]:
            if beta>1:
                for j in range(4):
                    assert (-1)**(n+j+1)*rho_derivative(beta,n,j,r)>0
                    checks['positive_beta_derivative_signs'] += 1
            elif n%2==0:
                assert rho_derivative(beta,n,1,r)>0
                checks['negative_even_monotonicity'] += 1
            else:
                assert rho(beta,n,r)>0 and rho_derivative(beta,n,2,r)>0
                checks['negative_odd_convexity'] += 1
                b=-beta
                err=abs(rho(-b,n,r)-b**(n+2)*rho(-1/b,n,-b*r))
                max_duality_error=max(max_duality_error,err)
                assert err<M('1e-32')
                checks['duality_identities'] += 1
        path=[-c,c/2,-c/3,M(0),c/7]
        exact=sum(beta*mp.log1p(r)-mp.log1p(beta*r) for r in path)
        poly=sum(sum((-1)**k*(beta**k-beta)/k*r**k for k in range(2,n+1)) for r in path)
        C=abs(beta)/(n+1)*(1/(1-c)+abs(beta)**n/(1-abs(beta)*c))
        bound=C*sum(abs(r)**(n+1) for r in path)
        assert abs(exact-poly)<=bound+M('1e-32')
        assert bound<=C*c**(n-1)*sum(r*r for r in path)+M('1e-32')
        checks['uniform_path_bounds'] += 1
    # Independent derivative check from the direct log / polynomial quotient.
    for n in [1,3,6]:
        r=c/2
        def direct(x):
            val=beta*mp.log1p(x)-mp.log1p(beta*x)
            val-=sum((-1)**k*(beta**k-beta)/k*x**k for k in range(2,n+1))
            return val/x**(n+1)
        err=abs(mp.diff(direct,r)-rho_derivative(beta,n,1,r))
        max_derivative_error=max(max_derivative_error,err)
        assert err<M('1e-30')
        checks['derivative_formula_checks'] += 1

# The finite remainder identity holds beyond the Taylor convergence interval.
# Check valid points outside |r|<min(1,1/|beta|) against direct logarithms.
for beta,r in [(M('1.1'),M('1.25')), (M(2),M('1.2')),
               (M('-.5'),M('1.5')), (M(-2),M('-.75'))]:
    assert 1+r>0 and 1+beta*r>0
    assert abs(r)>min(M(1),1/abs(beta))
    for n in range(1,8):
        direct=beta*mp.log1p(r)-mp.log1p(beta*r)
        direct-=sum((-1)**k*(beta**k-beta)/k*r**k for k in range(2,n+1))
        assert abs(direct-r**(n+1)*rho(beta,n,r))<M('1e-30')
        checks['whole_domain_identities'] += 1

thresholds={}
for m in range(4):
    b=bisect_decreasing(lambda b:psi(m,b),M('1.01'),M(4))
    assert abs(psi(m,b))<M('1e-18')
    thresholds[str(m)]={'odd_order':2*m+1,'b_m':mp.nstr(b,24),
        'residual':mp.nstr(psi(m,b),8)}
    if m>=1:assert b<2 and psi(m,M(2))<0

for b in [M('1.1'),M('1.5'),M(2),M(3),M(6)]:
    F=(b+1)/(2*(b-1))+2*b*mp.log(1-1/b)+2*mp.log(2)
    assert abs(F-psi(0,b))<M('1e-32')

local_slopes=[]
step=M('1e-5')
for n in [1,3,5,7]:
    roots=[]
    for b in [1-step,1+step]:
        rt=mp.findroot(lambda r:rho_derivative(-b,n,1,r),[-M('.01'),M('.01')])
        assert abs(rho_derivative(-b,n,1,rt))<M('1e-30')
        roots.append(rt)
    numerical=(roots[1]-roots[0])/(2*step)
    exact=-M((n+1)*(n+3))/(4*(n+2))
    assert abs(numerical-exact)<M('1e-7')
    assert roots[1]<roots[0]
    local_slopes.append({'n':n,'analytic':mp.nstr(exact,18),'central_difference':mp.nstr(numerical,18)})

print(json.dumps({'precision_digits':mp.mp.dps,'checks_passed':checks,
    'max_duality_absolute_error':mp.nstr(max_duality_error,10),
    'max_derivative_formula_absolute_error':mp.nstr(max_derivative_error,10),
    'thresholds':thresholds,'local_minimizer_slopes':local_slopes,
    'status':'all checks passed','qualification':'Numerical checks do not replace the analytic proofs.'},ensure_ascii=False,indent=2))
