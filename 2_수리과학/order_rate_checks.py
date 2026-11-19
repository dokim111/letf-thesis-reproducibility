#!/usr/bin/env python3
"""Check the two-sided order-rate bounds in Theorem 12.

Numerical checks supplement, and do not replace, the proof. Selected roots
are recomputed by 60-decimal quadrature after z=p(1-u), p=2m+3.
This file does not modify the earlier root table or the market data.
"""
from pathlib import Path
import csv
import json
import math
import mpmath as mp

HERE = Path(__file__).resolve().parent
mp.mp.dps = 60
C0 = mp.sqrt(1-mp.exp(-1))

def integrals_scaled(m, delta):
    p = mp.mpf(2*m+3)
    points = [mp.mpf(x) for x in (0, 1, 4, 12, 40, 100, 200) if x < p]
    points.append(p)
    weight = lambda z: (1-z/p)**(p-1)
    lhs = mp.quad(lambda z: weight(z)/(delta+z/p)**2, points)
    rhs = mp.quad(lambda z: weight(z)/(2-z/p)**2, points)
    return lhs, rhs

def log_root_equation(m, delta):
    lhs,rhs=integrals_scaled(m,delta)
    return m*mp.log1p(delta) - mp.log(lhs/rhs)/2

rows=list(csv.DictReader((HERE/'order_thresholds.csv').open(encoding='utf-8')))
all_bounds=[]
for row in rows:
    m=int(row['m'])
    if m<3:
        continue
    delta=float(row['b_m'])-1
    lower=math.log(m)/(2*m)
    upper=2*math.log(m)/m
    assert lower<delta<upper
    all_bounds.append(dict(m=m,delta=delta,lower=lower,upper=upper,
                           ratio_m_delta_log_m=m*delta/math.log(m)))

checked=[]
for m in [3,8,20,80,1000]:
    if m<=80:
        initial=mp.mpf(rows[m]['b_m'])-1
    else:
        initial=mp.lambertw(2*m)/m
    delta=mp.findroot(lambda x:log_root_equation(m,x),
                      (initial*mp.mpf('.99'),initial*mp.mpf('1.01')),
                      tol=mp.mpf('1e-50'))
    p=mp.mpf(2*m+3)
    b=1+delta
    scaled_I,scaled_J=integrals_scaled(m,delta)
    I=scaled_I/p;J=scaled_J/p
    lower_I=(1-mp.exp(-1))/(p*(delta+1/p)**2)
    tail_mass=1-(1-1/p)**p
    lower_product=b**m*(delta+1/p)
    upper_product=b**m*delta
    assert I>lower_I and J<1/p
    assert tail_mass>1-mp.exp(-1)
    assert lower_product>C0 and upper_product<2
    assert mp.log(m)/(2*m)<delta<2*mp.log(m)/m
    residual=abs(log_root_equation(m,delta))
    assert residual<mp.mpf('1e-48')
    checked.append(dict(m=m,b_m=mp.nstr(b,45),delta=mp.nstr(delta,45),
        root_log_residual=mp.nstr(residual,8),
        ratio_m_delta_log_m=float(m*delta/mp.log(m)),
        second_order_remainder=float(m*delta-mp.log(m)+mp.log(mp.log(m))),
        tail_mass=float(tail_mass),
        lower_product=float(lower_product),upper_product=float(upper_product),
        lower_I_margin=float(I-lower_I),upper_J_margin=float(1/p-J)))

grid_ratio=[r['ratio_m_delta_log_m'] for r in all_bounds if 8<=r['m']<=80]
report=dict(mp_dps=mp.mp.dps,c0=mp.nstr(C0,40),
    contradiction_uniform_bound=mp.nstr(mp.exp(-mp.mpf('.5')),40),
    grid_m_range=[3,80],all_grid_two_sided_bounds=True,
    ratio_range_m8_to80=[min(grid_ratio),max(grid_ratio)],
    independently_recomputed_roots=checked,numerics_are_not_proofs=True,
    asymptotic_statement='m*(b_m-1)=log(m)-log(log(m))+O(1); (b_m-1)~log(m)/m',
    finer_constant_not_claimed=True)
(HERE/'order_rate_checks_output.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
with (HERE/'order_rate_checks.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(all_bounds[0]));w.writeheader();w.writerows(all_bounds)
print(json.dumps(report,ensure_ascii=False,indent=2))
