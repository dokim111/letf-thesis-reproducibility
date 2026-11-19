#!/usr/bin/env python3
"""Reproduce order-threshold bounds, roots, and integral identities.

All outputs are deterministic.  SciPy roots over m=0,...,80 are checked
against independent 60-decimal mpmath quadrature at selected orders.
The quadrature verification is supplemental to the proofs in the paper.
"""
from pathlib import Path
import csv
import json
import math
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import mpmath as mp

HERE=Path(__file__).resolve().parent

def psi(m,b):
    factor=math.exp(-2*m*math.log(b))
    return quad(lambda u: u**(2*m+2)*(factor/(b-u)**2-1/(1+u)**2),
                0.0,1.0,epsabs=2e-13,epsrel=2e-13,limit=200)[0]

def upper(m):
    return brentq(lambda b:m*math.log(b)+math.log(b-1)-math.log(2),
                  1+1e-12,3.1,xtol=5e-15)

rows=[]
for m in range(81):
    bar=upper(m)
    root=brentq(lambda b:psi(m,b),1+1/(100*(m+1)),bar,xtol=5e-15)
    rows.append(dict(m=m,n=2*m+1,b_m=root,b_bar_m=bar,gap=bar-root,
                     scaled_gap=(root-1)*m/math.log(m) if m>=2 else None,
                     psi_residual=psi(m,root)))

mp.mp.dps=60

def psi_mp(m,b):
    # Direct representation on [0,1], evaluated without logarithmic subtraction.
    return mp.quad(lambda u: u**(2*m+2)*(b**(-2*m)/(b-u)**2-1/(1+u)**2),
                   [0,mp.mpf('.5'),mp.mpf('.9'),1])

def transformed_mp(m,b):
    split=1/b
    return b*mp.quad(lambda u:u**(2*m+2)/(1-u)**2,[0,split/2,split])-mp.quad(
        lambda u:u**(2*m+2)/(1+u)**2,[0,mp.mpf('.5'),1])

def difference_mp(m,b):
    split=1/b
    left=mp.quad(lambda u:u**(2*m+2)*(u*u-split*split)*(
        b/(1-u)**2-1/(1+u)**2),[0,split/2,split])
    right=mp.quad(lambda u:-u**(2*m+2)*(u*u-split*split)/(1+u)**2,
                  [split,(1+split)/2,1])
    return left+right

checked=[]
for m in [0,1,2,3,5,10,20,40,80]:
    init=mp.mpf(str(rows[m]['b_m']))
    root=mp.findroot(lambda b:psi_mp(m,b),(init-mp.mpf('1e-5'),init+mp.mpf('1e-5')),
                     tol=mp.mpf('1e-55'))
    bar_init=mp.mpf(str(rows[m]['b_bar_m']))
    bar=mp.findroot(lambda b:m*mp.log(b)+mp.log(b-1)-mp.log(2),
                   (bar_init-mp.mpf('1e-5'),bar_init+mp.mpf('1e-5')),
                   tol=mp.mpf('1e-55'))
    checked.append(dict(m=m,b_m=mp.nstr(root,45),b_bar_m=mp.nstr(bar,45),
                        residual=mp.nstr(abs(psi_mp(m,root)),8),
                        scipy_difference=float(abs(root-rows[m]['b_m']))))

identity_errors=[]
difference_errors=[]
negative_checks=[]
for m in [0,1,2,5,20,80]:
    for btext in ['1.05','1.2','1.8','2.5','3.0']:
        b=mp.mpf(btext)
        direct=psi_mp(m,b)
        identity_errors.append(abs(direct-transformed_mp(m,b)))
        diff=psi_mp(m+1,b)-direct/(b*b)
        difference_errors.append(abs(diff-difference_mp(m,b)))
        negative_checks.append(diff<0)

assert all(rows[m+1]['b_m']<rows[m]['b_m'] for m in range(80))
assert all(1<row['b_m']<row['b_bar_m'] for row in rows)
assert all(row['b_bar_m']-1<=2*math.log(row['m'])/row['m'] for row in rows if row['m']>=3)
assert all(negative_checks)
assert max(identity_errors)<mp.mpf('1e-48')
assert max(difference_errors)<mp.mpf('1e-48')
assert max(x['scipy_difference'] for x in checked)<5e-13

with (HERE/'order_thresholds.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows(rows)
report=dict(m_range=[0,80],mp_dps=mp.mp.dps,checked_roots=checked,
    maximum_change_of_variable_error=mp.nstr(max(identity_errors),8),
    maximum_signed_difference_identity_error=mp.nstr(max(difference_errors),8),
    strict_ordering_all_80_comparisons=True,strict_upper_bounds_all_81_orders=True,
    asymptotic_explicit_bound_all_m_ge_3=True,
    numerical_checks_are_not_proofs=True)
(HERE/'order_threshold_checks_output.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
