"""Archived interval routines for use with controlled document-level inputs."""
import time

import numpy as np
from scipy.stats import beta

SEED = 20261026
DRAWS = 20000
WEIGHTS=np.array([.3,.4,.3])


class IntegrityError(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise IntegrityError(message)


def _bootstrap(rows,cfg):
 result=[];drawfiles={}
 rng=np.random.default_rng(cfg['analysis']['seed'])
 for corpus in cfg['analysis']['corpora']:
  group=[x for x in rows if x['corpus']==corpus and x['level']=='r4']
  strata=[[r for r in sorted(group,key=lambda x:x['item_id']) if r['stratum']==s] for s in range(3)]
  if [len(x) for x in strata]!=[30,40,30]:raise IntegrityError('Bootstrap stratum counts changed')
  columns=['G_P','I']
  matrices=[np.array([[x[c] for c in columns] for x in block],dtype=float) for block in strata]
  draws=np.zeros((cfg['analysis']['bootstrap_draws'],2),dtype=float)
  for block in matrices:
   n=len(block)
   indices=rng.integers(0,n,size=(len(draws),n))
   draws+=(block[indices].sum(axis=1))/100
  drawfiles[corpus]=draws
  for j,name in enumerate(columns):
   observed=float(np.mean([x[name] for x in group]))
   primary=np.quantile(draws[:,j],cfg['analysis']['primary_quantiles'],method='linear')
   desc=np.quantile(draws[:,j],cfg['analysis']['descriptive_quantiles'],method='linear')
   result.append({'corpus':corpus,'level':'r4','contrast':name,'n':100,'estimate':observed,
     'ci_98_75_low':float(primary[0]),'ci_98_75_high':float(primary[1]),
     'ci_95_low':float(desc[0]),'ci_95_high':float(desc[1]),'draws':len(draws)})
 return result,drawfiles


def cp(k,n,alpha=.05):
    require(type(k) is int and type(n) is int and 0 <= k <= n and n > 0 and 0 < alpha < 1, 'Invalid exact-binomial inputs')
    return [0.0 if k==0 else float(beta.ppf(alpha/2,k,n-k+1)),
            1.0 if k==n else float(beta.ppf(1-alpha/2,k+1,n-k))]


def paired_bootstrap(lo, hi, strata, quiet=True):
    """Chunking bounds memory; progress never consumes RNG or changes draw order."""
    dl, du = np.zeros((DRAWS, lo.shape[1])), np.zeros((DRAWS, hi.shape[1]))
    rng = np.random.Generator(np.random.PCG64(SEED))
    start = time.monotonic()
    for h, (w, ix) in enumerate(zip(WEIGHTS, strata), 1):
        require(len(ix) > 0, 'Do not redistribute empty-stratum weights')
        for begin in range(0, DRAWS, 512):
            end = min(begin + 512, DRAWS)
            sampled = ix[rng.integers(0, len(ix), size=(end - begin, len(ix)))]
            dl[begin:end] += w * lo[sampled].mean(axis=1)
            du[begin:end] += w * hi[sampled].mean(axis=1)
        if not quiet:
            elapsed = time.monotonic() - start
            print(f'Paired bootstrap strata {h}/3; {DRAWS} draws; elapsed {elapsed:.1f}s; ETA {elapsed/h*(3-h):.1f}s', flush=True)
    return dl, du


def attenuation_cp(dlo, dhi, strata):
    """24 two-sided exact-binomial components across the two prespecified corpora."""
    require(np.all(dlo <= dhi) and np.all(dlo >= -2) and np.all(dhi <= 2), 'Invalid D bounds')
    alpha = .05 / 24
    lower, upper, details = 0., 0., []
    for h, (w, ix) in enumerate(zip(WEIGHTS, strata)):
        require(len(ix), 'Empty stratum: standardized CP unavailable')
        components, cl, cu = [], -2., -2.
        for threshold in (-1, 0, 1, 2):
            definite = int((dlo[ix] >= threshold).sum())
            possible = int((dhi[ix] >= threshold).sum())
            left = cp(definite, len(ix), alpha)[0]
            right = cp(possible, len(ix), alpha)[1]
            components.append({'threshold': threshold, 'definite': definite, 'possible': possible,
                               'n': len(ix), 'interval': [left, right]})
            cl += left
            cu += right
        lower += w * cl
        upper += w * cu
        details.append({'stratum': h, 'weight': float(w), 'thresholds': components, 'D_envelope': [cl, cu]})
    return {'envelope': [float(lower), float(upper)], 'alpha_per_component': alpha,
            'family_components': 24, 'family_alpha': .05, 'strata': details,
            'assumptions': 'Independent-document binomial sampling conditional on S1 and resolved labels; no human-error model.'}


def direction(bootstrap, conservative):
    if bootstrap[0] > 0 and conservative[0] > 0:
        return 'positive_supported'
    if bootstrap[1] < 0 and conservative[1] < 0:
        return 'negative_supported'
    return 'not_directionally_confirmed'
