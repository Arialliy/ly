"""Independent arithmetic audit; no images, training, packages, or GPU needed."""
import copy
import hashlib
import itertools
import json
import math
import pathlib
import random

OUT = pathlib.Path(__file__).resolve().parent
T, S, D, R = 4, 3, 3, 2

def dot(a, b):
    return sum(x*y for x, y in zip(a, b))

def make_case(seed, masked=False):
    rng = random.Random(seed)
    return {
        'h': [[[rng.uniform(-1, 1) for _ in range(D)] for _ in range(S)] for _ in range(T)],
        'logits': [[[rng.uniform(-2, 2) for _ in range(S)] for _ in range(S)] for _ in range(T-1)],
        'a': [[rng.uniform(-1, 1) for _ in range(D)] for _ in range(R)],
        'lam': [0.7, -0.4], 'w': [0.3, -0.5, 0.8], 'b': 0.15,
        'mask': [[[not masked or abs(x-y) <= 1 for y in range(S)] for x in range(S)] for _ in range(T-1)],
    }

def transitions(c):
    p=[]
    for t in range(T-1):
        rows=[]
        for x in range(S):
            logits=c['logits'][t][x]; mask=c['mask'][t][x]
            largest=max(l for l,m in zip(logits,mask) if m)
            e=[math.exp(l-largest) if m else 0.0 for l,m in zip(logits,mask)]
            rows.append([v/sum(e) for v in e])
        p.append(rows)
    return p

def enumerate_case(c, endpoint):
    p=transitions(c); records=[]
    for prefix in itertools.product(range(S), repeat=T-1):
        path=prefix+(endpoint,)
        q=math.prod(p[t][path[t+1]][path[t]] for t in range(T-1))
        v=[sum(c['h'][t][path[t]][d] for t in range(T)) for d in range(D)]
        z=[dot(a,v) for a in c['a']]
        score=c['b']+dot(c['w'],v)+dot(c['lam'],[u*u for u in z])
        records.append((path,q,v,z,score))
    m=[sum(q*z[r] for _,q,_,z,_ in records) for r in range(R)]
    n=[sum(q*z[r]*z[r] for _,q,_,z,_ in records) for r in range(R)]
    return records,m,n,sum(q*score for _,q,_,_,score in records)

def recurse(c,endpoint):
    p=transitions(c)
    z=[[[dot(a,c['h'][t][x]) for a in c['a']] for x in range(S)] for t in range(T)]
    m=copy.deepcopy(z[0]); n=[[u*u for u in row] for row in m]
    linear=[dot(c['w'],h) for h in c['h'][0]]
    for t in range(1,T):
        nm=[]; nn=[]; nl=[]
        for x in range(S):
            weights=p[t-1][x]
            bm=[sum(weights[y]*m[y][r] for y in range(S)) for r in range(R)]
            nm.append([z[t][x][r]+bm[r] for r in range(R)])
            nn.append([z[t][x][r]**2+2*z[t][x][r]*bm[r]+sum(weights[y]*n[y][r] for y in range(S)) for r in range(R)])
            nl.append(dot(c['w'],c['h'][t][x])+sum(weights[y]*linear[y] for y in range(S)))
        m,n,linear=nm,nn,nl
    score=c['b']+linear[endpoint]+dot(c['lam'],n[endpoint])
    return m[endpoint],n[endpoint],score

def derivative_logits(c,endpoint,t,x,y):
    p=transitions(c); rec,*_=enumerate_case(c,endpoint)
    if not c['mask'][t][x][y]: return 0.0
    return sum(q*score*((1.0 if path[t]==y else 0.0)-p[t][x][y])
               for path,q,_,_,score in rec if path[t+1]==x)

def derivative_h(c,endpoint,t,x,d):
    rec,*_=enumerate_case(c,endpoint)
    return sum(q*(c['w'][d]+2*sum(c['lam'][r]*c['a'][r][d]*z[r] for r in range(R)))
               for path,q,_,z,_ in rec if path[t]==x)

def main():
    errors={'normalization':0.0,'first_moment':0.0,'second_moment':0.0,'score':0.0}
    cases=[]
    for seed in range(20):
        c=make_case(seed,seed%2==1)
        for endpoint in range(S):
            records,m,n,score=enumerate_case(c,endpoint)
            rm,rn,rs=recurse(c,endpoint)
            errors['normalization']=max(errors['normalization'],abs(sum(q for _,q,_,_,_ in records)-1))
            errors['first_moment']=max(errors['first_moment'],max(abs(a-b) for a,b in zip(m,rm)))
            errors['second_moment']=max(errors['second_moment'],max(abs(a-b) for a,b in zip(n,rn)))
            errors['score']=max(errors['score'],abs(score-rs))
            if seed==0: cases.append({'endpoint':endpoint,'positive_paths':sum(q>0 for _,q,_,_,_ in records),'m':m,'n':n,'score':score})
    c=make_case(20261118,True); endpoint=2; eps=1e-5
    grad_errors={'logit':0.0,'feature':0.0}; checked={'logit':0,'feature':0}
    for name,tmax,last,deriv,key in [('logits',T-1,S,derivative_logits,'logit'),('h',T,D,derivative_h,'feature')]:
        for t in range(tmax):
            for x in range(S):
                for k in range(last):
                    old=c[name][t][x][k]
                    c[name][t][x][k]=old+eps; plus=recurse(c,endpoint)[2]
                    c[name][t][x][k]=old-eps; minus=recurse(c,endpoint)[2]
                    c[name][t][x][k]=old
                    actual=(plus-minus)/(2*eps); expected=deriv(c,endpoint,t,x,k)
                    grad_errors[key]=max(grad_errors[key],abs(actual-expected));checked[key]+=1
    result={
        'scope':'Independent new small-graph replication; not recovery of the original plan table.',
        'setting':{'T':T,'states':S,'D':D,'R':R,'random_seeds':20,'endpoints_per_seed':S,'masked_cases':10,'float':'Python float (double)'},
        'max_absolute_errors':errors,'gradient_max_absolute_errors':grad_errors,'gradient_components_checked':checked,'finite_difference_epsilon':eps,
        'first_seed_examples':cases,
        'checks':{'recurrence_matches_enumeration':max(errors.values())<1e-12,'finite_difference_matches_path_derivatives':max(grad_errors.values())<1e-7},
        'limits':['No GPU tensors or autodiff tested.','No image features, correspondence learning, or detection performance tested.','Linear w projection explicitly propagated in this implementation.'],
        'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
    }
    (OUT/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    assert all(result['checks'].values())

if __name__=='__main__': main()
