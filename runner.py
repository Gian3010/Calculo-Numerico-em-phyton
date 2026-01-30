
import numpy as np, pandas as pd, math, matplotlib.pyplot as plt
from metodos_raizes import bisection, newton, secant

# create outputs folders
import os
tables_dir = 'outputs/tables'
figs_dir = 'outputs/figures'
os.makedirs(tables_dir, exist_ok=True)
os.makedirs(figs_dir, exist_ok=True)

# --- Exercise I functions ---
def f1(x):
    return x*math.exp(-x)
def df1(x):
    return math.exp(-x) * (1 - x)

def f2(x):
    return x**3 - x - 3
def df2(x):
    return 3*x**2 - 1

def f3(x):
    return math.atan(x)
def df3(x):
    return 1/(1+x**2)

# helper to save table
def save_table(df, name):
    df.to_csv(os.path.join(tables_dir, name + '.csv'), index=False)

# helper to save plot of function with initial point(s)
def plot_function(func, xs, highlights=None, title='f(x)', fname='figure.png'):
    plt.figure(figsize=(6,4))
    plt.plot(xs, [func(x) for x in xs])
    plt.axhline(0, color='black', linewidth=0.5)
    if highlights:
        for x,h in highlights.items():
            plt.scatter([x],[func(x)], label=h)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, fname))
    plt.close()

xs = [i*0.01 for i in range(-500,501)]  # -5 to 5

# f1 tests (x0 = 2 and x0 = 0.5)
res_f1_x0_2 = newton(f1, df1, 2.0, eps=1e-8, maxit=50)
save_table(res_f1_x0_2['table'], 'newton_f1_x0_2')
plot_function(f1, xs, highlights={2.0:'x0=2'}, title='f1(x)=x e^{-x}', fname='f1_x0_2.png')

res_f1_x0_05 = newton(f1, df1, 0.5, eps=1e-8, maxit=50)
save_table(res_f1_x0_05['table'], 'newton_f1_x0_0.5')
plot_function(f1, xs, highlights={0.5:'x0=0.5'}, title='f1(x)=x e^{-x}', fname='f1_x0_0.5.png')

# f2 tests (x0 = 0.57 and 0.62)
res_f2_x0_057 = newton(f2, df2, 0.57, eps=1e-8, maxit=50)
save_table(res_f2_x0_057['table'], 'newton_f2_x0_0.57')
plot_function(f2, xs, highlights={0.57:'x0=0.57'}, title='f2(x)=x^3-x-3', fname='f2_x0_0.57.png')

res_f2_x0_062 = newton(f2, df2, 0.62, eps=1e-8, maxit=50)
save_table(res_f2_x0_062['table'], 'newton_f2_x0_0.62')
plot_function(f2, xs, highlights={0.62:'x0=0.62'}, title='f2(x)=x^3-x-3', fname='f2_x0_0.62.png')

# f3 tests (x0 = 1.45 and 1)
try:
    res_f3_x0_145 = newton(f3, df3, 1.45, eps=1e-8, maxit=50)
    save_table(res_f3_x0_145['table'], 'newton_f3_x0_1.45')
    plot_function(f3, xs, highlights={1.45:'x0=1.45'}, title='f3(x)=arctg(x)', fname='f3_x0_1.45.png')
except Exception as e:
    import pandas as pd
    pd.DataFrame([{'error':str(e)}]).to_csv(os.path.join(tables_dir,'newton_f3_x0_1.45_error.csv'), index=False)
try:
    res_f3_x0_1 = newton(f3, df3, 1.0, eps=1e-8, maxit=50)
    save_table(res_f3_x0_1['table'], 'newton_f3_x0_1.0')
    plot_function(f3, xs, highlights={1.0:'x0=1.0'}, title='f3(x)=arctg(x)', fname='f3_x0_1.0.png')
except Exception as e:
    import pandas as pd
    pd.DataFrame([{'error':str(e)}]).to_csv(os.path.join(tables_dir,'newton_f3_x0_1.0_error.csv'), index=False)

# --- Exercise II ---
eps0 = 8.85e-12
F = 1.5
p = 2e-5
q = 5e-5
r = 1.0

C = 4*math.pi*eps0*F/(p*q)  # note sign: we'll define f(x)= x/(x^2+r^2)^{3/2} - 4πε0 F/(pq)
const_term = 4*math.pi*eps0*F/(p*q)

def f_ex2(x):
    return x/(x**2 + r**2)**1.5 - const_term

def df_ex2(x):
    # derivative of x*(x^2+r^2)^(-3/2)
    return ( (x**2 + r**2)**(-3/2) + x * (-3/2)*(x**2 + r**2)**(-5/2) * (2*x) )

# 1) check existence: evaluate f(0) and f(1)
f0 = f_ex2(0.0)
f1 = f_ex2(1.0)
# Save these in a tiny csv
import pandas as pd
pd.DataFrame([{'x':0.0,'f(x)':f0},{'x':1.0,'f(x)':f1}]).to_csv(os.path.join(tables_dir,'ex2_f0_f1.csv'), index=False)

# 2) Bisection on [0,1]
res_ex2_bis = bisection(f_ex2, 0.0, 1.0, eps=1e-8, maxit=50)
save_table(res_ex2_bis['table'], 'ex2_bisection_0_1')

# 3) Newton x0 = 0.3
res_ex2_newton_03 = newton(f_ex2, df_ex2, 0.3, eps=1e-8, maxit=50)
save_table(res_ex2_newton_03['table'], 'ex2_newton_x0_0.3')
try:
    res_ex2_newton_07_safe = newton_safe(f_ex2, df_ex2, 0.7, eps=1e-4, maxit=50, max_step=0.5, damping=True)
    if 'table' in res_ex2_newton_07_safe:
        save_table(res_ex2_newton_07_safe['table'], 'ex2_newton_x0_0.7_safe')
    if res_ex2_newton_07_safe.get('root') is not None:
        # optionally save figure marking root
        pass
except Exception as e:
    import pandas as pd
    pd.DataFrame([{'error':str(e)}]).to_csv(os.path.join(tables_dir,'ex2_newton_x0_0.7_safe_error.csv'), index=False)

# 4) Secant x0=0.3, x1=0.6
res_ex2_secant = secant(f_ex2, 0.3, 0.6, eps=1e-8, maxit=50)
save_table(res_ex2_secant['table'], 'ex2_secant_0.3_0.6')

# 5) Newton x0=0.7, eps=1e-4, maxit=10
try:
    res_ex2_newton_07 = newton(f_ex2, df_ex2, 0.7, eps=1e-4, maxit=10)
    save_table(res_ex2_newton_07['table'], 'ex2_newton_x0_0.7_eps1e-4')
except Exception as e:
    import pandas as pd
    pd.DataFrame([{'error':str(e)}]).to_csv(os.path.join(tables_dir,'ex2_newton_x0_0.7_error.csv'), index=False)

# plot f_ex2 on [0,2]
xs2 = [i*0.01 for i in range(0,201)]
plot_function(f_ex2, xs2, highlights={0.3:'x0=0.3',0.7:'x0=0.7'}, title='Exercise II: f(x)', fname='ex2_function.png')

# Save summary of roots
summary = []
summary.append({'method':'bisection','root':res_ex2_bis.get('root') if isinstance(res_ex2_bis, dict) else None,'f(root)':res_ex2_bis.get('f_root') if isinstance(res_ex2_bis, dict) else None,'niter':res_ex2_bis.get('niter') if isinstance(res_ex2_bis, dict) else None})
summary.append({'method':'newton_0.3','root':res_ex2_newton_03.get('root') if isinstance(res_ex2_newton_03, dict) else None,'f(root)':res_ex2_newton_03.get('f_root') if isinstance(res_ex2_newton_03, dict) else None,'niter':res_ex2_newton_03.get('niter') if isinstance(res_ex2_newton_03, dict) else None})
summary.append({'method':'secant','root':res_ex2_secant.get('root') if isinstance(res_ex2_secant, dict) else None,'f(root)':res_ex2_secant.get('f_root') if isinstance(res_ex2_secant, dict) else None,'niter':res_ex2_secant.get('niter') if isinstance(res_ex2_secant, dict) else None})
if 'res_ex2_newton_07' in globals():
    summary.append({'method':'newton_0.7_eps1e-4','root':res_ex2_newton_07.get('root') if isinstance(res_ex2_newton_07, dict) else None,'f(root)':res_ex2_newton_07.get('f_root') if isinstance(res_ex2_newton_07, dict) else None,'niter':res_ex2_newton_07.get('niter') if isinstance(res_ex2_newton_07, dict) else None})
else:
    summary.append({'method':'newton_0.7_eps1e-4','root':None,'f(root)':None,'niter':None})
import pandas as pd
pd.DataFrame(summary).to_csv(os.path.join(tables_dir,'ex2_summary_roots.csv'), index=False)

print('Runner finished. Tables and figures saved.')
