# metodos_raizes.py
import math
import pandas as pd
from typing import Callable

def _df_from_rows(rows, cols):
    return pd.DataFrame(rows, columns=cols)

def bisection(f: Callable[[float], float], a: float, b: float, eps: float = 1e-8, maxit: int = 50):
    fa = f(a); fb = f(b)
    if fa == 0.0:
        rows = [(1, a, 0.0, None)]
        return {'root': a, 'f_root': 0.0, 'niter': 1, 'table': _df_from_rows(rows, ['k','xk','f(xk)','step'])}
    if fb == 0.0:
        rows = [(1, b, 0.0, None)]
        return {'root': b, 'f_root': 0.0, 'niter': 1, 'table': _df_from_rows(rows, ['k','xk','f(xk)','step'])}
    if fa * fb > 0:
        raise ValueError("f(a) e f(b) têm mesmo sinal; bissecção não aplicável.")

    rows = []
    prev_mid = None
    for k in range(1, maxit+1):
        mid = 0.5*(a + b)
        fm = f(mid)
        step = None if prev_mid is None else (mid - prev_mid)
        rows.append((k, mid, fm, step))
        # critérios de parada
        if abs(fm) <= eps or (b - a)/2.0 <= eps:
            return {'root': mid, 'f_root': fm, 'niter': k, 'table': _df_from_rows(rows, ['k','xk','f(xk)','step'])}
        # atualiza intervalo
        if fa * fm < 0:
            b = mid
            fb = fm
        else:
            a = mid
            fa = fm
        prev_mid = mid

    # retornamos último
    mid = 0.5*(a + b)
    fm = f(mid)
    return {'root': mid, 'f_root': fm, 'niter': maxit, 'table': _df_from_rows(rows, ['k','xk','f(xk)','step'])}


def newton(f: Callable[[float], float], fp: Callable[[float], float], x0: float, eps: float = 1e-8, maxit: int = 20, safe: bool = False):
    """
    safe: se True aplica damping simples para limitar tamanho de passo caso seja necessário.
    """
    rows = []
    xk = x0
    prev_x = None
    tol_fp = 1e-14

    for k in range(1, maxit+1):
        fk = f(xk)
        fpk = fp(xk)
        step = None if prev_x is None else (xk - prev_x)
        rows.append((k, xk, fk, fpk, step))

        if abs(fk) <= eps:
            return {'root': xk, 'f_root': fk, 'niter': k, 'table': _df_from_rows(rows, ['k','xk','f(xk)','f\'(xk)','step'])}

        if abs(fpk) < tol_fp:
            raise ZeroDivisionError(f"Derivada aproximadamente zero em iteração {k} (x = {xk}).")

        # passo de Newton
        delta = - fk / fpk
        # opcional: damping simples para segurança
        if safe and abs(delta) > 1.0:
            # reduz o passo por fator 1/2 até ficar aceitável (estratégia simples)
            factor = 1.0
            while abs(delta * factor) > 1.0:
                factor *= 0.5
                if factor < 1e-6:
                    break
            delta = delta * factor

        xnext = xk + delta
        step_next = xnext - xk

        if abs(step_next) <= eps:
            # registra x_{k+1} e sai
            rows.append((k+1, xnext, f(xnext), fp(xnext), step_next))
            return {'root': xnext, 'f_root': f(xnext), 'niter': k+1, 'table': _df_from_rows(rows, ['k','xk','f(xk)','f\'(xk)','step'])}

        prev_x = xk
        xk = xnext

    # atingiu maxit
    fk = f(xk)
    return {'root': xk, 'f_root': fk, 'niter': maxit, 'table': _df_from_rows(rows, ['k','xk','f(xk)','f\'(xk)','step'])}


def secant(f: Callable[[float], float], x0: float, x1: float, eps: float = 1e-8, maxit: int = 20):
    rows = []
    f0 = f(x0)
    rows.append((1, x0, f0, None))
    if abs(f0) <= eps:
        return {'root': x0, 'f_root': f0, 'niter': 1, 'table': _df_from_rows(rows, ['k','xk','f(xk)','step'])}
    f1 = f(x1)
    rows.append((2, x1, f1, x1 - x0))
    if abs(f1) <= eps:
        return {'root': x1, 'f_root': f1, 'niter': 2, 'table': _df_from_rows(rows, ['k','xk','f(xk)','step'])}

    x_prev, x_curr = x0, x1
    f_prev, f_curr = f0, f1
    tol_den = 1e-14

    for k in range(3, maxit+1):
        den = (f_curr - f_prev)
        if abs(den) < tol_den:
            raise ZeroDivisionError(f"Denominador muito pequeno na iteração {k}.")
        x_next = x_curr - f_curr * (x_curr - x_prev) / den
        f_next = f(x_next)
        step = x_next - x_curr
        rows.append((k, x_next, f_next, step))
        if abs(f_next) <= eps or abs(step) <= eps:
            return {'root': x_next, 'f_root': f_next, 'niter': k, 'table': _df_from_rows(rows, ['k','xk','f(xk)','step'])}
        x_prev, f_prev = x_curr, f_curr
        x_curr, f_curr = x_next, f_next

    return {'root': x_curr, 'f_root': f_curr, 'niter': maxit, 'table': _df_from_rows(rows, ['k','xk','f(xk)','step'])}
