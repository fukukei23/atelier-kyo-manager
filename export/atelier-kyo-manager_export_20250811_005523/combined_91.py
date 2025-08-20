
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\functions\theta.py ===
from .functions import defun, defun_wrapped

@defun
def _jacobi_theta2(ctx, z, q):
    extra1 = 10
    extra2 = 20
    # the loops below break when the fixed precision quantities
    # a and b go to zero;
    # right shifting small negative numbers by wp one obtains -1, not zero,
    # so the condition a**2 + b**2 > MIN is used to break the loops.
    MIN = 2
    if z == ctx.zero:
        if (not ctx._im(q)):
            wp = ctx.prec + extra1
            x = ctx.to_fixed(ctx._re(q), wp)
            x2 = (x*x) >> wp
            a = b = x2
            s = x2
            while abs(a) > MIN:
                b = (b*x2) >> wp
                a = (a*b) >> wp
                s += a
            s = (1 << (wp+1)) + (s << 1)
            s = ctx.ldexp(s, -wp)
        else:
            wp = ctx.prec + extra1
            xre = ctx.to_fixed(ctx._re(q), wp)
            xim = ctx.to_fixed(ctx._im(q), wp)
            x2re = (xre*xre - xim*xim) >> wp
            x2im = (xre*xim) >> (wp-1)
            are = bre = x2re
            aim = bim = x2im
            sre = (1<<wp) + are
            sim = aim
            while are**2 + aim**2 > MIN:
                bre, bim = (bre * x2re - bim * x2im) >> wp, \
                           (bre * x2im + bim * x2re) >> wp
                are, aim = (are * bre - aim * bim) >> wp,   \
                           (are * bim + aim * bre) >> wp
                sre += are
                sim += aim
            sre = (sre << 1)
            sim = (sim << 1)
            sre = ctx.ldexp(sre, -wp)
            sim = ctx.ldexp(sim, -wp)
            s = ctx.mpc(sre, sim)
    else:
        if (not ctx._im(q)) and (not ctx._im(z)):
            wp = ctx.prec + extra1
            x = ctx.to_fixed(ctx._re(q), wp)
            x2 = (x*x) >> wp
            a = b = x2
            c1, s1 = ctx.cos_sin(ctx._re(z), prec=wp)
            cn = c1 = ctx.to_fixed(c1, wp)
            sn = s1 = ctx.to_fixed(s1, wp)
            c2 = (c1*c1 - s1*s1) >> wp
            s2 = (c1 * s1) >> (wp - 1)
            cn, sn = (cn*c2 - sn*s2) >> wp, (sn*c2 + cn*s2) >> wp
            s = c1 + ((a * cn) >> wp)
            while abs(a) > MIN:
                b = (b*x2) >> wp
                a = (a*b) >> wp
                cn, sn = (cn*c2 - sn*s2) >> wp, (sn*c2 + cn*s2) >> wp
                s += (a * cn) >> wp
            s = (s << 1)
            s = ctx.ldexp(s, -wp)
            s *= ctx.nthroot(q, 4)
            return s
        # case z real, q complex
        elif not ctx._im(z):
            wp = ctx.prec + extra2
            xre = ctx.to_fixed(ctx._re(q), wp)
            xim = ctx.to_fixed(ctx._im(q), wp)
            x2re = (xre*xre - xim*xim) >> wp
            x2im = (xre*xim) >> (wp - 1)
            are = bre = x2re
            aim = bim = x2im
            c1, s1 = ctx.cos_sin(ctx._re(z), prec=wp)
            cn = c1 = ctx.to_fixed(c1, wp)
            sn = s1 = ctx.to_fixed(s1, wp)
            c2 = (c1*c1 - s1*s1) >> wp
            s2 = (c1 * s1) >> (wp - 1)
            cn, sn = (cn*c2 - sn*s2) >> wp, (sn*c2 + cn*s2) >> wp
            sre = c1 + ((are * cn) >> wp)
            sim = ((aim * cn) >> wp)
            while are**2 + aim**2 > MIN:
                bre, bim = (bre * x2re - bim * x2im) >> wp, \
                           (bre * x2im + bim * x2re) >> wp
                are, aim = (are * bre - aim * bim) >> wp,   \
                           (are * bim + aim * bre) >> wp
                cn, sn = (cn*c2 - sn*s2) >> wp, (sn*c2 + cn*s2) >> wp
                sre += ((are * cn) >> wp)
                sim += ((aim * cn) >> wp)
            sre = (sre << 1)
            sim = (sim << 1)
            sre = ctx.ldexp(sre, -wp)
            sim = ctx.ldexp(sim, -wp)
            s = ctx.mpc(sre, sim)
        #case z complex, q real
        elif not ctx._im(q):
            wp = ctx.prec + extra2
            x = ctx.to_fixed(ctx._re(q), wp)
            x2 = (x*x) >> wp
            a = b = x2
            prec0 = ctx.prec
            ctx.prec = wp
            c1, s1 = ctx.cos_sin(z)
            ctx.prec = prec0
            cnre = c1re = ctx.to_fixed(ctx._re(c1), wp)
            cnim = c1im = ctx.to_fixed(ctx._im(c1), wp)
            snre = s1re = ctx.to_fixed(ctx._re(s1), wp)
            snim = s1im = ctx.to_fixed(ctx._im(s1), wp)
            #c2 = (c1*c1 - s1*s1) >> wp
            c2re = (c1re*c1re - c1im*c1im - s1re*s1re + s1im*s1im) >> wp
            c2im = (c1re*c1im - s1re*s1im) >> (wp - 1)
            #s2 = (c1 * s1) >> (wp - 1)
            s2re = (c1re*s1re - c1im*s1im) >> (wp - 1)
            s2im = (c1re*s1im + c1im*s1re) >> (wp - 1)
            #cn, sn = (cn*c2 - sn*s2) >> wp, (sn*c2 + cn*s2) >> wp
            t1 = (cnre*c2re - cnim*c2im - snre*s2re + snim*s2im) >> wp
            t2 = (cnre*c2im + cnim*c2re - snre*s2im - snim*s2re) >> wp
            t3 = (snre*c2re - snim*c2im + cnre*s2re - cnim*s2im) >> wp
            t4 = (snre*c2im + snim*c2re + cnre*s2im + cnim*s2re) >> wp
            cnre = t1
            cnim = t2
            snre = t3
            snim = t4
            sre = c1re + ((a * cnre) >> wp)
            sim = c1im + ((a * cnim) >> wp)
            while abs(a) > MIN:
                b = (b*x2) >> wp
                a = (a*b) >> wp
                t1 = (cnre*c2re - cnim*c2im - snre*s2re + snim*s2im) >> wp
                t2 = (cnre*c2im + cnim*c2re - snre*s2im - snim*s2re) >> wp
                t3 = (snre*c2re - snim*c2im + cnre*s2re - cnim*s2im) >> wp
                t4 = (snre*c2im + snim*c2re + cnre*s2im + cnim*s2re) >> wp
                cnre = t1
                cnim = t2
                snre = t3
                snim = t4
                sre += ((a * cnre) >> wp)
                sim += ((a * cnim) >> wp)
            sre = (sre << 1)
            sim = (sim << 1)
            sre = ctx.ldexp(sre, -wp)
            sim = ctx.ldexp(sim, -wp)
            s = ctx.mpc(sre, sim)
        # case z and q complex
        else:
            wp = ctx.prec + extra2
            xre = ctx.to_fixed(ctx._re(q), wp)
            xim = ctx.to_fixed(ctx._im(q), wp)
            x2re = (xre*xre - xim*xim) >> wp
            x2im = (xre*xim) >> (wp - 1)
            are = bre = x2re
            aim = bim = x2im
            prec0 = ctx.prec
            ctx.prec = wp
            # cos(z), sin(z) with z complex
            c1, s1 = ctx.cos_sin(z)
            ctx.prec = prec0
            cnre = c1re = ctx.to_fixed(ctx._re(c1), wp)
            cnim = c1im = ctx.to_fixed(ctx._im(c1), wp)
            snre = s1re = ctx.to_fixed(ctx._re(s1), wp)
            snim = s1im = ctx.to_fixed(ctx._im(s1), wp)
            c2re = (c1re*c1re - c1im*c1im - s1re*s1re + s1im*s1im) >> wp
            c2im = (c1re*c1im - s1re*s1im) >> (wp - 1)
            s2re = (c1re*s1re - c1im*s1im) >> (wp - 1)
            s2im = (c1re*s1im + c1im*s1re) >> (wp - 1)
            t1 = (cnre*c2re - cnim*c2im - snre*s2re + snim*s2im) >> wp
            t2 = (cnre*c2im + cnim*c2re - snre*s2im - snim*s2re) >> wp
            t3 = (snre*c2re - snim*c2im + cnre*s2re - cnim*s2im) >> wp
            t4 = (snre*c2im + snim*c2re + cnre*s2im + cnim*s2re) >> wp
            cnre = t1
            cnim = t2
            snre = t3
            snim = t4
            n = 1
            termre = c1re
            termim = c1im
            sre = c1re + ((are * cnre - aim * cnim) >> wp)
            sim = c1im + ((are * cnim + aim * cnre) >> wp)
            n = 3
            termre = ((are * cnre - aim * cnim) >> wp)
            termim = ((are * cnim + aim * cnre) >> wp)
            sre = c1re + ((are * cnre - aim * cnim) >> wp)
            sim = c1im + ((are * cnim + aim * cnre) >> wp)
            n = 5
            while are**2 + aim**2 > MIN:
                bre, bim = (bre * x2re - bim * x2im) >> wp, \
                           (bre * x2im + bim * x2re) >> wp
                are, aim = (are * bre - aim * bim) >> wp,   \
                           (are * bim + aim * bre) >> wp
                #cn, sn = (cn*c1 - sn*s1) >> wp, (sn*c1 + cn*s1) >> wp
                t1 = (cnre*c2re - cnim*c2im - snre*s2re + snim*s2im) >> wp
                t2 = (cnre*c2im + cnim*c2re - snre*s2im - snim*s2re) >> wp
                t3 = (snre*c2re - snim*c2im + cnre*s2re - cnim*s2im) >> wp
                t4 = (snre*c2im + snim*c2re + cnre*s2im + cnim*s2re) >> wp
                cnre = t1
                cnim = t2
                snre = t3
                snim = t4
                termre = ((are * cnre - aim * cnim) >> wp)
                termim = ((aim * cnre + are * cnim) >> wp)
                sre += ((are * cnre - aim * cnim) >> wp)
                sim += ((aim * cnre + are * cnim) >> wp)
                n += 2
            sre = (sre << 1)
            sim = (sim << 1)
            sre = ctx.ldexp(sre, -wp)
            sim = ctx.ldexp(sim, -wp)
            s = ctx.mpc(sre, sim)
    s *= ctx.nthroot(q, 4)
    return s

@defun
def _djacobi_theta2(ctx, z, q, nd):
    MIN = 2
    extra1 = 10
    extra2 = 20
    if (not ctx._im(q)) and (not ctx._im(z)):
        wp = ctx.prec + extra1
        x = ctx.to_fixed(ctx._re(q), wp)
        x2 = (x*x) >> wp
        a = b = x2
        c1, s1 = ctx.cos_sin(ctx._re(z), prec=wp)
        cn = c1 = ctx.to_fixed(c1, wp)
        sn = s1 = ctx.to_fixed(s1, wp)
        c2 = (c1*c1 - s1*s1) >> wp
        s2 = (c1 * s1) >> (wp - 1)
        cn, sn = (cn*c2 - sn*s2) >> wp, (sn*c2 + cn*s2) >> wp
        if (nd&1):
            s = s1 + ((a * sn * 3**nd) >> wp)
        else:
            s = c1 + ((a * cn * 3**nd) >> wp)
        n = 2
        while abs(a) > MIN:
            b = (b*x2) >> wp
            a = (a*b) >> wp
            cn, sn = (cn*c2 - sn*s2) >> wp, (sn*c2 + cn*s2) >> wp
            if nd&1:
                s += (a * sn * (2*n+1)**nd) >> wp
            else:
                s += (a * cn * (2*n+1)**nd) >> wp
            n += 1
        s = -(s << 1)
        s = ctx.ldexp(s, -wp)
        # case z real, q complex
    elif not ctx._im(z):
        wp = ctx.prec + extra2
        xre = ctx.to_fixed(ctx._re(q), wp)
        xim = ctx.to_fixed(ctx._im(q), wp)
        x2re = (xre*xre - xim*xim) >> wp
        x2im = (xre*xim) >> (wp - 1)
        are = bre = x2re
        aim = bim = x2im
        c1, s1 = ctx.cos_sin(ctx._re(z), prec=wp)
        cn = c1 = ctx.to_fixed(c1, wp)
        sn = s1 = ctx.to_fixed(s1, wp)
        c2 = (c1*c1 - s1*s1) >> wp
        s2 = (c1 * s1) >> (wp - 1)
        cn, sn = (cn*c2 - sn*s2) >> wp, (sn*c2 + cn*s2) >> wp
        if (nd&1):
            sre = s1 + ((are * sn * 3**nd) >> wp)
            sim = ((aim * sn * 3**nd) >> wp)
        else:
            sre = c1 + ((are * cn * 3**nd) >> wp)
            sim = ((aim * cn * 3**nd) >> wp)
        n = 5
        while are**2 + aim**2 > MIN:
            bre, bim = (bre * x2re - bim * x2im) >> wp, \
                       (bre * x2im + bim * x2re) >> wp
            are, aim = (are * bre - aim * bim) >> wp,   \
                       (are * bim + aim * bre) >> wp
            cn, sn = (cn*c2 - sn*s2) >> wp, (sn*c2 + cn*s2) >> wp

            if (nd&1):
                sre += ((are * sn * n**nd) >> wp)
                sim += ((aim * sn * n**nd) >> wp)
            else:
                sre += ((are * cn * n**nd) >> wp)
                sim += ((aim * cn * n**nd) >> wp)
            n += 2
        sre = -(sre << 1)
        sim = -(sim << 1)
        sre = ctx.ldexp(sre, -wp)
        sim = ctx.ldexp(sim, -wp)
        s = ctx.mpc(sre, sim)
    #case z complex, q real
    elif not ctx._im(q):
        wp = ctx.prec + extra2
        x = ctx.to_fixed(ctx._re(q), wp)
        x2 = (x*x) >> wp
        a = b = x2
        prec0 = ctx.prec
        ctx.prec = wp
        c1, s1 = ctx.cos_sin(z)
        ctx.prec = prec0
        cnre = c1re = ctx.to_fixed(ctx._re(c1), wp)
        cnim = c1im = ctx.to_fixed(ctx._im(c1), wp)
        snre = s1re = ctx.to_fixed(ctx._re(s1), wp)
        snim = s1im = ctx.to_fixed(ctx._im(s1), wp)
        #c2 = (c1*c1 - s1*s1) >> wp
        c2re = (c1re*c1re - c1im*c1im - s1re*s1re + s1im*s1im) >> wp
        c2im = (c1re*c1im - s1re*s1im) >> (wp - 1)
        #s2 = (c1 * s1) >> (wp - 1)
        s2re = (c1re*s1re - c1im*s1im) >> (wp - 1)
        s2im = (c1re*s1im + c1im*s1re) >> (wp - 1)
        #cn, sn = (cn*c2 - sn*s2) >> wp, (sn*c2 + cn*s2) >> wp
        t1 = (cnre*c2re - cnim*c2im - snre*s2re + snim*s2im) >> wp
        t2 = (cnre*c2im + cnim*c2re - snre*s2im - snim*s2re) >> wp
        t3 = (snre*c2re - snim*c2im + cnre*s2re - cnim*s2im) >> wp
        t4 = (snre*c2im + snim*c2re + cnre*s2im + cnim*s2re) >> wp
        cnre = t1
        cnim = t2
        snre = t3
        snim = t4
        if (nd&1):
            sre = s1re + ((a * snre * 3**nd) >> wp)
            sim = s1im + ((a * snim * 3**nd) >> wp)
        else:
            sre = c1re + ((a * cnre * 3**nd) >> wp)
            sim = c1im + ((a * cnim * 3**nd) >> wp)
        n = 5
        while abs(a) > MIN:
            b = (b*x2) >> wp
            a = (a*b) >> wp
            t1 = (cnre*c2re - cnim*c2im - snre*s2re + snim*s2im) >> wp
            t2 = (cnre*c2im + cnim*c2re - snre*s2im - snim*s2re) >> wp
            t3 = (snre*c2re - snim*c2im + cnre*s2re - cnim*s2im) >> wp
            t4 = (snre*c2im + snim*c2re + cnre*s2im + cnim*s2re) >> wp
            cnre = t1
            cnim = t2
            snre = t3
            snim = t4
            if (nd&1):
                sre += ((a * snre * n**nd) >> wp)
                sim += ((a * snim * n**nd) >> wp)
            else:
                sre += ((a * cnre * n**nd) >> wp)
                sim += ((a * cnim * n**nd) >> wp)
            n += 2
        sre = -(sre << 1)
        sim = -(sim << 1)
        sre = ctx.ldexp(sre, -wp)
        sim = ctx.ldexp(sim, -wp)
        s = ctx.mpc(sre, sim)
    # case z and q complex
    else:
        wp = ctx.prec + extra2
        xre = ctx.to_fixed(ctx._re(q), wp)
        xim = ctx.to_fixed(ctx._im(q), wp)
        x2re = (xre*xre - xim*xim) >> wp
        x2im = (xre*xim) >> (wp - 1)
        are = bre = x2re
        aim = bim = x2im
        prec0 = ctx.prec
        ctx.prec = wp
        # cos(2*z), sin(2*z) with z complex
        c1, s1 = ctx.cos_sin(z)
        ctx.prec = prec0
        cnre = c1re = ctx.to_fixed(ctx._re(c1), wp)
        cnim = c1im = ctx.to_fixed(ctx._im(c1), wp)
        snre = s1re = ctx.to_fixed(ctx._re(s1), wp)
        snim = s1im = ctx.to_fixed(ctx._im(s1), wp)
        c2re = (c1re*c1re - c1im*c1im - s1re*s1re + s1im*s1im) >> wp
        c2im = (c1re*c1im - s1re*s1im) >> (wp - 1)
        s2re = (c1re*s1re - c1im*s1im) >> (wp - 1)
        s2im = (c1re*s1im + c1im*s1re) >> (wp - 1)
        t1 = (cnre*c2re - cnim*c2im - snre*s2re + snim*s2im) >> wp
        t2 = (cnre*c2im + cnim*c2re - snre*s2im - snim*s2re) >> wp
        t3 = (snre*c2re - snim*c2im + cnre*s2re - cnim*s2im) >> wp
        t4 = (snre*c2im + snim*c2re + cnre*s2im + cnim*s2re) >> wp
        cnre = t1
        cnim = t2
        snre = t3
        snim = t4
        if (nd&1):
            sre = s1re + (((are * snre - aim * snim) * 3**nd) >> wp)
            sim = s1im + (((are * snim + aim * snre)* 3**nd) >> wp)
        else:
            sre = c1re + (((are * cnre - aim * cnim) * 3**nd) >> wp)
            sim = c1im + (((are * cnim + aim * cnre)* 3**nd) >> wp)
        n = 5
        while are**2 + aim**2 > MIN:
            bre, bim = (bre * x2re - bim * x2im) >> wp, \
                       (bre * x2im + bim * x2re) >> wp
            are, aim = (are * bre - aim * bim) >> wp,   \
                       (are * bim + aim * bre) >> wp
            #cn, sn = (cn*c1 - sn*s1) >> wp, (sn*c1 + cn*s1) >> wp
            t1 = (cnre*c2re - cnim*c2im - snre*s2re + snim*s2im) >> wp
            t2 = (cnre*c2im + cnim*c2re - snre*s2im - snim*s2re) >> wp
            t3 = (snre*c2re - snim*c2im + cnre*s2re - cnim*s2im) >> wp
            t4 = (snre*c2im + snim*c2re + cnre*s2im + cnim*s2re) >> wp
            cnre = t1
            cnim = t2
            snre = t3
            snim = t4
            if (nd&1):
                sre += (((are * snre - aim * snim) * n**nd) >> wp)
                sim += (((aim * snre + are * snim) * n**nd) >> wp)
            else:
                sre += (((are * cnre - aim * cnim) * n**nd) >> wp)
                sim += (((aim * cnre + are * cnim) * n**nd) >> wp)
            n += 2
        sre = -(sre << 1)
        sim = -(sim << 1)
        sre = ctx.ldexp(sre, -wp)
        sim = ctx.ldexp(sim, -wp)
        s = ctx.mpc(sre, sim)
    s *= ctx.nthroot(q, 4)
    if (nd&1):
        return (-1)**(nd//2) * s
    else:
        return (-1)**(1 + nd//2) * s

@defun
def _jacobi_theta3(ctx, z, q):
    extra1 = 10
    extra2 = 20
    MIN = 2
    if z == ctx.zero:
        if not ctx._im(q):
            wp = ctx.prec + extra1
            x = ctx.to_fixed(ctx._re(q), wp)
            s = x
            a = b = x
            x2 = (x*x) >> wp
            while abs(a) > MIN:
                b = (b*x2) >> wp
                a = (a*b) >> wp
                s += a
            s = (1 << wp) + (s << 1)
            s = ctx.ldexp(s, -wp)
            return s
        else:
            wp = ctx.prec + extra1
            xre = ctx.to_fixed(ctx._re(q), wp)
            xim = ctx.to_fixed(ctx._im(q), wp)
            x2re = (xre*xre - xim*xim) >> wp
            x2im = (xre*xim) >> (wp - 1)
            sre = are = bre = xre
            sim = aim = bim = xim
            while are**2 + aim**2 > MIN:
                bre, bim = (bre * x2re - bim * x2im) >> wp, \
                           (bre * x2im + bim * x2re) >> wp
                are, aim = (are * bre - aim * bim) >> wp,   \
                           (are * bim + aim * bre) >> wp
                sre += are
                sim += aim
            sre = (1 << wp) + (sre << 1)
            sim = (sim << 1)
            sre = ctx.ldexp(sre, -wp)
            sim = ctx.ldexp(sim, -wp)
            s = ctx.mpc(sre, sim)
            return s
    else:
        if (not ctx._im(q)) and (not ctx._im(z)):
            s = 0
            wp = ctx.prec + extra1
            x = ctx.to_fixed(ctx._re(q), wp)
            a = b = x
            x2 = (x*x) >> wp
            c1, s1 = ctx.cos_sin(ctx._re(z)*2, prec=wp)
            c1 = ctx.to_fixed(c1, wp)
            s1 = ctx.to_fixed(s1, wp)
            cn = c1
            sn = s1
            s += (a * cn) >> wp
            while abs(a) > MIN:
                b = (b*x2) >> wp
                a = (a*b) >> wp
                cn, sn = (cn*c1 - sn*s1) >> wp, (sn*c1 + cn*s1) >> wp
                s += (a * cn) >> wp
            s = (1 << wp) + (s << 1)
            s = ctx.ldexp(s, -wp)
            return s
        # case z real, q complex
        elif not ctx._im(z):
            wp = ctx.prec + extra2
            xre = ctx.to_fixed(ctx._re(q), wp)
            xim = ctx.to_fixed(ctx._im(q), wp)
            x2re = (xre*xre - xim*xim) >> wp
            x2im = (xre*xim) >> (wp - 1)
            are = bre = xre
            aim = bim = xim
            c1, s1 = ctx.cos_sin(ctx._re(z)*2, prec=wp)
            c1 = ctx.to_fixed(c1, wp)
            s1 = ctx.to_fixed(s1, wp)
            cn = c1
            sn = s1
            sre = (are * cn) >> wp
            sim = (aim * cn) >> wp
            while are**2 + aim**2 > MIN:
                bre, bim = (bre * x2re - bim * x2im) >> wp, \
                           (bre * x2im + bim * x2re) >> wp
                are, aim = (are * bre - aim * bim) >> wp,   \
                           (are * bim + aim * bre) >> wp
                cn, sn = (cn*c1 - sn*s1) >> wp, (sn*c1 + cn*s1) >> wp
                sre += (are * cn) >> wp
                sim += (aim * cn) >> wp
            sre = (1 << wp) + (sre << 1)
            sim = (sim << 1)
            sre = ctx.ldexp(sre, -wp)
            sim = ctx.ldexp(sim, -wp)
            s = ctx.mpc(sre, sim)
            return s
        #case z complex, q real
        elif not ctx._im(q):
            wp = ctx.prec + extra2
            x = ctx.to_fixed(ctx._re(q), wp)
            a = b = x
            x2 = (x*x) >> wp
            prec0 = ctx.prec
            ctx.prec = wp
            c1, s1 = ctx.cos_sin(2*z)
            ctx.prec = prec0
            cnre = c1re = ctx.to_fixed(ctx._re(c1), wp)
            cnim = c1im = ctx.to_fixed(ctx._im(c1), wp)
            snre = s1re = ctx.to_fixed(ctx._re(s1), wp)
            snim = s1im = ctx.to_fixed(ctx._im(s1), wp)
            sre = (a * cnre) >> wp
            sim = (a * cnim) >> wp
            while abs(a) > MIN:
                b = (b*x2) >> wp
                a = (a*b) >> wp
                t1 = (cnre*c1re - cnim*c1im - snre*s1re + snim*s1im) >> wp
                t2 = (cnre*c1im + cnim*c1re - snre*s1im - snim*s1re) >> wp
                t3 = (snre*c1re - snim*c1im + cnre*s1re - cnim*s1im) >> wp
                t4 = (snre*c1im + snim*c1re + cnre*s1im + cnim*s1re) >> wp
                cnre = t1
                cnim = t2
                snre = t3
                snim = t4
                sre += (a * cnre) >> wp
                sim += (a * cnim) >> wp
            sre = (1 << wp) + (sre << 1)
            sim = (sim << 1)
            sre = ctx.ldexp(sre, -wp)
            sim = ctx.ldexp(sim, -wp)
            s = ctx.mpc(sre, sim)
            return s
        # case z and q complex
        else:
            wp = ctx.prec + extra2
            xre = ctx.to_fixed(ctx._re(q), wp)
            xim = ctx.to_fixed(ctx._im(q), wp)
            x2re = (xre*xre - xim*xim) >> wp
            x2im = (xre*xim) >> (wp - 1)
            are = bre = xre
            aim = bim = xim
            prec0 = ctx.prec
            ctx.prec = wp
            # cos(2*z), sin(2*z) with z complex
            c1, s1 = ctx.cos_sin(2*z)
            ctx.prec = prec0
            cnre = c1re = ctx.to_fixed(ctx._re(c1), wp)
            cnim = c1im = ctx.to_fixed(ctx._im(c1), wp)
            snre = s1re = ctx.to_fixed(ctx._re(s1), wp)
            snim = s1im = ctx.to_fixed(ctx._im(s1), wp)
            sre = (are * cnre - aim * cnim) >> wp
            sim = (aim * cnre + are * cnim) >> wp
            while are**2 + aim**2 > MIN:
                bre, bim = (bre * x2re - bim * x2im) >> wp, \
                           (bre * x2im + bim * x2re) >> wp
                are, aim = (are * bre - aim * bim) >> wp,   \
                           (are * bim + aim * bre) >> wp
                t1 = (cnre*c1re - cnim*c1im - snre*s1re + snim*s1im) >> wp
                t2 = (cnre*c1im + cnim*c1re - snre*s1im - snim*s1re) >> wp
                t3 = (snre*c1re - snim*c1im + cnre*s1re - cnim*s1im) >> wp
                t4 = (snre*c1im + snim*c1re + cnre*s1im + cnim*s1re) >> wp
                cnre = t1
                cnim = t2
                snre = t3
                snim = t4
                sre += (are * cnre - aim * cnim) >> wp
                sim += (aim * cnre + are * cnim) >> wp
            sre = (1 << wp) + (sre << 1)
            sim = (sim << 1)
            sre = ctx.ldexp(sre, -wp)
            sim = ctx.ldexp(sim, -wp)
            s = ctx.mpc(sre, sim)
            return s

@defun
def _djacobi_theta3(ctx, z, q, nd):
    """nd=1,2,3 order of the derivative with respect to z"""
    MIN = 2
    extra1 = 10
    extra2 = 20
    if (not ctx._im(q)) and (not ctx._im(z)):
        s = 0
        wp = ctx.prec + extra1
        x = ctx.to_fixed(ctx._re(q), wp)
        a = b = x
        x2 = (x*x) >> wp
        c1, s1 = ctx.cos_sin(ctx._re(z)*2, prec=wp)
        c1 = ctx.to_fixed(c1, wp)
        s1 = ctx.to_fixed(s1, wp)
        cn = c1
        sn = s1
        if (nd&1):
            s += (a * sn) >> wp
        else:
            s += (a * cn) >> wp
        n = 2
        while abs(a) > MIN:
            b = (b*x2) >> wp
            a = (a*b) >> wp
            cn, sn = (cn*c1 - sn*s1) >> wp, (sn*c1 + cn*s1) >> wp
            if nd&1:
                s += (a * sn * n**nd) >> wp
            else:
                s += (a * cn * n**nd) >> wp
            n += 1
        s = -(s << (nd+1))
        s = ctx.ldexp(s, -wp)
    # case z real, q complex
    elif not ctx._im(z):
        wp = ctx.prec + extra2
        xre = ctx.to_fixed(ctx._re(q), wp)
        xim = ctx.to_fixed(ctx._im(q), wp)
        x2re = (xre*xre - xim*xim) >> wp
        x2im = (xre*xim) >> (wp - 1)
        are = bre = xre
        aim = bim = xim
        c1, s1 = ctx.cos_sin(ctx._re(z)*2, prec=wp)
        c1 = ctx.to_fixed(c1, wp)
        s1 = ctx.to_fixed(s1, wp)
        cn = c1
        sn = s1
        if (nd&1):
            sre = (are * sn) >> wp
            sim = (aim * sn) >> wp
        else:
            sre = (are * cn) >> wp
            sim = (aim * cn) >> wp
        n = 2
        while are**2 + aim**2 > MIN:
            bre, bim = (bre * x2re - bim * x2im) >> wp, \
                       (bre * x2im + bim * x2re) >> wp
            are, aim = (are * bre - aim * bim) >> wp,   \
                       (are * bim + aim * bre) >> wp
            cn, sn = (cn*c1 - sn*s1) >> wp, (sn*c1 + cn*s1) >> wp
            if nd&1:
                sre += (are * sn * n**nd) >> wp
                sim += (aim * sn * n**nd) >> wp
            else:
                sre += (are * cn * n**nd) >> wp
                sim += (aim * cn * n**nd) >> wp
            n += 1
        sre = -(sre << (nd+1))
        sim = -(sim << (nd+1))
        sre = ctx.ldexp(sre, -wp)
        sim = ctx.ldexp(sim, -wp)
        s = ctx.mpc(sre, sim)
    #case z complex, q real
    elif not ctx._im(q):
        wp = ctx.prec + extra2
        x = ctx.to_fixed(ctx._re(q), wp)
        a = b = x
        x2 = (x*x) >> wp
        prec0 = ctx.prec
        ctx.prec = wp
        c1, s1 = ctx.cos_sin(2*z)
        ctx.prec = prec0
        cnre = c1re = ctx.to_fixed(ctx._re(c1), wp)
        cnim = c1im = ctx.to_fixed(ctx._im(c1), wp)
        snre = s1re = ctx.to_fixed(ctx._re(s1), wp)
        snim = s1im = ctx.to_fixed(ctx._im(s1), wp)
        if (nd&1):
            sre = (a * snre) >> wp
            sim = (a * snim) >> wp
        else:
            sre = (a * cnre) >> wp
            sim = (a * cnim) >> wp
        n = 2
        while abs(a) > MIN:
            b = (b*x2) >> wp
            a = (a*b) >> wp
            t1 = (cnre*c1re - cnim*c1im - snre*s1re + snim*s1im) >> wp
            t2 = (cnre*c1im + cnim*c1re - snre*s1im - snim*s1re) >> wp
            t3 = (snre*c1re - snim*c1im + cnre*s1re - cnim*s1im) >> wp
            t4 = (snre*c1im + snim*c1re + cnre*s1im + cnim*s1re) >> wp
            cnre = t1
            cnim = t2
            snre = t3
            snim = t4
            if (nd&1):
                sre += (a * snre * n**nd) >> wp
                sim += (a * snim * n**nd) >> wp
            else:
                sre += (a * cnre * n**nd) >> wp
                sim += (a * cnim * n**nd) >> wp
            n += 1
        sre = -(sre << (nd+1))
        sim = -(sim << (nd+1))
        sre = ctx.ldexp(sre, -wp)
        sim = ctx.ldexp(sim, -wp)
        s = ctx.mpc(sre, sim)
    # case z and q complex
    else:
        wp = ctx.prec + extra2
        xre = ctx.to_fixed(ctx._re(q), wp)
        xim = ctx.to_fixed(ctx._im(q), wp)
        x2re = (xre*xre - xim*xim) >> wp
        x2im = (xre*xim) >> (wp - 1)
        are = bre = xre
        aim = bim = xim
        prec0 = ctx.prec
        ctx.prec = wp
        # cos(2*z), sin(2*z) with z complex
        c1, s1 = ctx.cos_sin(2*z)
        ctx.prec = prec0
        cnre = c1re = ctx.to_fixed(ctx._re(c1), wp)
        cnim = c1im = ctx.to_fixed(ctx._im(c1), wp)
        snre = s1re = ctx.to_fixed(ctx._re(s1), wp)
        snim = s1im = ctx.to_fixed(ctx._im(s1), wp)
        if (nd&1):
            sre = (are * snre - aim * snim) >> wp
            sim = (aim * snre + are * snim) >> wp
        else:
            sre = (are * cnre - aim * cnim) >> wp
            sim = (aim * cnre + are * cnim) >> wp
        n = 2
        while are**2 + aim**2 > MIN:
            bre, bim = (bre * x2re - bim * x2im) >> wp, \
                       (bre * x2im + bim * x2re) >> wp
            are, aim = (are * bre - aim * bim) >> wp,   \
                       (are * bim + aim * bre) >> wp
            t1 = (cnre*c1re - cnim*c1im - snre*s1re + snim*s1im) >> wp
            t2 = (cnre*c1im + cnim*c1re - snre*s1im - snim*s1re) >> wp
            t3 = (snre*c1re - snim*c1im + cnre*s1re - cnim*s1im) >> wp
            t4 = (snre*c1im + snim*c1re + cnre*s1im + cnim*s1re) >> wp
            cnre = t1
            cnim = t2
            snre = t3
            snim = t4
            if(nd&1):
                sre += ((are * snre - aim * snim) * n**nd) >> wp
                sim += ((aim * snre + are * snim) * n**nd) >> wp
            else:
                sre += ((are * cnre - aim * cnim) * n**nd) >> wp
                sim += ((aim * cnre + are * cnim) * n**nd) >> wp
            n += 1
        sre = -(sre << (nd+1))
        sim = -(sim << (nd+1))
        sre = ctx.ldexp(sre, -wp)
        sim = ctx.ldexp(sim, -wp)
        s = ctx.mpc(sre, sim)
    if (nd&1):
        return (-1)**(nd//2) * s
    else:
        return (-1)**(1 + nd//2) * s

@defun
def _jacobi_theta2a(ctx, z, q):
    """
    case ctx._im(z) != 0
    theta(2, z, q) =
    q**1/4 * Sum(q**(n*n + n) * exp(j*(2*n + 1)*z), n=-inf, inf)
    max term for minimum (2*n+1)*log(q).real - 2* ctx._im(z)
    n0 = int(ctx._im(z)/log(q).real - 1/2)
    theta(2, z, q) =
    q**1/4 * Sum(q**(n*n + n) * exp(j*(2*n + 1)*z), n=n0, inf) +
    q**1/4 * Sum(q**(n*n + n) * exp(j*(2*n + 1)*z), n, n0-1, -inf)
    """
    n = n0 = int(ctx._im(z)/ctx._re(ctx.log(q)) - 1/2)
    e2 = ctx.expj(2*z)
    e = e0 = ctx.expj((2*n+1)*z)
    a = q**(n*n + n)
    # leading term
    term = a * e
    s = term
    eps1 = ctx.eps*abs(term)
    while 1:
        n += 1
        e = e * e2
        term = q**(n*n + n) * e
        if abs(term) < eps1:
            break
        s += term
    e = e0
    e2 = ctx.expj(-2*z)
    n = n0
    while 1:
        n -= 1
        e = e * e2
        term = q**(n*n + n) * e
        if abs(term) < eps1:
            break
        s += term
    s = s * ctx.nthroot(q, 4)
    return s

@defun
def _jacobi_theta3a(ctx, z, q):
    """
    case ctx._im(z) != 0
    theta3(z, q) = Sum(q**(n*n) * exp(j*2*n*z), n, -inf, inf)
    max term for n*abs(log(q).real) + ctx._im(z) ~= 0
    n0 = int(- ctx._im(z)/abs(log(q).real))
    """
    n = n0 = int(-ctx._im(z)/abs(ctx._re(ctx.log(q))))
    e2 = ctx.expj(2*z)
    e = e0 = ctx.expj(2*n*z)
    s = term = q**(n*n) * e
    eps1 = ctx.eps*abs(term)
    while 1:
        n += 1
        e = e * e2
        term = q**(n*n) * e
        if abs(term) < eps1:
            break
        s += term
    e = e0
    e2 = ctx.expj(-2*z)
    n = n0
    while 1:
        n -= 1
        e = e * e2
        term = q**(n*n) * e
        if abs(term) < eps1:
            break
        s += term
    return s

@defun
def _djacobi_theta2a(ctx, z, q, nd):
    """
    case ctx._im(z) != 0
    dtheta(2, z, q, nd) =
    j* q**1/4 * Sum(q**(n*n + n) * (2*n+1)*exp(j*(2*n + 1)*z), n=-inf, inf)
    max term for (2*n0+1)*log(q).real - 2* ctx._im(z) ~= 0
    n0 = int(ctx._im(z)/log(q).real - 1/2)
    """
    n = n0 = int(ctx._im(z)/ctx._re(ctx.log(q)) - 1/2)
    e2 = ctx.expj(2*z)
    e = e0 = ctx.expj((2*n + 1)*z)
    a = q**(n*n + n)
    # leading term
    term = (2*n+1)**nd * a * e
    s = term
    eps1 = ctx.eps*abs(term)
    while 1:
        n += 1
        e = e * e2
        term = (2*n+1)**nd * q**(n*n + n) * e
        if abs(term) < eps1:
            break
        s += term
    e = e0
    e2 = ctx.expj(-2*z)
    n = n0
    while 1:
        n -= 1
        e = e * e2
        term = (2*n+1)**nd * q**(n*n + n) * e
        if abs(term) < eps1:
            break
        s += term
    return ctx.j**nd * s * ctx.nthroot(q, 4)

@defun
def _djacobi_theta3a(ctx, z, q, nd):
    """
    case ctx._im(z) != 0
    djtheta3(z, q, nd) = (2*j)**nd *
      Sum(q**(n*n) * n**nd * exp(j*2*n*z), n, -inf, inf)
    max term for minimum n*abs(log(q).real) + ctx._im(z)
    """
    n = n0 = int(-ctx._im(z)/abs(ctx._re(ctx.log(q))))
    e2 = ctx.expj(2*z)
    e = e0 = ctx.expj(2*n*z)
    a = q**(n*n) * e
    s = term = n**nd * a
    if n != 0:
        eps1 = ctx.eps*abs(term)
    else:
        eps1 = ctx.eps*abs(a)
    while 1:
        n += 1
        e = e * e2
        a = q**(n*n) * e
        term = n**nd * a
        if n != 0:
            aterm = abs(term)
        else:
            aterm = abs(a)
        if aterm < eps1:
            break
        s += term
    e = e0
    e2 = ctx.expj(-2*z)
    n = n0
    while 1:
        n -= 1
        e = e * e2
        a = q**(n*n) * e
        term = n**nd * a
        if n != 0:
            aterm = abs(term)
        else:
            aterm = abs(a)
        if aterm < eps1:
            break
        s += term
    return (2*ctx.j)**nd * s

@defun
def jtheta(ctx, n, z, q, derivative=0):
    if derivative:
        return ctx._djtheta(n, z, q, derivative)

    z = ctx.convert(z)
    q = ctx.convert(q)

    # Implementation note
    # If ctx._im(z) is close to zero, _jacobi_theta2 and _jacobi_theta3
    # are used,
    # which compute the series starting from n=0 using fixed precision
    # numbers;
    # otherwise  _jacobi_theta2a and _jacobi_theta3a are used, which compute
    # the series starting from n=n0, which is the largest term.

    # TODO: write _jacobi_theta2a and _jacobi_theta3a using fixed-point

    if abs(q) > ctx.THETA_Q_LIM:
        raise ValueError('abs(q) > THETA_Q_LIM = %f' % ctx.THETA_Q_LIM)

    extra = 10
    if z:
        M = ctx.mag(z)
        if M > 5 or (n == 1 and M < -5):
            extra += 2*abs(M)
    cz = 0.5
    extra2 = 50
    prec0 = ctx.prec
    try:
        ctx.prec += extra
        if n == 1:
            if ctx._im(z):
                if abs(ctx._im(z)) < cz * abs(ctx._re(ctx.log(q))):
                    ctx.dps += extra2
                    res = ctx._jacobi_theta2(z - ctx.pi/2, q)
                else:
                    ctx.dps += 10
                    res = ctx._jacobi_theta2a(z - ctx.pi/2, q)
            else:
                res = ctx._jacobi_theta2(z - ctx.pi/2, q)
        elif n == 2:
            if ctx._im(z):
                if abs(ctx._im(z)) < cz * abs(ctx._re(ctx.log(q))):
                    ctx.dps += extra2
                    res = ctx._jacobi_theta2(z, q)
                else:
                    ctx.dps += 10
                    res = ctx._jacobi_theta2a(z, q)
            else:
                res = ctx._jacobi_theta2(z, q)
        elif n == 3:
            if ctx._im(z):
                if abs(ctx._im(z)) < cz * abs(ctx._re(ctx.log(q))):
                    ctx.dps += extra2
                    res = ctx._jacobi_theta3(z, q)
                else:
                    ctx.dps += 10
                    res = ctx._jacobi_theta3a(z, q)
            else:
                res = ctx._jacobi_theta3(z, q)
        elif n == 4:
            if ctx._im(z):
                if abs(ctx._im(z)) < cz * abs(ctx._re(ctx.log(q))):
                    ctx.dps += extra2
                    res = ctx._jacobi_theta3(z, -q)
                else:
                    ctx.dps += 10
                    res = ctx._jacobi_theta3a(z, -q)
            else:
                res = ctx._jacobi_theta3(z, -q)
        else:
            raise ValueError
    finally:
        ctx.prec = prec0
    return res

@defun
def _djtheta(ctx, n, z, q, derivative=1):
    z = ctx.convert(z)
    q = ctx.convert(q)
    nd = int(derivative)

    if abs(q) > ctx.THETA_Q_LIM:
        raise ValueError('abs(q) > THETA_Q_LIM = %f' % ctx.THETA_Q_LIM)
    extra = 10 + ctx.prec * nd // 10
    if z:
        M = ctx.mag(z)
        if M > 5 or (n != 1 and M < -5):
            extra += 2*abs(M)
    cz = 0.5
    extra2 = 50
    prec0 = ctx.prec
    try:
        ctx.prec += extra
        if n == 1:
            if ctx._im(z):
                if abs(ctx._im(z)) < cz * abs(ctx._re(ctx.log(q))):
                    ctx.dps += extra2
                    res = ctx._djacobi_theta2(z - ctx.pi/2, q, nd)
                else:
                    ctx.dps += 10
                    res = ctx._djacobi_theta2a(z - ctx.pi/2, q, nd)
            else:
                res = ctx._djacobi_theta2(z - ctx.pi/2, q, nd)
        elif n == 2:
            if ctx._im(z):
                if abs(ctx._im(z)) < cz * abs(ctx._re(ctx.log(q))):
                    ctx.dps += extra2
                    res = ctx._djacobi_theta2(z, q, nd)
                else:
                    ctx.dps += 10
                    res = ctx._djacobi_theta2a(z, q, nd)
            else:
                res = ctx._djacobi_theta2(z, q, nd)
        elif n == 3:
            if ctx._im(z):
                if abs(ctx._im(z)) < cz * abs(ctx._re(ctx.log(q))):
                    ctx.dps += extra2
                    res = ctx._djacobi_theta3(z, q, nd)
                else:
                    ctx.dps += 10
                    res = ctx._djacobi_theta3a(z, q, nd)
            else:
                res = ctx._djacobi_theta3(z, q, nd)
        elif n == 4:
            if ctx._im(z):
                if abs(ctx._im(z)) < cz * abs(ctx._re(ctx.log(q))):
                    ctx.dps += extra2
                    res = ctx._djacobi_theta3(z, -q, nd)
                else:
                    ctx.dps += 10
                    res = ctx._djacobi_theta3a(z, -q, nd)
            else:
                res = ctx._djacobi_theta3(z, -q, nd)
        else:
            raise ValueError
    finally:
        ctx.prec = prec0
    return +res

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\urllib3\connection.py ===
from __future__ import annotations

import datetime
import http.client
import logging
import os
import re
import socket
import sys
import threading
import typing
import warnings
from http.client import HTTPConnection as _HTTPConnection
from http.client import HTTPException as HTTPException  # noqa: F401
from http.client import ResponseNotReady
from socket import timeout as SocketTimeout

if typing.TYPE_CHECKING:
    from .response import HTTPResponse
    from .util.ssl_ import _TYPE_PEER_CERT_RET_DICT
    from .util.ssltransport import SSLTransport

from ._collections import HTTPHeaderDict
from .http2 import probe as http2_probe
from .util.response import assert_header_parsing
from .util.timeout import _DEFAULT_TIMEOUT, _TYPE_TIMEOUT, Timeout
from .util.util import to_str
from .util.wait import wait_for_read

try:  # Compiled with SSL?
    import ssl

    BaseSSLError = ssl.SSLError
except (ImportError, AttributeError):
    ssl = None  # type: ignore[assignment]

    class BaseSSLError(BaseException):  # type: ignore[no-redef]
        pass


from ._base_connection import _TYPE_BODY
from ._base_connection import ProxyConfig as ProxyConfig
from ._base_connection import _ResponseOptions as _ResponseOptions
from ._version import __version__
from .exceptions import (
    ConnectTimeoutError,
    HeaderParsingError,
    NameResolutionError,
    NewConnectionError,
    ProxyError,
    SystemTimeWarning,
)
from .util import SKIP_HEADER, SKIPPABLE_HEADERS, connection, ssl_
from .util.request import body_to_chunks
from .util.ssl_ import assert_fingerprint as _assert_fingerprint
from .util.ssl_ import (
    create_urllib3_context,
    is_ipaddress,
    resolve_cert_reqs,
    resolve_ssl_version,
    ssl_wrap_socket,
)
from .util.ssl_match_hostname import CertificateError, match_hostname
from .util.url import Url

# Not a no-op, we're adding this to the namespace so it can be imported.
ConnectionError = ConnectionError
BrokenPipeError = BrokenPipeError


log = logging.getLogger(__name__)

port_by_scheme = {"http": 80, "https": 443}

# When it comes time to update this value as a part of regular maintenance
# (ie test_recent_date is failing) update it to ~6 months before the current date.
RECENT_DATE = datetime.date(2023, 6, 1)

_CONTAINS_CONTROL_CHAR_RE = re.compile(r"[^-!#$%&'*+.^_`|~0-9a-zA-Z]")


class HTTPConnection(_HTTPConnection):
    """
    Based on :class:`http.client.HTTPConnection` but provides an extra constructor
    backwards-compatibility layer between older and newer Pythons.

    Additional keyword parameters are used to configure attributes of the connection.
    Accepted parameters include:

    - ``source_address``: Set the source address for the current connection.
    - ``socket_options``: Set specific options on the underlying socket. If not specified, then
      defaults are loaded from ``HTTPConnection.default_socket_options`` which includes disabling
      Nagle's algorithm (sets TCP_NODELAY to 1) unless the connection is behind a proxy.

      For example, if you wish to enable TCP Keep Alive in addition to the defaults,
      you might pass:

      .. code-block:: python

         HTTPConnection.default_socket_options + [
             (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
         ]

      Or you may want to disable the defaults by passing an empty list (e.g., ``[]``).
    """

    default_port: typing.ClassVar[int] = port_by_scheme["http"]  # type: ignore[misc]

    #: Disable Nagle's algorithm by default.
    #: ``[(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)]``
    default_socket_options: typing.ClassVar[connection._TYPE_SOCKET_OPTIONS] = [
        (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    ]

    #: Whether this connection verifies the host's certificate.
    is_verified: bool = False

    #: Whether this proxy connection verified the proxy host's certificate.
    # If no proxy is currently connected to the value will be ``None``.
    proxy_is_verified: bool | None = None

    blocksize: int
    source_address: tuple[str, int] | None
    socket_options: connection._TYPE_SOCKET_OPTIONS | None

    _has_connected_to_proxy: bool
    _response_options: _ResponseOptions | None
    _tunnel_host: str | None
    _tunnel_port: int | None
    _tunnel_scheme: str | None

    def __init__(
        self,
        host: str,
        port: int | None = None,
        *,
        timeout: _TYPE_TIMEOUT = _DEFAULT_TIMEOUT,
        source_address: tuple[str, int] | None = None,
        blocksize: int = 16384,
        socket_options: None | (
            connection._TYPE_SOCKET_OPTIONS
        ) = default_socket_options,
        proxy: Url | None = None,
        proxy_config: ProxyConfig | None = None,
    ) -> None:
        super().__init__(
            host=host,
            port=port,
            timeout=Timeout.resolve_default_timeout(timeout),
            source_address=source_address,
            blocksize=blocksize,
        )
        self.socket_options = socket_options
        self.proxy = proxy
        self.proxy_config = proxy_config

        self._has_connected_to_proxy = False
        self._response_options = None
        self._tunnel_host: str | None = None
        self._tunnel_port: int | None = None
        self._tunnel_scheme: str | None = None

    @property
    def host(self) -> str:
        """
        Getter method to remove any trailing dots that indicate the hostname is an FQDN.

        In general, SSL certificates don't include the trailing dot indicating a
        fully-qualified domain name, and thus, they don't validate properly when
        checked against a domain name that includes the dot. In addition, some
        servers may not expect to receive the trailing dot when provided.

        However, the hostname with trailing dot is critical to DNS resolution; doing a
        lookup with the trailing dot will properly only resolve the appropriate FQDN,
        whereas a lookup without a trailing dot will search the system's search domain
        list. Thus, it's important to keep the original host around for use only in
        those cases where it's appropriate (i.e., when doing DNS lookup to establish the
        actual TCP connection across which we're going to send HTTP requests).
        """
        return self._dns_host.rstrip(".")

    @host.setter
    def host(self, value: str) -> None:
        """
        Setter for the `host` property.

        We assume that only urllib3 uses the _dns_host attribute; httplib itself
        only uses `host`, and it seems reasonable that other libraries follow suit.
        """
        self._dns_host = value

    def _new_conn(self) -> socket.socket:
        """Establish a socket connection and set nodelay settings on it.

        :return: New socket connection.
        """
        try:
            sock = connection.create_connection(
                (self._dns_host, self.port),
                self.timeout,
                source_address=self.source_address,
                socket_options=self.socket_options,
            )
        except socket.gaierror as e:
            raise NameResolutionError(self.host, self, e) from e
        except SocketTimeout as e:
            raise ConnectTimeoutError(
                self,
                f"Connection to {self.host} timed out. (connect timeout={self.timeout})",
            ) from e

        except OSError as e:
            raise NewConnectionError(
                self, f"Failed to establish a new connection: {e}"
            ) from e

        sys.audit("http.client.connect", self, self.host, self.port)

        return sock

    def set_tunnel(
        self,
        host: str,
        port: int | None = None,
        headers: typing.Mapping[str, str] | None = None,
        scheme: str = "http",
    ) -> None:
        if scheme not in ("http", "https"):
            raise ValueError(
                f"Invalid proxy scheme for tunneling: {scheme!r}, must be either 'http' or 'https'"
            )
        super().set_tunnel(host, port=port, headers=headers)
        self._tunnel_scheme = scheme

    if sys.version_info < (3, 11, 4):

        def _tunnel(self) -> None:
            _MAXLINE = http.client._MAXLINE  # type: ignore[attr-defined]
            connect = b"CONNECT %s:%d HTTP/1.0\r\n" % (  # type: ignore[str-format]
                self._tunnel_host.encode("ascii"),  # type: ignore[union-attr]
                self._tunnel_port,
            )
            headers = [connect]
            for header, value in self._tunnel_headers.items():  # type: ignore[attr-defined]
                headers.append(f"{header}: {value}\r\n".encode("latin-1"))
            headers.append(b"\r\n")
            # Making a single send() call instead of one per line encourages
            # the host OS to use a more optimal packet size instead of
            # potentially emitting a series of small packets.
            self.send(b"".join(headers))
            del headers

            response = self.response_class(self.sock, method=self._method)  # type: ignore[attr-defined]
            try:
                (version, code, message) = response._read_status()  # type: ignore[attr-defined]

                if code != http.HTTPStatus.OK:
                    self.close()
                    raise OSError(f"Tunnel connection failed: {code} {message.strip()}")
                while True:
                    line = response.fp.readline(_MAXLINE + 1)
                    if len(line) > _MAXLINE:
                        raise http.client.LineTooLong("header line")
                    if not line:
                        # for sites which EOF without sending a trailer
                        break
                    if line in (b"\r\n", b"\n", b""):
                        break

                    if self.debuglevel > 0:
                        print("header:", line.decode())
            finally:
                response.close()

    def connect(self) -> None:
        self.sock = self._new_conn()
        if self._tunnel_host:
            # If we're tunneling it means we're connected to our proxy.
            self._has_connected_to_proxy = True

            # TODO: Fix tunnel so it doesn't depend on self.sock state.
            self._tunnel()

        # If there's a proxy to be connected to we are fully connected.
        # This is set twice (once above and here) due to forwarding proxies
        # not using tunnelling.
        self._has_connected_to_proxy = bool(self.proxy)

        if self._has_connected_to_proxy:
            self.proxy_is_verified = False

    @property
    def is_closed(self) -> bool:
        return self.sock is None

    @property
    def is_connected(self) -> bool:
        if self.sock is None:
            return False
        return not wait_for_read(self.sock, timeout=0.0)

    @property
    def has_connected_to_proxy(self) -> bool:
        return self._has_connected_to_proxy

    @property
    def proxy_is_forwarding(self) -> bool:
        """
        Return True if a forwarding proxy is configured, else return False
        """
        return bool(self.proxy) and self._tunnel_host is None

    @property
    def proxy_is_tunneling(self) -> bool:
        """
        Return True if a tunneling proxy is configured, else return False
        """
        return self._tunnel_host is not None

    def close(self) -> None:
        try:
            super().close()
        finally:
            # Reset all stateful properties so connection
            # can be re-used without leaking prior configs.
            self.sock = None
            self.is_verified = False
            self.proxy_is_verified = None
            self._has_connected_to_proxy = False
            self._response_options = None
            self._tunnel_host = None
            self._tunnel_port = None
            self._tunnel_scheme = None

    def putrequest(
        self,
        method: str,
        url: str,
        skip_host: bool = False,
        skip_accept_encoding: bool = False,
    ) -> None:
        """"""
        # Empty docstring because the indentation of CPython's implementation
        # is broken but we don't want this method in our documentation.
        match = _CONTAINS_CONTROL_CHAR_RE.search(method)
        if match:
            raise ValueError(
                f"Method cannot contain non-token characters {method!r} (found at least {match.group()!r})"
            )

        return super().putrequest(
            method, url, skip_host=skip_host, skip_accept_encoding=skip_accept_encoding
        )

    def putheader(self, header: str, *values: str) -> None:  # type: ignore[override]
        """"""
        if not any(isinstance(v, str) and v == SKIP_HEADER for v in values):
            super().putheader(header, *values)
        elif to_str(header.lower()) not in SKIPPABLE_HEADERS:
            skippable_headers = "', '".join(
                [str.title(header) for header in sorted(SKIPPABLE_HEADERS)]
            )
            raise ValueError(
                f"urllib3.util.SKIP_HEADER only supports '{skippable_headers}'"
            )

    # `request` method's signature intentionally violates LSP.
    # urllib3's API is different from `http.client.HTTPConnection` and the subclassing is only incidental.
    def request(  # type: ignore[override]
        self,
        method: str,
        url: str,
        body: _TYPE_BODY | None = None,
        headers: typing.Mapping[str, str] | None = None,
        *,
        chunked: bool = False,
        preload_content: bool = True,
        decode_content: bool = True,
        enforce_content_length: bool = True,
    ) -> None:
        # Update the inner socket's timeout value to send the request.
        # This only triggers if the connection is re-used.
        if self.sock is not None:
            self.sock.settimeout(self.timeout)

        # Store these values to be fed into the HTTPResponse
        # object later. TODO: Remove this in favor of a real
        # HTTP lifecycle mechanism.

        # We have to store these before we call .request()
        # because sometimes we can still salvage a response
        # off the wire even if we aren't able to completely
        # send the request body.
        self._response_options = _ResponseOptions(
            request_method=method,
            request_url=url,
            preload_content=preload_content,
            decode_content=decode_content,
            enforce_content_length=enforce_content_length,
        )

        if headers is None:
            headers = {}
        header_keys = frozenset(to_str(k.lower()) for k in headers)
        skip_accept_encoding = "accept-encoding" in header_keys
        skip_host = "host" in header_keys
        self.putrequest(
            method, url, skip_accept_encoding=skip_accept_encoding, skip_host=skip_host
        )

        # Transform the body into an iterable of sendall()-able chunks
        # and detect if an explicit Content-Length is doable.
        chunks_and_cl = body_to_chunks(body, method=method, blocksize=self.blocksize)
        chunks = chunks_and_cl.chunks
        content_length = chunks_and_cl.content_length

        # When chunked is explicit set to 'True' we respect that.
        if chunked:
            if "transfer-encoding" not in header_keys:
                self.putheader("Transfer-Encoding", "chunked")
        else:
            # Detect whether a framing mechanism is already in use. If so
            # we respect that value, otherwise we pick chunked vs content-length
            # depending on the type of 'body'.
            if "content-length" in header_keys:
                chunked = False
            elif "transfer-encoding" in header_keys:
                chunked = True

            # Otherwise we go off the recommendation of 'body_to_chunks()'.
            else:
                chunked = False
                if content_length is None:
                    if chunks is not None:
                        chunked = True
                        self.putheader("Transfer-Encoding", "chunked")
                else:
                    self.putheader("Content-Length", str(content_length))

        # Now that framing headers are out of the way we send all the other headers.
        if "user-agent" not in header_keys:
            self.putheader("User-Agent", _get_default_user_agent())
        for header, value in headers.items():
            self.putheader(header, value)
        self.endheaders()

        # If we're given a body we start sending that in chunks.
        if chunks is not None:
            for chunk in chunks:
                # Sending empty chunks isn't allowed for TE: chunked
                # as it indicates the end of the body.
                if not chunk:
                    continue
                if isinstance(chunk, str):
                    chunk = chunk.encode("utf-8")
                if chunked:
                    self.send(b"%x\r\n%b\r\n" % (len(chunk), chunk))
                else:
                    self.send(chunk)

        # Regardless of whether we have a body or not, if we're in
        # chunked mode we want to send an explicit empty chunk.
        if chunked:
            self.send(b"0\r\n\r\n")

    def request_chunked(
        self,
        method: str,
        url: str,
        body: _TYPE_BODY | None = None,
        headers: typing.Mapping[str, str] | None = None,
    ) -> None:
        """
        Alternative to the common request method, which sends the
        body with chunked encoding and not as one block
        """
        warnings.warn(
            "HTTPConnection.request_chunked() is deprecated and will be removed "
            "in urllib3 v2.1.0. Instead use HTTPConnection.request(..., chunked=True).",
            category=DeprecationWarning,
            stacklevel=2,
        )
        self.request(method, url, body=body, headers=headers, chunked=True)

    def getresponse(  # type: ignore[override]
        self,
    ) -> HTTPResponse:
        """
        Get the response from the server.

        If the HTTPConnection is in the correct state, returns an instance of HTTPResponse or of whatever object is returned by the response_class variable.

        If a request has not been sent or if a previous response has not be handled, ResponseNotReady is raised. If the HTTP response indicates that the connection should be closed, then it will be closed before the response is returned. When the connection is closed, the underlying socket is closed.
        """
        # Raise the same error as http.client.HTTPConnection
        if self._response_options is None:
            raise ResponseNotReady()

        # Reset this attribute for being used again.
        resp_options = self._response_options
        self._response_options = None

        # Since the connection's timeout value may have been updated
        # we need to set the timeout on the socket.
        self.sock.settimeout(self.timeout)

        # This is needed here to avoid circular import errors
        from .response import HTTPResponse

        # Save a reference to the shutdown function before ownership is passed
        # to httplib_response
        # TODO should we implement it everywhere?
        _shutdown = getattr(self.sock, "shutdown", None)

        # Get the response from http.client.HTTPConnection
        httplib_response = super().getresponse()

        try:
            assert_header_parsing(httplib_response.msg)
        except (HeaderParsingError, TypeError) as hpe:
            log.warning(
                "Failed to parse headers (url=%s): %s",
                _url_from_connection(self, resp_options.request_url),
                hpe,
                exc_info=True,
            )

        headers = HTTPHeaderDict(httplib_response.msg.items())

        response = HTTPResponse(
            body=httplib_response,
            headers=headers,
            status=httplib_response.status,
            version=httplib_response.version,
            version_string=getattr(self, "_http_vsn_str", "HTTP/?"),
            reason=httplib_response.reason,
            preload_content=resp_options.preload_content,
            decode_content=resp_options.decode_content,
            original_response=httplib_response,
            enforce_content_length=resp_options.enforce_content_length,
            request_method=resp_options.request_method,
            request_url=resp_options.request_url,
            sock_shutdown=_shutdown,
        )
        return response


class HTTPSConnection(HTTPConnection):
    """
    Many of the parameters to this constructor are passed to the underlying SSL
    socket by means of :py:func:`urllib3.util.ssl_wrap_socket`.
    """

    default_port = port_by_scheme["https"]  # type: ignore[misc]

    cert_reqs: int | str | None = None
    ca_certs: str | None = None
    ca_cert_dir: str | None = None
    ca_cert_data: None | str | bytes = None
    ssl_version: int | str | None = None
    ssl_minimum_version: int | None = None
    ssl_maximum_version: int | None = None
    assert_fingerprint: str | None = None
    _connect_callback: typing.Callable[..., None] | None = None

    def __init__(
        self,
        host: str,
        port: int | None = None,
        *,
        timeout: _TYPE_TIMEOUT = _DEFAULT_TIMEOUT,
        source_address: tuple[str, int] | None = None,
        blocksize: int = 16384,
        socket_options: None | (
            connection._TYPE_SOCKET_OPTIONS
        ) = HTTPConnection.default_socket_options,
        proxy: Url | None = None,
        proxy_config: ProxyConfig | None = None,
        cert_reqs: int | str | None = None,
        assert_hostname: None | str | typing.Literal[False] = None,
        assert_fingerprint: str | None = None,
        server_hostname: str | None = None,
        ssl_context: ssl.SSLContext | None = None,
        ca_certs: str | None = None,
        ca_cert_dir: str | None = None,
        ca_cert_data: None | str | bytes = None,
        ssl_minimum_version: int | None = None,
        ssl_maximum_version: int | None = None,
        ssl_version: int | str | None = None,  # Deprecated
        cert_file: str | None = None,
        key_file: str | None = None,
        key_password: str | None = None,
    ) -> None:
        super().__init__(
            host,
            port=port,
            timeout=timeout,
            source_address=source_address,
            blocksize=blocksize,
            socket_options=socket_options,
            proxy=proxy,
            proxy_config=proxy_config,
        )

        self.key_file = key_file
        self.cert_file = cert_file
        self.key_password = key_password
        self.ssl_context = ssl_context
        self.server_hostname = server_hostname
        self.assert_hostname = assert_hostname
        self.assert_fingerprint = assert_fingerprint
        self.ssl_version = ssl_version
        self.ssl_minimum_version = ssl_minimum_version
        self.ssl_maximum_version = ssl_maximum_version
        self.ca_certs = ca_certs and os.path.expanduser(ca_certs)
        self.ca_cert_dir = ca_cert_dir and os.path.expanduser(ca_cert_dir)
        self.ca_cert_data = ca_cert_data

        # cert_reqs depends on ssl_context so calculate last.
        if cert_reqs is None:
            if self.ssl_context is not None:
                cert_reqs = self.ssl_context.verify_mode
            else:
                cert_reqs = resolve_cert_reqs(None)
        self.cert_reqs = cert_reqs
        self._connect_callback = None

    def set_cert(
        self,
        key_file: str | None = None,
        cert_file: str | None = None,
        cert_reqs: int | str | None = None,
        key_password: str | None = None,
        ca_certs: str | None = None,
        assert_hostname: None | str | typing.Literal[False] = None,
        assert_fingerprint: str | None = None,
        ca_cert_dir: str | None = None,
        ca_cert_data: None | str | bytes = None,
    ) -> None:
        """
        This method should only be called once, before the connection is used.
        """
        warnings.warn(
            "HTTPSConnection.set_cert() is deprecated and will be removed "
            "in urllib3 v2.1.0. Instead provide the parameters to the "
            "HTTPSConnection constructor.",
            category=DeprecationWarning,
            stacklevel=2,
        )

        # If cert_reqs is not provided we'll assume CERT_REQUIRED unless we also
        # have an SSLContext object in which case we'll use its verify_mode.
        if cert_reqs is None:
            if self.ssl_context is not None:
                cert_reqs = self.ssl_context.verify_mode
            else:
                cert_reqs = resolve_cert_reqs(None)

        self.key_file = key_file
        self.cert_file = cert_file
        self.cert_reqs = cert_reqs
        self.key_password = key_password
        self.assert_hostname = assert_hostname
        self.assert_fingerprint = assert_fingerprint
        self.ca_certs = ca_certs and os.path.expanduser(ca_certs)
        self.ca_cert_dir = ca_cert_dir and os.path.expanduser(ca_cert_dir)
        self.ca_cert_data = ca_cert_data

    def connect(self) -> None:
        # Today we don't need to be doing this step before the /actual/ socket
        # connection, however in the future we'll need to decide whether to
        # create a new socket or re-use an existing "shared" socket as a part
        # of the HTTP/2 handshake dance.
        if self._tunnel_host is not None and self._tunnel_port is not None:
            probe_http2_host = self._tunnel_host
            probe_http2_port = self._tunnel_port
        else:
            probe_http2_host = self.host
            probe_http2_port = self.port

        # Check if the target origin supports HTTP/2.
        # If the value comes back as 'None' it means that the current thread
        # is probing for HTTP/2 support. Otherwise, we're waiting for another
        # probe to complete, or we get a value right away.
        target_supports_http2: bool | None
        if "h2" in ssl_.ALPN_PROTOCOLS:
            target_supports_http2 = http2_probe.acquire_and_get(
                host=probe_http2_host, port=probe_http2_port
            )
        else:
            # If HTTP/2 isn't going to be offered it doesn't matter if
            # the target supports HTTP/2. Don't want to make a probe.
            target_supports_http2 = False

        if self._connect_callback is not None:
            self._connect_callback(
                "before connect",
                thread_id=threading.get_ident(),
                target_supports_http2=target_supports_http2,
            )

        try:
            sock: socket.socket | ssl.SSLSocket
            self.sock = sock = self._new_conn()
            server_hostname: str = self.host
            tls_in_tls = False

            # Do we need to establish a tunnel?
            if self.proxy_is_tunneling:
                # We're tunneling to an HTTPS origin so need to do TLS-in-TLS.
                if self._tunnel_scheme == "https":
                    # _connect_tls_proxy will verify and assign proxy_is_verified
                    self.sock = sock = self._connect_tls_proxy(self.host, sock)
                    tls_in_tls = True
                elif self._tunnel_scheme == "http":
                    self.proxy_is_verified = False

                # If we're tunneling it means we're connected to our proxy.
                self._has_connected_to_proxy = True

                self._tunnel()
                # Override the host with the one we're requesting data from.
                server_hostname = typing.cast(str, self._tunnel_host)

            if self.server_hostname is not None:
                server_hostname = self.server_hostname

            is_time_off = datetime.date.today() < RECENT_DATE
            if is_time_off:
                warnings.warn(
                    (
                        f"System time is way off (before {RECENT_DATE}). This will probably "
                        "lead to SSL verification errors"
                    ),
                    SystemTimeWarning,
                )

            # Remove trailing '.' from fqdn hostnames to allow certificate validation
            server_hostname_rm_dot = server_hostname.rstrip(".")

            sock_and_verified = _ssl_wrap_socket_and_match_hostname(
                sock=sock,
                cert_reqs=self.cert_reqs,
                ssl_version=self.ssl_version,
                ssl_minimum_version=self.ssl_minimum_version,
                ssl_maximum_version=self.ssl_maximum_version,
                ca_certs=self.ca_certs,
                ca_cert_dir=self.ca_cert_dir,
                ca_cert_data=self.ca_cert_data,
                cert_file=self.cert_file,
                key_file=self.key_file,
                key_password=self.key_password,
                server_hostname=server_hostname_rm_dot,
                ssl_context=self.ssl_context,
                tls_in_tls=tls_in_tls,
                assert_hostname=self.assert_hostname,
                assert_fingerprint=self.assert_fingerprint,
            )
            self.sock = sock_and_verified.socket

        # If an error occurs during connection/handshake we may need to release
        # our lock so another connection can probe the origin.
        except BaseException:
            if self._connect_callback is not None:
                self._connect_callback(
                    "after connect failure",
                    thread_id=threading.get_ident(),
                    target_supports_http2=target_supports_http2,
                )

            if target_supports_http2 is None:
                http2_probe.set_and_release(
                    host=probe_http2_host, port=probe_http2_port, supports_http2=None
                )
            raise

        # If this connection doesn't know if the origin supports HTTP/2
        # we report back to the HTTP/2 probe our result.
        if target_supports_http2 is None:
            supports_http2 = sock_and_verified.socket.selected_alpn_protocol() == "h2"
            http2_probe.set_and_release(
                host=probe_http2_host,
                port=probe_http2_port,
                supports_http2=supports_http2,
            )

        # Forwarding proxies can never have a verified target since
        # the proxy is the one doing the verification. Should instead
        # use a CONNECT tunnel in order to verify the target.
        # See: https://github.com/urllib3/urllib3/issues/3267.
        if self.proxy_is_forwarding:
            self.is_verified = False
        else:
            self.is_verified = sock_and_verified.is_verified

        # If there's a proxy to be connected to we are fully connected.
        # This is set twice (once above and here) due to forwarding proxies
        # not using tunnelling.
        self._has_connected_to_proxy = bool(self.proxy)

        # Set `self.proxy_is_verified` unless it's already set while
        # establishing a tunnel.
        if self._has_connected_to_proxy and self.proxy_is_verified is None:
            self.proxy_is_verified = sock_and_verified.is_verified

    def _connect_tls_proxy(self, hostname: str, sock: socket.socket) -> ssl.SSLSocket:
        """
        Establish a TLS connection to the proxy using the provided SSL context.
        """
        # `_connect_tls_proxy` is called when self._tunnel_host is truthy.
        proxy_config = typing.cast(ProxyConfig, self.proxy_config)
        ssl_context = proxy_config.ssl_context
        sock_and_verified = _ssl_wrap_socket_and_match_hostname(
            sock,
            cert_reqs=self.cert_reqs,
            ssl_version=self.ssl_version,
            ssl_minimum_version=self.ssl_minimum_version,
            ssl_maximum_version=self.ssl_maximum_version,
            ca_certs=self.ca_certs,
            ca_cert_dir=self.ca_cert_dir,
            ca_cert_data=self.ca_cert_data,
            server_hostname=hostname,
            ssl_context=ssl_context,
            assert_hostname=proxy_config.assert_hostname,
            assert_fingerprint=proxy_config.assert_fingerprint,
            # Features that aren't implemented for proxies yet:
            cert_file=None,
            key_file=None,
            key_password=None,
            tls_in_tls=False,
        )
        self.proxy_is_verified = sock_and_verified.is_verified
        return sock_and_verified.socket  # type: ignore[return-value]


class _WrappedAndVerifiedSocket(typing.NamedTuple):
    """
    Wrapped socket and whether the connection is
    verified after the TLS handshake
    """

    socket: ssl.SSLSocket | SSLTransport
    is_verified: bool


def _ssl_wrap_socket_and_match_hostname(
    sock: socket.socket,
    *,
    cert_reqs: None | str | int,
    ssl_version: None | str | int,
    ssl_minimum_version: int | None,
    ssl_maximum_version: int | None,
    cert_file: str | None,
    key_file: str | None,
    key_password: str | None,
    ca_certs: str | None,
    ca_cert_dir: str | None,
    ca_cert_data: None | str | bytes,
    assert_hostname: None | str | typing.Literal[False],
    assert_fingerprint: str | None,
    server_hostname: str | None,
    ssl_context: ssl.SSLContext | None,
    tls_in_tls: bool = False,
) -> _WrappedAndVerifiedSocket:
    """Logic for constructing an SSLContext from all TLS parameters, passing
    that down into ssl_wrap_socket, and then doing certificate verification
    either via hostname or fingerprint. This function exists to guarantee
    that both proxies and targets have the same behavior when connecting via TLS.
    """
    default_ssl_context = False
    if ssl_context is None:
        default_ssl_context = True
        context = create_urllib3_context(
            ssl_version=resolve_ssl_version(ssl_version),
            ssl_minimum_version=ssl_minimum_version,
            ssl_maximum_version=ssl_maximum_version,
            cert_reqs=resolve_cert_reqs(cert_reqs),
        )
    else:
        context = ssl_context

    context.verify_mode = resolve_cert_reqs(cert_reqs)

    # In some cases, we want to verify hostnames ourselves
    if (
        # `ssl` can't verify fingerprints or alternate hostnames
        assert_fingerprint
        or assert_hostname
        # assert_hostname can be set to False to disable hostname checking
        or assert_hostname is False
        # We still support OpenSSL 1.0.2, which prevents us from verifying
        # hostnames easily: https://github.com/pyca/pyopenssl/pull/933
        or ssl_.IS_PYOPENSSL
        or not ssl_.HAS_NEVER_CHECK_COMMON_NAME
    ):
        context.check_hostname = False

    # Try to load OS default certs if none are given. We need to do the hasattr() check
    # for custom pyOpenSSL SSLContext objects because they don't support
    # load_default_certs().
    if (
        not ca_certs
        and not ca_cert_dir
        and not ca_cert_data
        and default_ssl_context
        and hasattr(context, "load_default_certs")
    ):
        context.load_default_certs()

    # Ensure that IPv6 addresses are in the proper format and don't have a
    # scope ID. Python's SSL module fails to recognize scoped IPv6 addresses
    # and interprets them as DNS hostnames.
    if server_hostname is not None:
        normalized = server_hostname.strip("[]")
        if "%" in normalized:
            normalized = normalized[: normalized.rfind("%")]
        if is_ipaddress(normalized):
            server_hostname = normalized

    ssl_sock = ssl_wrap_socket(
        sock=sock,
        keyfile=key_file,
        certfile=cert_file,
        key_password=key_password,
        ca_certs=ca_certs,
        ca_cert_dir=ca_cert_dir,
        ca_cert_data=ca_cert_data,
        server_hostname=server_hostname,
        ssl_context=context,
        tls_in_tls=tls_in_tls,
    )

    try:
        if assert_fingerprint:
            _assert_fingerprint(
                ssl_sock.getpeercert(binary_form=True), assert_fingerprint
            )
        elif (
            context.verify_mode != ssl.CERT_NONE
            and not context.check_hostname
            and assert_hostname is not False
        ):
            cert: _TYPE_PEER_CERT_RET_DICT = ssl_sock.getpeercert()  # type: ignore[assignment]

            # Need to signal to our match_hostname whether to use 'commonName' or not.
            # If we're using our own constructed SSLContext we explicitly set 'False'
            # because PyPy hard-codes 'True' from SSLContext.hostname_checks_common_name.
            if default_ssl_context:
                hostname_checks_common_name = False
            else:
                hostname_checks_common_name = (
                    getattr(context, "hostname_checks_common_name", False) or False
                )

            _match_hostname(
                cert,
                assert_hostname or server_hostname,  # type: ignore[arg-type]
                hostname_checks_common_name,
            )

        return _WrappedAndVerifiedSocket(
            socket=ssl_sock,
            is_verified=context.verify_mode == ssl.CERT_REQUIRED
            or bool(assert_fingerprint),
        )
    except BaseException:
        ssl_sock.close()
        raise


def _match_hostname(
    cert: _TYPE_PEER_CERT_RET_DICT | None,
    asserted_hostname: str,
    hostname_checks_common_name: bool = False,
) -> None:
    # Our upstream implementation of ssl.match_hostname()
    # only applies this normalization to IP addresses so it doesn't
    # match DNS SANs so we do the same thing!
    stripped_hostname = asserted_hostname.strip("[]")
    if is_ipaddress(stripped_hostname):
        asserted_hostname = stripped_hostname

    try:
        match_hostname(cert, asserted_hostname, hostname_checks_common_name)
    except CertificateError as e:
        log.warning(
            "Certificate did not match expected hostname: %s. Certificate: %s",
            asserted_hostname,
            cert,
        )
        # Add cert to exception and reraise so client code can inspect
        # the cert when catching the exception, if they want to
        e._peer_cert = cert  # type: ignore[attr-defined]
        raise


def _wrap_proxy_error(err: Exception, proxy_scheme: str | None) -> ProxyError:
    # Look for the phrase 'wrong version number', if found
    # then we should warn the user that we're very sure that
    # this proxy is HTTP-only and they have a configuration issue.
    error_normalized = " ".join(re.split("[^a-z]", str(err).lower()))
    is_likely_http_proxy = (
        "wrong version number" in error_normalized
        or "unknown protocol" in error_normalized
        or "record layer failure" in error_normalized
    )
    http_proxy_warning = (
        ". Your proxy appears to only use HTTP and not HTTPS, "
        "try changing your proxy URL to be HTTP. See: "
        "https://urllib3.readthedocs.io/en/latest/advanced-usage.html"
        "#https-proxy-error-http-proxy"
    )
    new_err = ProxyError(
        f"Unable to connect to proxy"
        f"{http_proxy_warning if is_likely_http_proxy and proxy_scheme == 'https' else ''}",
        err,
    )
    new_err.__cause__ = err
    return new_err


def _get_default_user_agent() -> str:
    return f"python-urllib3/{__version__}"


class DummyConnection:
    """Used to detect a failed ConnectionCls import."""


if not ssl:
    HTTPSConnection = DummyConnection  # type: ignore[misc, assignment] # noqa: F811


VerifiedHTTPSConnection = HTTPSConnection


def _url_from_connection(
    conn: HTTPConnection | HTTPSConnection, path: str | None = None
) -> str:
    """Returns the URL from a given connection. This is mainly used for testing and logging."""

    scheme = "https" if isinstance(conn, HTTPSConnection) else "http"

    return Url(scheme=scheme, host=conn.host, port=conn.port, path=path).url

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_core\_io_windows.py ===
from __future__ import annotations

import enum
import itertools
import socket
import sys
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Literal,
    Protocol,
    TypeVar,
    cast,
)

import attrs
from outcome import Value

from .. import _core
from ._io_common import wake_all
from ._run import _public
from ._windows_cffi import (
    INVALID_HANDLE_VALUE,
    AFDPollFlags,
    CData,
    CompletionModes,
    CType,
    ErrorCodes,
    FileFlags,
    Handle,
    IoControlCodes,
    WSAIoctls,
    _handle,
    _Overlapped,
    ffi,
    kernel32,
    ntdll,
    raise_winerror,
    ws2_32,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from typing_extensions import Buffer, TypeAlias

    from .._file_io import _HasFileNo
    from ._traps import Abort, RaiseCancelT
    from ._unbounded_queue import UnboundedQueue

EventResult: TypeAlias = int
T = TypeVar("T")

# There's a lot to be said about the overall design of a Windows event
# loop. See
#
#    https://github.com/python-trio/trio/issues/52
#
# for discussion. This now just has some lower-level notes:
#
# How IOCP fits together:
#
# The general model is that you call some function like ReadFile or WriteFile
# to tell the kernel that you want it to perform some operation, and the
# kernel goes off and does that in the background, then at some point later it
# sends you a notification that the operation is complete. There are some more
# exotic APIs that don't quite fit this pattern, but most APIs do.
#
# Each background operation is tracked using an OVERLAPPED struct, that
# uniquely identifies that particular operation.
#
# An "IOCP" (or "I/O completion port") is an object that lets the kernel send
# us these notifications -- basically it's just a kernel->userspace queue.
#
# Each IOCP notification is represented by an OVERLAPPED_ENTRY struct, which
# contains 3 fields:
# - The "completion key". This is an opaque integer that we pick, and use
#   however is convenient.
# - pointer to the OVERLAPPED struct for the completed operation.
# - dwNumberOfBytesTransferred (an integer).
#
# And in addition, for regular I/O, the OVERLAPPED structure gets filled in
# with:
# - result code (named "Internal")
# - number of bytes transferred (named "InternalHigh"); usually redundant
#   with dwNumberOfBytesTransferred.
#
# There are also some other entries in OVERLAPPED which only matter on input:
# - Offset and OffsetHigh which are inputs to {Read,Write}File and
#   otherwise always zero
# - hEvent which is for if you aren't using IOCP; we always set it to zero.
#
# That describes the usual pattern for operations and the usual meaning of
# these struct fields, but really these are just some arbitrary chunks of
# bytes that get passed back and forth, so some operations like to overload
# them to mean something else.
#
# You can also directly queue an OVERLAPPED_ENTRY object to an IOCP by calling
# PostQueuedCompletionStatus. When you use this you get to set all the
# OVERLAPPED_ENTRY fields to arbitrary values.
#
# You can request to cancel any operation if you know which handle it was
# issued on + the OVERLAPPED struct that identifies it (via CancelIoEx). This
# request might fail because the operation has already completed, or it might
# be queued to happen in the background, so you only find out whether it
# succeeded or failed later, when we get back the notification for the
# operation being complete.
#
# There are three types of operations that we support:
#
# == Regular I/O operations on handles (e.g. files or named pipes) ==
#
# Implemented by: register_with_iocp, wait_overlapped
#
# To use these, you have to register the handle with your IOCP first. Once
# it's registered, any operations on that handle will automatically send
# completion events to that IOCP, with a completion key that you specify *when
# the handle is registered* (so you can't use different completion keys for
# different operations).
#
# We give these two dedicated completion keys: CKeys.WAIT_OVERLAPPED for
# regular operations, and CKeys.LATE_CANCEL that's used to make
# wait_overlapped cancellable even if the user forgot to call
# register_with_iocp. The problem here is that after we request the cancel,
# wait_overlapped keeps blocking until it sees the completion notification...
# but if the user forgot to register_with_iocp, then the completion will never
# come, so the cancellation will never resolve. To avoid this, whenever we try
# to cancel an I/O operation and the cancellation fails, we use
# PostQueuedCompletionStatus to send a CKeys.LATE_CANCEL notification. If this
# arrives before the real completion, we assume the user forgot to call
# register_with_iocp on their handle, and raise an error accordingly.
#
# == Socket state notifications ==
#
# Implemented by: wait_readable, wait_writable
#
# The public APIs that windows provides for this are all really awkward and
# don't integrate with IOCP. So we drop down to a lower level, and talk
# directly to the socket device driver in the kernel, which is called "AFD".
# Unfortunately, this is a totally undocumented internal API. Fortunately
# libuv also does this, so we can be pretty confident that MS won't break it
# on us, and there is a *little* bit of information out there if you go
# digging.
#
# Basically: we open a magic file that refers to the AFD driver, register the
# magic file with our IOCP, and then we can issue regular overlapped I/O
# operations on that handle. Specifically, the operation we use is called
# IOCTL_AFD_POLL, which lets us pass in a buffer describing which events we're
# interested in on a given socket (readable, writable, etc.). Later, when the
# operation completes, the kernel rewrites the buffer we passed in to record
# which events happened, and uses IOCP as normal to notify us that this
# operation has completed.
#
# Unfortunately, the Windows kernel seems to have bugs if you try to issue
# multiple simultaneous IOCTL_AFD_POLL operations on the same socket (see
# https://github.com/python-trio/trio/wiki/notes-to-self#afd-labpy).
# So if a user calls wait_readable and
# wait_writable at the same time, we have to combine those into a single
# IOCTL_AFD_POLL. This means we can't just use the wait_overlapped machinery.
# Instead we have some dedicated code to handle these operations, and a
# dedicated completion key CKeys.AFD_POLL.
#
# Sources of information:
# - https://github.com/python-trio/trio/issues/52
# - Wepoll: https://github.com/piscisaureus/wepoll/
# - libuv: https://github.com/libuv/libuv/
# - ReactOS: https://github.com/reactos/reactos/
# - Ancient leaked copies of the Windows NT and Winsock source code:
#   https://github.com/pustladi/Windows-2000/blob/661d000d50637ed6fab2329d30e31775046588a9/private/net/sockets/winsock2/wsp/msafd/select.c#L59-L655
#   https://github.com/metoo10987/WinNT4/blob/f5c14e6b42c8f45c20fe88d14c61f9d6e0386b8e/private/ntos/afd/poll.c#L68-L707
# - The WSAEventSelect docs (this exposes a finer-grained set of events than
#   select(), so if you squint you can treat it as a source of information on
#   the fine-grained AFD poll types)
#
#
# == Everything else ==
#
# There are also some weirder APIs for interacting with IOCP. For example, the
# "Job" API lets you specify an IOCP handle and "completion key", and then in
# the future whenever certain events happen it sends uses IOCP to send a
# notification. These notifications don't correspond to any particular
# operation; they're just spontaneous messages you get. The
# "dwNumberOfBytesTransferred" field gets repurposed to carry an identifier
# for the message type (e.g. JOB_OBJECT_MSG_EXIT_PROCESS), and the
# "lpOverlapped" field gets repurposed to carry some arbitrary data that
# depends on the message type (e.g. the pid of the process that exited).
#
# To handle these, we have monitor_completion_key, where we hand out an
# unassigned completion key, let users set it up however they want, and then
# get any events that arrive on that key.
#
# (Note: monitor_completion_key is not documented or fully baked; expect it to
# change in the future.)


# Our completion keys
class CKeys(enum.IntEnum):
    AFD_POLL = 0
    WAIT_OVERLAPPED = 1
    LATE_CANCEL = 2
    FORCE_WAKEUP = 3
    USER_DEFINED = 4  # and above


# AFD_POLL has a finer-grained set of events than other APIs. We collapse them
# down into Unix-style "readable" and "writable".
#
# Note: AFD_POLL_LOCAL_CLOSE isn't a reliable substitute for notify_closing(),
# because even if the user closes the socket *handle*, the socket *object*
# could still remain open, e.g. if the socket was dup'ed (possibly into
# another process). Explicitly calling notify_closing() guarantees that
# everyone waiting on the *handle* wakes up, which is what you'd expect.
#
# However, we can't avoid getting LOCAL_CLOSE notifications -- the kernel
# delivers them whether we ask for them or not -- so better to include them
# here for documentation, and so that when we check (delivered & requested) we
# get a match.

READABLE_FLAGS = (
    AFDPollFlags.AFD_POLL_RECEIVE
    | AFDPollFlags.AFD_POLL_ACCEPT
    | AFDPollFlags.AFD_POLL_DISCONNECT  # other side sent an EOF
    | AFDPollFlags.AFD_POLL_ABORT
    | AFDPollFlags.AFD_POLL_LOCAL_CLOSE
)

WRITABLE_FLAGS = (
    AFDPollFlags.AFD_POLL_SEND
    | AFDPollFlags.AFD_POLL_CONNECT_FAIL
    | AFDPollFlags.AFD_POLL_ABORT
    | AFDPollFlags.AFD_POLL_LOCAL_CLOSE
)


# Annoyingly, while the API makes it *seem* like you can happily issue as many
# independent AFD_POLL operations as you want without them interfering with
# each other, in fact if you issue two AFD_POLL operations for the same socket
# at the same time with notification going to the same IOCP port, then Windows
# gets super confused. For example, if we issue one operation from
# wait_readable, and another independent operation from wait_writable, then
# Windows may complete the wait_writable operation when the socket becomes
# readable.
#
# To avoid this, we have to coalesce all the operations on a single socket
# into one, and when the set of waiters changes we have to throw away the old
# operation and start a new one.
@attrs.define(eq=False)
class AFDWaiters:
    read_task: _core.Task | None = None
    write_task: _core.Task | None = None
    current_op: AFDPollOp | None = None


# Just used for internal type checking.
class _AFDHandle(Protocol):
    Handle: Handle
    Status: int
    Events: int


# Just used for internal type checking.
class _AFDPollInfo(Protocol):
    Timeout: int
    NumberOfHandles: int
    Exclusive: int
    Handles: list[_AFDHandle]


# We also need to bundle up all the info for a single op into a standalone
# object, because we need to keep all these objects alive until the operation
# finishes, even if we're throwing it away.
@attrs.frozen(eq=False)
class AFDPollOp:
    lpOverlapped: CData
    poll_info: _AFDPollInfo
    waiters: AFDWaiters
    afd_group: AFDGroup


# The Windows kernel has a weird issue when using AFD handles. If you have N
# instances of wait_readable/wait_writable registered with a single AFD handle,
# then cancelling any one of them takes something like O(N**2) time. So if we
# used just a single AFD handle, then cancellation would quickly become very
# expensive, e.g. a program with N active sockets would take something like
# O(N**3) time to unwind after control-C. The solution is to spread our sockets
# out over multiple AFD handles, so that N doesn't grow too large for any
# individual handle.
MAX_AFD_GROUP_SIZE = 500  # at 1000, the cubic scaling is just starting to bite


@attrs.define(eq=False)
class AFDGroup:
    size: int
    handle: Handle


assert not TYPE_CHECKING or sys.platform == "win32"


@attrs.frozen(eq=False)
class _WindowsStatistics:
    tasks_waiting_read: int
    tasks_waiting_write: int
    tasks_waiting_overlapped: int
    completion_key_monitors: int
    backend: Literal["windows"] = attrs.field(init=False, default="windows")


# Maximum number of events to dequeue from the completion port on each pass
# through the run loop. Somewhat arbitrary. Should be large enough to collect
# a good set of tasks on each loop, but not so large to waste tons of memory.
# (Each WindowsIOManager holds a buffer whose size is ~32x this number.)
MAX_EVENTS = 1000


def _check(success: T) -> T:
    if not success:
        raise_winerror()
    return success


def _get_underlying_socket(
    sock: _HasFileNo | int | Handle,
    *,
    which: WSAIoctls = WSAIoctls.SIO_BASE_HANDLE,
) -> Handle:
    if hasattr(sock, "fileno"):
        sock = sock.fileno()
    base_ptr = ffi.new("HANDLE *")
    out_size = ffi.new("DWORD *")
    failed = ws2_32.WSAIoctl(
        ffi.cast("SOCKET", sock),
        which,
        ffi.NULL,
        0,
        base_ptr,
        ffi.sizeof("HANDLE"),
        out_size,
        ffi.NULL,
        ffi.NULL,
    )
    if failed:
        code = ws2_32.WSAGetLastError()
        raise_winerror(code)
    return Handle(base_ptr[0])


def _get_base_socket(sock: _HasFileNo | int | Handle) -> Handle:
    # There is a development kit for LSPs called Komodia Redirector.
    # It does some unusual (some might say evil) things like intercepting
    # SIO_BASE_HANDLE (fails) and SIO_BSP_HANDLE_SELECT (returns the same
    # socket) in a misguided attempt to prevent bypassing it. It's been used
    # in malware including the infamous Lenovo Superfish incident from 2015,
    # but unfortunately is also used in some legitimate products such as
    # parental control tools and Astrill VPN. Komodia happens to not
    # block SIO_BSP_HANDLE_POLL, so we'll try SIO_BASE_HANDLE and fall back
    # to SIO_BSP_HANDLE_POLL if it doesn't work.
    # References:
    # - https://github.com/piscisaureus/wepoll/blob/0598a791bf9cbbf480793d778930fc635b044980/wepoll.c#L2223
    # - https://github.com/tokio-rs/mio/issues/1314

    while True:
        try:
            # If this is not a Komodia-intercepted socket, we can just use
            # SIO_BASE_HANDLE.
            return _get_underlying_socket(sock)
        except OSError as ex:
            if ex.winerror == ErrorCodes.ERROR_NOT_SOCKET:
                # SIO_BASE_HANDLE might fail even without LSP intervention,
                # if we get something that's not a socket.
                raise
            if hasattr(sock, "fileno"):
                sock = sock.fileno()
            sock = _handle(sock)
            next_sock = _get_underlying_socket(
                sock,
                which=WSAIoctls.SIO_BSP_HANDLE_POLL,
            )
            if next_sock == sock:
                # If BSP_HANDLE_POLL returns the same socket we already had,
                # then there's no layering going on and we need to fail
                # to prevent an infinite loop.
                raise RuntimeError(
                    "Unexpected network configuration detected: "
                    "SIO_BASE_HANDLE failed and SIO_BSP_HANDLE_POLL didn't "
                    "return a different socket. Please file a bug at "
                    "https://github.com/python-trio/trio/issues/new, "
                    "and include the output of running: "
                    "netsh winsock show catalog",
                ) from ex
            # Otherwise we've gotten at least one layer deeper, so
            # loop back around to keep digging.
            sock = next_sock


def _afd_helper_handle() -> Handle:
    # The "AFD" driver is exposed at the NT path "\Device\Afd". We're using
    # the Win32 CreateFile, though, so we have to pass a Win32 path. \\.\ is
    # how Win32 refers to the NT \GLOBAL??\ directory, and GLOBALROOT is a
    # symlink inside that directory that points to the root of the NT path
    # system. So by sticking that in front of the NT path, we get a Win32
    # path. Alternatively, we could use NtCreateFile directly, since it takes
    # an NT path. But we already wrap CreateFileW so this was easier.
    # References:
    #   https://blogs.msdn.microsoft.com/jeremykuhne/2016/05/02/dos-to-nt-a-paths-journey/
    #   https://stackoverflow.com/a/21704022
    #
    # I'm actually not sure what the \Trio part at the end of the path does.
    # Wepoll uses \Device\Afd\Wepoll, so I just copied them. (I'm guessing it
    # might be visible in some debug tools, and is otherwise arbitrary?)
    rawname = r"\\.\GLOBALROOT\Device\Afd\Trio".encode("utf-16le") + b"\0\0"
    rawname_buf = ffi.from_buffer(rawname)

    handle = kernel32.CreateFileW(
        ffi.cast("LPCWSTR", rawname_buf),
        FileFlags.SYNCHRONIZE,
        FileFlags.FILE_SHARE_READ | FileFlags.FILE_SHARE_WRITE,
        ffi.NULL,  # no security attributes
        FileFlags.OPEN_EXISTING,
        FileFlags.FILE_FLAG_OVERLAPPED,
        ffi.NULL,  # no template file
    )
    if handle == INVALID_HANDLE_VALUE:  # pragma: no cover
        raise_winerror()
    return handle


@attrs.frozen(slots=False)
class CompletionKeyEventInfo:
    lpOverlapped: CData
    dwNumberOfBytesTransferred: int


class WindowsIOManager:
    def __init__(self) -> None:
        # If this method raises an exception, then __del__ could run on a
        # half-initialized object. So we initialize everything that __del__
        # touches to safe values up front, before we do anything that can
        # fail.
        self._iocp = None
        self._all_afd_handles: list[Handle] = []

        self._iocp = _check(
            kernel32.CreateIoCompletionPort(INVALID_HANDLE_VALUE, ffi.NULL, 0, 0),
        )
        self._events = ffi.new("OVERLAPPED_ENTRY[]", MAX_EVENTS)

        self._vacant_afd_groups: set[AFDGroup] = set()
        # {lpOverlapped: AFDPollOp}
        self._afd_ops: dict[CData, AFDPollOp] = {}
        # {socket handle: AFDWaiters}
        self._afd_waiters: dict[Handle, AFDWaiters] = {}

        # {lpOverlapped: task}
        self._overlapped_waiters: dict[CData, _core.Task] = {}
        self._posted_too_late_to_cancel: set[CData] = set()

        self._completion_key_queues: dict[int, UnboundedQueue[object]] = {}
        self._completion_key_counter = itertools.count(CKeys.USER_DEFINED)

        with socket.socket() as s:
            # We assume we're not working with any LSP that changes
            # how select() is supposed to work. Validate this by
            # ensuring that the result of SIO_BSP_HANDLE_SELECT (the
            # LSP-hookable mechanism for "what should I use for
            # select()?") matches that of SIO_BASE_HANDLE ("what is
            # the real non-hooked underlying socket here?").
            #
            # This doesn't work for Komodia-based LSPs; see the comments
            # in _get_base_socket() for details. But we have special
            # logic for those, so we just skip this check if
            # SIO_BASE_HANDLE fails.

            # LSPs can in theory override this, but we believe that it never
            # actually happens in the wild (except Komodia)
            select_handle = _get_underlying_socket(
                s,
                which=WSAIoctls.SIO_BSP_HANDLE_SELECT,
            )
            try:
                # LSPs shouldn't override this...
                base_handle = _get_underlying_socket(s, which=WSAIoctls.SIO_BASE_HANDLE)
            except OSError:
                # But Komodia-based LSPs do anyway, in a way that causes
                # a failure with WSAEFAULT. We have special handling for
                # them in _get_base_socket(). Make sure it works.
                _get_base_socket(s)
            else:
                if base_handle != select_handle:
                    raise RuntimeError(
                        "Unexpected network configuration detected: "
                        "SIO_BASE_HANDLE and SIO_BSP_HANDLE_SELECT differ. "
                        "Please file a bug at "
                        "https://github.com/python-trio/trio/issues/new, "
                        "and include the output of running: "
                        "netsh winsock show catalog",
                    )

    def close(self) -> None:
        try:
            if self._iocp is not None:
                iocp = self._iocp
                self._iocp = None
                _check(kernel32.CloseHandle(iocp))
        finally:
            while self._all_afd_handles:
                afd_handle = self._all_afd_handles.pop()
                _check(kernel32.CloseHandle(afd_handle))

    def __del__(self) -> None:
        self.close()

    def statistics(self) -> _WindowsStatistics:
        tasks_waiting_read = 0
        tasks_waiting_write = 0
        for waiter in self._afd_waiters.values():
            if waiter.read_task is not None:
                tasks_waiting_read += 1
            if waiter.write_task is not None:
                tasks_waiting_write += 1
        return _WindowsStatistics(
            tasks_waiting_read=tasks_waiting_read,
            tasks_waiting_write=tasks_waiting_write,
            tasks_waiting_overlapped=len(self._overlapped_waiters),
            completion_key_monitors=len(self._completion_key_queues),
        )

    def force_wakeup(self) -> None:
        assert self._iocp is not None
        _check(
            kernel32.PostQueuedCompletionStatus(
                self._iocp,
                0,
                CKeys.FORCE_WAKEUP,
                ffi.NULL,
            ),
        )

    def get_events(self, timeout: float) -> EventResult:
        received = ffi.new("PULONG")
        milliseconds = round(1000 * timeout)
        if timeout > 0 and milliseconds == 0:
            milliseconds = 1
        try:
            assert self._iocp is not None
            _check(
                kernel32.GetQueuedCompletionStatusEx(
                    self._iocp,
                    self._events,
                    MAX_EVENTS,
                    received,
                    milliseconds,
                    0,
                ),
            )
        except OSError as exc:
            if exc.winerror != ErrorCodes.WAIT_TIMEOUT:  # pragma: no cover
                raise
            return 0
        result = received[0]
        assert isinstance(result, int)
        return result

    def process_events(self, received: EventResult) -> None:
        for i in range(received):
            entry = self._events[i]
            if entry.lpCompletionKey == CKeys.AFD_POLL:
                lpo = entry.lpOverlapped
                op = self._afd_ops.pop(lpo)
                waiters = op.waiters
                if waiters.current_op is not op:
                    # Stale op, nothing to do
                    pass
                else:
                    waiters.current_op = None
                    # I don't think this can happen, so if it does let's crash
                    # and get a debug trace.
                    if lpo.Internal != 0:  # pragma: no cover
                        code = ntdll.RtlNtStatusToDosError(lpo.Internal)
                        raise_winerror(code)
                    flags = op.poll_info.Handles[0].Events
                    if waiters.read_task and flags & READABLE_FLAGS:
                        _core.reschedule(waiters.read_task)
                        waiters.read_task = None
                    if waiters.write_task and flags & WRITABLE_FLAGS:
                        _core.reschedule(waiters.write_task)
                        waiters.write_task = None
                    self._refresh_afd(op.poll_info.Handles[0].Handle)
            elif entry.lpCompletionKey == CKeys.WAIT_OVERLAPPED:
                # Regular I/O event, dispatch on lpOverlapped
                waiter = self._overlapped_waiters.pop(entry.lpOverlapped)
                overlapped = entry.lpOverlapped
                transferred = entry.dwNumberOfBytesTransferred
                info = CompletionKeyEventInfo(
                    lpOverlapped=overlapped,
                    dwNumberOfBytesTransferred=transferred,
                )
                _core.reschedule(waiter, Value(info))
            elif entry.lpCompletionKey == CKeys.LATE_CANCEL:
                # Post made by a regular I/O event's abort_fn
                # after it failed to cancel the I/O. If we still
                # have a waiter with this lpOverlapped, we didn't
                # get the regular I/O completion and almost
                # certainly the user forgot to call
                # register_with_iocp.
                self._posted_too_late_to_cancel.remove(entry.lpOverlapped)
                try:
                    waiter = self._overlapped_waiters.pop(entry.lpOverlapped)
                except KeyError:
                    # Looks like the actual completion got here before this
                    # fallback post did -- we're in the "expected" case of
                    # too-late-to-cancel, where the user did nothing wrong.
                    # Nothing more to do.
                    pass
                else:
                    exc = _core.TrioInternalError(
                        f"Failed to cancel overlapped I/O in {waiter.name} and didn't "
                        "receive the completion either. Did you forget to "
                        "call register_with_iocp()?",
                    )
                    # Raising this out of handle_io ensures that
                    # the user will see our message even if some
                    # other task is in an uncancellable wait due
                    # to the same underlying forgot-to-register
                    # issue (if their CancelIoEx succeeds, we
                    # have no way of noticing that their completion
                    # won't arrive). Unfortunately it loses the
                    # task traceback. If you're debugging this
                    # error and can't tell where it's coming from,
                    # try changing this line to
                    # _core.reschedule(waiter, outcome.Error(exc))
                    raise exc
            elif entry.lpCompletionKey == CKeys.FORCE_WAKEUP:
                pass
            else:
                # dispatch on lpCompletionKey
                queue = self._completion_key_queues[entry.lpCompletionKey]
                overlapped = int(ffi.cast("uintptr_t", entry.lpOverlapped))
                transferred = entry.dwNumberOfBytesTransferred
                info = CompletionKeyEventInfo(
                    lpOverlapped=overlapped,
                    dwNumberOfBytesTransferred=transferred,
                )
                queue.put_nowait(info)

    def _register_with_iocp(self, handle_: int | CData, completion_key: int) -> None:
        handle = _handle(handle_)
        assert self._iocp is not None
        _check(kernel32.CreateIoCompletionPort(handle, self._iocp, completion_key, 0))
        # Supposedly this makes things slightly faster, by disabling the
        # ability to do WaitForSingleObject(handle). We would never want to do
        # that anyway, so might as well get the extra speed (if any).
        # Ref: http://www.lenholgate.com/blog/2009/09/interesting-blog-posts-on-high-performance-servers.html
        _check(
            kernel32.SetFileCompletionNotificationModes(
                handle,
                CompletionModes.FILE_SKIP_SET_EVENT_ON_HANDLE,
            ),
        )

    ################################################################
    # AFD stuff
    ################################################################

    def _refresh_afd(self, base_handle: Handle) -> None:
        waiters = self._afd_waiters[base_handle]
        if waiters.current_op is not None:
            afd_group = waiters.current_op.afd_group
            try:
                _check(
                    kernel32.CancelIoEx(
                        afd_group.handle,
                        waiters.current_op.lpOverlapped,
                    ),
                )
            except OSError as exc:
                if exc.winerror != ErrorCodes.ERROR_NOT_FOUND:
                    # I don't think this is possible, so if it happens let's
                    # crash noisily.
                    raise  # pragma: no cover
            waiters.current_op = None
            afd_group.size -= 1
            self._vacant_afd_groups.add(afd_group)

        flags = 0
        if waiters.read_task is not None:
            flags |= READABLE_FLAGS
        if waiters.write_task is not None:
            flags |= WRITABLE_FLAGS

        if not flags:
            del self._afd_waiters[base_handle]
        else:
            try:
                afd_group = self._vacant_afd_groups.pop()
            except KeyError:
                afd_group = AFDGroup(0, _afd_helper_handle())
                self._register_with_iocp(afd_group.handle, CKeys.AFD_POLL)
                self._all_afd_handles.append(afd_group.handle)
            self._vacant_afd_groups.add(afd_group)

            lpOverlapped = ffi.new("LPOVERLAPPED")

            poll_info = cast("_AFDPollInfo", ffi.new("AFD_POLL_INFO *"))
            poll_info.Timeout = 2**63 - 1  # INT64_MAX
            poll_info.NumberOfHandles = 1
            poll_info.Exclusive = 0
            poll_info.Handles[0].Handle = base_handle
            poll_info.Handles[0].Status = 0
            poll_info.Handles[0].Events = flags

            try:
                _check(
                    kernel32.DeviceIoControl(
                        afd_group.handle,
                        IoControlCodes.IOCTL_AFD_POLL,
                        cast("CType", poll_info),
                        ffi.sizeof("AFD_POLL_INFO"),
                        cast("CType", poll_info),
                        ffi.sizeof("AFD_POLL_INFO"),
                        ffi.NULL,
                        lpOverlapped,
                    ),
                )
            except OSError as exc:
                if exc.winerror != ErrorCodes.ERROR_IO_PENDING:
                    # This could happen if the socket handle got closed behind
                    # our back while a wait_* call was pending, and we tried
                    # to re-issue the call. Clear our state and wake up any
                    # pending calls.
                    del self._afd_waiters[base_handle]
                    # Do this last, because it could raise.
                    wake_all(waiters, exc)
                    return
            op = AFDPollOp(lpOverlapped, poll_info, waiters, afd_group)
            waiters.current_op = op
            self._afd_ops[lpOverlapped] = op
            afd_group.size += 1
            if afd_group.size >= MAX_AFD_GROUP_SIZE:
                self._vacant_afd_groups.remove(afd_group)

    async def _afd_poll(self, sock: _HasFileNo | int, mode: str) -> None:
        base_handle = _get_base_socket(sock)
        waiters = self._afd_waiters.get(base_handle)
        if waiters is None:
            waiters = AFDWaiters()
            self._afd_waiters[base_handle] = waiters
        if getattr(waiters, mode) is not None:
            raise _core.BusyResourceError
        setattr(waiters, mode, _core.current_task())
        # Could potentially raise if the handle is somehow invalid; that's OK,
        # we let it escape.
        self._refresh_afd(base_handle)

        def abort_fn(_: RaiseCancelT) -> Abort:
            setattr(waiters, mode, None)
            self._refresh_afd(base_handle)
            return _core.Abort.SUCCEEDED

        await _core.wait_task_rescheduled(abort_fn)

    @_public
    async def wait_readable(self, sock: _HasFileNo | int) -> None:
        """Block until the kernel reports that the given object is readable.

        On Unix systems, ``sock`` must either be an integer file descriptor,
        or else an object with a ``.fileno()`` method which returns an
        integer file descriptor. Any kind of file descriptor can be passed,
        though the exact semantics will depend on your kernel. For example,
        this probably won't do anything useful for on-disk files.

        On Windows systems, ``sock`` must either be an integer ``SOCKET``
        handle, or else an object with a ``.fileno()`` method which returns
        an integer ``SOCKET`` handle. File descriptors aren't supported,
        and neither are handles that refer to anything besides a
        ``SOCKET``.

        :raises trio.BusyResourceError:
            if another task is already waiting for the given socket to
            become readable.
        :raises trio.ClosedResourceError:
            if another task calls :func:`notify_closing` while this
            function is still working.
        """
        await self._afd_poll(sock, "read_task")

    @_public
    async def wait_writable(self, sock: _HasFileNo | int) -> None:
        """Block until the kernel reports that the given object is writable.

        See `wait_readable` for the definition of ``sock``.

        :raises trio.BusyResourceError:
            if another task is already waiting for the given socket to
            become writable.
        :raises trio.ClosedResourceError:
            if another task calls :func:`notify_closing` while this
            function is still working.
        """
        await self._afd_poll(sock, "write_task")

    @_public
    def notify_closing(self, handle: Handle | int | _HasFileNo) -> None:
        """Notify waiters of the given object that it will be closed.

        Call this before closing a file descriptor (on Unix) or socket (on
        Windows). This will cause any `wait_readable` or `wait_writable`
        calls on the given object to immediately wake up and raise
        `~trio.ClosedResourceError`.

        This doesn't actually close the object – you still have to do that
        yourself afterwards. Also, you want to be careful to make sure no
        new tasks start waiting on the object in between when you call this
        and when it's actually closed. So to close something properly, you
        usually want to do these steps in order:

        1. Explicitly mark the object as closed, so that any new attempts
           to use it will abort before they start.
        2. Call `notify_closing` to wake up any already-existing users.
        3. Actually close the object.

        It's also possible to do them in a different order if that's more
        convenient, *but only if* you make sure not to have any checkpoints in
        between the steps. This way they all happen in a single atomic
        step, so other tasks won't be able to tell what order they happened
        in anyway.
        """
        handle = _get_base_socket(handle)
        waiters = self._afd_waiters.get(handle)
        if waiters is not None:
            wake_all(waiters, _core.ClosedResourceError())
            self._refresh_afd(handle)

    ################################################################
    # Regular overlapped operations
    ################################################################

    @_public
    def register_with_iocp(self, handle: int | CData) -> None:
        """TODO: these are implemented, but are currently more of a sketch than
        anything real. See `#26
        <https://github.com/python-trio/trio/issues/26>`__ and `#52
        <https://github.com/python-trio/trio/issues/52>`__.
        """
        self._register_with_iocp(handle, CKeys.WAIT_OVERLAPPED)

    @_public
    async def wait_overlapped(
        self,
        handle_: int | CData,
        lpOverlapped: CData | int,
    ) -> object:
        """TODO: these are implemented, but are currently more of a sketch than
        anything real. See `#26
        <https://github.com/python-trio/trio/issues/26>`__ and `#52
        <https://github.com/python-trio/trio/issues/52>`__.
        """
        handle = _handle(handle_)
        if isinstance(lpOverlapped, int):  # TODO: test this line
            lpOverlapped = ffi.cast("LPOVERLAPPED", lpOverlapped)
        if lpOverlapped in self._overlapped_waiters:  # TODO: test this line
            raise _core.BusyResourceError(
                "another task is already waiting on that lpOverlapped",
            )
        task = _core.current_task()
        self._overlapped_waiters[lpOverlapped] = task
        raise_cancel = None

        def abort(raise_cancel_: RaiseCancelT) -> Abort:
            nonlocal raise_cancel
            raise_cancel = raise_cancel_
            try:
                _check(kernel32.CancelIoEx(handle, lpOverlapped))
            except OSError as exc:
                if exc.winerror == ErrorCodes.ERROR_NOT_FOUND:
                    assert self._iocp is not None
                    # Too late to cancel. If this happens because the
                    # operation is already completed, we don't need to do
                    # anything; we'll get a notification of that completion
                    # soon. But another possibility is that the operation was
                    # performed on a handle that wasn't registered with our
                    # IOCP (ie, the user forgot to call register_with_iocp),
                    # in which case we're just never going to see the
                    # completion. To avoid an uncancellable infinite sleep in
                    # the latter case, we'll PostQueuedCompletionStatus here,
                    # and if our post arrives before the original completion
                    # does, we'll assume the handle wasn't registered.
                    _check(
                        kernel32.PostQueuedCompletionStatus(
                            self._iocp,
                            0,
                            CKeys.LATE_CANCEL,
                            lpOverlapped,
                        ),
                    )
                    # Keep the lpOverlapped referenced so its address
                    # doesn't get reused until our posted completion
                    # status has been processed. Otherwise, we can
                    # get confused about which completion goes with
                    # which I/O.
                    self._posted_too_late_to_cancel.add(lpOverlapped)
                else:  # pragma: no cover
                    raise _core.TrioInternalError(
                        "CancelIoEx failed with unexpected error",
                    ) from exc
            return _core.Abort.FAILED

        # TODO: what type does this return?
        info = await _core.wait_task_rescheduled(abort)
        lpOverlappedTyped = cast("_Overlapped", lpOverlapped)
        if lpOverlappedTyped.Internal != 0:
            # the lpOverlapped reports the error as an NT status code,
            # which we must convert back to a Win32 error code before
            # it will produce the right sorts of exceptions
            code = ntdll.RtlNtStatusToDosError(lpOverlappedTyped.Internal)
            if code == ErrorCodes.ERROR_OPERATION_ABORTED:
                if raise_cancel is not None:
                    raise_cancel()
                else:
                    # We didn't request this cancellation, so assume
                    # it happened due to the underlying handle being
                    # closed before the operation could complete.
                    raise _core.ClosedResourceError("another task closed this resource")
            else:
                raise_winerror(code)
        return info

    async def _perform_overlapped(
        self,
        handle: int | CData,
        submit_fn: Callable[[_Overlapped], None],
    ) -> _Overlapped:
        # submit_fn(lpOverlapped) submits some I/O
        # it may raise an OSError with ERROR_IO_PENDING
        # the handle must already be registered using
        # register_with_iocp(handle)
        # This always does a schedule point, but it's possible that the
        # operation will not be cancellable, depending on how Windows is
        # feeling today. So we need to check for cancellation manually.
        await _core.checkpoint_if_cancelled()
        lpOverlapped = cast("_Overlapped", ffi.new("LPOVERLAPPED"))
        try:
            submit_fn(lpOverlapped)
        except OSError as exc:
            if exc.winerror != ErrorCodes.ERROR_IO_PENDING:
                raise
        await self.wait_overlapped(handle, cast("CData", lpOverlapped))
        return lpOverlapped

    @_public
    async def write_overlapped(
        self,
        handle: int | CData,
        data: Buffer,
        file_offset: int = 0,
    ) -> int:
        """TODO: these are implemented, but are currently more of a sketch than
        anything real. See `#26
        <https://github.com/python-trio/trio/issues/26>`__ and `#52
        <https://github.com/python-trio/trio/issues/52>`__.
        """
        with ffi.from_buffer(data) as cbuf:

            def submit_write(lpOverlapped: _Overlapped) -> None:
                # yes, these are the real documented names
                offset_fields = lpOverlapped.DUMMYUNIONNAME.DUMMYSTRUCTNAME
                offset_fields.Offset = file_offset & 0xFFFFFFFF
                offset_fields.OffsetHigh = file_offset >> 32
                _check(
                    kernel32.WriteFile(
                        _handle(handle),
                        ffi.cast("LPCVOID", cbuf),
                        len(cbuf),
                        ffi.NULL,
                        lpOverlapped,
                    ),
                )

            lpOverlapped = await self._perform_overlapped(handle, submit_write)
            # this is "number of bytes transferred"
            return lpOverlapped.InternalHigh

    @_public
    async def readinto_overlapped(
        self,
        handle: int | CData,
        buffer: Buffer,
        file_offset: int = 0,
    ) -> int:
        """TODO: these are implemented, but are currently more of a sketch than
        anything real. See `#26
        <https://github.com/python-trio/trio/issues/26>`__ and `#52
        <https://github.com/python-trio/trio/issues/52>`__.
        """
        with ffi.from_buffer(buffer, require_writable=True) as cbuf:

            def submit_read(lpOverlapped: _Overlapped) -> None:
                offset_fields = lpOverlapped.DUMMYUNIONNAME.DUMMYSTRUCTNAME
                offset_fields.Offset = file_offset & 0xFFFFFFFF
                offset_fields.OffsetHigh = file_offset >> 32
                _check(
                    kernel32.ReadFile(
                        _handle(handle),
                        ffi.cast("LPVOID", cbuf),
                        len(cbuf),
                        ffi.NULL,
                        lpOverlapped,
                    ),
                )

            lpOverlapped = await self._perform_overlapped(handle, submit_read)
            return lpOverlapped.InternalHigh

    ################################################################
    # Raw IOCP operations
    ################################################################

    @_public
    def current_iocp(self) -> int:
        """TODO: these are implemented, but are currently more of a sketch than
        anything real. See `#26
        <https://github.com/python-trio/trio/issues/26>`__ and `#52
        <https://github.com/python-trio/trio/issues/52>`__.
        """
        assert self._iocp is not None
        return int(ffi.cast("uintptr_t", self._iocp))

    @contextmanager
    @_public
    def monitor_completion_key(self) -> Iterator[tuple[int, UnboundedQueue[object]]]:
        """TODO: these are implemented, but are currently more of a sketch than
        anything real. See `#26
        <https://github.com/python-trio/trio/issues/26>`__ and `#52
        <https://github.com/python-trio/trio/issues/52>`__.
        """
        key = next(self._completion_key_counter)
        queue = _core.UnboundedQueue[object]()
        self._completion_key_queues[key] = queue
        try:
            yield (key, queue)
        finally:
            del self._completion_key_queues[key]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\requests\models.py ===
"""
requests.models
~~~~~~~~~~~~~~~

This module contains the primary objects that power Requests.
"""

import datetime

# Import encoding now, to avoid implicit import later.
# Implicit import within threads may cause LookupError when standard library is in a ZIP,
# such as in Embedded Python. See https://github.com/psf/requests/issues/3578.
import encodings.idna  # noqa: F401
from io import UnsupportedOperation

from urllib3.exceptions import (
    DecodeError,
    LocationParseError,
    ProtocolError,
    ReadTimeoutError,
    SSLError,
)
from urllib3.fields import RequestField
from urllib3.filepost import encode_multipart_formdata
from urllib3.util import parse_url

from ._internal_utils import to_native_string, unicode_is_ascii
from .auth import HTTPBasicAuth
from .compat import (
    Callable,
    JSONDecodeError,
    Mapping,
    basestring,
    builtin_str,
    chardet,
    cookielib,
)
from .compat import json as complexjson
from .compat import urlencode, urlsplit, urlunparse
from .cookies import _copy_cookie_jar, cookiejar_from_dict, get_cookie_header
from .exceptions import (
    ChunkedEncodingError,
    ConnectionError,
    ContentDecodingError,
    HTTPError,
    InvalidJSONError,
    InvalidURL,
)
from .exceptions import JSONDecodeError as RequestsJSONDecodeError
from .exceptions import MissingSchema
from .exceptions import SSLError as RequestsSSLError
from .exceptions import StreamConsumedError
from .hooks import default_hooks
from .status_codes import codes
from .structures import CaseInsensitiveDict
from .utils import (
    check_header_validity,
    get_auth_from_url,
    guess_filename,
    guess_json_utf,
    iter_slices,
    parse_header_links,
    requote_uri,
    stream_decode_response_unicode,
    super_len,
    to_key_val_list,
)

#: The set of HTTP status codes that indicate an automatically
#: processable redirect.
REDIRECT_STATI = (
    codes.moved,  # 301
    codes.found,  # 302
    codes.other,  # 303
    codes.temporary_redirect,  # 307
    codes.permanent_redirect,  # 308
)

DEFAULT_REDIRECT_LIMIT = 30
CONTENT_CHUNK_SIZE = 10 * 1024
ITER_CHUNK_SIZE = 512


class RequestEncodingMixin:
    @property
    def path_url(self):
        """Build the path URL to use."""

        url = []

        p = urlsplit(self.url)

        path = p.path
        if not path:
            path = "/"

        url.append(path)

        query = p.query
        if query:
            url.append("?")
            url.append(query)

        return "".join(url)

    @staticmethod
    def _encode_params(data):
        """Encode parameters in a piece of data.

        Will successfully encode parameters when passed as a dict or a list of
        2-tuples. Order is retained if data is a list of 2-tuples but arbitrary
        if parameters are supplied as a dict.
        """

        if isinstance(data, (str, bytes)):
            return data
        elif hasattr(data, "read"):
            return data
        elif hasattr(data, "__iter__"):
            result = []
            for k, vs in to_key_val_list(data):
                if isinstance(vs, basestring) or not hasattr(vs, "__iter__"):
                    vs = [vs]
                for v in vs:
                    if v is not None:
                        result.append(
                            (
                                k.encode("utf-8") if isinstance(k, str) else k,
                                v.encode("utf-8") if isinstance(v, str) else v,
                            )
                        )
            return urlencode(result, doseq=True)
        else:
            return data

    @staticmethod
    def _encode_files(files, data):
        """Build the body for a multipart/form-data request.

        Will successfully encode files when passed as a dict or a list of
        tuples. Order is retained if data is a list of tuples but arbitrary
        if parameters are supplied as a dict.
        The tuples may be 2-tuples (filename, fileobj), 3-tuples (filename, fileobj, contentype)
        or 4-tuples (filename, fileobj, contentype, custom_headers).
        """
        if not files:
            raise ValueError("Files must be provided.")
        elif isinstance(data, basestring):
            raise ValueError("Data must not be a string.")

        new_fields = []
        fields = to_key_val_list(data or {})
        files = to_key_val_list(files or {})

        for field, val in fields:
            if isinstance(val, basestring) or not hasattr(val, "__iter__"):
                val = [val]
            for v in val:
                if v is not None:
                    # Don't call str() on bytestrings: in Py3 it all goes wrong.
                    if not isinstance(v, bytes):
                        v = str(v)

                    new_fields.append(
                        (
                            field.decode("utf-8")
                            if isinstance(field, bytes)
                            else field,
                            v.encode("utf-8") if isinstance(v, str) else v,
                        )
                    )

        for k, v in files:
            # support for explicit filename
            ft = None
            fh = None
            if isinstance(v, (tuple, list)):
                if len(v) == 2:
                    fn, fp = v
                elif len(v) == 3:
                    fn, fp, ft = v
                else:
                    fn, fp, ft, fh = v
            else:
                fn = guess_filename(v) or k
                fp = v

            if isinstance(fp, (str, bytes, bytearray)):
                fdata = fp
            elif hasattr(fp, "read"):
                fdata = fp.read()
            elif fp is None:
                continue
            else:
                fdata = fp

            rf = RequestField(name=k, data=fdata, filename=fn, headers=fh)
            rf.make_multipart(content_type=ft)
            new_fields.append(rf)

        body, content_type = encode_multipart_formdata(new_fields)

        return body, content_type


class RequestHooksMixin:
    def register_hook(self, event, hook):
        """Properly register a hook."""

        if event not in self.hooks:
            raise ValueError(f'Unsupported event specified, with event name "{event}"')

        if isinstance(hook, Callable):
            self.hooks[event].append(hook)
        elif hasattr(hook, "__iter__"):
            self.hooks[event].extend(h for h in hook if isinstance(h, Callable))

    def deregister_hook(self, event, hook):
        """Deregister a previously registered hook.
        Returns True if the hook existed, False if not.
        """

        try:
            self.hooks[event].remove(hook)
            return True
        except ValueError:
            return False


class Request(RequestHooksMixin):
    """A user-created :class:`Request <Request>` object.

    Used to prepare a :class:`PreparedRequest <PreparedRequest>`, which is sent to the server.

    :param method: HTTP method to use.
    :param url: URL to send.
    :param headers: dictionary of headers to send.
    :param files: dictionary of {filename: fileobject} files to multipart upload.
    :param data: the body to attach to the request. If a dictionary or
        list of tuples ``[(key, value)]`` is provided, form-encoding will
        take place.
    :param json: json for the body to attach to the request (if files or data is not specified).
    :param params: URL parameters to append to the URL. If a dictionary or
        list of tuples ``[(key, value)]`` is provided, form-encoding will
        take place.
    :param auth: Auth handler or (user, pass) tuple.
    :param cookies: dictionary or CookieJar of cookies to attach to this request.
    :param hooks: dictionary of callback hooks, for internal usage.

    Usage::

      >>> import requests
      >>> req = requests.Request('GET', 'https://httpbin.org/get')
      >>> req.prepare()
      <PreparedRequest [GET]>
    """

    def __init__(
        self,
        method=None,
        url=None,
        headers=None,
        files=None,
        data=None,
        params=None,
        auth=None,
        cookies=None,
        hooks=None,
        json=None,
    ):
        # Default empty dicts for dict params.
        data = [] if data is None else data
        files = [] if files is None else files
        headers = {} if headers is None else headers
        params = {} if params is None else params
        hooks = {} if hooks is None else hooks

        self.hooks = default_hooks()
        for k, v in list(hooks.items()):
            self.register_hook(event=k, hook=v)

        self.method = method
        self.url = url
        self.headers = headers
        self.files = files
        self.data = data
        self.json = json
        self.params = params
        self.auth = auth
        self.cookies = cookies

    def __repr__(self):
        return f"<Request [{self.method}]>"

    def prepare(self):
        """Constructs a :class:`PreparedRequest <PreparedRequest>` for transmission and returns it."""
        p = PreparedRequest()
        p.prepare(
            method=self.method,
            url=self.url,
            headers=self.headers,
            files=self.files,
            data=self.data,
            json=self.json,
            params=self.params,
            auth=self.auth,
            cookies=self.cookies,
            hooks=self.hooks,
        )
        return p


class PreparedRequest(RequestEncodingMixin, RequestHooksMixin):
    """The fully mutable :class:`PreparedRequest <PreparedRequest>` object,
    containing the exact bytes that will be sent to the server.

    Instances are generated from a :class:`Request <Request>` object, and
    should not be instantiated manually; doing so may produce undesirable
    effects.

    Usage::

      >>> import requests
      >>> req = requests.Request('GET', 'https://httpbin.org/get')
      >>> r = req.prepare()
      >>> r
      <PreparedRequest [GET]>

      >>> s = requests.Session()
      >>> s.send(r)
      <Response [200]>
    """

    def __init__(self):
        #: HTTP verb to send to the server.
        self.method = None
        #: HTTP URL to send the request to.
        self.url = None
        #: dictionary of HTTP headers.
        self.headers = None
        # The `CookieJar` used to create the Cookie header will be stored here
        # after prepare_cookies is called
        self._cookies = None
        #: request body to send to the server.
        self.body = None
        #: dictionary of callback hooks, for internal usage.
        self.hooks = default_hooks()
        #: integer denoting starting position of a readable file-like body.
        self._body_position = None

    def prepare(
        self,
        method=None,
        url=None,
        headers=None,
        files=None,
        data=None,
        params=None,
        auth=None,
        cookies=None,
        hooks=None,
        json=None,
    ):
        """Prepares the entire request with the given parameters."""

        self.prepare_method(method)
        self.prepare_url(url, params)
        self.prepare_headers(headers)
        self.prepare_cookies(cookies)
        self.prepare_body(data, files, json)
        self.prepare_auth(auth, url)

        # Note that prepare_auth must be last to enable authentication schemes
        # such as OAuth to work on a fully prepared request.

        # This MUST go after prepare_auth. Authenticators could add a hook
        self.prepare_hooks(hooks)

    def __repr__(self):
        return f"<PreparedRequest [{self.method}]>"

    def copy(self):
        p = PreparedRequest()
        p.method = self.method
        p.url = self.url
        p.headers = self.headers.copy() if self.headers is not None else None
        p._cookies = _copy_cookie_jar(self._cookies)
        p.body = self.body
        p.hooks = self.hooks
        p._body_position = self._body_position
        return p

    def prepare_method(self, method):
        """Prepares the given HTTP method."""
        self.method = method
        if self.method is not None:
            self.method = to_native_string(self.method.upper())

    @staticmethod
    def _get_idna_encoded_host(host):
        import idna

        try:
            host = idna.encode(host, uts46=True).decode("utf-8")
        except idna.IDNAError:
            raise UnicodeError
        return host

    def prepare_url(self, url, params):
        """Prepares the given HTTP URL."""
        #: Accept objects that have string representations.
        #: We're unable to blindly call unicode/str functions
        #: as this will include the bytestring indicator (b'')
        #: on python 3.x.
        #: https://github.com/psf/requests/pull/2238
        if isinstance(url, bytes):
            url = url.decode("utf8")
        else:
            url = str(url)

        # Remove leading whitespaces from url
        url = url.lstrip()

        # Don't do any URL preparation for non-HTTP schemes like `mailto`,
        # `data` etc to work around exceptions from `url_parse`, which
        # handles RFC 3986 only.
        if ":" in url and not url.lower().startswith("http"):
            self.url = url
            return

        # Support for unicode domain names and paths.
        try:
            scheme, auth, host, port, path, query, fragment = parse_url(url)
        except LocationParseError as e:
            raise InvalidURL(*e.args)

        if not scheme:
            raise MissingSchema(
                f"Invalid URL {url!r}: No scheme supplied. "
                f"Perhaps you meant https://{url}?"
            )

        if not host:
            raise InvalidURL(f"Invalid URL {url!r}: No host supplied")

        # In general, we want to try IDNA encoding the hostname if the string contains
        # non-ASCII characters. This allows users to automatically get the correct IDNA
        # behaviour. For strings containing only ASCII characters, we need to also verify
        # it doesn't start with a wildcard (*), before allowing the unencoded hostname.
        if not unicode_is_ascii(host):
            try:
                host = self._get_idna_encoded_host(host)
            except UnicodeError:
                raise InvalidURL("URL has an invalid label.")
        elif host.startswith(("*", ".")):
            raise InvalidURL("URL has an invalid label.")

        # Carefully reconstruct the network location
        netloc = auth or ""
        if netloc:
            netloc += "@"
        netloc += host
        if port:
            netloc += f":{port}"

        # Bare domains aren't valid URLs.
        if not path:
            path = "/"

        if isinstance(params, (str, bytes)):
            params = to_native_string(params)

        enc_params = self._encode_params(params)
        if enc_params:
            if query:
                query = f"{query}&{enc_params}"
            else:
                query = enc_params

        url = requote_uri(urlunparse([scheme, netloc, path, None, query, fragment]))
        self.url = url

    def prepare_headers(self, headers):
        """Prepares the given HTTP headers."""

        self.headers = CaseInsensitiveDict()
        if headers:
            for header in headers.items():
                # Raise exception on invalid header value.
                check_header_validity(header)
                name, value = header
                self.headers[to_native_string(name)] = value

    def prepare_body(self, data, files, json=None):
        """Prepares the given HTTP body data."""

        # Check if file, fo, generator, iterator.
        # If not, run through normal process.

        # Nottin' on you.
        body = None
        content_type = None

        if not data and json is not None:
            # urllib3 requires a bytes-like body. Python 2's json.dumps
            # provides this natively, but Python 3 gives a Unicode string.
            content_type = "application/json"

            try:
                body = complexjson.dumps(json, allow_nan=False)
            except ValueError as ve:
                raise InvalidJSONError(ve, request=self)

            if not isinstance(body, bytes):
                body = body.encode("utf-8")

        is_stream = all(
            [
                hasattr(data, "__iter__"),
                not isinstance(data, (basestring, list, tuple, Mapping)),
            ]
        )

        if is_stream:
            try:
                length = super_len(data)
            except (TypeError, AttributeError, UnsupportedOperation):
                length = None

            body = data

            if getattr(body, "tell", None) is not None:
                # Record the current file position before reading.
                # This will allow us to rewind a file in the event
                # of a redirect.
                try:
                    self._body_position = body.tell()
                except OSError:
                    # This differentiates from None, allowing us to catch
                    # a failed `tell()` later when trying to rewind the body
                    self._body_position = object()

            if files:
                raise NotImplementedError(
                    "Streamed bodies and files are mutually exclusive."
                )

            if length:
                self.headers["Content-Length"] = builtin_str(length)
            else:
                self.headers["Transfer-Encoding"] = "chunked"
        else:
            # Multi-part file uploads.
            if files:
                (body, content_type) = self._encode_files(files, data)
            else:
                if data:
                    body = self._encode_params(data)
                    if isinstance(data, basestring) or hasattr(data, "read"):
                        content_type = None
                    else:
                        content_type = "application/x-www-form-urlencoded"

            self.prepare_content_length(body)

            # Add content-type if it wasn't explicitly provided.
            if content_type and ("content-type" not in self.headers):
                self.headers["Content-Type"] = content_type

        self.body = body

    def prepare_content_length(self, body):
        """Prepare Content-Length header based on request method and body"""
        if body is not None:
            length = super_len(body)
            if length:
                # If length exists, set it. Otherwise, we fallback
                # to Transfer-Encoding: chunked.
                self.headers["Content-Length"] = builtin_str(length)
        elif (
            self.method not in ("GET", "HEAD")
            and self.headers.get("Content-Length") is None
        ):
            # Set Content-Length to 0 for methods that can have a body
            # but don't provide one. (i.e. not GET or HEAD)
            self.headers["Content-Length"] = "0"

    def prepare_auth(self, auth, url=""):
        """Prepares the given HTTP auth data."""

        # If no Auth is explicitly provided, extract it from the URL first.
        if auth is None:
            url_auth = get_auth_from_url(self.url)
            auth = url_auth if any(url_auth) else None

        if auth:
            if isinstance(auth, tuple) and len(auth) == 2:
                # special-case basic HTTP auth
                auth = HTTPBasicAuth(*auth)

            # Allow auth to make its changes.
            r = auth(self)

            # Update self to reflect the auth changes.
            self.__dict__.update(r.__dict__)

            # Recompute Content-Length
            self.prepare_content_length(self.body)

    def prepare_cookies(self, cookies):
        """Prepares the given HTTP cookie data.

        This function eventually generates a ``Cookie`` header from the
        given cookies using cookielib. Due to cookielib's design, the header
        will not be regenerated if it already exists, meaning this function
        can only be called once for the life of the
        :class:`PreparedRequest <PreparedRequest>` object. Any subsequent calls
        to ``prepare_cookies`` will have no actual effect, unless the "Cookie"
        header is removed beforehand.
        """
        if isinstance(cookies, cookielib.CookieJar):
            self._cookies = cookies
        else:
            self._cookies = cookiejar_from_dict(cookies)

        cookie_header = get_cookie_header(self._cookies, self)
        if cookie_header is not None:
            self.headers["Cookie"] = cookie_header

    def prepare_hooks(self, hooks):
        """Prepares the given hooks."""
        # hooks can be passed as None to the prepare method and to this
        # method. To prevent iterating over None, simply use an empty list
        # if hooks is False-y
        hooks = hooks or []
        for event in hooks:
            self.register_hook(event, hooks[event])


class Response:
    """The :class:`Response <Response>` object, which contains a
    server's response to an HTTP request.
    """

    __attrs__ = [
        "_content",
        "status_code",
        "headers",
        "url",
        "history",
        "encoding",
        "reason",
        "cookies",
        "elapsed",
        "request",
    ]

    def __init__(self):
        self._content = False
        self._content_consumed = False
        self._next = None

        #: Integer Code of responded HTTP Status, e.g. 404 or 200.
        self.status_code = None

        #: Case-insensitive Dictionary of Response Headers.
        #: For example, ``headers['content-encoding']`` will return the
        #: value of a ``'Content-Encoding'`` response header.
        self.headers = CaseInsensitiveDict()

        #: File-like object representation of response (for advanced usage).
        #: Use of ``raw`` requires that ``stream=True`` be set on the request.
        #: This requirement does not apply for use internally to Requests.
        self.raw = None

        #: Final URL location of Response.
        self.url = None

        #: Encoding to decode with when accessing r.text.
        self.encoding = None

        #: A list of :class:`Response <Response>` objects from
        #: the history of the Request. Any redirect responses will end
        #: up here. The list is sorted from the oldest to the most recent request.
        self.history = []

        #: Textual reason of responded HTTP Status, e.g. "Not Found" or "OK".
        self.reason = None

        #: A CookieJar of Cookies the server sent back.
        self.cookies = cookiejar_from_dict({})

        #: The amount of time elapsed between sending the request
        #: and the arrival of the response (as a timedelta).
        #: This property specifically measures the time taken between sending
        #: the first byte of the request and finishing parsing the headers. It
        #: is therefore unaffected by consuming the response content or the
        #: value of the ``stream`` keyword argument.
        self.elapsed = datetime.timedelta(0)

        #: The :class:`PreparedRequest <PreparedRequest>` object to which this
        #: is a response.
        self.request = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __getstate__(self):
        # Consume everything; accessing the content attribute makes
        # sure the content has been fully read.
        if not self._content_consumed:
            self.content

        return {attr: getattr(self, attr, None) for attr in self.__attrs__}

    def __setstate__(self, state):
        for name, value in state.items():
            setattr(self, name, value)

        # pickled objects do not have .raw
        setattr(self, "_content_consumed", True)
        setattr(self, "raw", None)

    def __repr__(self):
        return f"<Response [{self.status_code}]>"

    def __bool__(self):
        """Returns True if :attr:`status_code` is less than 400.

        This attribute checks if the status code of the response is between
        400 and 600 to see if there was a client error or a server error. If
        the status code, is between 200 and 400, this will return True. This
        is **not** a check to see if the response code is ``200 OK``.
        """
        return self.ok

    def __nonzero__(self):
        """Returns True if :attr:`status_code` is less than 400.

        This attribute checks if the status code of the response is between
        400 and 600 to see if there was a client error or a server error. If
        the status code, is between 200 and 400, this will return True. This
        is **not** a check to see if the response code is ``200 OK``.
        """
        return self.ok

    def __iter__(self):
        """Allows you to use a response as an iterator."""
        return self.iter_content(128)

    @property
    def ok(self):
        """Returns True if :attr:`status_code` is less than 400, False if not.

        This attribute checks if the status code of the response is between
        400 and 600 to see if there was a client error or a server error. If
        the status code is between 200 and 400, this will return True. This
        is **not** a check to see if the response code is ``200 OK``.
        """
        try:
            self.raise_for_status()
        except HTTPError:
            return False
        return True

    @property
    def is_redirect(self):
        """True if this Response is a well-formed HTTP redirect that could have
        been processed automatically (by :meth:`Session.resolve_redirects`).
        """
        return "location" in self.headers and self.status_code in REDIRECT_STATI

    @property
    def is_permanent_redirect(self):
        """True if this Response one of the permanent versions of redirect."""
        return "location" in self.headers and self.status_code in (
            codes.moved_permanently,
            codes.permanent_redirect,
        )

    @property
    def next(self):
        """Returns a PreparedRequest for the next request in a redirect chain, if there is one."""
        return self._next

    @property
    def apparent_encoding(self):
        """The apparent encoding, provided by the charset_normalizer or chardet libraries."""
        if chardet is not None:
            return chardet.detect(self.content)["encoding"]
        else:
            # If no character detection library is available, we'll fall back
            # to a standard Python utf-8 str.
            return "utf-8"

    def iter_content(self, chunk_size=1, decode_unicode=False):
        """Iterates over the response data.  When stream=True is set on the
        request, this avoids reading the content at once into memory for
        large responses.  The chunk size is the number of bytes it should
        read into memory.  This is not necessarily the length of each item
        returned as decoding can take place.

        chunk_size must be of type int or None. A value of None will
        function differently depending on the value of `stream`.
        stream=True will read data as it arrives in whatever size the
        chunks are received. If stream=False, data is returned as
        a single chunk.

        If decode_unicode is True, content will be decoded using the best
        available encoding based on the response.
        """

        def generate():
            # Special case for urllib3.
            if hasattr(self.raw, "stream"):
                try:
                    yield from self.raw.stream(chunk_size, decode_content=True)
                except ProtocolError as e:
                    raise ChunkedEncodingError(e)
                except DecodeError as e:
                    raise ContentDecodingError(e)
                except ReadTimeoutError as e:
                    raise ConnectionError(e)
                except SSLError as e:
                    raise RequestsSSLError(e)
            else:
                # Standard file-like object.
                while True:
                    chunk = self.raw.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk

            self._content_consumed = True

        if self._content_consumed and isinstance(self._content, bool):
            raise StreamConsumedError()
        elif chunk_size is not None and not isinstance(chunk_size, int):
            raise TypeError(
                f"chunk_size must be an int, it is instead a {type(chunk_size)}."
            )
        # simulate reading small chunks of the content
        reused_chunks = iter_slices(self._content, chunk_size)

        stream_chunks = generate()

        chunks = reused_chunks if self._content_consumed else stream_chunks

        if decode_unicode:
            chunks = stream_decode_response_unicode(chunks, self)

        return chunks

    def iter_lines(
        self, chunk_size=ITER_CHUNK_SIZE, decode_unicode=False, delimiter=None
    ):
        """Iterates over the response data, one line at a time.  When
        stream=True is set on the request, this avoids reading the
        content at once into memory for large responses.

        .. note:: This method is not reentrant safe.
        """

        pending = None

        for chunk in self.iter_content(
            chunk_size=chunk_size, decode_unicode=decode_unicode
        ):
            if pending is not None:
                chunk = pending + chunk

            if delimiter:
                lines = chunk.split(delimiter)
            else:
                lines = chunk.splitlines()

            if lines and lines[-1] and chunk and lines[-1][-1] == chunk[-1]:
                pending = lines.pop()
            else:
                pending = None

            yield from lines

        if pending is not None:
            yield pending

    @property
    def content(self):
        """Content of the response, in bytes."""

        if self._content is False:
            # Read the contents.
            if self._content_consumed:
                raise RuntimeError("The content for this response was already consumed")

            if self.status_code == 0 or self.raw is None:
                self._content = None
            else:
                self._content = b"".join(self.iter_content(CONTENT_CHUNK_SIZE)) or b""

        self._content_consumed = True
        # don't need to release the connection; that's been handled by urllib3
        # since we exhausted the data.
        return self._content

    @property
    def text(self):
        """Content of the response, in unicode.

        If Response.encoding is None, encoding will be guessed using
        ``charset_normalizer`` or ``chardet``.

        The encoding of the response content is determined based solely on HTTP
        headers, following RFC 2616 to the letter. If you can take advantage of
        non-HTTP knowledge to make a better guess at the encoding, you should
        set ``r.encoding`` appropriately before accessing this property.
        """

        # Try charset from content-type
        content = None
        encoding = self.encoding

        if not self.content:
            return ""

        # Fallback to auto-detected encoding.
        if self.encoding is None:
            encoding = self.apparent_encoding

        # Decode unicode from given encoding.
        try:
            content = str(self.content, encoding, errors="replace")
        except (LookupError, TypeError):
            # A LookupError is raised if the encoding was not found which could
            # indicate a misspelling or similar mistake.
            #
            # A TypeError can be raised if encoding is None
            #
            # So we try blindly encoding.
            content = str(self.content, errors="replace")

        return content

    def json(self, **kwargs):
        r"""Decodes the JSON response body (if any) as a Python object.

        This may return a dictionary, list, etc. depending on what is in the response.

        :param \*\*kwargs: Optional arguments that ``json.loads`` takes.
        :raises requests.exceptions.JSONDecodeError: If the response body does not
            contain valid json.
        """

        if not self.encoding and self.content and len(self.content) > 3:
            # No encoding set. JSON RFC 4627 section 3 states we should expect
            # UTF-8, -16 or -32. Detect which one to use; If the detection or
            # decoding fails, fall back to `self.text` (using charset_normalizer to make
            # a best guess).
            encoding = guess_json_utf(self.content)
            if encoding is not None:
                try:
                    return complexjson.loads(self.content.decode(encoding), **kwargs)
                except UnicodeDecodeError:
                    # Wrong UTF codec detected; usually because it's not UTF-8
                    # but some other 8-bit codec.  This is an RFC violation,
                    # and the server didn't bother to tell us what codec *was*
                    # used.
                    pass
                except JSONDecodeError as e:
                    raise RequestsJSONDecodeError(e.msg, e.doc, e.pos)

        try:
            return complexjson.loads(self.text, **kwargs)
        except JSONDecodeError as e:
            # Catch JSON-related errors and raise as requests.JSONDecodeError
            # This aliases json.JSONDecodeError and simplejson.JSONDecodeError
            raise RequestsJSONDecodeError(e.msg, e.doc, e.pos)

    @property
    def links(self):
        """Returns the parsed header links of the response, if any."""

        header = self.headers.get("link")

        resolved_links = {}

        if header:
            links = parse_header_links(header)

            for link in links:
                key = link.get("rel") or link.get("url")
                resolved_links[key] = link

        return resolved_links

    def raise_for_status(self):
        """Raises :class:`HTTPError`, if one occurred."""

        http_error_msg = ""
        if isinstance(self.reason, bytes):
            # We attempt to decode utf-8 first because some servers
            # choose to localize their reason strings. If the string
            # isn't utf-8, we fall back to iso-8859-1 for all other
            # encodings. (See PR #3538)
            try:
                reason = self.reason.decode("utf-8")
            except UnicodeDecodeError:
                reason = self.reason.decode("iso-8859-1")
        else:
            reason = self.reason

        if 400 <= self.status_code < 500:
            http_error_msg = (
                f"{self.status_code} Client Error: {reason} for url: {self.url}"
            )

        elif 500 <= self.status_code < 600:
            http_error_msg = (
                f"{self.status_code} Server Error: {reason} for url: {self.url}"
            )

        if http_error_msg:
            raise HTTPError(http_error_msg, response=self)

    def close(self):
        """Releases the connection back to the pool. Once this method has been
        called the underlying ``raw`` object must not be accessed again.

        *Note: Should not normally need to be called explicitly.*
        """
        if not self._content_consumed:
            self.raw.close()

        release_conn = getattr(self.raw, "release_conn", None)
        if release_conn is not None:
            release_conn()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\codeprinter.py ===
from __future__ import annotations
from typing import Any

from functools import wraps

from sympy.core import Add, Mul, Pow, S, sympify, Float
from sympy.core.basic import Basic
from sympy.core.expr import Expr, UnevaluatedExpr
from sympy.core.function import Lambda
from sympy.core.mul import _keep_coeff
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import re
from sympy.printing.str import StrPrinter
from sympy.printing.precedence import precedence, PRECEDENCE


class requires:
    """ Decorator for registering requirements on print methods. """
    def __init__(self, **kwargs):
        self._req = kwargs

    def __call__(self, method):
        def _method_wrapper(self_, *args, **kwargs):
            for k, v in self._req.items():
                getattr(self_, k).update(v)
            return method(self_, *args, **kwargs)
        return wraps(method)(_method_wrapper)


class AssignmentError(Exception):
    """
    Raised if an assignment variable for a loop is missing.
    """
    pass

class PrintMethodNotImplementedError(NotImplementedError):
    """
    Raised if a _print_* method is missing in the Printer.
    """
    pass

def _convert_python_lists(arg):
    if isinstance(arg, list):
        from sympy.codegen.abstract_nodes import List
        return List(*(_convert_python_lists(e) for e in arg))
    elif isinstance(arg, tuple):
        return tuple(_convert_python_lists(e) for e in arg)
    else:
        return arg


class CodePrinter(StrPrinter):
    """
    The base class for code-printing subclasses.
    """

    _operators = {
        'and': '&&',
        'or': '||',
        'not': '!',
    }

    _default_settings: dict[str, Any] = {
        'order': None,
        'full_prec': 'auto',
        'error_on_reserved': False,
        'reserved_word_suffix': '_',
        'human': True,
        'inline': False,
        'allow_unknown_functions': False,
        'strict': None  # True or False; None => True if human == True
    }

    # Functions which are "simple" to rewrite to other functions that
    # may be supported
    # function_to_rewrite : (function_to_rewrite_to, iterable_with_other_functions_required)
    _rewriteable_functions = {
            'cot': ('tan', []),
            'csc': ('sin', []),
            'sec': ('cos', []),
            'acot': ('atan', []),
            'acsc': ('asin', []),
            'asec': ('acos', []),
            'coth': ('exp', []),
            'csch': ('exp', []),
            'sech': ('exp', []),
            'acoth': ('log', []),
            'acsch': ('log', []),
            'asech': ('log', []),
            'catalan': ('gamma', []),
            'fibonacci': ('sqrt', []),
            'lucas': ('sqrt', []),
            'beta': ('gamma', []),
            'sinc': ('sin', ['Piecewise']),
            'Mod': ('floor', []),
            'factorial': ('gamma', []),
            'factorial2': ('gamma', ['Piecewise']),
            'subfactorial': ('uppergamma', []),
            'RisingFactorial': ('gamma', ['Piecewise']),
            'FallingFactorial': ('gamma', ['Piecewise']),
            'binomial': ('gamma', []),
            'frac': ('floor', []),
            'Max': ('Piecewise', []),
            'Min': ('Piecewise', []),
            'Heaviside': ('Piecewise', []),
            'erf2': ('erf', []),
            'erfc': ('erf', []),
            'Li': ('li', []),
            'Ei': ('li', []),
            'dirichlet_eta': ('zeta', []),
            'riemann_xi': ('zeta', ['gamma']),
            'SingularityFunction': ('Piecewise', []),
    }

    def __init__(self, settings=None):
        super().__init__(settings=settings)
        if self._settings.get('strict', True) == None:
            # for backwards compatibility, human=False need not to throw:
            self._settings['strict'] = self._settings.get('human', True) == True
        if not hasattr(self, 'reserved_words'):
            self.reserved_words = set()

    def _handle_UnevaluatedExpr(self, expr):
        return expr.replace(re, lambda arg: arg if isinstance(
            arg, UnevaluatedExpr) and arg.args[0].is_real else re(arg))

    def doprint(self, expr, assign_to=None):
        """
        Print the expression as code.

        Parameters
        ----------
        expr : Expression
            The expression to be printed.

        assign_to : Symbol, string, MatrixSymbol, list of strings or Symbols (optional)
            If provided, the printed code will set the expression to a variable or multiple variables
            with the name or names given in ``assign_to``.
        """
        from sympy.matrices.expressions.matexpr import MatrixSymbol
        from sympy.codegen.ast import CodeBlock, Assignment

        def _handle_assign_to(expr, assign_to):
            if assign_to is None:
                return sympify(expr)
            if isinstance(assign_to, (list, tuple)):
                if len(expr) != len(assign_to):
                    raise ValueError('Failed to assign an expression of length {} to {} variables'.format(len(expr), len(assign_to)))
                return CodeBlock(*[_handle_assign_to(lhs, rhs) for lhs, rhs in zip(expr, assign_to)])
            if isinstance(assign_to, str):
                if expr.is_Matrix:
                    assign_to = MatrixSymbol(assign_to, *expr.shape)
                else:
                    assign_to = Symbol(assign_to)
            elif not isinstance(assign_to, Basic):
                raise TypeError("{} cannot assign to object of type {}".format(
                        type(self).__name__, type(assign_to)))
            return Assignment(assign_to, expr)

        expr = _convert_python_lists(expr)
        expr = _handle_assign_to(expr, assign_to)

        # Remove re(...) nodes due to UnevaluatedExpr.is_real always is None:
        expr = self._handle_UnevaluatedExpr(expr)

        # keep a set of expressions that are not strictly translatable to Code
        # and number constants that must be declared and initialized
        self._not_supported = set()
        self._number_symbols = set()

        lines = self._print(expr).splitlines()

        # format the output
        if self._settings["human"]:
            frontlines = []
            if self._not_supported:
                frontlines.append(self._get_comment(
                        "Not supported in {}:".format(self.language)))
                for expr in sorted(self._not_supported, key=str):
                    frontlines.append(self._get_comment(type(expr).__name__))
            for name, value in sorted(self._number_symbols, key=str):
                frontlines.append(self._declare_number_const(name, value))
            lines = frontlines + lines
            lines = self._format_code(lines)
            result = "\n".join(lines)
        else:
            lines = self._format_code(lines)
            num_syms = {(k, self._print(v)) for k, v in self._number_symbols}
            result = (num_syms, self._not_supported, "\n".join(lines))
        self._not_supported = set()
        self._number_symbols = set()
        return result

    def _doprint_loops(self, expr, assign_to=None):
        # Here we print an expression that contains Indexed objects, they
        # correspond to arrays in the generated code.  The low-level implementation
        # involves looping over array elements and possibly storing results in temporary
        # variables or accumulate it in the assign_to object.

        if self._settings.get('contract', True):
            from sympy.tensor import get_contraction_structure
            # Setup loops over non-dummy indices  --  all terms need these
            indices = self._get_expression_indices(expr, assign_to)
            # Setup loops over dummy indices  --  each term needs separate treatment
            dummies = get_contraction_structure(expr)
        else:
            indices = []
            dummies = {None: (expr,)}
        openloop, closeloop = self._get_loop_opening_ending(indices)

        # terms with no summations first
        if None in dummies:
            text = StrPrinter.doprint(self, Add(*dummies[None]))
        else:
            # If all terms have summations we must initialize array to Zero
            text = StrPrinter.doprint(self, 0)

        # skip redundant assignments (where lhs == rhs)
        lhs_printed = self._print(assign_to)
        lines = []
        if text != lhs_printed:
            lines.extend(openloop)
            if assign_to is not None:
                text = self._get_statement("%s = %s" % (lhs_printed, text))
            lines.append(text)
            lines.extend(closeloop)

        # then terms with summations
        for d in dummies:
            if isinstance(d, tuple):
                indices = self._sort_optimized(d, expr)
                openloop_d, closeloop_d = self._get_loop_opening_ending(
                    indices)

                for term in dummies[d]:
                    if term in dummies and not ([list(f.keys()) for f in dummies[term]]
                            == [[None] for f in dummies[term]]):
                        # If one factor in the term has it's own internal
                        # contractions, those must be computed first.
                        # (temporary variables?)
                        raise NotImplementedError(
                            "FIXME: no support for contractions in factor yet")
                    else:

                        # We need the lhs expression as an accumulator for
                        # the loops, i.e
                        #
                        # for (int d=0; d < dim; d++){
                        #    lhs[] = lhs[] + term[][d]
                        # }           ^.................. the accumulator
                        #
                        # We check if the expression already contains the
                        # lhs, and raise an exception if it does, as that
                        # syntax is currently undefined.  FIXME: What would be
                        # a good interpretation?
                        if assign_to is None:
                            raise AssignmentError(
                                "need assignment variable for loops")
                        if term.has(assign_to):
                            raise ValueError("FIXME: lhs present in rhs,\
                                this is undefined in CodePrinter")

                        lines.extend(openloop)
                        lines.extend(openloop_d)
                        text = "%s = %s" % (lhs_printed, StrPrinter.doprint(
                            self, assign_to + term))
                        lines.append(self._get_statement(text))
                        lines.extend(closeloop_d)
                        lines.extend(closeloop)

        return "\n".join(lines)

    def _get_expression_indices(self, expr, assign_to):
        from sympy.tensor import get_indices
        rinds, junk = get_indices(expr)
        linds, junk = get_indices(assign_to)

        # support broadcast of scalar
        if linds and not rinds:
            rinds = linds
        if rinds != linds:
            raise ValueError("lhs indices must match non-dummy"
                    " rhs indices in %s" % expr)

        return self._sort_optimized(rinds, assign_to)

    def _sort_optimized(self, indices, expr):

        from sympy.tensor.indexed import Indexed

        if not indices:
            return []

        # determine optimized loop order by giving a score to each index
        # the index with the highest score are put in the innermost loop.
        score_table = {}
        for i in indices:
            score_table[i] = 0

        arrays = expr.atoms(Indexed)
        for arr in arrays:
            for p, ind in enumerate(arr.indices):
                try:
                    score_table[ind] += self._rate_index_position(p)
                except KeyError:
                    pass

        return sorted(indices, key=lambda x: score_table[x])

    def _rate_index_position(self, p):
        """function to calculate score based on position among indices

        This method is used to sort loops in an optimized order, see
        CodePrinter._sort_optimized()
        """
        raise NotImplementedError("This function must be implemented by "
                                  "subclass of CodePrinter.")

    def _get_statement(self, codestring):
        """Formats a codestring with the proper line ending."""
        raise NotImplementedError("This function must be implemented by "
                                  "subclass of CodePrinter.")

    def _get_comment(self, text):
        """Formats a text string as a comment."""
        raise NotImplementedError("This function must be implemented by "
                                  "subclass of CodePrinter.")

    def _declare_number_const(self, name, value):
        """Declare a numeric constant at the top of a function"""
        raise NotImplementedError("This function must be implemented by "
                                  "subclass of CodePrinter.")

    def _format_code(self, lines):
        """Take in a list of lines of code, and format them accordingly.

        This may include indenting, wrapping long lines, etc..."""
        raise NotImplementedError("This function must be implemented by "
                                  "subclass of CodePrinter.")

    def _get_loop_opening_ending(self, indices):
        """Returns a tuple (open_lines, close_lines) containing lists
        of codelines"""
        raise NotImplementedError("This function must be implemented by "
                                  "subclass of CodePrinter.")

    def _print_Dummy(self, expr):
        if expr.name.startswith('Dummy_'):
            return '_' + expr.name
        else:
            return '%s_%d' % (expr.name, expr.dummy_index)

    def _print_Idx(self, expr):
        return self._print(expr.label)

    def _print_CodeBlock(self, expr):
        return '\n'.join([self._print(i) for i in expr.args])

    def _print_String(self, string):
        return str(string)

    def _print_QuotedString(self, arg):
        return '"%s"' % arg.text

    def _print_Comment(self, string):
        return self._get_comment(str(string))

    def _print_Assignment(self, expr):
        from sympy.codegen.ast import Assignment
        from sympy.functions.elementary.piecewise import Piecewise
        from sympy.matrices.expressions.matexpr import MatrixSymbol
        from sympy.tensor.indexed import IndexedBase
        lhs = expr.lhs
        rhs = expr.rhs
        # We special case assignments that take multiple lines
        if isinstance(expr.rhs, Piecewise):
            # Here we modify Piecewise so each expression is now
            # an Assignment, and then continue on the print.
            expressions = []
            conditions = []
            for (e, c) in rhs.args:
                expressions.append(Assignment(lhs, e))
                conditions.append(c)
            temp = Piecewise(*zip(expressions, conditions))
            return self._print(temp)
        elif isinstance(lhs, MatrixSymbol):
            # Here we form an Assignment for each element in the array,
            # printing each one.
            lines = []
            for (i, j) in self._traverse_matrix_indices(lhs):
                temp = Assignment(lhs[i, j], rhs[i, j])
                code0 = self._print(temp)
                lines.append(code0)
            return "\n".join(lines)
        elif self._settings.get("contract", False) and (lhs.has(IndexedBase) or
                rhs.has(IndexedBase)):
            # Here we check if there is looping to be done, and if so
            # print the required loops.
            return self._doprint_loops(rhs, lhs)
        else:
            lhs_code = self._print(lhs)
            rhs_code = self._print(rhs)
            return self._get_statement("%s = %s" % (lhs_code, rhs_code))

    def _print_AugmentedAssignment(self, expr):
        lhs_code = self._print(expr.lhs)
        rhs_code = self._print(expr.rhs)
        return self._get_statement("{} {} {}".format(
            *(self._print(arg) for arg in [lhs_code, expr.op, rhs_code])))

    def _print_FunctionCall(self, expr):
        return '%s(%s)' % (
            expr.name,
            ', '.join((self._print(arg) for arg in expr.function_args)))

    def _print_Variable(self, expr):
        return self._print(expr.symbol)

    def _print_Symbol(self, expr):
        name = super()._print_Symbol(expr)

        if name in self.reserved_words:
            if self._settings['error_on_reserved']:
                msg = ('This expression includes the symbol "{}" which is a '
                       'reserved keyword in this language.')
                raise ValueError(msg.format(name))
            return name + self._settings['reserved_word_suffix']
        else:
            return name

    def _can_print(self, name):
        """ Check if function ``name`` is either a known function or has its own
            printing method. Used to check if rewriting is possible."""
        return name in self.known_functions or getattr(self, '_print_{}'.format(name), False)

    def _print_Function(self, expr):
        if expr.func.__name__ in self.known_functions:
            cond_func = self.known_functions[expr.func.__name__]
            if isinstance(cond_func, str):
                return "%s(%s)" % (cond_func, self.stringify(expr.args, ", "))
            else:
                for cond, func in cond_func:
                    if cond(*expr.args):
                        break
                if func is not None:
                    try:
                        return func(*[self.parenthesize(item, 0) for item in expr.args])
                    except TypeError:
                        return "%s(%s)" % (func, self.stringify(expr.args, ", "))
        elif hasattr(expr, '_imp_') and isinstance(expr._imp_, Lambda):
            # inlined function
            return self._print(expr._imp_(*expr.args))
        elif expr.func.__name__ in self._rewriteable_functions:
            # Simple rewrite to supported function possible
            target_f, required_fs = self._rewriteable_functions[expr.func.__name__]
            if self._can_print(target_f) and all(self._can_print(f) for f in required_fs):
                return '(' + self._print(expr.rewrite(target_f)) + ')'

        if expr.is_Function and self._settings.get('allow_unknown_functions', False):
            return '%s(%s)' % (self._print(expr.func), ', '.join(map(self._print, expr.args)))
        else:
            return self._print_not_supported(expr)

    _print_Expr = _print_Function

    def _print_Derivative(self, expr):
        obj, *wrt_order_pairs = expr.args
        for func_arg in obj.args:
            if not func_arg.is_Symbol:
                raise ValueError("%s._print_Derivative(...) only supports functions with symbols as arguments." %
                                 self.__class__.__name__)
        meth_name = '_print_Derivative_%s' % obj.func.__name__
        pmeth = getattr(self, meth_name, None)
        if pmeth is None:
            if self._settings.get('strict', False):
                raise PrintMethodNotImplementedError(
                    f"Unsupported by {type(self)}: {type(expr)}" +
                    f"\nPrinter has no method: {meth_name}" +
                    "\nSet the printer option 'strict' to False in order to generate partially printed code."
                )
            return self._print_not_supported(expr)
        orders = dict(wrt_order_pairs)
        seq_orders = [orders[arg] for arg in obj.args]
        return pmeth(obj.args, seq_orders)

    # Don't inherit the str-printer method for Heaviside to the code printers
    _print_Heaviside = None

    def _print_NumberSymbol(self, expr):
        if self._settings.get("inline", False):
            return self._print(Float(expr.evalf(self._settings["precision"])))
        else:
            # A Number symbol that is not implemented here or with _printmethod
            # is registered and evaluated
            self._number_symbols.add((expr,
                Float(expr.evalf(self._settings["precision"]))))
            return str(expr)

    def _print_Catalan(self, expr):
        return self._print_NumberSymbol(expr)
    def _print_EulerGamma(self, expr):
        return self._print_NumberSymbol(expr)
    def _print_GoldenRatio(self, expr):
        return self._print_NumberSymbol(expr)
    def _print_TribonacciConstant(self, expr):
        return self._print_NumberSymbol(expr)
    def _print_Exp1(self, expr):
        return self._print_NumberSymbol(expr)
    def _print_Pi(self, expr):
        return self._print_NumberSymbol(expr)

    def _print_And(self, expr):
        PREC = precedence(expr)
        return (" %s " % self._operators['and']).join(self.parenthesize(a, PREC)
                for a in sorted(expr.args, key=default_sort_key))

    def _print_Or(self, expr):
        PREC = precedence(expr)
        return (" %s " % self._operators['or']).join(self.parenthesize(a, PREC)
                for a in sorted(expr.args, key=default_sort_key))

    def _print_Xor(self, expr):
        if self._operators.get('xor') is None:
            return self._print(expr.to_nnf())
        PREC = precedence(expr)
        return (" %s " % self._operators['xor']).join(self.parenthesize(a, PREC)
                for a in expr.args)

    def _print_Equivalent(self, expr):
        if self._operators.get('equivalent') is None:
            return self._print(expr.to_nnf())
        PREC = precedence(expr)
        return (" %s " % self._operators['equivalent']).join(self.parenthesize(a, PREC)
                for a in expr.args)

    def _print_Not(self, expr):
        PREC = precedence(expr)
        return self._operators['not'] + self.parenthesize(expr.args[0], PREC)

    def _print_BooleanFunction(self, expr):
        return self._print(expr.to_nnf())

    def _print_isnan(self, arg):
        return 'isnan(%s)' % self._print(*arg.args)

    def _print_isinf(self, arg):
        return 'isinf(%s)' % self._print(*arg.args)

    def _print_Mul(self, expr):

        prec = precedence(expr)

        c, e = expr.as_coeff_Mul()
        if c < 0:
            expr = _keep_coeff(-c, e)
            sign = "-"
        else:
            sign = ""

        a = []  # items in the numerator
        b = []  # items that are in the denominator (if any)

        pow_paren = []  # Will collect all pow with more than one base element and exp = -1

        if self.order not in ('old', 'none'):
            args = expr.as_ordered_factors()
        else:
            # use make_args in case expr was something like -x -> x
            args = Mul.make_args(expr)

        # Gather args for numerator/denominator
        for item in args:
            if item.is_commutative and item.is_Pow and item.exp.is_Rational and item.exp.is_negative:
                if item.exp != -1:
                    b.append(Pow(item.base, -item.exp, evaluate=False))
                else:
                    if len(item.args[0].args) != 1 and isinstance(item.base, Mul):   # To avoid situations like #14160
                        pow_paren.append(item)
                    b.append(Pow(item.base, -item.exp))
            else:
                a.append(item)

        a = a or [S.One]

        if len(a) == 1 and sign == "-":
            # Unary minus does not have a SymPy class, and hence there's no
            # precedence weight associated with it, Python's unary minus has
            # an operator precedence between multiplication and exponentiation,
            # so we use this to compute a weight.
            a_str = [self.parenthesize(a[0], 0.5*(PRECEDENCE["Pow"]+PRECEDENCE["Mul"]))]
        else:
            a_str = [self.parenthesize(x, prec) for x in a]
        b_str = [self.parenthesize(x, prec) for x in b]

        # To parenthesize Pow with exp = -1 and having more than one Symbol
        for item in pow_paren:
            if item.base in b:
                b_str[b.index(item.base)] = "(%s)" % b_str[b.index(item.base)]

        if not b:
            return sign + '*'.join(a_str)
        elif len(b) == 1:
            return sign + '*'.join(a_str) + "/" + b_str[0]
        else:
            return sign + '*'.join(a_str) + "/(%s)" % '*'.join(b_str)

    def _print_not_supported(self, expr):
        if self._settings.get('strict', False):
            raise PrintMethodNotImplementedError(
                f"Unsupported by {type(self)}: {type(expr)}" +
                "\nSet the printer option 'strict' to False in order to generate partially printed code."
            )
        try:
            self._not_supported.add(expr)
        except TypeError:
            # not hashable
            pass
        return self.emptyPrinter(expr)

    # The following can not be simply translated into C or Fortran
    _print_Basic = _print_not_supported
    _print_ComplexInfinity = _print_not_supported
    _print_ExprCondPair = _print_not_supported
    _print_GeometryEntity = _print_not_supported
    _print_Infinity = _print_not_supported
    _print_Integral = _print_not_supported
    _print_Interval = _print_not_supported
    _print_AccumulationBounds = _print_not_supported
    _print_Limit = _print_not_supported
    _print_MatrixBase = _print_not_supported
    _print_DeferredVector = _print_not_supported
    _print_NaN = _print_not_supported
    _print_NegativeInfinity = _print_not_supported
    _print_Order = _print_not_supported
    _print_RootOf = _print_not_supported
    _print_RootsOf = _print_not_supported
    _print_RootSum = _print_not_supported
    _print_Uniform = _print_not_supported
    _print_Unit = _print_not_supported
    _print_Wild = _print_not_supported
    _print_WildFunction = _print_not_supported
    _print_Relational = _print_not_supported


# Code printer functions. These are included in this file so that they can be
# imported in the top-level __init__.py without importing the sympy.codegen
# module.

def ccode(expr, assign_to=None, standard='c99', **settings):
    """Converts an expr to a string of c code

    Parameters
    ==========

    expr : Expr
        A SymPy expression to be converted.
    assign_to : optional
        When given, the argument is used as the name of the variable to which
        the expression is assigned. Can be a string, ``Symbol``,
        ``MatrixSymbol``, or ``Indexed`` type. This is helpful in case of
        line-wrapping, or for expressions that generate multi-line statements.
    standard : str, optional
        String specifying the standard. If your compiler supports a more modern
        standard you may set this to 'c99' to allow the printer to use more math
        functions. [default='c89'].
    precision : integer, optional
        The precision for numbers such as pi [default=17].
    user_functions : dict, optional
        A dictionary where the keys are string representations of either
        ``FunctionClass`` or ``UndefinedFunction`` instances and the values
        are their desired C string representations. Alternatively, the
        dictionary value can be a list of tuples i.e. [(argument_test,
        cfunction_string)] or [(argument_test, cfunction_formater)]. See below
        for examples.
    dereference : iterable, optional
        An iterable of symbols that should be dereferenced in the printed code
        expression. These would be values passed by address to the function.
        For example, if ``dereference=[a]``, the resulting code would print
        ``(*a)`` instead of ``a``.
    human : bool, optional
        If True, the result is a single string that may contain some constant
        declarations for the number symbols. If False, the same information is
        returned in a tuple of (symbols_to_declare, not_supported_functions,
        code_text). [default=True].
    contract: bool, optional
        If True, ``Indexed`` instances are assumed to obey tensor contraction
        rules and the corresponding nested loops over indices are generated.
        Setting contract=False will not generate loops, instead the user is
        responsible to provide values for the indices in the code.
        [default=True].

    Examples
    ========

    >>> from sympy import ccode, symbols, Rational, sin, ceiling, Abs, Function
    >>> x, tau = symbols("x, tau")
    >>> expr = (2*tau)**Rational(7, 2)
    >>> ccode(expr)
    '8*M_SQRT2*pow(tau, 7.0/2.0)'
    >>> ccode(expr, math_macros={})
    '8*sqrt(2)*pow(tau, 7.0/2.0)'
    >>> ccode(sin(x), assign_to="s")
    's = sin(x);'
    >>> from sympy.codegen.ast import real, float80
    >>> ccode(expr, type_aliases={real: float80})
    '8*M_SQRT2l*powl(tau, 7.0L/2.0L)'

    Simple custom printing can be defined for certain types by passing a
    dictionary of {"type" : "function"} to the ``user_functions`` kwarg.
    Alternatively, the dictionary value can be a list of tuples i.e.
    [(argument_test, cfunction_string)].

    >>> custom_functions = {
    ...   "ceiling": "CEIL",
    ...   "Abs": [(lambda x: not x.is_integer, "fabs"),
    ...           (lambda x: x.is_integer, "ABS")],
    ...   "func": "f"
    ... }
    >>> func = Function('func')
    >>> ccode(func(Abs(x) + ceiling(x)), standard='C89', user_functions=custom_functions)
    'f(fabs(x) + CEIL(x))'

    or if the C-function takes a subset of the original arguments:

    >>> ccode(2**x + 3**x, standard='C99', user_functions={'Pow': [
    ...   (lambda b, e: b == 2, lambda b, e: 'exp2(%s)' % e),
    ...   (lambda b, e: b != 2, 'pow')]})
    'exp2(x) + pow(3, x)'

    ``Piecewise`` expressions are converted into conditionals. If an
    ``assign_to`` variable is provided an if statement is created, otherwise
    the ternary operator is used. Note that if the ``Piecewise`` lacks a
    default term, represented by ``(expr, True)`` then an error will be thrown.
    This is to prevent generating an expression that may not evaluate to
    anything.

    >>> from sympy import Piecewise
    >>> expr = Piecewise((x + 1, x > 0), (x, True))
    >>> print(ccode(expr, tau, standard='C89'))
    if (x > 0) {
    tau = x + 1;
    }
    else {
    tau = x;
    }

    Support for loops is provided through ``Indexed`` types. With
    ``contract=True`` these expressions will be turned into loops, whereas
    ``contract=False`` will just print the assignment expression that should be
    looped over:

    >>> from sympy import Eq, IndexedBase, Idx
    >>> len_y = 5
    >>> y = IndexedBase('y', shape=(len_y,))
    >>> t = IndexedBase('t', shape=(len_y,))
    >>> Dy = IndexedBase('Dy', shape=(len_y-1,))
    >>> i = Idx('i', len_y-1)
    >>> e=Eq(Dy[i], (y[i+1]-y[i])/(t[i+1]-t[i]))
    >>> ccode(e.rhs, assign_to=e.lhs, contract=False, standard='C89')
    'Dy[i] = (y[i + 1] - y[i])/(t[i + 1] - t[i]);'

    Matrices are also supported, but a ``MatrixSymbol`` of the same dimensions
    must be provided to ``assign_to``. Note that any expression that can be
    generated normally can also exist inside a Matrix:

    >>> from sympy import Matrix, MatrixSymbol
    >>> mat = Matrix([x**2, Piecewise((x + 1, x > 0), (x, True)), sin(x)])
    >>> A = MatrixSymbol('A', 3, 1)
    >>> print(ccode(mat, A, standard='C89'))
    A[0] = pow(x, 2);
    if (x > 0) {
       A[1] = x + 1;
    }
    else {
       A[1] = x;
    }
    A[2] = sin(x);
    """
    from sympy.printing.c import c_code_printers
    return c_code_printers[standard.lower()](settings).doprint(expr, assign_to)

def print_ccode(expr, **settings):
    """Prints C representation of the given expression."""
    print(ccode(expr, **settings))

def fcode(expr, assign_to=None, **settings):
    """Converts an expr to a string of fortran code

    Parameters
    ==========

    expr : Expr
        A SymPy expression to be converted.
    assign_to : optional
        When given, the argument is used as the name of the variable to which
        the expression is assigned. Can be a string, ``Symbol``,
        ``MatrixSymbol``, or ``Indexed`` type. This is helpful in case of
        line-wrapping, or for expressions that generate multi-line statements.
    precision : integer, optional
        DEPRECATED. Use type_mappings instead. The precision for numbers such
        as pi [default=17].
    user_functions : dict, optional
        A dictionary where keys are ``FunctionClass`` instances and values are
        their string representations. Alternatively, the dictionary value can
        be a list of tuples i.e. [(argument_test, cfunction_string)]. See below
        for examples.
    human : bool, optional
        If True, the result is a single string that may contain some constant
        declarations for the number symbols. If False, the same information is
        returned in a tuple of (symbols_to_declare, not_supported_functions,
        code_text). [default=True].
    contract: bool, optional
        If True, ``Indexed`` instances are assumed to obey tensor contraction
        rules and the corresponding nested loops over indices are generated.
        Setting contract=False will not generate loops, instead the user is
        responsible to provide values for the indices in the code.
        [default=True].
    source_format : optional
        The source format can be either 'fixed' or 'free'. [default='fixed']
    standard : integer, optional
        The Fortran standard to be followed. This is specified as an integer.
        Acceptable standards are 66, 77, 90, 95, 2003, and 2008. Default is 77.
        Note that currently the only distinction internally is between
        standards before 95, and those 95 and after. This may change later as
        more features are added.
    name_mangling : bool, optional
        If True, then the variables that would become identical in
        case-insensitive Fortran are mangled by appending different number
        of ``_`` at the end. If False, SymPy Will not interfere with naming of
        variables. [default=True]

    Examples
    ========

    >>> from sympy import fcode, symbols, Rational, sin, ceiling, floor
    >>> x, tau = symbols("x, tau")
    >>> fcode((2*tau)**Rational(7, 2))
    '      8*sqrt(2.0d0)*tau**(7.0d0/2.0d0)'
    >>> fcode(sin(x), assign_to="s")
    '      s = sin(x)'

    Custom printing can be defined for certain types by passing a dictionary of
    "type" : "function" to the ``user_functions`` kwarg. Alternatively, the
    dictionary value can be a list of tuples i.e. [(argument_test,
    cfunction_string)].

    >>> custom_functions = {
    ...   "ceiling": "CEIL",
    ...   "floor": [(lambda x: not x.is_integer, "FLOOR1"),
    ...             (lambda x: x.is_integer, "FLOOR2")]
    ... }
    >>> fcode(floor(x) + ceiling(x), user_functions=custom_functions)
    '      CEIL(x) + FLOOR1(x)'

    ``Piecewise`` expressions are converted into conditionals. If an
    ``assign_to`` variable is provided an if statement is created, otherwise
    the ternary operator is used. Note that if the ``Piecewise`` lacks a
    default term, represented by ``(expr, True)`` then an error will be thrown.
    This is to prevent generating an expression that may not evaluate to
    anything.

    >>> from sympy import Piecewise
    >>> expr = Piecewise((x + 1, x > 0), (x, True))
    >>> print(fcode(expr, tau))
          if (x > 0) then
             tau = x + 1
          else
             tau = x
          end if

    Support for loops is provided through ``Indexed`` types. With
    ``contract=True`` these expressions will be turned into loops, whereas
    ``contract=False`` will just print the assignment expression that should be
    looped over:

    >>> from sympy import Eq, IndexedBase, Idx
    >>> len_y = 5
    >>> y = IndexedBase('y', shape=(len_y,))
    >>> t = IndexedBase('t', shape=(len_y,))
    >>> Dy = IndexedBase('Dy', shape=(len_y-1,))
    >>> i = Idx('i', len_y-1)
    >>> e=Eq(Dy[i], (y[i+1]-y[i])/(t[i+1]-t[i]))
    >>> fcode(e.rhs, assign_to=e.lhs, contract=False)
    '      Dy(i) = (y(i + 1) - y(i))/(t(i + 1) - t(i))'

    Matrices are also supported, but a ``MatrixSymbol`` of the same dimensions
    must be provided to ``assign_to``. Note that any expression that can be
    generated normally can also exist inside a Matrix:

    >>> from sympy import Matrix, MatrixSymbol
    >>> mat = Matrix([x**2, Piecewise((x + 1, x > 0), (x, True)), sin(x)])
    >>> A = MatrixSymbol('A', 3, 1)
    >>> print(fcode(mat, A))
          A(1, 1) = x**2
             if (x > 0) then
          A(2, 1) = x + 1
             else
          A(2, 1) = x
             end if
          A(3, 1) = sin(x)
    """
    from sympy.printing.fortran import FCodePrinter
    return FCodePrinter(settings).doprint(expr, assign_to)


def print_fcode(expr, **settings):
    """Prints the Fortran representation of the given expression.

       See fcode for the meaning of the optional arguments.
    """
    print(fcode(expr, **settings))

def cxxcode(expr, assign_to=None, standard='c++11', **settings):
    """ C++ equivalent of :func:`~.ccode`. """
    from sympy.printing.cxx import cxx_code_printers
    return cxx_code_printers[standard.lower()](settings).doprint(expr, assign_to)


def rust_code(expr, assign_to=None, **settings):
    """Converts an expr to a string of Rust code

    Parameters
    ==========

    expr : Expr
        A SymPy expression to be converted.
    assign_to : optional
        When given, the argument is used as the name of the variable to which
        the expression is assigned. Can be a string, ``Symbol``,
        ``MatrixSymbol``, or ``Indexed`` type. This is helpful in case of
        line-wrapping, or for expressions that generate multi-line statements.
    precision : integer, optional
        The precision for numbers such as pi [default=15].
    user_functions : dict, optional
        A dictionary where the keys are string representations of either
        ``FunctionClass`` or ``UndefinedFunction`` instances and the values
        are their desired C string representations. Alternatively, the
        dictionary value can be a list of tuples i.e. [(argument_test,
        cfunction_string)].  See below for examples.
    dereference : iterable, optional
        An iterable of symbols that should be dereferenced in the printed code
        expression. These would be values passed by address to the function.
        For example, if ``dereference=[a]``, the resulting code would print
        ``(*a)`` instead of ``a``.
    human : bool, optional
        If True, the result is a single string that may contain some constant
        declarations for the number symbols. If False, the same information is
        returned in a tuple of (symbols_to_declare, not_supported_functions,
        code_text). [default=True].
    contract: bool, optional
        If True, ``Indexed`` instances are assumed to obey tensor contraction
        rules and the corresponding nested loops over indices are generated.
        Setting contract=False will not generate loops, instead the user is
        responsible to provide values for the indices in the code.
        [default=True].

    Examples
    ========

    >>> from sympy import rust_code, symbols, Rational, sin, ceiling, Abs, Function
    >>> x, tau = symbols("x, tau")
    >>> rust_code((2*tau)**Rational(7, 2))
    '8.0*1.4142135623731*tau.powf(7_f64/2.0)'
    >>> rust_code(sin(x), assign_to="s")
    's = x.sin();'

    Simple custom printing can be defined for certain types by passing a
    dictionary of {"type" : "function"} to the ``user_functions`` kwarg.
    Alternatively, the dictionary value can be a list of tuples i.e.
    [(argument_test, cfunction_string)].

    >>> custom_functions = {
    ...   "ceiling": "CEIL",
    ...   "Abs": [(lambda x: not x.is_integer, "fabs", 4),
    ...           (lambda x: x.is_integer, "ABS", 4)],
    ...   "func": "f"
    ... }
    >>> func = Function('func')
    >>> rust_code(func(Abs(x) + ceiling(x)), user_functions=custom_functions)
    '(fabs(x) + x.ceil()).f()'

    ``Piecewise`` expressions are converted into conditionals. If an
    ``assign_to`` variable is provided an if statement is created, otherwise
    the ternary operator is used. Note that if the ``Piecewise`` lacks a
    default term, represented by ``(expr, True)`` then an error will be thrown.
    This is to prevent generating an expression that may not evaluate to
    anything.

    >>> from sympy import Piecewise
    >>> expr = Piecewise((x + 1, x > 0), (x, True))
    >>> print(rust_code(expr, tau))
    tau = if (x > 0.0) {
        x + 1
    } else {
        x
    };

    Support for loops is provided through ``Indexed`` types. With
    ``contract=True`` these expressions will be turned into loops, whereas
    ``contract=False`` will just print the assignment expression that should be
    looped over:

    >>> from sympy import Eq, IndexedBase, Idx
    >>> len_y = 5
    >>> y = IndexedBase('y', shape=(len_y,))
    >>> t = IndexedBase('t', shape=(len_y,))
    >>> Dy = IndexedBase('Dy', shape=(len_y-1,))
    >>> i = Idx('i', len_y-1)
    >>> e=Eq(Dy[i], (y[i+1]-y[i])/(t[i+1]-t[i]))
    >>> rust_code(e.rhs, assign_to=e.lhs, contract=False)
    'Dy[i] = (y[i + 1] - y[i])/(t[i + 1] - t[i]);'

    Matrices are also supported, but a ``MatrixSymbol`` of the same dimensions
    must be provided to ``assign_to``. Note that any expression that can be
    generated normally can also exist inside a Matrix:

    >>> from sympy import Matrix, MatrixSymbol
    >>> mat = Matrix([x**2, Piecewise((x + 1, x > 0), (x, True)), sin(x)])
    >>> A = MatrixSymbol('A', 3, 1)
    >>> print(rust_code(mat, A))
    A = [x.powi(2), if (x > 0.0) {
        x + 1
    } else {
        x
    }, x.sin()];
    """
    from sympy.printing.rust import RustCodePrinter
    printer = RustCodePrinter(settings)
    expr = printer._rewrite_known_functions(expr)
    if isinstance(expr, Expr):
        for src_func, dst_func in printer.function_overrides.values():
            expr = expr.replace(src_func, dst_func)
    return printer.doprint(expr, assign_to)


def print_rust_code(expr, **settings):
    """Prints Rust representation of the given expression."""
    print(rust_code(expr, **settings))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\requests\models.py ===
"""
requests.models
~~~~~~~~~~~~~~~

This module contains the primary objects that power Requests.
"""

import datetime

# Import encoding now, to avoid implicit import later.
# Implicit import within threads may cause LookupError when standard library is in a ZIP,
# such as in Embedded Python. See https://github.com/psf/requests/issues/3578.
import encodings.idna  # noqa: F401
from io import UnsupportedOperation

from pip._vendor.urllib3.exceptions import (
    DecodeError,
    LocationParseError,
    ProtocolError,
    ReadTimeoutError,
    SSLError,
)
from pip._vendor.urllib3.fields import RequestField
from pip._vendor.urllib3.filepost import encode_multipart_formdata
from pip._vendor.urllib3.util import parse_url

from ._internal_utils import to_native_string, unicode_is_ascii
from .auth import HTTPBasicAuth
from .compat import (
    Callable,
    JSONDecodeError,
    Mapping,
    basestring,
    builtin_str,
    chardet,
    cookielib,
)
from .compat import json as complexjson
from .compat import urlencode, urlsplit, urlunparse
from .cookies import _copy_cookie_jar, cookiejar_from_dict, get_cookie_header
from .exceptions import (
    ChunkedEncodingError,
    ConnectionError,
    ContentDecodingError,
    HTTPError,
    InvalidJSONError,
    InvalidURL,
)
from .exceptions import JSONDecodeError as RequestsJSONDecodeError
from .exceptions import MissingSchema
from .exceptions import SSLError as RequestsSSLError
from .exceptions import StreamConsumedError
from .hooks import default_hooks
from .status_codes import codes
from .structures import CaseInsensitiveDict
from .utils import (
    check_header_validity,
    get_auth_from_url,
    guess_filename,
    guess_json_utf,
    iter_slices,
    parse_header_links,
    requote_uri,
    stream_decode_response_unicode,
    super_len,
    to_key_val_list,
)

#: The set of HTTP status codes that indicate an automatically
#: processable redirect.
REDIRECT_STATI = (
    codes.moved,  # 301
    codes.found,  # 302
    codes.other,  # 303
    codes.temporary_redirect,  # 307
    codes.permanent_redirect,  # 308
)

DEFAULT_REDIRECT_LIMIT = 30
CONTENT_CHUNK_SIZE = 10 * 1024
ITER_CHUNK_SIZE = 512


class RequestEncodingMixin:
    @property
    def path_url(self):
        """Build the path URL to use."""

        url = []

        p = urlsplit(self.url)

        path = p.path
        if not path:
            path = "/"

        url.append(path)

        query = p.query
        if query:
            url.append("?")
            url.append(query)

        return "".join(url)

    @staticmethod
    def _encode_params(data):
        """Encode parameters in a piece of data.

        Will successfully encode parameters when passed as a dict or a list of
        2-tuples. Order is retained if data is a list of 2-tuples but arbitrary
        if parameters are supplied as a dict.
        """

        if isinstance(data, (str, bytes)):
            return data
        elif hasattr(data, "read"):
            return data
        elif hasattr(data, "__iter__"):
            result = []
            for k, vs in to_key_val_list(data):
                if isinstance(vs, basestring) or not hasattr(vs, "__iter__"):
                    vs = [vs]
                for v in vs:
                    if v is not None:
                        result.append(
                            (
                                k.encode("utf-8") if isinstance(k, str) else k,
                                v.encode("utf-8") if isinstance(v, str) else v,
                            )
                        )
            return urlencode(result, doseq=True)
        else:
            return data

    @staticmethod
    def _encode_files(files, data):
        """Build the body for a multipart/form-data request.

        Will successfully encode files when passed as a dict or a list of
        tuples. Order is retained if data is a list of tuples but arbitrary
        if parameters are supplied as a dict.
        The tuples may be 2-tuples (filename, fileobj), 3-tuples (filename, fileobj, contentype)
        or 4-tuples (filename, fileobj, contentype, custom_headers).
        """
        if not files:
            raise ValueError("Files must be provided.")
        elif isinstance(data, basestring):
            raise ValueError("Data must not be a string.")

        new_fields = []
        fields = to_key_val_list(data or {})
        files = to_key_val_list(files or {})

        for field, val in fields:
            if isinstance(val, basestring) or not hasattr(val, "__iter__"):
                val = [val]
            for v in val:
                if v is not None:
                    # Don't call str() on bytestrings: in Py3 it all goes wrong.
                    if not isinstance(v, bytes):
                        v = str(v)

                    new_fields.append(
                        (
                            field.decode("utf-8")
                            if isinstance(field, bytes)
                            else field,
                            v.encode("utf-8") if isinstance(v, str) else v,
                        )
                    )

        for k, v in files:
            # support for explicit filename
            ft = None
            fh = None
            if isinstance(v, (tuple, list)):
                if len(v) == 2:
                    fn, fp = v
                elif len(v) == 3:
                    fn, fp, ft = v
                else:
                    fn, fp, ft, fh = v
            else:
                fn = guess_filename(v) or k
                fp = v

            if isinstance(fp, (str, bytes, bytearray)):
                fdata = fp
            elif hasattr(fp, "read"):
                fdata = fp.read()
            elif fp is None:
                continue
            else:
                fdata = fp

            rf = RequestField(name=k, data=fdata, filename=fn, headers=fh)
            rf.make_multipart(content_type=ft)
            new_fields.append(rf)

        body, content_type = encode_multipart_formdata(new_fields)

        return body, content_type


class RequestHooksMixin:
    def register_hook(self, event, hook):
        """Properly register a hook."""

        if event not in self.hooks:
            raise ValueError(f'Unsupported event specified, with event name "{event}"')

        if isinstance(hook, Callable):
            self.hooks[event].append(hook)
        elif hasattr(hook, "__iter__"):
            self.hooks[event].extend(h for h in hook if isinstance(h, Callable))

    def deregister_hook(self, event, hook):
        """Deregister a previously registered hook.
        Returns True if the hook existed, False if not.
        """

        try:
            self.hooks[event].remove(hook)
            return True
        except ValueError:
            return False


class Request(RequestHooksMixin):
    """A user-created :class:`Request <Request>` object.

    Used to prepare a :class:`PreparedRequest <PreparedRequest>`, which is sent to the server.

    :param method: HTTP method to use.
    :param url: URL to send.
    :param headers: dictionary of headers to send.
    :param files: dictionary of {filename: fileobject} files to multipart upload.
    :param data: the body to attach to the request. If a dictionary or
        list of tuples ``[(key, value)]`` is provided, form-encoding will
        take place.
    :param json: json for the body to attach to the request (if files or data is not specified).
    :param params: URL parameters to append to the URL. If a dictionary or
        list of tuples ``[(key, value)]`` is provided, form-encoding will
        take place.
    :param auth: Auth handler or (user, pass) tuple.
    :param cookies: dictionary or CookieJar of cookies to attach to this request.
    :param hooks: dictionary of callback hooks, for internal usage.

    Usage::

      >>> import requests
      >>> req = requests.Request('GET', 'https://httpbin.org/get')
      >>> req.prepare()
      <PreparedRequest [GET]>
    """

    def __init__(
        self,
        method=None,
        url=None,
        headers=None,
        files=None,
        data=None,
        params=None,
        auth=None,
        cookies=None,
        hooks=None,
        json=None,
    ):
        # Default empty dicts for dict params.
        data = [] if data is None else data
        files = [] if files is None else files
        headers = {} if headers is None else headers
        params = {} if params is None else params
        hooks = {} if hooks is None else hooks

        self.hooks = default_hooks()
        for k, v in list(hooks.items()):
            self.register_hook(event=k, hook=v)

        self.method = method
        self.url = url
        self.headers = headers
        self.files = files
        self.data = data
        self.json = json
        self.params = params
        self.auth = auth
        self.cookies = cookies

    def __repr__(self):
        return f"<Request [{self.method}]>"

    def prepare(self):
        """Constructs a :class:`PreparedRequest <PreparedRequest>` for transmission and returns it."""
        p = PreparedRequest()
        p.prepare(
            method=self.method,
            url=self.url,
            headers=self.headers,
            files=self.files,
            data=self.data,
            json=self.json,
            params=self.params,
            auth=self.auth,
            cookies=self.cookies,
            hooks=self.hooks,
        )
        return p


class PreparedRequest(RequestEncodingMixin, RequestHooksMixin):
    """The fully mutable :class:`PreparedRequest <PreparedRequest>` object,
    containing the exact bytes that will be sent to the server.

    Instances are generated from a :class:`Request <Request>` object, and
    should not be instantiated manually; doing so may produce undesirable
    effects.

    Usage::

      >>> import requests
      >>> req = requests.Request('GET', 'https://httpbin.org/get')
      >>> r = req.prepare()
      >>> r
      <PreparedRequest [GET]>

      >>> s = requests.Session()
      >>> s.send(r)
      <Response [200]>
    """

    def __init__(self):
        #: HTTP verb to send to the server.
        self.method = None
        #: HTTP URL to send the request to.
        self.url = None
        #: dictionary of HTTP headers.
        self.headers = None
        # The `CookieJar` used to create the Cookie header will be stored here
        # after prepare_cookies is called
        self._cookies = None
        #: request body to send to the server.
        self.body = None
        #: dictionary of callback hooks, for internal usage.
        self.hooks = default_hooks()
        #: integer denoting starting position of a readable file-like body.
        self._body_position = None

    def prepare(
        self,
        method=None,
        url=None,
        headers=None,
        files=None,
        data=None,
        params=None,
        auth=None,
        cookies=None,
        hooks=None,
        json=None,
    ):
        """Prepares the entire request with the given parameters."""

        self.prepare_method(method)
        self.prepare_url(url, params)
        self.prepare_headers(headers)
        self.prepare_cookies(cookies)
        self.prepare_body(data, files, json)
        self.prepare_auth(auth, url)

        # Note that prepare_auth must be last to enable authentication schemes
        # such as OAuth to work on a fully prepared request.

        # This MUST go after prepare_auth. Authenticators could add a hook
        self.prepare_hooks(hooks)

    def __repr__(self):
        return f"<PreparedRequest [{self.method}]>"

    def copy(self):
        p = PreparedRequest()
        p.method = self.method
        p.url = self.url
        p.headers = self.headers.copy() if self.headers is not None else None
        p._cookies = _copy_cookie_jar(self._cookies)
        p.body = self.body
        p.hooks = self.hooks
        p._body_position = self._body_position
        return p

    def prepare_method(self, method):
        """Prepares the given HTTP method."""
        self.method = method
        if self.method is not None:
            self.method = to_native_string(self.method.upper())

    @staticmethod
    def _get_idna_encoded_host(host):
        from pip._vendor import idna

        try:
            host = idna.encode(host, uts46=True).decode("utf-8")
        except idna.IDNAError:
            raise UnicodeError
        return host

    def prepare_url(self, url, params):
        """Prepares the given HTTP URL."""
        #: Accept objects that have string representations.
        #: We're unable to blindly call unicode/str functions
        #: as this will include the bytestring indicator (b'')
        #: on python 3.x.
        #: https://github.com/psf/requests/pull/2238
        if isinstance(url, bytes):
            url = url.decode("utf8")
        else:
            url = str(url)

        # Remove leading whitespaces from url
        url = url.lstrip()

        # Don't do any URL preparation for non-HTTP schemes like `mailto`,
        # `data` etc to work around exceptions from `url_parse`, which
        # handles RFC 3986 only.
        if ":" in url and not url.lower().startswith("http"):
            self.url = url
            return

        # Support for unicode domain names and paths.
        try:
            scheme, auth, host, port, path, query, fragment = parse_url(url)
        except LocationParseError as e:
            raise InvalidURL(*e.args)

        if not scheme:
            raise MissingSchema(
                f"Invalid URL {url!r}: No scheme supplied. "
                f"Perhaps you meant https://{url}?"
            )

        if not host:
            raise InvalidURL(f"Invalid URL {url!r}: No host supplied")

        # In general, we want to try IDNA encoding the hostname if the string contains
        # non-ASCII characters. This allows users to automatically get the correct IDNA
        # behaviour. For strings containing only ASCII characters, we need to also verify
        # it doesn't start with a wildcard (*), before allowing the unencoded hostname.
        if not unicode_is_ascii(host):
            try:
                host = self._get_idna_encoded_host(host)
            except UnicodeError:
                raise InvalidURL("URL has an invalid label.")
        elif host.startswith(("*", ".")):
            raise InvalidURL("URL has an invalid label.")

        # Carefully reconstruct the network location
        netloc = auth or ""
        if netloc:
            netloc += "@"
        netloc += host
        if port:
            netloc += f":{port}"

        # Bare domains aren't valid URLs.
        if not path:
            path = "/"

        if isinstance(params, (str, bytes)):
            params = to_native_string(params)

        enc_params = self._encode_params(params)
        if enc_params:
            if query:
                query = f"{query}&{enc_params}"
            else:
                query = enc_params

        url = requote_uri(urlunparse([scheme, netloc, path, None, query, fragment]))
        self.url = url

    def prepare_headers(self, headers):
        """Prepares the given HTTP headers."""

        self.headers = CaseInsensitiveDict()
        if headers:
            for header in headers.items():
                # Raise exception on invalid header value.
                check_header_validity(header)
                name, value = header
                self.headers[to_native_string(name)] = value

    def prepare_body(self, data, files, json=None):
        """Prepares the given HTTP body data."""

        # Check if file, fo, generator, iterator.
        # If not, run through normal process.

        # Nottin' on you.
        body = None
        content_type = None

        if not data and json is not None:
            # urllib3 requires a bytes-like body. Python 2's json.dumps
            # provides this natively, but Python 3 gives a Unicode string.
            content_type = "application/json"

            try:
                body = complexjson.dumps(json, allow_nan=False)
            except ValueError as ve:
                raise InvalidJSONError(ve, request=self)

            if not isinstance(body, bytes):
                body = body.encode("utf-8")

        is_stream = all(
            [
                hasattr(data, "__iter__"),
                not isinstance(data, (basestring, list, tuple, Mapping)),
            ]
        )

        if is_stream:
            try:
                length = super_len(data)
            except (TypeError, AttributeError, UnsupportedOperation):
                length = None

            body = data

            if getattr(body, "tell", None) is not None:
                # Record the current file position before reading.
                # This will allow us to rewind a file in the event
                # of a redirect.
                try:
                    self._body_position = body.tell()
                except OSError:
                    # This differentiates from None, allowing us to catch
                    # a failed `tell()` later when trying to rewind the body
                    self._body_position = object()

            if files:
                raise NotImplementedError(
                    "Streamed bodies and files are mutually exclusive."
                )

            if length:
                self.headers["Content-Length"] = builtin_str(length)
            else:
                self.headers["Transfer-Encoding"] = "chunked"
        else:
            # Multi-part file uploads.
            if files:
                (body, content_type) = self._encode_files(files, data)
            else:
                if data:
                    body = self._encode_params(data)
                    if isinstance(data, basestring) or hasattr(data, "read"):
                        content_type = None
                    else:
                        content_type = "application/x-www-form-urlencoded"

            self.prepare_content_length(body)

            # Add content-type if it wasn't explicitly provided.
            if content_type and ("content-type" not in self.headers):
                self.headers["Content-Type"] = content_type

        self.body = body

    def prepare_content_length(self, body):
        """Prepare Content-Length header based on request method and body"""
        if body is not None:
            length = super_len(body)
            if length:
                # If length exists, set it. Otherwise, we fallback
                # to Transfer-Encoding: chunked.
                self.headers["Content-Length"] = builtin_str(length)
        elif (
            self.method not in ("GET", "HEAD")
            and self.headers.get("Content-Length") is None
        ):
            # Set Content-Length to 0 for methods that can have a body
            # but don't provide one. (i.e. not GET or HEAD)
            self.headers["Content-Length"] = "0"

    def prepare_auth(self, auth, url=""):
        """Prepares the given HTTP auth data."""

        # If no Auth is explicitly provided, extract it from the URL first.
        if auth is None:
            url_auth = get_auth_from_url(self.url)
            auth = url_auth if any(url_auth) else None

        if auth:
            if isinstance(auth, tuple) and len(auth) == 2:
                # special-case basic HTTP auth
                auth = HTTPBasicAuth(*auth)

            # Allow auth to make its changes.
            r = auth(self)

            # Update self to reflect the auth changes.
            self.__dict__.update(r.__dict__)

            # Recompute Content-Length
            self.prepare_content_length(self.body)

    def prepare_cookies(self, cookies):
        """Prepares the given HTTP cookie data.

        This function eventually generates a ``Cookie`` header from the
        given cookies using cookielib. Due to cookielib's design, the header
        will not be regenerated if it already exists, meaning this function
        can only be called once for the life of the
        :class:`PreparedRequest <PreparedRequest>` object. Any subsequent calls
        to ``prepare_cookies`` will have no actual effect, unless the "Cookie"
        header is removed beforehand.
        """
        if isinstance(cookies, cookielib.CookieJar):
            self._cookies = cookies
        else:
            self._cookies = cookiejar_from_dict(cookies)

        cookie_header = get_cookie_header(self._cookies, self)
        if cookie_header is not None:
            self.headers["Cookie"] = cookie_header

    def prepare_hooks(self, hooks):
        """Prepares the given hooks."""
        # hooks can be passed as None to the prepare method and to this
        # method. To prevent iterating over None, simply use an empty list
        # if hooks is False-y
        hooks = hooks or []
        for event in hooks:
            self.register_hook(event, hooks[event])


class Response:
    """The :class:`Response <Response>` object, which contains a
    server's response to an HTTP request.
    """

    __attrs__ = [
        "_content",
        "status_code",
        "headers",
        "url",
        "history",
        "encoding",
        "reason",
        "cookies",
        "elapsed",
        "request",
    ]

    def __init__(self):
        self._content = False
        self._content_consumed = False
        self._next = None

        #: Integer Code of responded HTTP Status, e.g. 404 or 200.
        self.status_code = None

        #: Case-insensitive Dictionary of Response Headers.
        #: For example, ``headers['content-encoding']`` will return the
        #: value of a ``'Content-Encoding'`` response header.
        self.headers = CaseInsensitiveDict()

        #: File-like object representation of response (for advanced usage).
        #: Use of ``raw`` requires that ``stream=True`` be set on the request.
        #: This requirement does not apply for use internally to Requests.
        self.raw = None

        #: Final URL location of Response.
        self.url = None

        #: Encoding to decode with when accessing r.text.
        self.encoding = None

        #: A list of :class:`Response <Response>` objects from
        #: the history of the Request. Any redirect responses will end
        #: up here. The list is sorted from the oldest to the most recent request.
        self.history = []

        #: Textual reason of responded HTTP Status, e.g. "Not Found" or "OK".
        self.reason = None

        #: A CookieJar of Cookies the server sent back.
        self.cookies = cookiejar_from_dict({})

        #: The amount of time elapsed between sending the request
        #: and the arrival of the response (as a timedelta).
        #: This property specifically measures the time taken between sending
        #: the first byte of the request and finishing parsing the headers. It
        #: is therefore unaffected by consuming the response content or the
        #: value of the ``stream`` keyword argument.
        self.elapsed = datetime.timedelta(0)

        #: The :class:`PreparedRequest <PreparedRequest>` object to which this
        #: is a response.
        self.request = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __getstate__(self):
        # Consume everything; accessing the content attribute makes
        # sure the content has been fully read.
        if not self._content_consumed:
            self.content

        return {attr: getattr(self, attr, None) for attr in self.__attrs__}

    def __setstate__(self, state):
        for name, value in state.items():
            setattr(self, name, value)

        # pickled objects do not have .raw
        setattr(self, "_content_consumed", True)
        setattr(self, "raw", None)

    def __repr__(self):
        return f"<Response [{self.status_code}]>"

    def __bool__(self):
        """Returns True if :attr:`status_code` is less than 400.

        This attribute checks if the status code of the response is between
        400 and 600 to see if there was a client error or a server error. If
        the status code, is between 200 and 400, this will return True. This
        is **not** a check to see if the response code is ``200 OK``.
        """
        return self.ok

    def __nonzero__(self):
        """Returns True if :attr:`status_code` is less than 400.

        This attribute checks if the status code of the response is between
        400 and 600 to see if there was a client error or a server error. If
        the status code, is between 200 and 400, this will return True. This
        is **not** a check to see if the response code is ``200 OK``.
        """
        return self.ok

    def __iter__(self):
        """Allows you to use a response as an iterator."""
        return self.iter_content(128)

    @property
    def ok(self):
        """Returns True if :attr:`status_code` is less than 400, False if not.

        This attribute checks if the status code of the response is between
        400 and 600 to see if there was a client error or a server error. If
        the status code is between 200 and 400, this will return True. This
        is **not** a check to see if the response code is ``200 OK``.
        """
        try:
            self.raise_for_status()
        except HTTPError:
            return False
        return True

    @property
    def is_redirect(self):
        """True if this Response is a well-formed HTTP redirect that could have
        been processed automatically (by :meth:`Session.resolve_redirects`).
        """
        return "location" in self.headers and self.status_code in REDIRECT_STATI

    @property
    def is_permanent_redirect(self):
        """True if this Response one of the permanent versions of redirect."""
        return "location" in self.headers and self.status_code in (
            codes.moved_permanently,
            codes.permanent_redirect,
        )

    @property
    def next(self):
        """Returns a PreparedRequest for the next request in a redirect chain, if there is one."""
        return self._next

    @property
    def apparent_encoding(self):
        """The apparent encoding, provided by the charset_normalizer or chardet libraries."""
        if chardet is not None:
            return chardet.detect(self.content)["encoding"]
        else:
            # If no character detection library is available, we'll fall back
            # to a standard Python utf-8 str.
            return "utf-8"

    def iter_content(self, chunk_size=1, decode_unicode=False):
        """Iterates over the response data.  When stream=True is set on the
        request, this avoids reading the content at once into memory for
        large responses.  The chunk size is the number of bytes it should
        read into memory.  This is not necessarily the length of each item
        returned as decoding can take place.

        chunk_size must be of type int or None. A value of None will
        function differently depending on the value of `stream`.
        stream=True will read data as it arrives in whatever size the
        chunks are received. If stream=False, data is returned as
        a single chunk.

        If decode_unicode is True, content will be decoded using the best
        available encoding based on the response.
        """

        def generate():
            # Special case for urllib3.
            if hasattr(self.raw, "stream"):
                try:
                    yield from self.raw.stream(chunk_size, decode_content=True)
                except ProtocolError as e:
                    raise ChunkedEncodingError(e)
                except DecodeError as e:
                    raise ContentDecodingError(e)
                except ReadTimeoutError as e:
                    raise ConnectionError(e)
                except SSLError as e:
                    raise RequestsSSLError(e)
            else:
                # Standard file-like object.
                while True:
                    chunk = self.raw.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk

            self._content_consumed = True

        if self._content_consumed and isinstance(self._content, bool):
            raise StreamConsumedError()
        elif chunk_size is not None and not isinstance(chunk_size, int):
            raise TypeError(
                f"chunk_size must be an int, it is instead a {type(chunk_size)}."
            )
        # simulate reading small chunks of the content
        reused_chunks = iter_slices(self._content, chunk_size)

        stream_chunks = generate()

        chunks = reused_chunks if self._content_consumed else stream_chunks

        if decode_unicode:
            chunks = stream_decode_response_unicode(chunks, self)

        return chunks

    def iter_lines(
        self, chunk_size=ITER_CHUNK_SIZE, decode_unicode=False, delimiter=None
    ):
        """Iterates over the response data, one line at a time.  When
        stream=True is set on the request, this avoids reading the
        content at once into memory for large responses.

        .. note:: This method is not reentrant safe.
        """

        pending = None

        for chunk in self.iter_content(
            chunk_size=chunk_size, decode_unicode=decode_unicode
        ):
            if pending is not None:
                chunk = pending + chunk

            if delimiter:
                lines = chunk.split(delimiter)
            else:
                lines = chunk.splitlines()

            if lines and lines[-1] and chunk and lines[-1][-1] == chunk[-1]:
                pending = lines.pop()
            else:
                pending = None

            yield from lines

        if pending is not None:
            yield pending

    @property
    def content(self):
        """Content of the response, in bytes."""

        if self._content is False:
            # Read the contents.
            if self._content_consumed:
                raise RuntimeError("The content for this response was already consumed")

            if self.status_code == 0 or self.raw is None:
                self._content = None
            else:
                self._content = b"".join(self.iter_content(CONTENT_CHUNK_SIZE)) or b""

        self._content_consumed = True
        # don't need to release the connection; that's been handled by urllib3
        # since we exhausted the data.
        return self._content

    @property
    def text(self):
        """Content of the response, in unicode.

        If Response.encoding is None, encoding will be guessed using
        ``charset_normalizer`` or ``chardet``.

        The encoding of the response content is determined based solely on HTTP
        headers, following RFC 2616 to the letter. If you can take advantage of
        non-HTTP knowledge to make a better guess at the encoding, you should
        set ``r.encoding`` appropriately before accessing this property.
        """

        # Try charset from content-type
        content = None
        encoding = self.encoding

        if not self.content:
            return ""

        # Fallback to auto-detected encoding.
        if self.encoding is None:
            encoding = self.apparent_encoding

        # Decode unicode from given encoding.
        try:
            content = str(self.content, encoding, errors="replace")
        except (LookupError, TypeError):
            # A LookupError is raised if the encoding was not found which could
            # indicate a misspelling or similar mistake.
            #
            # A TypeError can be raised if encoding is None
            #
            # So we try blindly encoding.
            content = str(self.content, errors="replace")

        return content

    def json(self, **kwargs):
        r"""Returns the json-encoded content of a response, if any.

        :param \*\*kwargs: Optional arguments that ``json.loads`` takes.
        :raises requests.exceptions.JSONDecodeError: If the response body does not
            contain valid json.
        """

        if not self.encoding and self.content and len(self.content) > 3:
            # No encoding set. JSON RFC 4627 section 3 states we should expect
            # UTF-8, -16 or -32. Detect which one to use; If the detection or
            # decoding fails, fall back to `self.text` (using charset_normalizer to make
            # a best guess).
            encoding = guess_json_utf(self.content)
            if encoding is not None:
                try:
                    return complexjson.loads(self.content.decode(encoding), **kwargs)
                except UnicodeDecodeError:
                    # Wrong UTF codec detected; usually because it's not UTF-8
                    # but some other 8-bit codec.  This is an RFC violation,
                    # and the server didn't bother to tell us what codec *was*
                    # used.
                    pass
                except JSONDecodeError as e:
                    raise RequestsJSONDecodeError(e.msg, e.doc, e.pos)

        try:
            return complexjson.loads(self.text, **kwargs)
        except JSONDecodeError as e:
            # Catch JSON-related errors and raise as requests.JSONDecodeError
            # This aliases json.JSONDecodeError and simplejson.JSONDecodeError
            raise RequestsJSONDecodeError(e.msg, e.doc, e.pos)

    @property
    def links(self):
        """Returns the parsed header links of the response, if any."""

        header = self.headers.get("link")

        resolved_links = {}

        if header:
            links = parse_header_links(header)

            for link in links:
                key = link.get("rel") or link.get("url")
                resolved_links[key] = link

        return resolved_links

    def raise_for_status(self):
        """Raises :class:`HTTPError`, if one occurred."""

        http_error_msg = ""
        if isinstance(self.reason, bytes):
            # We attempt to decode utf-8 first because some servers
            # choose to localize their reason strings. If the string
            # isn't utf-8, we fall back to iso-8859-1 for all other
            # encodings. (See PR #3538)
            try:
                reason = self.reason.decode("utf-8")
            except UnicodeDecodeError:
                reason = self.reason.decode("iso-8859-1")
        else:
            reason = self.reason

        if 400 <= self.status_code < 500:
            http_error_msg = (
                f"{self.status_code} Client Error: {reason} for url: {self.url}"
            )

        elif 500 <= self.status_code < 600:
            http_error_msg = (
                f"{self.status_code} Server Error: {reason} for url: {self.url}"
            )

        if http_error_msg:
            raise HTTPError(http_error_msg, response=self)

    def close(self):
        """Releases the connection back to the pool. Once this method has been
        called the underlying ``raw`` object must not be accessed again.

        *Note: Should not normally need to be called explicitly.*
        """
        if not self._content_consumed:
            self.raw.close()

        release_conn = getattr(self.raw, "release_conn", None)
        if release_conn is not None:
            release_conn()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\repmatrix.py ===
from collections import defaultdict

from operator import index as index_

from sympy.core.expr import Expr
from sympy.core.kind import Kind, NumberKind, UndefinedKind
from sympy.core.numbers import Integer, Rational
from sympy.core.sympify import _sympify, SympifyError
from sympy.core.singleton import S
from sympy.polys.domains import ZZ, QQ, GF, EXRAW
from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.exceptions import DMNonInvertibleMatrixError
from sympy.polys.polyerrors import CoercionFailed, NotInvertible
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence
from sympy.utilities.misc import filldedent, as_int

from .exceptions import ShapeError, NonSquareMatrixError, NonInvertibleMatrixError
from .matrixbase import classof, MatrixBase
from .kind import MatrixKind


class RepMatrix(MatrixBase):
    """Matrix implementation based on DomainMatrix as an internal representation.

    The RepMatrix class is a superclass for Matrix, ImmutableMatrix,
    SparseMatrix and ImmutableSparseMatrix which are the main usable matrix
    classes in SymPy. Most methods on this class are simply forwarded to
    DomainMatrix.
    """

    #
    # MatrixBase is the common superclass for all of the usable explicit matrix
    # classes in SymPy. The idea is that MatrixBase is an abstract class though
    # and that subclasses will implement the lower-level methods.
    #
    # RepMatrix is a subclass of MatrixBase that uses DomainMatrix as an
    # internal representation and delegates lower-level methods to
    # DomainMatrix. All of SymPy's standard explicit matrix classes subclass
    # RepMatrix and so use DomainMatrix internally.
    #
    # A RepMatrix uses an internal DomainMatrix with the domain set to ZZ, QQ
    # or EXRAW. The EXRAW domain is equivalent to the previous implementation
    # of Matrix that used Expr for the elements. The ZZ and QQ domains are used
    # when applicable just because they are compatible with the previous
    # implementation but are much more efficient. Other domains such as QQ[x]
    # are not used because they differ from Expr in some way (e.g. automatic
    # expansion of powers and products).
    #

    _rep: DomainMatrix

    def __eq__(self, other):
        # Skip sympify for mutable matrices...
        if not isinstance(other, RepMatrix):
            try:
                other = _sympify(other)
            except SympifyError:
                return NotImplemented
            if not isinstance(other, RepMatrix):
                return NotImplemented

        return self._rep.unify_eq(other._rep)

    def to_DM(self, domain=None, **kwargs):
        """Convert to a :class:`~.DomainMatrix`.

        Examples
        ========

        >>> from sympy import Matrix
        >>> M = Matrix([[1, 2], [3, 4]])
        >>> M.to_DM()
        DomainMatrix({0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}, (2, 2), ZZ)

        The :meth:`DomainMatrix.to_Matrix` method can be used to convert back:

        >>> M.to_DM().to_Matrix() == M
        True

        The domain can be given explicitly or otherwise it will be chosen by
        :func:`construct_domain`. Any keyword arguments (besides ``domain``)
        are passed to :func:`construct_domain`:

        >>> from sympy import QQ, symbols
        >>> x = symbols('x')
        >>> M = Matrix([[x, 1], [1, x]])
        >>> M
        Matrix([
        [x, 1],
        [1, x]])
        >>> M.to_DM().domain
        ZZ[x]
        >>> M.to_DM(field=True).domain
        ZZ(x)
        >>> M.to_DM(domain=QQ[x]).domain
        QQ[x]

        See Also
        ========

        DomainMatrix
        DomainMatrix.to_Matrix
        DomainMatrix.convert_to
        DomainMatrix.choose_domain
        construct_domain
        """
        if domain is not None:
            if kwargs:
                raise TypeError("Options cannot be used with domain parameter")
            return self._rep.convert_to(domain)

        rep = self._rep
        dom = rep.domain

        # If the internal DomainMatrix is already ZZ or QQ then we can maybe
        # bypass calling construct_domain or performing any conversions. Some
        # kwargs might affect this though e.g. field=True (not sure if there
        # are others).
        if not kwargs:
            if dom.is_ZZ:
                return rep.copy()
            elif dom.is_QQ:
                # All elements might be integers
                try:
                    return rep.convert_to(ZZ)
                except CoercionFailed:
                    pass
                return rep.copy()

        # Let construct_domain choose a domain
        rep_dom = rep.choose_domain(**kwargs)

        # XXX: There should be an option to construct_domain to choose EXRAW
        # instead of EX. At least converting to EX does not initially trigger
        # EX.simplify which is what we want here but should probably be
        # considered a bug in EX. Perhaps also this could be handled in
        # DomainMatrix.choose_domain rather than here...
        if rep_dom.domain.is_EX:
            rep_dom = rep_dom.convert_to(EXRAW)

        return rep_dom

    @classmethod
    def _unify_element_sympy(cls, rep, element):
        domain = rep.domain
        element = _sympify(element)

        if domain != EXRAW:
            # The domain can only be ZZ, QQ or EXRAW
            if element.is_Integer:
                new_domain = domain
            elif element.is_Rational:
                new_domain = QQ
            else:
                new_domain = EXRAW

            # XXX: This converts the domain for all elements in the matrix
            # which can be slow. This happens e.g. if __setitem__ changes one
            # element to something that does not fit in the domain
            if new_domain != domain:
                rep = rep.convert_to(new_domain)
                domain = new_domain

            if domain != EXRAW:
                element = new_domain.from_sympy(element)

        if domain == EXRAW and not isinstance(element, Expr):
            sympy_deprecation_warning(
                """
                non-Expr objects in a Matrix is deprecated. Matrix represents
                a mathematical matrix. To represent a container of non-numeric
                entities, Use a list of lists, TableForm, NumPy array, or some
                other data structure instead.
                """,
                deprecated_since_version="1.9",
                active_deprecations_target="deprecated-non-expr-in-matrix",
                stacklevel=4,
            )

        return rep, element

    @classmethod
    def _dod_to_DomainMatrix(cls, rows, cols, dod, types):

        if not all(issubclass(typ, Expr) for typ in types):
            sympy_deprecation_warning(
                """
                non-Expr objects in a Matrix is deprecated. Matrix represents
                a mathematical matrix. To represent a container of non-numeric
                entities, Use a list of lists, TableForm, NumPy array, or some
                other data structure instead.
                """,
                deprecated_since_version="1.9",
                active_deprecations_target="deprecated-non-expr-in-matrix",
                stacklevel=6,
            )

        rep = DomainMatrix(dod, (rows, cols), EXRAW)

        if all(issubclass(typ, Rational) for typ in types):
            if all(issubclass(typ, Integer) for typ in types):
                rep = rep.convert_to(ZZ)
            else:
                rep = rep.convert_to(QQ)

        return rep

    @classmethod
    def _flat_list_to_DomainMatrix(cls, rows, cols, flat_list):

        elements_dod = defaultdict(dict)
        for n, element in enumerate(flat_list):
            if element != 0:
                i, j = divmod(n, cols)
                elements_dod[i][j] = element

        types = set(map(type, flat_list))

        rep = cls._dod_to_DomainMatrix(rows, cols, elements_dod, types)
        return rep

    @classmethod
    def _smat_to_DomainMatrix(cls, rows, cols, smat):

        elements_dod = defaultdict(dict)
        for (i, j), element in smat.items():
            if element != 0:
                elements_dod[i][j] = element

        types = set(map(type, smat.values()))

        rep = cls._dod_to_DomainMatrix(rows, cols, elements_dod, types)
        return rep

    def flat(self):
        return self._rep.to_sympy().to_list_flat()

    def _eval_tolist(self):
        return self._rep.to_sympy().to_list()

    def _eval_todok(self):
        return self._rep.to_sympy().to_dok()

    @classmethod
    def _eval_from_dok(cls, rows, cols, dok):
        return cls._fromrep(cls._smat_to_DomainMatrix(rows, cols, dok))

    def _eval_values(self):
        return list(self._eval_iter_values())

    def _eval_iter_values(self):
        rep = self._rep
        K = rep.domain
        values = rep.iter_values()
        if not K.is_EXRAW:
            values = map(K.to_sympy, values)
        return values

    def _eval_iter_items(self):
        rep = self._rep
        K = rep.domain
        to_sympy = K.to_sympy
        items = rep.iter_items()
        if not K.is_EXRAW:
            items = ((i, to_sympy(v)) for i, v in items)
        return items

    def copy(self):
        return self._fromrep(self._rep.copy())

    @property
    def kind(self) -> MatrixKind:
        domain = self._rep.domain
        element_kind: Kind
        if domain in (ZZ, QQ):
            element_kind = NumberKind
        elif domain == EXRAW:
            kinds = {e.kind for e in self.values()}
            if len(kinds) == 1:
                [element_kind] = kinds
            else:
                element_kind = UndefinedKind
        else: # pragma: no cover
            raise RuntimeError("Domain should only be ZZ, QQ or EXRAW")
        return MatrixKind(element_kind)

    def _eval_has(self, *patterns):
        # if the matrix has any zeros, see if S.Zero
        # has the pattern.  If _smat is full length,
        # the matrix has no zeros.
        zhas = False
        dok = self.todok()
        if len(dok) != self.rows*self.cols:
            zhas = S.Zero.has(*patterns)
        return zhas or any(value.has(*patterns) for value in dok.values())

    def _eval_is_Identity(self):
        if not all(self[i, i] == 1 for i in range(self.rows)):
            return False
        return len(self.todok()) == self.rows

    def _eval_is_symmetric(self, simpfunc):
        diff = (self - self.T).applyfunc(simpfunc)
        return len(diff.values()) == 0

    def _eval_transpose(self):
        """Returns the transposed SparseMatrix of this SparseMatrix.

        Examples
        ========

        >>> from sympy import SparseMatrix
        >>> a = SparseMatrix(((1, 2), (3, 4)))
        >>> a
        Matrix([
        [1, 2],
        [3, 4]])
        >>> a.T
        Matrix([
        [1, 3],
        [2, 4]])
        """
        return self._fromrep(self._rep.transpose())

    def _eval_col_join(self, other):
        return self._fromrep(self._rep.vstack(other._rep))

    def _eval_row_join(self, other):
        return self._fromrep(self._rep.hstack(other._rep))

    def _eval_extract(self, rowsList, colsList):
        return self._fromrep(self._rep.extract(rowsList, colsList))

    def __getitem__(self, key):
        return _getitem_RepMatrix(self, key)

    @classmethod
    def _eval_zeros(cls, rows, cols):
        rep = DomainMatrix.zeros((rows, cols), ZZ)
        return cls._fromrep(rep)

    @classmethod
    def _eval_eye(cls, rows, cols):
        rep = DomainMatrix.eye((rows, cols), ZZ)
        return cls._fromrep(rep)

    def _eval_add(self, other):
        return classof(self, other)._fromrep(self._rep + other._rep)

    def _eval_matrix_mul(self, other):
        return classof(self, other)._fromrep(self._rep * other._rep)

    def _eval_matrix_mul_elementwise(self, other):
        selfrep, otherrep = self._rep.unify(other._rep)
        newrep = selfrep.mul_elementwise(otherrep)
        return classof(self, other)._fromrep(newrep)

    def _eval_scalar_mul(self, other):
        rep, other = self._unify_element_sympy(self._rep, other)
        return self._fromrep(rep.scalarmul(other))

    def _eval_scalar_rmul(self, other):
        rep, other = self._unify_element_sympy(self._rep, other)
        return self._fromrep(rep.rscalarmul(other))

    def _eval_Abs(self):
        return self._fromrep(self._rep.applyfunc(abs))

    def _eval_conjugate(self):
        rep = self._rep
        domain = rep.domain
        if domain in (ZZ, QQ):
            return self.copy()
        else:
            return self._fromrep(rep.applyfunc(lambda e: e.conjugate()))

    def equals(self, other, failing_expression=False):
        """Applies ``equals`` to corresponding elements of the matrices,
        trying to prove that the elements are equivalent, returning True
        if they are, False if any pair is not, and None (or the first
        failing expression if failing_expression is True) if it cannot
        be decided if the expressions are equivalent or not. This is, in
        general, an expensive operation.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.abc import x
        >>> A = Matrix([x*(x - 1), 0])
        >>> B = Matrix([x**2 - x, 0])
        >>> A == B
        False
        >>> A.simplify() == B.simplify()
        True
        >>> A.equals(B)
        True
        >>> A.equals(2)
        False

        See Also
        ========
        sympy.core.expr.Expr.equals
        """
        if self.shape != getattr(other, 'shape', None):
            return False

        rv = True
        for i in range(self.rows):
            for j in range(self.cols):
                ans = self[i, j].equals(other[i, j], failing_expression)
                if ans is False:
                    return False
                elif ans is not True and rv is True:
                    rv = ans
        return rv

    def inv_mod(M, m):
        r"""
        Returns the inverse of the integer matrix ``M`` modulo ``m``.

        Examples
        ========

        >>> from sympy import Matrix
        >>> A = Matrix(2, 2, [1, 2, 3, 4])
        >>> A.inv_mod(5)
        Matrix([
        [3, 1],
        [4, 2]])
        >>> A.inv_mod(3)
        Matrix([
        [1, 1],
        [0, 1]])

        """

        if not M.is_square:
            raise NonSquareMatrixError()

        try:
            m = as_int(m)
        except ValueError:
            raise TypeError("inv_mod: modulus m must be an integer")

        K = GF(m, symmetric=False)

        try:
            dM = M.to_DM(K)
        except CoercionFailed:
            raise ValueError("inv_mod: matrix entries must be integers")

        if K.is_Field:
            try:
                dMi = dM.inv()
            except DMNonInvertibleMatrixError as exc:
                msg = f'Matrix is not invertible (mod {m})'
                raise NonInvertibleMatrixError(msg) from exc
        else:
            dMadj, det = dM.adj_det()
            try:
                detinv = 1 / det
            except NotInvertible:
                msg = f'Matrix is not invertible (mod {m})'
                raise NonInvertibleMatrixError(msg)
            dMi = dMadj * detinv

        return dMi.to_Matrix()

    def lll(self, delta=0.75):
        """LLL-reduced basis for the rowspace of a matrix of integers.

        Performs the Lenstra–Lenstra–Lovász (LLL) basis reduction algorithm.

        The implementation is provided by :class:`~DomainMatrix`. See
        :meth:`~DomainMatrix.lll` for more details.

        Examples
        ========

        >>> from sympy import Matrix
        >>> M = Matrix([[1, 0, 0, 0, -20160],
        ...             [0, 1, 0, 0, 33768],
        ...             [0, 0, 1, 0, 39578],
        ...             [0, 0, 0, 1, 47757]])
        >>> M.lll()
        Matrix([
        [ 10, -3,  -2,  8,  -4],
        [  3, -9,   8,  1, -11],
        [ -3, 13,  -9, -3,  -9],
        [-12, -7, -11,  9,  -1]])

        See Also
        ========

        lll_transform
        sympy.polys.matrices.domainmatrix.DomainMatrix.lll
        """
        delta = QQ.from_sympy(_sympify(delta))
        dM = self._rep.convert_to(ZZ)
        basis = dM.lll(delta=delta)
        return self._fromrep(basis)

    def lll_transform(self, delta=0.75):
        """LLL-reduced basis and transformation matrix.

        Performs the Lenstra–Lenstra–Lovász (LLL) basis reduction algorithm.

        The implementation is provided by :class:`~DomainMatrix`. See
        :meth:`~DomainMatrix.lll_transform` for more details.

        Examples
        ========

        >>> from sympy import Matrix
        >>> M = Matrix([[1, 0, 0, 0, -20160],
        ...             [0, 1, 0, 0, 33768],
        ...             [0, 0, 1, 0, 39578],
        ...             [0, 0, 0, 1, 47757]])
        >>> B, T = M.lll_transform()
        >>> B
        Matrix([
        [ 10, -3,  -2,  8,  -4],
        [  3, -9,   8,  1, -11],
        [ -3, 13,  -9, -3,  -9],
        [-12, -7, -11,  9,  -1]])
        >>> T
        Matrix([
        [ 10, -3,  -2,  8],
        [  3, -9,   8,  1],
        [ -3, 13,  -9, -3],
        [-12, -7, -11,  9]])

        The transformation matrix maps the original basis to the LLL-reduced
        basis:

        >>> T * M == B
        True

        See Also
        ========

        lll
        sympy.polys.matrices.domainmatrix.DomainMatrix.lll_transform
        """
        delta = QQ.from_sympy(_sympify(delta))
        dM = self._rep.convert_to(ZZ)
        basis, transform = dM.lll_transform(delta=delta)
        B = self._fromrep(basis)
        T = self._fromrep(transform)
        return B, T


class MutableRepMatrix(RepMatrix):
    """Mutable matrix based on DomainMatrix as the internal representation"""

    #
    # MutableRepMatrix is a subclass of RepMatrix that adds/overrides methods
    # to make the instances mutable. MutableRepMatrix is a superclass for both
    # MutableDenseMatrix and MutableSparseMatrix.
    #

    is_zero = False

    def __new__(cls, *args, **kwargs):
        return cls._new(*args, **kwargs)

    @classmethod
    def _new(cls, *args, copy=True, **kwargs):
        if copy is False:
            # The input was rows, cols, [list].
            # It should be used directly without creating a copy.
            if len(args) != 3:
                raise TypeError("'copy=False' requires a matrix be initialized as rows,cols,[list]")
            rows, cols, flat_list = args
        else:
            rows, cols, flat_list = cls._handle_creation_inputs(*args, **kwargs)
            flat_list = list(flat_list) # create a shallow copy

        rep = cls._flat_list_to_DomainMatrix(rows, cols, flat_list)

        return cls._fromrep(rep)

    @classmethod
    def _fromrep(cls, rep):
        obj = super().__new__(cls)
        obj.rows, obj.cols = rep.shape
        obj._rep = rep
        return obj

    def copy(self):
        return self._fromrep(self._rep.copy())

    def as_mutable(self):
        return self.copy()

    def __setitem__(self, key, value):
        """

        Examples
        ========

        >>> from sympy import Matrix, I, zeros, ones
        >>> m = Matrix(((1, 2+I), (3, 4)))
        >>> m
        Matrix([
        [1, 2 + I],
        [3,     4]])
        >>> m[1, 0] = 9
        >>> m
        Matrix([
        [1, 2 + I],
        [9,     4]])
        >>> m[1, 0] = [[0, 1]]

        To replace row r you assign to position r*m where m
        is the number of columns:

        >>> M = zeros(4)
        >>> m = M.cols
        >>> M[3*m] = ones(1, m)*2; M
        Matrix([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [2, 2, 2, 2]])

        And to replace column c you can assign to position c:

        >>> M[2] = ones(m, 1)*4; M
        Matrix([
        [0, 0, 4, 0],
        [0, 0, 4, 0],
        [0, 0, 4, 0],
        [2, 2, 4, 2]])
        """
        rv = self._setitem(key, value)
        if rv is not None:
            i, j, value = rv
            self._rep, value = self._unify_element_sympy(self._rep, value)
            self._rep.rep.setitem(i, j, value)

    def _eval_col_del(self, col):
        self._rep = DomainMatrix.hstack(self._rep[:,:col], self._rep[:,col+1:])
        self.cols -= 1

    def _eval_row_del(self, row):
        self._rep = DomainMatrix.vstack(self._rep[:row,:], self._rep[row+1:, :])
        self.rows -= 1

    def _eval_col_insert(self, col, other):
        other = self._new(other)
        return self.hstack(self[:,:col], other, self[:,col:])

    def _eval_row_insert(self, row, other):
        other = self._new(other)
        return self.vstack(self[:row,:], other, self[row:,:])

    def col_op(self, j, f):
        """In-place operation on col j using two-arg functor whose args are
        interpreted as (self[i, j], i).

        Examples
        ========

        >>> from sympy import eye
        >>> M = eye(3)
        >>> M.col_op(1, lambda v, i: v + 2*M[i, 0]); M
        Matrix([
        [1, 2, 0],
        [0, 1, 0],
        [0, 0, 1]])

        See Also
        ========
        col
        row_op
        """
        for i in range(self.rows):
            self[i, j] = f(self[i, j], i)

    def col_swap(self, i, j):
        """Swap the two given columns of the matrix in-place.

        Examples
        ========

        >>> from sympy import Matrix
        >>> M = Matrix([[1, 0], [1, 0]])
        >>> M
        Matrix([
        [1, 0],
        [1, 0]])
        >>> M.col_swap(0, 1)
        >>> M
        Matrix([
        [0, 1],
        [0, 1]])

        See Also
        ========

        col
        row_swap
        """
        for k in range(0, self.rows):
            self[k, i], self[k, j] = self[k, j], self[k, i]

    def row_op(self, i, f):
        """In-place operation on row ``i`` using two-arg functor whose args are
        interpreted as ``(self[i, j], j)``.

        Examples
        ========

        >>> from sympy import eye
        >>> M = eye(3)
        >>> M.row_op(1, lambda v, j: v + 2*M[0, j]); M
        Matrix([
        [1, 0, 0],
        [2, 1, 0],
        [0, 0, 1]])

        See Also
        ========
        row
        zip_row_op
        col_op

        """
        for j in range(self.cols):
            self[i, j] = f(self[i, j], j)

    #The next three methods give direct support for the most common row operations inplace.
    def row_mult(self,i,factor):
        """Multiply the given row by the given factor in-place.

        Examples
        ========

        >>> from sympy import eye
        >>> M = eye(3)
        >>> M.row_mult(1,7); M
        Matrix([
        [1, 0, 0],
        [0, 7, 0],
        [0, 0, 1]])

        """
        for j in range(self.cols):
            self[i,j] *= factor

    def row_add(self,s,t,k):
        """Add k times row s (source) to row t (target) in place.

        Examples
        ========

        >>> from sympy import eye
        >>> M = eye(3)
        >>> M.row_add(0, 2,3); M
        Matrix([
        [1, 0, 0],
        [0, 1, 0],
        [3, 0, 1]])
        """

        for j in range(self.cols):
            self[t,j] += k*self[s,j]

    def row_swap(self, i, j):
        """Swap the two given rows of the matrix in-place.

        Examples
        ========

        >>> from sympy import Matrix
        >>> M = Matrix([[0, 1], [1, 0]])
        >>> M
        Matrix([
        [0, 1],
        [1, 0]])
        >>> M.row_swap(0, 1)
        >>> M
        Matrix([
        [1, 0],
        [0, 1]])

        See Also
        ========

        row
        col_swap
        """
        for k in range(0, self.cols):
            self[i, k], self[j, k] = self[j, k], self[i, k]

    def zip_row_op(self, i, k, f):
        """In-place operation on row ``i`` using two-arg functor whose args are
        interpreted as ``(self[i, j], self[k, j])``.

        Examples
        ========

        >>> from sympy import eye
        >>> M = eye(3)
        >>> M.zip_row_op(1, 0, lambda v, u: v + 2*u); M
        Matrix([
        [1, 0, 0],
        [2, 1, 0],
        [0, 0, 1]])

        See Also
        ========
        row
        row_op
        col_op

        """
        for j in range(self.cols):
            self[i, j] = f(self[i, j], self[k, j])

    def copyin_list(self, key, value):
        """Copy in elements from a list.

        Parameters
        ==========

        key : slice
            The section of this matrix to replace.
        value : iterable
            The iterable to copy values from.

        Examples
        ========

        >>> from sympy import eye
        >>> I = eye(3)
        >>> I[:2, 0] = [1, 2] # col
        >>> I
        Matrix([
        [1, 0, 0],
        [2, 1, 0],
        [0, 0, 1]])
        >>> I[1, :2] = [[3, 4]]
        >>> I
        Matrix([
        [1, 0, 0],
        [3, 4, 0],
        [0, 0, 1]])

        See Also
        ========

        copyin_matrix
        """
        if not is_sequence(value):
            raise TypeError("`value` must be an ordered iterable, not %s." % type(value))
        return self.copyin_matrix(key, type(self)(value))

    def copyin_matrix(self, key, value):
        """Copy in values from a matrix into the given bounds.

        Parameters
        ==========

        key : slice
            The section of this matrix to replace.
        value : Matrix
            The matrix to copy values from.

        Examples
        ========

        >>> from sympy import Matrix, eye
        >>> M = Matrix([[0, 1], [2, 3], [4, 5]])
        >>> I = eye(3)
        >>> I[:3, :2] = M
        >>> I
        Matrix([
        [0, 1, 0],
        [2, 3, 0],
        [4, 5, 1]])
        >>> I[0, 1] = M
        >>> I
        Matrix([
        [0, 0, 1],
        [2, 2, 3],
        [4, 4, 5]])

        See Also
        ========

        copyin_list
        """
        rlo, rhi, clo, chi = self.key2bounds(key)
        shape = value.shape
        dr, dc = rhi - rlo, chi - clo
        if shape != (dr, dc):
            raise ShapeError(filldedent("The Matrix `value` doesn't have the "
                                        "same dimensions "
                                        "as the in sub-Matrix given by `key`."))

        for i in range(value.rows):
            for j in range(value.cols):
                self[i + rlo, j + clo] = value[i, j]

    def fill(self, value):
        """Fill self with the given value.

        Notes
        =====

        Unless many values are going to be deleted (i.e. set to zero)
        this will create a matrix that is slower than a dense matrix in
        operations.

        Examples
        ========

        >>> from sympy import SparseMatrix
        >>> M = SparseMatrix.zeros(3); M
        Matrix([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]])
        >>> M.fill(1); M
        Matrix([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]])

        See Also
        ========

        zeros
        ones
        """
        value = _sympify(value)
        if not value:
            self._rep = DomainMatrix.zeros(self.shape, EXRAW)
        else:
            elements_dod = {i: dict.fromkeys(range(self.cols), value) for i in range(self.rows)}
            self._rep = DomainMatrix(elements_dod, self.shape, EXRAW)


def _getitem_RepMatrix(self, key):
    """Return portion of self defined by key. If the key involves a slice
    then a list will be returned (if key is a single slice) or a matrix
    (if key was a tuple involving a slice).

    Examples
    ========

    >>> from sympy import Matrix, I
    >>> m = Matrix([
    ... [1, 2 + I],
    ... [3, 4    ]])

    If the key is a tuple that does not involve a slice then that element
    is returned:

    >>> m[1, 0]
    3

    When a tuple key involves a slice, a matrix is returned. Here, the
    first column is selected (all rows, column 0):

    >>> m[:, 0]
    Matrix([
    [1],
    [3]])

    If the slice is not a tuple then it selects from the underlying
    list of elements that are arranged in row order and a list is
    returned if a slice is involved:

    >>> m[0]
    1
    >>> m[::2]
    [1, 3]
    """
    if isinstance(key, tuple):
        i, j = key
        try:
            return self._rep.getitem_sympy(index_(i), index_(j))
        except (TypeError, IndexError):
            if (isinstance(i, Expr) and not i.is_number) or (isinstance(j, Expr) and not j.is_number):
                if ((j < 0) is True) or ((j >= self.shape[1]) is True) or\
                   ((i < 0) is True) or ((i >= self.shape[0]) is True):
                    raise ValueError("index out of boundary")
                from sympy.matrices.expressions.matexpr import MatrixElement
                return MatrixElement(self, i, j)

            if isinstance(i, slice):
                i = range(self.rows)[i]
            elif is_sequence(i):
                pass
            else:
                i = [i]
            if isinstance(j, slice):
                j = range(self.cols)[j]
            elif is_sequence(j):
                pass
            else:
                j = [j]
            return self.extract(i, j)

    else:
        # Index/slice like a flattened list
        rows, cols = self.shape

        # Raise the appropriate exception:
        if not rows * cols:
            return [][key]

        rep = self._rep.rep
        domain = rep.domain
        is_slice = isinstance(key, slice)

        if is_slice:
            values = [rep.getitem(*divmod(n, cols)) for n in range(rows * cols)[key]]
        else:
            values = [rep.getitem(*divmod(index_(key), cols))]

        if domain != EXRAW:
            to_sympy = domain.to_sympy
            values = [to_sympy(val) for val in values]

        if is_slice:
            return values
        else:
            return values[0]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\gpt2\gpt2_helper.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
# This script helps onnx conversion and validation for GPT2 model with past state.
import logging
import os
import pickle
import random
import shutil
import tempfile
import time
from pathlib import Path

import numpy
import onnx
import torch
from benchmark_helper import Precision
from float16 import float_to_float16_max_diff
from fusion_options import FusionOptions
from io_binding_helper import IOBindingHelper
from onnx_model import OnnxModel
from optimizer import optimize_model
from torch_onnx_export_helper import torch_onnx_export
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Model, TFGPT2Model

logger = logging.getLogger(__name__)

PRETRAINED_GPT2_MODELS = ["distilgpt2", "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]

DEFAULT_TOLERANCE = {
    Precision.FLOAT32: 0.0005,
    Precision.FLOAT16: 0.2,
    Precision.INT8: 3.0,
}


class GPT2ModelNoPastState(GPT2Model):
    """Here we wrap a class to disable past state output."""

    def __init__(self, config):
        super().__init__(config)

    def forward(self, input_ids):
        return super().forward(input_ids, use_cache=False, return_dict=False)


class TFGPT2ModelNoPastState(TFGPT2Model):
    """Here we wrap a class to disable past state output."""

    def __init__(self, config):
        config.use_cache = False
        super().__init__(config)

    def forward(self, input_ids):
        return super().call(input_ids, use_cache=False)


class MyGPT2Model(GPT2Model):
    """Here we wrap a class for Onnx model conversion for GPT2Model with past state."""

    def __init__(self, config):
        super().__init__(config)

    @staticmethod
    def post_process(result, num_layer):
        if isinstance(result[1][0], (tuple, list)):
            assert len(result[1]) == num_layer and len(result[1][0]) == 2
            # assert len(result[1][0][0].shape) == 4 and result[1][0][0].shape == result[1][0][1].shape
            present = []
            for i in range(num_layer):
                # Since transformers v4.*, past key and values are separated outputs.
                # Here we concate them into one tensor to be compatible with Attention operator.
                present.append(
                    torch.cat(
                        (result[1][i][0].unsqueeze(0), result[1][i][1].unsqueeze(0)),
                        dim=0,
                    )
                )
            return (result[0], tuple(present))

        return result

    def forward(self, input_ids, position_ids, attention_mask, *past):
        result = super().forward(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past,
            return_dict=False,
        )
        return MyGPT2Model.post_process(result, self.config.n_layer)


class MyGPT2LMHeadModel(GPT2LMHeadModel):
    """Here we wrap a class for Onnx model conversion for GPT2LMHeadModel with past state."""

    def __init__(self, config):
        super().__init__(config)

    def forward(self, input_ids, position_ids, attention_mask, *past):
        result = super().forward(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past,
            return_dict=False,
        )

        return MyGPT2Model.post_process(result, self.config.n_layer)


class MyGPT2LMHeadModel_NoPadding(GPT2LMHeadModel):  # noqa: N801
    """Here we wrap a class for Onnx model conversion for GPT2LMHeadModel with past state and no padding.
    When you always use batch_size=1 in inference, there is no padding in inputs. In such case, position_ids
    and attention_mask need no be in inputs.
    """

    def __init__(self, config):
        super().__init__(config)

    def forward(self, input_ids, *past):
        result = super().forward(input_ids, past_key_values=past, return_dict=False)

        return MyGPT2Model.post_process(result, self.config.n_layer)


# Maps model class name to a tuple of model class, name of first output and use padding or not
MODEL_CLASSES = {
    "GPT2LMHeadModel": (MyGPT2LMHeadModel, "logits", True),
    "GPT2LMHeadModel_NoPadding": (MyGPT2LMHeadModel_NoPadding, "logits", False),
    "GPT2Model": (MyGPT2Model, "last_state", True),
}


class Gpt2Inputs:
    def __init__(self, input_ids, position_ids, attention_mask, past):
        self.input_ids: torch.LongTensor = input_ids
        self.position_ids: torch.LongTensor = position_ids
        self.attention_mask: torch.LongTensor | torch.FloatTensor | torch.HalfTensor = attention_mask
        self.past: list[torch.FloatTensor] | list[torch.HalfTensor] = past

    def to_list(self) -> list:
        input_list = [v for v in [self.input_ids, self.position_ids, self.attention_mask] if v is not None]
        if self.past:
            input_list.extend(self.past)

        return input_list

    def to_tuple(self) -> tuple:
        return tuple(v for v in [self.input_ids, self.position_ids, self.attention_mask, self.past] if v is not None)

    def to_fp32(self):
        # For attention mask, only convert fp16 to fp32, and keep the original type if it is integer.
        attention_mask = None
        if self.attention_mask is not None:
            attention_mask = (
                self.attention_mask.to(dtype=torch.float32)
                if (self.attention_mask.dtype == torch.float16)
                else self.attention_mask
            )

        past = [p.to(dtype=torch.float32) for p in self.past]
        return Gpt2Inputs(self.input_ids, self.position_ids, attention_mask, past)


class Gpt2Helper:
    """A helper class for Gpt2 model conversion, inference and verification."""

    @staticmethod
    def get_dummy_inputs(
        batch_size: int,
        past_sequence_length: int,
        sequence_length: int,
        num_attention_heads: int,
        hidden_size: int,
        num_layer: int,
        vocab_size: int,
        device: torch.device,
        float16: bool = False,
        has_position_ids: bool = True,
        has_attention_mask: bool = True,
        input_ids_dtype: torch.dtype = torch.int32,
        position_ids_dtype: torch.dtype = torch.int32,
        attention_mask_dtype: torch.dtype = torch.int32,
        left_side_padding: bool = True,
    ) -> Gpt2Inputs:
        """Create random inputs for GPT2 model.
        Returns torch tensors of input_ids, position_ids, attention_mask and a list of past state tensors.
        """
        float_type = torch.float16 if float16 else torch.float32
        past_shape = [
            2,
            batch_size,
            num_attention_heads,
            past_sequence_length,
            int(hidden_size / num_attention_heads),
        ]

        past = [(torch.rand(past_shape, dtype=float_type, device=device) * 2.0 - 1.0) for _ in range(num_layer)]
        input_ids = torch.randint(
            low=0,
            high=vocab_size - 1,
            size=(batch_size, sequence_length),
            dtype=input_ids_dtype,
            device=device,
        )

        attention_mask = None
        if has_attention_mask:
            total_sequence_length = past_sequence_length + sequence_length
            attention_mask = torch.ones(
                [batch_size, total_sequence_length],
                dtype=attention_mask_dtype,
                device=device,
            )

            if total_sequence_length >= 2:
                for i in range(batch_size):
                    padding_length = random.randint(0, total_sequence_length - 1)
                    if left_side_padding:
                        attention_mask[i, :padding_length] = 0
                    else:  # right side padding
                        attention_mask[i, total_sequence_length - padding_length :] = 0

        # Deduce position_ids from attention mask
        position_ids = None
        if has_position_ids:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(position_ids < 0, 0)
            position_ids = position_ids[:, past_sequence_length:].to(position_ids_dtype)

        return Gpt2Inputs(input_ids, position_ids, attention_mask, past)

    @staticmethod
    def get_output_shapes(
        batch_size: int,
        past_sequence_length: int,
        sequence_length: int,
        config: GPT2Config,
        model_class: str = "GPT2LMHeadModel",
    ) -> dict[str, list[int]]:
        """Returns a dictionary with output name as key, and shape as value."""
        num_attention_heads = config.num_attention_heads
        hidden_size = config.hidden_size
        num_layer = config.num_hidden_layers
        vocab_size = config.vocab_size

        output_name = MODEL_CLASSES[model_class][1]

        last_state_shape = [
            batch_size,
            sequence_length,
            vocab_size if output_name == "logits" else hidden_size,
        ]
        present_state_shape = [
            2,
            batch_size,
            num_attention_heads,
            past_sequence_length + sequence_length,
            int(hidden_size / num_attention_heads),
        ]

        output_shapes = {output_name: last_state_shape}
        for i in range(num_layer):
            output_shapes["present_" + str(i)] = present_state_shape

        return output_shapes

    @staticmethod
    def auto_increase_buffer_size(output_buffers, output_shapes):
        for key in output_shapes:
            assert key in output_buffers
            buffer = output_buffers[key]
            if numpy.prod(output_shapes[key]) > buffer.nelement():
                output_buffers[key] = torch.empty(
                    numpy.prod(output_shapes[key]),
                    dtype=buffer.dtype,
                    device=buffer.device,
                )

    @staticmethod
    def get_output_buffers(output_shapes, device, is_float16=False):
        """Returns a dictionary of output name as key, and 1D tensor as value. The tensor has enough space for given shape."""
        data_type = torch.float16 if is_float16 else torch.float32

        output_buffers = {}
        for name, shape in output_shapes.items():
            output_buffers[name] = torch.empty(numpy.prod(shape), dtype=data_type, device=device)
        return output_buffers

    @staticmethod
    def diff_outputs(torch_outputs, ort_outputs, relative=False):
        """Returns the maximum difference between PyTorch and OnnxRuntime outputs."""
        expected_outputs = torch_outputs[0].cpu().numpy()
        diff = numpy.abs(expected_outputs - ort_outputs[0])
        if relative:
            return numpy.amax(diff / (numpy.abs(expected_outputs) + 1e-6))
        else:
            return numpy.amax(diff)

    @staticmethod
    def compare_outputs(torch_outputs, ort_outputs, rtol=1e-03, atol=1e-03, **kwargs):
        """Returns True if torch and ORT outputs are close for given thresholds, and False otherwise.
        Note: need kwargs since Gpt2BeamSearchHelper.compare_outputs has an extra parameter model_class
        """
        is_close = numpy.allclose(ort_outputs[0], torch_outputs[0].cpu().numpy(), rtol=rtol, atol=atol)
        logger.debug(f"PyTorch and OnnxRuntime output 0 (last_state) are close: {is_close}")

        is_all_close = is_close
        num_layers = len(ort_outputs) - 1

        for layer in range(num_layers):
            is_close = numpy.allclose(
                ort_outputs[1 + layer],
                torch_outputs[1][layer].cpu().numpy(),
                rtol=rtol,
                atol=atol,
            )
            logger.debug(f"PyTorch and OnnxRuntime layer {layer} state (present_{layer}) are close:{is_close}")
            is_all_close = is_all_close and is_close

        if not is_all_close:
            max_abs_diff = Gpt2Helper.diff_outputs(torch_outputs, ort_outputs)
            logger.info(f"PyTorch and OnnxRuntime results are not all close: max_abs_diff={max_abs_diff:.5f}")

        return is_all_close

    @staticmethod
    def compare_outputs_v2(torch_outputs, ort_outputs, atol=1e-06):
        """Compare outputs from PyTorch and OnnxRuntime

        Args:
            torch_outputs (Tuple[Torch.Tensor]): PyTorch model output
            ort_outputs (List[numpy.ndarray]): OnnxRuntime output
            atol (float, optional): Absolute tollerance. Defaults to 1e-06.

        Returns:
            is_all_close(bool): whether all elements are close.
            max_abs_diff(float): maximum absolute difference.
            messages(str): a list of debug message for each output
        """
        is_all_close = True
        is_top1_matched = False
        max_diffs = []
        messages = []
        for i in range(len(ort_outputs)):
            ort_output = ort_outputs[i]
            torch_output = (torch_outputs[0] if i == 0 else torch_outputs[1][i - 1]).cpu().numpy()
            is_close = numpy.allclose(ort_output, torch_output, atol=atol, rtol=0)
            max_diffs.append(numpy.amax(numpy.abs(torch_output - ort_output)))
            is_all_close = is_all_close and is_close

            if numpy.isnan(torch_output).any():
                logger.debug(f"PyTorch output {i} has nan")
            if numpy.isinf(torch_output).any():
                logger.debug(f"PyTorch output {i} has inf")
            if numpy.isnan(ort_output).any():
                logger.debug(f"ORT output {i} has nan")
            if numpy.isinf(ort_output).any():
                logger.debug(f"ORT output {i} has inf")

            diff = numpy.fabs(ort_output - torch_output)
            idx = numpy.unravel_index(diff.argmax(), diff.shape)
            messages.append(
                f"diff={diff[idx]:.9f} index={idx} ort={ort_output[idx]:.9f} torch={float(torch_output[idx]):.9f}"
            )

            if i == 0:  # logits
                ort_max_index = numpy.unravel_index(numpy.argmax(ort_output, axis=None), ort_output.shape)
                torch_max_index = numpy.unravel_index(numpy.argmax(torch_output, axis=None), torch_output.shape)
                is_top1_matched = numpy.array_equal(ort_max_index, torch_max_index)

        max_diff_output_index = max_diffs.index(max(max_diffs))
        return (
            is_all_close,
            max(max_diffs),
            max_diff_output_index,
            messages,
            is_top1_matched,
        )

    @staticmethod
    def export_onnx(
        model,
        device,
        onnx_model_path: str,
        verbose: bool = False,
        use_external_data_format: bool = False,
        has_position_ids: bool = True,
        has_attention_mask: bool = True,
        input_ids_dtype: torch.dtype = torch.int32,
        position_ids_dtype: torch.dtype = torch.int32,
        attention_mask_dtype: torch.dtype = torch.int32,
    ):
        """Export GPT-2 model with past state to ONNX model."""
        config: GPT2Config = model.config
        num_layer = config.n_layer
        dummy_inputs = Gpt2Helper.get_dummy_inputs(
            batch_size=1,
            past_sequence_length=1,
            sequence_length=1,
            num_attention_heads=config.num_attention_heads,
            hidden_size=config.hidden_size,
            num_layer=num_layer,
            vocab_size=config.vocab_size,
            device=device,
            float16=False,
            has_position_ids=has_position_ids,
            has_attention_mask=has_attention_mask,
            input_ids_dtype=input_ids_dtype,
            position_ids_dtype=position_ids_dtype,
            attention_mask_dtype=attention_mask_dtype,
        )
        input_list = dummy_inputs.to_list()

        with torch.no_grad():
            outputs = model(*input_list)

        past_names = [f"past_{i}" for i in range(num_layer)]
        present_names = [f"present_{i}" for i in range(num_layer)]

        # GPT2Model outputs last_state; GPT2LMHeadModel outputs logits (prediction_scores)
        assert outputs[0].shape[2] == config.vocab_size or outputs[0].shape[2] == config.hidden_size
        output_names = ["logits" if outputs[0].shape[2] == config.vocab_size else "last_state", *present_names]

        # Shape of input tensors:
        #    input_ids: (batch_size, seq_len)
        #    past_{i}:  (2, batch_size, num_heads, past_seq_len, hidden_size/num_heads)
        #    attention_mask: (batch_size, past_seq_len + seq_len)
        # Shape of output tensors:
        #    last_state: (batch_size, seq_len, hidden_size)
        #      or logits: (batch_size, seq_len, vocab_size)
        #    present_{i}:  (2, batch_size, num_heads, past_seq_len + seq_len, hidden_size/num_heads)
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            output_names[0]: {0: "batch_size", 1: "seq_len"},
        }
        for name in past_names:
            dynamic_axes[name] = {1: "batch_size", 3: "past_seq_len"}
        for name in present_names:
            dynamic_axes[name] = {1: "batch_size", 3: "total_seq_len"}

        input_names = ["input_ids"]
        if has_position_ids:
            dynamic_axes["position_ids"] = {0: "batch_size", 1: "seq_len"}
            input_names.append("position_ids")
        if has_attention_mask:
            dynamic_axes["attention_mask"] = {0: "batch_size", 1: "total_seq_len"}
            input_names.append("attention_mask")
        input_names.extend(past_names)

        assert len(outputs) == 2 and len(outputs[1]) == num_layer

        logger.info(
            f"Shapes: input_ids={dummy_inputs.input_ids.shape} past={dummy_inputs.past[0].shape} output={outputs[0].shape} present={outputs[1][0].shape}"
        )

        Path(onnx_model_path).parent.mkdir(parents=True, exist_ok=True)

        if use_external_data_format:
            # We let PyTorch export onnx to a temp directory first, then convert external data to one file.
            with tempfile.TemporaryDirectory() as tmp_dir_name:
                temp_onnx_model_path = os.path.join(tmp_dir_name, "gpt2.onnx")
                Path(temp_onnx_model_path).parent.mkdir(parents=True, exist_ok=True)

                torch_onnx_export(
                    model,
                    args=tuple(input_list),
                    f=temp_onnx_model_path,
                    export_params=True,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    opset_version=11,
                    do_constant_folding=True,
                    use_external_data_format=True,
                    verbose=verbose,
                )

                model = onnx.load_model(temp_onnx_model_path, load_external_data=True)
                OnnxModel.save(
                    model,
                    onnx_model_path,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                )
        else:
            torch_onnx_export(
                model,
                args=tuple(input_list),
                f=onnx_model_path,
                export_params=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=11,
                do_constant_folding=True,
                use_external_data_format=False,
                verbose=verbose,
            )

    @staticmethod
    def optimize_onnx(
        onnx_model_path,
        optimized_model_path,
        is_float16,
        num_attention_heads,
        hidden_size,
        use_external_data_format=False,
        auto_mixed_precision=False,
        stage=0,
        **kwargs,
    ):
        """Optimize ONNX model with an option to convert it to use mixed precision."""
        optimization_options = FusionOptions("gpt2")

        m = optimize_model(
            onnx_model_path,
            model_type="gpt2",
            num_heads=num_attention_heads,
            hidden_size=hidden_size,
            opt_level=0,
            optimization_options=optimization_options,
            use_gpu=False,
        )

        if is_float16:
            if auto_mixed_precision:
                Gpt2Helper.auto_mixed_precision(m)
            else:
                if "keep_io_types" not in kwargs:
                    kwargs["keep_io_types"] = False
                m.convert_float_to_float16(use_symbolic_shape_infer=True, **kwargs)

        m.save_model_to_file(optimized_model_path, use_external_data_format)
        return m

    @staticmethod
    def auto_mixed_precision(
        onnx_model: OnnxModel,
        op_block_list: list[str] = [  # noqa: B006
            "Add",
            "LayerNormalization",
            "SkipLayerNormalization",
            "FastGelu",
            "EmbedLayerNormalization",
        ],
    ):
        """Convert GPT-2 model to mixed precision.
           It detects whether original model has fp16 weights, and set parameters for float16 conversion automatically.
        Args:
            onnx_model (OnnxModel): optimized ONNX model
            op_block_list (List[str], optional): operators to compute in fp32. Defaults to ["Add", "LayerNormalization",
                                                 "SkipLayerNormalization", "FastGelu", "EmbedLayerNormalization"]
        Returns:
            parameters(dict): a dictionary of parameters used in float16 conversion
        """
        op_full_set = {node.op_type for node in onnx_model.nodes()}
        fp32_op_set = set(op_block_list)
        fp16_op_set = op_full_set.difference(fp32_op_set)
        logger.info(f"fp32 op: {fp32_op_set} fp16 op: {fp16_op_set}")

        # logits is the first output
        logits_output_name = onnx_model.graph().output[0].name

        # We use the weight in last MatMul node to detect whether the model is stored with float16 weights from training.
        is_weight_fp16_precision = False
        output_name_to_node = onnx_model.output_name_to_node()
        assert logits_output_name in output_name_to_node
        node = output_name_to_node[logits_output_name]
        last_matmul_node = None
        if node.op_type == "MatMul":
            last_matmul_node = node
            logger.info(f"Found last MatMul node for logits: {node.name}")
            initializer = None
            for input in node.input:
                initializer = onnx_model.get_initializer(input)
                if initializer is not None:
                    break

            # when the max difference of value after converting float to float16 is lower than a threshold (1e-6),
            # we can deduce that the weights are stored in float16 precision.
            max_diff = float_to_float16_max_diff(initializer)
            logger.debug(f"max diff of converting weights in last MatMul node {node.name}: {max_diff}")
            is_weight_fp16_precision = max_diff < 1e-6
        else:
            logger.warning(f"Failed to find MatMul node for logits. Found {node.op_type} of node {node.name}")

        keep_io_types = []
        node_block_list = []
        if (not is_weight_fp16_precision) and (last_matmul_node is not None):
            # When original weight is float32 precision, keep logits and last MatMul in float32 could get better precision.
            keep_io_types = [logits_output_name]
            node_block_list = [last_matmul_node.name]

        parameters = {
            "keep_io_types": keep_io_types,
            "op_block_list": op_block_list,
            "node_block_list": node_block_list,
            "force_fp16_initializers": is_weight_fp16_precision,
        }

        logger.info(f"auto_mixed_precision parameters: {parameters}")
        onnx_model.convert_float_to_float16(use_symbolic_shape_infer=True, **parameters)

        return parameters

    @staticmethod
    def pytorch_inference(model, inputs: Gpt2Inputs, total_runs: int = 0):
        """Run inference of PyTorch model, and returns average latency in ms when total_runs > 0 besides outputs."""
        logger.debug("start pytorch_inference")

        # Convert it to fp32 as the PyTroch model cannot deal with half input.
        input_list = inputs.to_fp32().to_list()

        with torch.no_grad():
            outputs = model(*input_list)

        if total_runs == 0:
            return outputs

        latency = []
        with torch.no_grad():
            for _ in range(total_runs):
                start = time.time()
                outputs = model(*input_list)
                latency.append(time.time() - start)

        average_latency = sum(latency) * 1000 / len(latency)
        logger.debug("PyTorch inference time = {} ms".format(format(average_latency, ".2f")))  # noqa: G001

        return outputs, average_latency

    @staticmethod
    def onnxruntime_inference(ort_session, inputs: Gpt2Inputs, total_runs: int = 0):
        """Run inference of ONNX model, and returns average latency in ms when total_runs > 0 besides outputs."""
        logger.debug("start onnxruntime_inference")

        ort_inputs = {"input_ids": numpy.ascontiguousarray(inputs.input_ids.cpu().numpy())}

        if inputs.past is not None:
            for i, past_i in enumerate(inputs.past):
                ort_inputs[f"past_{i}"] = numpy.ascontiguousarray(past_i.cpu().numpy())

        if inputs.attention_mask is not None:
            ort_inputs["attention_mask"] = numpy.ascontiguousarray(inputs.attention_mask.cpu().numpy())

        if inputs.position_ids is not None:
            ort_inputs["position_ids"] = numpy.ascontiguousarray(inputs.position_ids.cpu().numpy())

        ort_outputs = ort_session.run(None, ort_inputs)
        if total_runs == 0:
            return ort_outputs

        latency = []
        for _ in range(total_runs):
            start = time.time()
            ort_outputs = ort_session.run(None, ort_inputs)
            latency.append(time.time() - start)

        average_latency = sum(latency) * 1000 / len(latency)
        logger.debug("OnnxRuntime Inference time = {} ms".format(format(average_latency, ".2f")))  # noqa: G001

        return ort_outputs, average_latency

    @staticmethod
    def prepare_io_binding(
        ort_session,
        input_ids,
        position_ids,
        attention_mask,
        past,
        output_buffers,
        output_shapes,
    ):
        """Returnas IO binding object for a session."""
        return IOBindingHelper.prepare_io_binding(
            ort_session,
            input_ids,
            position_ids,
            attention_mask,
            past,
            output_buffers,
            output_shapes,
        )

    @staticmethod
    def get_outputs_from_io_binding_buffer(ort_session, output_buffers, output_shapes, return_numpy=True):
        """Copy results to cpu. Returns a list of numpy array."""
        return IOBindingHelper.get_outputs_from_io_binding_buffer(
            ort_session, output_buffers, output_shapes, return_numpy
        )

    @staticmethod
    def onnxruntime_inference_with_binded_io(
        ort_session,
        inputs: Gpt2Inputs,
        output_buffers: dict[str, torch.Tensor],
        output_shapes: dict[str, list[int]],
        total_runs: int = 0,
        return_numpy: bool = True,
        include_copy_output_latency: bool = False,
    ):
        """Inference with IO binding. Returns outputs, and optional latency when total_runs > 0."""
        logger.debug("start onnxruntime_inference_with_binded_io")

        # Bind inputs and outputs to onnxruntime session
        io_binding = Gpt2Helper.prepare_io_binding(
            ort_session,
            inputs.input_ids,
            inputs.position_ids,
            inputs.attention_mask,
            inputs.past,
            output_buffers,
            output_shapes,
        )

        # Run onnxruntime with io binding
        ort_session.run_with_iobinding(io_binding)

        # Copy results to cpu for verification
        ort_outputs = Gpt2Helper.get_outputs_from_io_binding_buffer(
            ort_session, output_buffers, output_shapes, return_numpy
        )

        if total_runs == 0:
            return ort_outputs

        latency = []
        for _ in range(total_runs):
            start = time.time()
            # Run onnxruntime with io binding
            ort_session.run_with_iobinding(io_binding)
            if include_copy_output_latency:
                _ = Gpt2Helper.get_outputs_from_io_binding_buffer(
                    ort_session, output_buffers, output_shapes, return_numpy
                )
            latency.append(time.time() - start)

        average_latency = sum(latency) * 1000 / len(latency)
        logger.debug("OnnxRuntime with IO binding inference time = %.2f ms", average_latency)

        return ort_outputs, average_latency

    @staticmethod
    def save_outputs(i, ort_outputs, torch_outputs):
        with open(f"ort_outputs_{i}.pickle", "wb") as f:
            pickle.dump(ort_outputs, f)
        logger.info(f"ORT output are saved to ort_outputs_{i}.pickle")

        with open(f"torch_outputs_{i}.pickle", "wb") as f:
            pickle.dump(torch_outputs, f)
        logger.info(f"Torch output are saved to torch_outputs_{i}.pickle")

    @staticmethod
    def save_inputs(i, dummy_inputs, ort_outputs, torch_outputs):
        with open(f"dummy_inputs_{i}.pickle", "wb") as f:
            pickle.dump(dummy_inputs, f)
        logger.info(f"inputs are saved to dummy_inputs_{i}.pickle")

    @staticmethod
    def test_parity(
        ort_session,
        model,
        device,
        is_float16=False,
        rtol=5e-4,
        atol=5e-4,
        test_cases_per_run=10000,
        total_runs=1,
        use_io_binding=True,
        model_class="GPT2LMHeadModel",
        has_position_ids=True,
        has_attention_mask=True,
        input_ids_dtype=torch.int32,
        position_ids_dtype=torch.int32,
        attention_mask_dtype=torch.int32,
        stage=0,
        verbose=False,
        enable_pickle_output=False,
    ):
        """Generate random inputs and compare the results of PyTorch and Onnx Runtime."""

        config: GPT2Config = model.config

        logger.info(
            f"Running parity test (atol={atol}, test_cases={test_cases_per_run}, runs={total_runs}, use_io_binding={use_io_binding}, model_class={model_class}, is_float16={is_float16}) ..."
        )

        max_batch_size = 8
        max_past_seq_len = 4  # Do not use large number here for higher chance of hitting empty past (past_seq_len=0)
        max_seq_len = 2

        output_buffers = None
        if use_io_binding:
            max_output_shapes = Gpt2Helper.get_output_shapes(
                max_batch_size, max_past_seq_len, max_seq_len, config, model_class
            )
            output_buffers = Gpt2Helper.get_output_buffers(max_output_shapes, device, is_float16)

        passed_test_cases = 0
        top1_matched_cases = 0

        max_abs_diff_list = []
        top1_matched_cases_per_run = [0] * total_runs
        total_test_cases = test_cases_per_run * total_runs
        for i in range(total_test_cases):
            run_id = int(i / test_cases_per_run)
            sequence_length = random.randint(1, max_seq_len)
            past_sequence_length = 0 if (stage == 1) else random.randint(0, max_past_seq_len)
            batch_size = random.randint(1, max_batch_size)

            logger.debug(
                f"Running parity test for batch_size={batch_size} past_sequence_length={past_sequence_length}..."
            )
            dummy_inputs = Gpt2Helper.get_dummy_inputs(
                batch_size,
                past_sequence_length,
                sequence_length,
                config.num_attention_heads,
                config.hidden_size,
                config.n_layer,
                config.vocab_size,
                device,
                is_float16,
                has_position_ids,
                has_attention_mask,
                input_ids_dtype=input_ids_dtype,
                position_ids_dtype=position_ids_dtype,
                attention_mask_dtype=attention_mask_dtype,
                left_side_padding=True,
            )
            outputs = Gpt2Helper.pytorch_inference(model, dummy_inputs)
            if use_io_binding:
                ort_outputs = Gpt2Helper.onnxruntime_inference(ort_session, dummy_inputs)
            else:
                output_shapes = Gpt2Helper.get_output_shapes(
                    batch_size,
                    past_sequence_length,
                    sequence_length,
                    config,
                    model_class,
                )
                ort_outputs = Gpt2Helper.onnxruntime_inference_with_binded_io(
                    ort_session, dummy_inputs, output_buffers, output_shapes
                )

            (
                is_all_close,
                max_abs_diff,
                max_diff_output_index,
                messages,
                is_top1_matched,
            ) = Gpt2Helper.compare_outputs_v2(outputs, ort_outputs, atol=atol)
            if not numpy.isnan(max_abs_diff):
                max_abs_diff_list.append(max_abs_diff)
            if is_all_close:
                passed_test_cases += 1

            if is_top1_matched:
                top1_matched_cases += 1
                top1_matched_cases_per_run[run_id] += 1

            if verbose and not is_all_close:
                logger.info(
                    f"test_case={i} batch_size={batch_size} past_sequence_length={past_sequence_length} sequence_length={sequence_length} MaxDiff={max_abs_diff}"
                )
                for i, message in enumerate(messages):  # noqa: PLW2901
                    logger.info(f"\t{i}: Name={ort_session.get_outputs()[i].name}, {message}")

            # Collect data for debugging
            if enable_pickle_output and (numpy.isnan(max_abs_diff) or max_abs_diff > 100 * atol):
                Gpt2Helper.save_inputs(i, dummy_inputs)
                Gpt2Helper.save_outputs(i, ort_outputs, outputs)

        if max_abs_diff_list:
            result = {
                f"max_diff_percentile_{p}": f"{numpy.percentile(max_abs_diff_list, p):.5f}" for p in [50, 90, 95, 99]
            }
        else:
            result = {f"max_diff_percentile_{p}": "nan" for p in [50, 90, 95, 99]}

        result["top1_match_rate"] = top1_matched_cases * 1.0 / total_test_cases
        result["top1_match_rate_per_run"] = [x * 1.0 / test_cases_per_run for x in top1_matched_cases_per_run]
        result["diff_pass_rate"] = passed_test_cases * 1.0 / total_test_cases
        result["nan_rate"] = (total_test_cases - len(max_abs_diff_list)) * 1.0 / total_test_cases

        logger.info(
            f"Parity Test Cases={total_test_cases}; Passed={passed_test_cases}; Nan={total_test_cases - len(max_abs_diff_list)}; Top1_Matched={top1_matched_cases}"
        )

        if passed_test_cases > 0.95 * total_test_cases:
            logger.info(f"Parity is good: passed rate={int(passed_test_cases * 100 / total_test_cases):.0f}%")

        return result

    @staticmethod
    def test_performance(
        ort_session,
        model,
        device,
        is_float16=False,
        total_runs=100,
        use_io_binding=True,
        model_class="GPT2LMHeadModel",
        has_position_ids=True,
        has_attention_mask=True,
        input_ids_dtype=torch.int32,
        position_ids_dtype=torch.int32,
        attention_mask_dtype=torch.int32,
        batch_size=8,
        sequence_length=1,
        past_sequence_length=32,
    ):
        """Generate random inputs and measure average latency of Onnx Runtime."""

        config: GPT2Config = model.config

        output_buffers = None
        if use_io_binding:
            output_shapes = Gpt2Helper.get_output_shapes(
                batch_size, past_sequence_length, sequence_length, config, model_class
            )
            output_buffers = Gpt2Helper.get_output_buffers(output_shapes, device, is_float16)

        dummy_inputs = Gpt2Helper.get_dummy_inputs(
            batch_size,
            past_sequence_length,
            sequence_length,
            config.num_attention_heads,
            config.hidden_size,
            config.n_layer,
            config.vocab_size,
            device,
            is_float16,
            has_position_ids,
            has_attention_mask,
            input_ids_dtype=input_ids_dtype,
            position_ids_dtype=position_ids_dtype,
            attention_mask_dtype=attention_mask_dtype,
        )

        if use_io_binding:
            _, latency = Gpt2Helper.onnxruntime_inference(ort_session, dummy_inputs, total_runs)
        else:
            _, latency = Gpt2Helper.onnxruntime_inference_with_binded_io(
                ort_session, dummy_inputs, output_buffers, output_shapes, total_runs
            )

        return latency

    @staticmethod
    def torchscript(model, config, device, has_position_ids=True, has_attention_mask=True):
        """JIT trace for TorchScript."""
        input_list = Gpt2Helper.get_dummy_inputs(
            batch_size=1,
            past_sequence_length=1,
            sequence_length=1,
            num_attention_heads=config.num_attention_heads,
            hidden_size=config.hidden_size,
            num_layer=config.n_layer,
            vocab_size=config.vocab_size,
            device=device,
            float16=False,
            has_position_ids=has_position_ids,
            has_attention_mask=has_attention_mask,
        ).to_list()
        return torch.jit.trace(model, input_list)

    @staticmethod
    def get_onnx_paths(
        output_dir,
        model_name_or_path,
        model_class: str = "GPT2LMHeadModel",
        has_past=True,
        new_folder=False,
        remove_existing=["raw", "fp32", "fp16", "int8"],  # noqa: B006
    ):
        """Build a  path name for given model based on given attributes."""
        model_name = model_name_or_path
        if os.path.isdir(model_name_or_path):
            model_name = Path(model_name_or_path).parts[-1]
        else:
            model_name.split("/")[-1]

        if model_class != "GPT2LMHeadModel":
            model_name += "_" + model_class

        if has_past:
            model_name += "_past"

        if new_folder:
            suffix = {"raw": "", "fp32": "_fp32", "fp16": "_fp16", "int8": "_int8"}
            # Remove the directories if existed.
            for model_type in ["raw", "fp32", "fp16", "int8"]:
                new_dir = os.path.join(output_dir, model_name + suffix[model_type])
                if os.path.exists(new_dir):
                    if model_type in remove_existing:
                        try:
                            shutil.rmtree(new_dir)
                            logger.info(f"Removed the existed directory: {new_dir}")
                        except OSError as e:
                            logger.info(f"Failed to remove the directory {new_dir}: {e.strerror}")
                    else:
                        logger.info(f"Directory for {model_type} existed: {new_dir}")

            # store each model to its own directory (for external data format).
            return {
                "raw": os.path.join(os.path.join(output_dir, model_name), model_name + ".onnx"),
                "fp32": os.path.join(
                    os.path.join(output_dir, model_name + "_fp32"),
                    model_name + "_fp32.onnx",
                ),
                "fp16": os.path.join(
                    os.path.join(output_dir, model_name + "_fp16"),
                    model_name + "_fp16.onnx",
                ),
                "int8": os.path.join(
                    os.path.join(output_dir, model_name + "_int8"),
                    model_name + "_int8.onnx",
                ),
            }

        return {
            "raw": os.path.join(output_dir, model_name + ".onnx"),
            "fp32": os.path.join(output_dir, model_name + "_fp32.onnx"),
            "fp16": os.path.join(output_dir, model_name + "_fp16.onnx"),
            "int8": os.path.join(output_dir, model_name + "_int8.onnx"),
        }

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\distlib\metadata.py ===
# -*- coding: utf-8 -*-
#
# Copyright (C) 2012 The Python Software Foundation.
# See LICENSE.txt and CONTRIBUTORS.txt.
#
"""Implementation of the Metadata for Python packages PEPs.

Supports all metadata formats (1.0, 1.1, 1.2, 1.3/2.1 and 2.2).
"""
from __future__ import unicode_literals

import codecs
from email import message_from_file
import json
import logging
import re

from . import DistlibException, __version__
from .compat import StringIO, string_types, text_type
from .markers import interpret
from .util import extract_by_key, get_extras
from .version import get_scheme, PEP440_VERSION_RE

logger = logging.getLogger(__name__)


class MetadataMissingError(DistlibException):
    """A required metadata is missing"""


class MetadataConflictError(DistlibException):
    """Attempt to read or write metadata fields that are conflictual."""


class MetadataUnrecognizedVersionError(DistlibException):
    """Unknown metadata version number."""


class MetadataInvalidError(DistlibException):
    """A metadata value is invalid"""


# public API of this module
__all__ = ['Metadata', 'PKG_INFO_ENCODING', 'PKG_INFO_PREFERRED_VERSION']

# Encoding used for the PKG-INFO files
PKG_INFO_ENCODING = 'utf-8'

# preferred version. Hopefully will be changed
# to 1.2 once PEP 345 is supported everywhere
PKG_INFO_PREFERRED_VERSION = '1.1'

_LINE_PREFIX_1_2 = re.compile('\n       \\|')
_LINE_PREFIX_PRE_1_2 = re.compile('\n        ')
_241_FIELDS = ('Metadata-Version', 'Name', 'Version', 'Platform', 'Summary', 'Description', 'Keywords', 'Home-page',
               'Author', 'Author-email', 'License')

_314_FIELDS = ('Metadata-Version', 'Name', 'Version', 'Platform', 'Supported-Platform', 'Summary', 'Description',
               'Keywords', 'Home-page', 'Author', 'Author-email', 'License', 'Classifier', 'Download-URL', 'Obsoletes',
               'Provides', 'Requires')

_314_MARKERS = ('Obsoletes', 'Provides', 'Requires', 'Classifier', 'Download-URL')

_345_FIELDS = ('Metadata-Version', 'Name', 'Version', 'Platform', 'Supported-Platform', 'Summary', 'Description',
               'Keywords', 'Home-page', 'Author', 'Author-email', 'Maintainer', 'Maintainer-email', 'License',
               'Classifier', 'Download-URL', 'Obsoletes-Dist', 'Project-URL', 'Provides-Dist', 'Requires-Dist',
               'Requires-Python', 'Requires-External')

_345_MARKERS = ('Provides-Dist', 'Requires-Dist', 'Requires-Python', 'Obsoletes-Dist', 'Requires-External',
                'Maintainer', 'Maintainer-email', 'Project-URL')

_426_FIELDS = ('Metadata-Version', 'Name', 'Version', 'Platform', 'Supported-Platform', 'Summary', 'Description',
               'Keywords', 'Home-page', 'Author', 'Author-email', 'Maintainer', 'Maintainer-email', 'License',
               'Classifier', 'Download-URL', 'Obsoletes-Dist', 'Project-URL', 'Provides-Dist', 'Requires-Dist',
               'Requires-Python', 'Requires-External', 'Private-Version', 'Obsoleted-By', 'Setup-Requires-Dist',
               'Extension', 'Provides-Extra')

_426_MARKERS = ('Private-Version', 'Provides-Extra', 'Obsoleted-By', 'Setup-Requires-Dist', 'Extension')

# See issue #106: Sometimes 'Requires' and 'Provides' occur wrongly in
# the metadata. Include them in the tuple literal below to allow them
# (for now).
# Ditto for Obsoletes - see issue #140.
_566_FIELDS = _426_FIELDS + ('Description-Content-Type', 'Requires', 'Provides', 'Obsoletes')

_566_MARKERS = ('Description-Content-Type', )

_643_MARKERS = ('Dynamic', 'License-File')

_643_FIELDS = _566_FIELDS + _643_MARKERS

_ALL_FIELDS = set()
_ALL_FIELDS.update(_241_FIELDS)
_ALL_FIELDS.update(_314_FIELDS)
_ALL_FIELDS.update(_345_FIELDS)
_ALL_FIELDS.update(_426_FIELDS)
_ALL_FIELDS.update(_566_FIELDS)
_ALL_FIELDS.update(_643_FIELDS)

EXTRA_RE = re.compile(r'''extra\s*==\s*("([^"]+)"|'([^']+)')''')


def _version2fieldlist(version):
    if version == '1.0':
        return _241_FIELDS
    elif version == '1.1':
        return _314_FIELDS
    elif version == '1.2':
        return _345_FIELDS
    elif version in ('1.3', '2.1'):
        # avoid adding field names if already there
        return _345_FIELDS + tuple(f for f in _566_FIELDS if f not in _345_FIELDS)
    elif version == '2.0':
        raise ValueError('Metadata 2.0 is withdrawn and not supported')
        # return _426_FIELDS
    elif version == '2.2':
        return _643_FIELDS
    raise MetadataUnrecognizedVersionError(version)


def _best_version(fields):
    """Detect the best version depending on the fields used."""

    def _has_marker(keys, markers):
        return any(marker in keys for marker in markers)

    keys = [key for key, value in fields.items() if value not in ([], 'UNKNOWN', None)]
    possible_versions = ['1.0', '1.1', '1.2', '1.3', '2.1', '2.2']  # 2.0 removed

    # first let's try to see if a field is not part of one of the version
    for key in keys:
        if key not in _241_FIELDS and '1.0' in possible_versions:
            possible_versions.remove('1.0')
            logger.debug('Removed 1.0 due to %s', key)
        if key not in _314_FIELDS and '1.1' in possible_versions:
            possible_versions.remove('1.1')
            logger.debug('Removed 1.1 due to %s', key)
        if key not in _345_FIELDS and '1.2' in possible_versions:
            possible_versions.remove('1.2')
            logger.debug('Removed 1.2 due to %s', key)
        if key not in _566_FIELDS and '1.3' in possible_versions:
            possible_versions.remove('1.3')
            logger.debug('Removed 1.3 due to %s', key)
        if key not in _566_FIELDS and '2.1' in possible_versions:
            if key != 'Description':  # In 2.1, description allowed after headers
                possible_versions.remove('2.1')
                logger.debug('Removed 2.1 due to %s', key)
        if key not in _643_FIELDS and '2.2' in possible_versions:
            possible_versions.remove('2.2')
            logger.debug('Removed 2.2 due to %s', key)
        # if key not in _426_FIELDS and '2.0' in possible_versions:
        # possible_versions.remove('2.0')
        # logger.debug('Removed 2.0 due to %s', key)

    # possible_version contains qualified versions
    if len(possible_versions) == 1:
        return possible_versions[0]  # found !
    elif len(possible_versions) == 0:
        logger.debug('Out of options - unknown metadata set: %s', fields)
        raise MetadataConflictError('Unknown metadata set')

    # let's see if one unique marker is found
    is_1_1 = '1.1' in possible_versions and _has_marker(keys, _314_MARKERS)
    is_1_2 = '1.2' in possible_versions and _has_marker(keys, _345_MARKERS)
    is_2_1 = '2.1' in possible_versions and _has_marker(keys, _566_MARKERS)
    # is_2_0 = '2.0' in possible_versions and _has_marker(keys, _426_MARKERS)
    is_2_2 = '2.2' in possible_versions and _has_marker(keys, _643_MARKERS)
    if int(is_1_1) + int(is_1_2) + int(is_2_1) + int(is_2_2) > 1:
        raise MetadataConflictError('You used incompatible 1.1/1.2/2.1/2.2 fields')

    # we have the choice, 1.0, or 1.2, 2.1 or 2.2
    #   - 1.0 has a broken Summary field but works with all tools
    #   - 1.1 is to avoid
    #   - 1.2 fixes Summary but has little adoption
    #   - 2.1 adds more features
    #   - 2.2 is the latest
    if not is_1_1 and not is_1_2 and not is_2_1 and not is_2_2:
        # we couldn't find any specific marker
        if PKG_INFO_PREFERRED_VERSION in possible_versions:
            return PKG_INFO_PREFERRED_VERSION
    if is_1_1:
        return '1.1'
    if is_1_2:
        return '1.2'
    if is_2_1:
        return '2.1'
    # if is_2_2:
    # return '2.2'

    return '2.2'


# This follows the rules about transforming keys as described in
# https://www.python.org/dev/peps/pep-0566/#id17
_ATTR2FIELD = {name.lower().replace("-", "_"): name for name in _ALL_FIELDS}
_FIELD2ATTR = {field: attr for attr, field in _ATTR2FIELD.items()}

_PREDICATE_FIELDS = ('Requires-Dist', 'Obsoletes-Dist', 'Provides-Dist')
_VERSIONS_FIELDS = ('Requires-Python', )
_VERSION_FIELDS = ('Version', )
_LISTFIELDS = ('Platform', 'Classifier', 'Obsoletes', 'Requires', 'Provides', 'Obsoletes-Dist', 'Provides-Dist',
               'Requires-Dist', 'Requires-External', 'Project-URL', 'Supported-Platform', 'Setup-Requires-Dist',
               'Provides-Extra', 'Extension', 'License-File')
_LISTTUPLEFIELDS = ('Project-URL', )

_ELEMENTSFIELD = ('Keywords', )

_UNICODEFIELDS = ('Author', 'Maintainer', 'Summary', 'Description')

_MISSING = object()

_FILESAFE = re.compile('[^A-Za-z0-9.]+')


def _get_name_and_version(name, version, for_filename=False):
    """Return the distribution name with version.

    If for_filename is true, return a filename-escaped form."""
    if for_filename:
        # For both name and version any runs of non-alphanumeric or '.'
        # characters are replaced with a single '-'.  Additionally any
        # spaces in the version string become '.'
        name = _FILESAFE.sub('-', name)
        version = _FILESAFE.sub('-', version.replace(' ', '.'))
    return '%s-%s' % (name, version)


class LegacyMetadata(object):
    """The legacy metadata of a release.

    Supports versions 1.0, 1.1, 1.2, 2.0 and 1.3/2.1 (auto-detected). You can
    instantiate the class with one of these arguments (or none):
    - *path*, the path to a metadata file
    - *fileobj* give a file-like object with metadata as content
    - *mapping* is a dict-like object
    - *scheme* is a version scheme name
    """

    # TODO document the mapping API and UNKNOWN default key

    def __init__(self, path=None, fileobj=None, mapping=None, scheme='default'):
        if [path, fileobj, mapping].count(None) < 2:
            raise TypeError('path, fileobj and mapping are exclusive')
        self._fields = {}
        self.requires_files = []
        self._dependencies = None
        self.scheme = scheme
        if path is not None:
            self.read(path)
        elif fileobj is not None:
            self.read_file(fileobj)
        elif mapping is not None:
            self.update(mapping)
            self.set_metadata_version()

    def set_metadata_version(self):
        self._fields['Metadata-Version'] = _best_version(self._fields)

    def _write_field(self, fileobj, name, value):
        fileobj.write('%s: %s\n' % (name, value))

    def __getitem__(self, name):
        return self.get(name)

    def __setitem__(self, name, value):
        return self.set(name, value)

    def __delitem__(self, name):
        field_name = self._convert_name(name)
        try:
            del self._fields[field_name]
        except KeyError:
            raise KeyError(name)

    def __contains__(self, name):
        return (name in self._fields or self._convert_name(name) in self._fields)

    def _convert_name(self, name):
        if name in _ALL_FIELDS:
            return name
        name = name.replace('-', '_').lower()
        return _ATTR2FIELD.get(name, name)

    def _default_value(self, name):
        if name in _LISTFIELDS or name in _ELEMENTSFIELD:
            return []
        return 'UNKNOWN'

    def _remove_line_prefix(self, value):
        if self.metadata_version in ('1.0', '1.1'):
            return _LINE_PREFIX_PRE_1_2.sub('\n', value)
        else:
            return _LINE_PREFIX_1_2.sub('\n', value)

    def __getattr__(self, name):
        if name in _ATTR2FIELD:
            return self[name]
        raise AttributeError(name)

    #
    # Public API
    #

    def get_fullname(self, filesafe=False):
        """
        Return the distribution name with version.

        If filesafe is true, return a filename-escaped form.
        """
        return _get_name_and_version(self['Name'], self['Version'], filesafe)

    def is_field(self, name):
        """return True if name is a valid metadata key"""
        name = self._convert_name(name)
        return name in _ALL_FIELDS

    def is_multi_field(self, name):
        name = self._convert_name(name)
        return name in _LISTFIELDS

    def read(self, filepath):
        """Read the metadata values from a file path."""
        fp = codecs.open(filepath, 'r', encoding='utf-8')
        try:
            self.read_file(fp)
        finally:
            fp.close()

    def read_file(self, fileob):
        """Read the metadata values from a file object."""
        msg = message_from_file(fileob)
        self._fields['Metadata-Version'] = msg['metadata-version']

        # When reading, get all the fields we can
        for field in _ALL_FIELDS:
            if field not in msg:
                continue
            if field in _LISTFIELDS:
                # we can have multiple lines
                values = msg.get_all(field)
                if field in _LISTTUPLEFIELDS and values is not None:
                    values = [tuple(value.split(',')) for value in values]
                self.set(field, values)
            else:
                # single line
                value = msg[field]
                if value is not None and value != 'UNKNOWN':
                    self.set(field, value)

        # PEP 566 specifies that the body be used for the description, if
        # available
        body = msg.get_payload()
        self["Description"] = body if body else self["Description"]
        # logger.debug('Attempting to set metadata for %s', self)
        # self.set_metadata_version()

    def write(self, filepath, skip_unknown=False):
        """Write the metadata fields to filepath."""
        fp = codecs.open(filepath, 'w', encoding='utf-8')
        try:
            self.write_file(fp, skip_unknown)
        finally:
            fp.close()

    def write_file(self, fileobject, skip_unknown=False):
        """Write the PKG-INFO format data to a file object."""
        self.set_metadata_version()

        for field in _version2fieldlist(self['Metadata-Version']):
            values = self.get(field)
            if skip_unknown and values in ('UNKNOWN', [], ['UNKNOWN']):
                continue
            if field in _ELEMENTSFIELD:
                self._write_field(fileobject, field, ','.join(values))
                continue
            if field not in _LISTFIELDS:
                if field == 'Description':
                    if self.metadata_version in ('1.0', '1.1'):
                        values = values.replace('\n', '\n        ')
                    else:
                        values = values.replace('\n', '\n       |')
                values = [values]

            if field in _LISTTUPLEFIELDS:
                values = [','.join(value) for value in values]

            for value in values:
                self._write_field(fileobject, field, value)

    def update(self, other=None, **kwargs):
        """Set metadata values from the given iterable `other` and kwargs.

        Behavior is like `dict.update`: If `other` has a ``keys`` method,
        they are looped over and ``self[key]`` is assigned ``other[key]``.
        Else, ``other`` is an iterable of ``(key, value)`` iterables.

        Keys that don't match a metadata field or that have an empty value are
        dropped.
        """

        def _set(key, value):
            if key in _ATTR2FIELD and value:
                self.set(self._convert_name(key), value)

        if not other:
            # other is None or empty container
            pass
        elif hasattr(other, 'keys'):
            for k in other.keys():
                _set(k, other[k])
        else:
            for k, v in other:
                _set(k, v)

        if kwargs:
            for k, v in kwargs.items():
                _set(k, v)

    def set(self, name, value):
        """Control then set a metadata field."""
        name = self._convert_name(name)

        if ((name in _ELEMENTSFIELD or name == 'Platform') and not isinstance(value, (list, tuple))):
            if isinstance(value, string_types):
                value = [v.strip() for v in value.split(',')]
            else:
                value = []
        elif (name in _LISTFIELDS and not isinstance(value, (list, tuple))):
            if isinstance(value, string_types):
                value = [value]
            else:
                value = []

        if logger.isEnabledFor(logging.WARNING):
            project_name = self['Name']

            scheme = get_scheme(self.scheme)
            if name in _PREDICATE_FIELDS and value is not None:
                for v in value:
                    # check that the values are valid
                    if not scheme.is_valid_matcher(v.split(';')[0]):
                        logger.warning("'%s': '%s' is not valid (field '%s')", project_name, v, name)
            # FIXME this rejects UNKNOWN, is that right?
            elif name in _VERSIONS_FIELDS and value is not None:
                if not scheme.is_valid_constraint_list(value):
                    logger.warning("'%s': '%s' is not a valid version (field '%s')", project_name, value, name)
            elif name in _VERSION_FIELDS and value is not None:
                if not scheme.is_valid_version(value):
                    logger.warning("'%s': '%s' is not a valid version (field '%s')", project_name, value, name)

        if name in _UNICODEFIELDS:
            if name == 'Description':
                value = self._remove_line_prefix(value)

        self._fields[name] = value

    def get(self, name, default=_MISSING):
        """Get a metadata field."""
        name = self._convert_name(name)
        if name not in self._fields:
            if default is _MISSING:
                default = self._default_value(name)
            return default
        if name in _UNICODEFIELDS:
            value = self._fields[name]
            return value
        elif name in _LISTFIELDS:
            value = self._fields[name]
            if value is None:
                return []
            res = []
            for val in value:
                if name not in _LISTTUPLEFIELDS:
                    res.append(val)
                else:
                    # That's for Project-URL
                    res.append((val[0], val[1]))
            return res

        elif name in _ELEMENTSFIELD:
            value = self._fields[name]
            if isinstance(value, string_types):
                return value.split(',')
        return self._fields[name]

    def check(self, strict=False):
        """Check if the metadata is compliant. If strict is True then raise if
        no Name or Version are provided"""
        self.set_metadata_version()

        # XXX should check the versions (if the file was loaded)
        missing, warnings = [], []

        for attr in ('Name', 'Version'):  # required by PEP 345
            if attr not in self:
                missing.append(attr)

        if strict and missing != []:
            msg = 'missing required metadata: %s' % ', '.join(missing)
            raise MetadataMissingError(msg)

        for attr in ('Home-page', 'Author'):
            if attr not in self:
                missing.append(attr)

        # checking metadata 1.2 (XXX needs to check 1.1, 1.0)
        if self['Metadata-Version'] != '1.2':
            return missing, warnings

        scheme = get_scheme(self.scheme)

        def are_valid_constraints(value):
            for v in value:
                if not scheme.is_valid_matcher(v.split(';')[0]):
                    return False
            return True

        for fields, controller in ((_PREDICATE_FIELDS, are_valid_constraints),
                                   (_VERSIONS_FIELDS, scheme.is_valid_constraint_list), (_VERSION_FIELDS,
                                                                                         scheme.is_valid_version)):
            for field in fields:
                value = self.get(field, None)
                if value is not None and not controller(value):
                    warnings.append("Wrong value for '%s': %s" % (field, value))

        return missing, warnings

    def todict(self, skip_missing=False):
        """Return fields as a dict.

        Field names will be converted to use the underscore-lowercase style
        instead of hyphen-mixed case (i.e. home_page instead of Home-page).
        This is as per https://www.python.org/dev/peps/pep-0566/#id17.
        """
        self.set_metadata_version()

        fields = _version2fieldlist(self['Metadata-Version'])

        data = {}

        for field_name in fields:
            if not skip_missing or field_name in self._fields:
                key = _FIELD2ATTR[field_name]
                if key != 'project_url':
                    data[key] = self[field_name]
                else:
                    data[key] = [','.join(u) for u in self[field_name]]

        return data

    def add_requirements(self, requirements):
        if self['Metadata-Version'] == '1.1':
            # we can't have 1.1 metadata *and* Setuptools requires
            for field in ('Obsoletes', 'Requires', 'Provides'):
                if field in self:
                    del self[field]
        self['Requires-Dist'] += requirements

    # Mapping API
    # TODO could add iter* variants

    def keys(self):
        return list(_version2fieldlist(self['Metadata-Version']))

    def __iter__(self):
        for key in self.keys():
            yield key

    def values(self):
        return [self[key] for key in self.keys()]

    def items(self):
        return [(key, self[key]) for key in self.keys()]

    def __repr__(self):
        return '<%s %s %s>' % (self.__class__.__name__, self.name, self.version)


METADATA_FILENAME = 'pydist.json'
WHEEL_METADATA_FILENAME = 'metadata.json'
LEGACY_METADATA_FILENAME = 'METADATA'


class Metadata(object):
    """
    The metadata of a release. This implementation uses 2.1
    metadata where possible. If not possible, it wraps a LegacyMetadata
    instance which handles the key-value metadata format.
    """

    METADATA_VERSION_MATCHER = re.compile(r'^\d+(\.\d+)*$')

    NAME_MATCHER = re.compile('^[0-9A-Z]([0-9A-Z_.-]*[0-9A-Z])?$', re.I)

    FIELDNAME_MATCHER = re.compile('^[A-Z]([0-9A-Z-]*[0-9A-Z])?$', re.I)

    VERSION_MATCHER = PEP440_VERSION_RE

    SUMMARY_MATCHER = re.compile('.{1,2047}')

    METADATA_VERSION = '2.0'

    GENERATOR = 'distlib (%s)' % __version__

    MANDATORY_KEYS = {
        'name': (),
        'version': (),
        'summary': ('legacy', ),
    }

    INDEX_KEYS = ('name version license summary description author '
                  'author_email keywords platform home_page classifiers '
                  'download_url')

    DEPENDENCY_KEYS = ('extras run_requires test_requires build_requires '
                       'dev_requires provides meta_requires obsoleted_by '
                       'supports_environments')

    SYNTAX_VALIDATORS = {
        'metadata_version': (METADATA_VERSION_MATCHER, ()),
        'name': (NAME_MATCHER, ('legacy', )),
        'version': (VERSION_MATCHER, ('legacy', )),
        'summary': (SUMMARY_MATCHER, ('legacy', )),
        'dynamic': (FIELDNAME_MATCHER, ('legacy', )),
    }

    __slots__ = ('_legacy', '_data', 'scheme')

    def __init__(self, path=None, fileobj=None, mapping=None, scheme='default'):
        if [path, fileobj, mapping].count(None) < 2:
            raise TypeError('path, fileobj and mapping are exclusive')
        self._legacy = None
        self._data = None
        self.scheme = scheme
        # import pdb; pdb.set_trace()
        if mapping is not None:
            try:
                self._validate_mapping(mapping, scheme)
                self._data = mapping
            except MetadataUnrecognizedVersionError:
                self._legacy = LegacyMetadata(mapping=mapping, scheme=scheme)
                self.validate()
        else:
            data = None
            if path:
                with open(path, 'rb') as f:
                    data = f.read()
            elif fileobj:
                data = fileobj.read()
            if data is None:
                # Initialised with no args - to be added
                self._data = {
                    'metadata_version': self.METADATA_VERSION,
                    'generator': self.GENERATOR,
                }
            else:
                if not isinstance(data, text_type):
                    data = data.decode('utf-8')
                try:
                    self._data = json.loads(data)
                    self._validate_mapping(self._data, scheme)
                except ValueError:
                    # Note: MetadataUnrecognizedVersionError does not
                    # inherit from ValueError (it's a DistlibException,
                    # which should not inherit from ValueError).
                    # The ValueError comes from the json.load - if that
                    # succeeds and we get a validation error, we want
                    # that to propagate
                    self._legacy = LegacyMetadata(fileobj=StringIO(data), scheme=scheme)
                    self.validate()

    common_keys = set(('name', 'version', 'license', 'keywords', 'summary'))

    none_list = (None, list)
    none_dict = (None, dict)

    mapped_keys = {
        'run_requires': ('Requires-Dist', list),
        'build_requires': ('Setup-Requires-Dist', list),
        'dev_requires': none_list,
        'test_requires': none_list,
        'meta_requires': none_list,
        'extras': ('Provides-Extra', list),
        'modules': none_list,
        'namespaces': none_list,
        'exports': none_dict,
        'commands': none_dict,
        'classifiers': ('Classifier', list),
        'source_url': ('Download-URL', None),
        'metadata_version': ('Metadata-Version', None),
    }

    del none_list, none_dict

    def __getattribute__(self, key):
        common = object.__getattribute__(self, 'common_keys')
        mapped = object.__getattribute__(self, 'mapped_keys')
        if key in mapped:
            lk, maker = mapped[key]
            if self._legacy:
                if lk is None:
                    result = None if maker is None else maker()
                else:
                    result = self._legacy.get(lk)
            else:
                value = None if maker is None else maker()
                if key not in ('commands', 'exports', 'modules', 'namespaces', 'classifiers'):
                    result = self._data.get(key, value)
                else:
                    # special cases for PEP 459
                    sentinel = object()
                    result = sentinel
                    d = self._data.get('extensions')
                    if d:
                        if key == 'commands':
                            result = d.get('python.commands', value)
                        elif key == 'classifiers':
                            d = d.get('python.details')
                            if d:
                                result = d.get(key, value)
                        else:
                            d = d.get('python.exports')
                            if not d:
                                d = self._data.get('python.exports')
                            if d:
                                result = d.get(key, value)
                    if result is sentinel:
                        result = value
        elif key not in common:
            result = object.__getattribute__(self, key)
        elif self._legacy:
            result = self._legacy.get(key)
        else:
            result = self._data.get(key)
        return result

    def _validate_value(self, key, value, scheme=None):
        if key in self.SYNTAX_VALIDATORS:
            pattern, exclusions = self.SYNTAX_VALIDATORS[key]
            if (scheme or self.scheme) not in exclusions:
                m = pattern.match(value)
                if not m:
                    raise MetadataInvalidError("'%s' is an invalid value for "
                                               "the '%s' property" % (value, key))

    def __setattr__(self, key, value):
        self._validate_value(key, value)
        common = object.__getattribute__(self, 'common_keys')
        mapped = object.__getattribute__(self, 'mapped_keys')
        if key in mapped:
            lk, _ = mapped[key]
            if self._legacy:
                if lk is None:
                    raise NotImplementedError
                self._legacy[lk] = value
            elif key not in ('commands', 'exports', 'modules', 'namespaces', 'classifiers'):
                self._data[key] = value
            else:
                # special cases for PEP 459
                d = self._data.setdefault('extensions', {})
                if key == 'commands':
                    d['python.commands'] = value
                elif key == 'classifiers':
                    d = d.setdefault('python.details', {})
                    d[key] = value
                else:
                    d = d.setdefault('python.exports', {})
                    d[key] = value
        elif key not in common:
            object.__setattr__(self, key, value)
        else:
            if key == 'keywords':
                if isinstance(value, string_types):
                    value = value.strip()
                    if value:
                        value = value.split()
                    else:
                        value = []
            if self._legacy:
                self._legacy[key] = value
            else:
                self._data[key] = value

    @property
    def name_and_version(self):
        return _get_name_and_version(self.name, self.version, True)

    @property
    def provides(self):
        if self._legacy:
            result = self._legacy['Provides-Dist']
        else:
            result = self._data.setdefault('provides', [])
        s = '%s (%s)' % (self.name, self.version)
        if s not in result:
            result.append(s)
        return result

    @provides.setter
    def provides(self, value):
        if self._legacy:
            self._legacy['Provides-Dist'] = value
        else:
            self._data['provides'] = value

    def get_requirements(self, reqts, extras=None, env=None):
        """
        Base method to get dependencies, given a set of extras
        to satisfy and an optional environment context.
        :param reqts: A list of sometimes-wanted dependencies,
                      perhaps dependent on extras and environment.
        :param extras: A list of optional components being requested.
        :param env: An optional environment for marker evaluation.
        """
        if self._legacy:
            result = reqts
        else:
            result = []
            extras = get_extras(extras or [], self.extras)
            for d in reqts:
                if 'extra' not in d and 'environment' not in d:
                    # unconditional
                    include = True
                else:
                    if 'extra' not in d:
                        # Not extra-dependent - only environment-dependent
                        include = True
                    else:
                        include = d.get('extra') in extras
                    if include:
                        # Not excluded because of extras, check environment
                        marker = d.get('environment')
                        if marker:
                            include = interpret(marker, env)
                if include:
                    result.extend(d['requires'])
            for key in ('build', 'dev', 'test'):
                e = ':%s:' % key
                if e in extras:
                    extras.remove(e)
                    # A recursive call, but it should terminate since 'test'
                    # has been removed from the extras
                    reqts = self._data.get('%s_requires' % key, [])
                    result.extend(self.get_requirements(reqts, extras=extras, env=env))
        return result

    @property
    def dictionary(self):
        if self._legacy:
            return self._from_legacy()
        return self._data

    @property
    def dependencies(self):
        if self._legacy:
            raise NotImplementedError
        else:
            return extract_by_key(self._data, self.DEPENDENCY_KEYS)

    @dependencies.setter
    def dependencies(self, value):
        if self._legacy:
            raise NotImplementedError
        else:
            self._data.update(value)

    def _validate_mapping(self, mapping, scheme):
        if mapping.get('metadata_version') != self.METADATA_VERSION:
            raise MetadataUnrecognizedVersionError()
        missing = []
        for key, exclusions in self.MANDATORY_KEYS.items():
            if key not in mapping:
                if scheme not in exclusions:
                    missing.append(key)
        if missing:
            msg = 'Missing metadata items: %s' % ', '.join(missing)
            raise MetadataMissingError(msg)
        for k, v in mapping.items():
            self._validate_value(k, v, scheme)

    def validate(self):
        if self._legacy:
            missing, warnings = self._legacy.check(True)
            if missing or warnings:
                logger.warning('Metadata: missing: %s, warnings: %s', missing, warnings)
        else:
            self._validate_mapping(self._data, self.scheme)

    def todict(self):
        if self._legacy:
            return self._legacy.todict(True)
        else:
            result = extract_by_key(self._data, self.INDEX_KEYS)
            return result

    def _from_legacy(self):
        assert self._legacy and not self._data
        result = {
            'metadata_version': self.METADATA_VERSION,
            'generator': self.GENERATOR,
        }
        lmd = self._legacy.todict(True)  # skip missing ones
        for k in ('name', 'version', 'license', 'summary', 'description', 'classifier'):
            if k in lmd:
                if k == 'classifier':
                    nk = 'classifiers'
                else:
                    nk = k
                result[nk] = lmd[k]
        kw = lmd.get('Keywords', [])
        if kw == ['']:
            kw = []
        result['keywords'] = kw
        keys = (('requires_dist', 'run_requires'), ('setup_requires_dist', 'build_requires'))
        for ok, nk in keys:
            if ok in lmd and lmd[ok]:
                result[nk] = [{'requires': lmd[ok]}]
        result['provides'] = self.provides
        # author = {}
        # maintainer = {}
        return result

    LEGACY_MAPPING = {
        'name': 'Name',
        'version': 'Version',
        ('extensions', 'python.details', 'license'): 'License',
        'summary': 'Summary',
        'description': 'Description',
        ('extensions', 'python.project', 'project_urls', 'Home'): 'Home-page',
        ('extensions', 'python.project', 'contacts', 0, 'name'): 'Author',
        ('extensions', 'python.project', 'contacts', 0, 'email'): 'Author-email',
        'source_url': 'Download-URL',
        ('extensions', 'python.details', 'classifiers'): 'Classifier',
    }

    def _to_legacy(self):

        def process_entries(entries):
            reqts = set()
            for e in entries:
                extra = e.get('extra')
                env = e.get('environment')
                rlist = e['requires']
                for r in rlist:
                    if not env and not extra:
                        reqts.add(r)
                    else:
                        marker = ''
                        if extra:
                            marker = 'extra == "%s"' % extra
                        if env:
                            if marker:
                                marker = '(%s) and %s' % (env, marker)
                            else:
                                marker = env
                        reqts.add(';'.join((r, marker)))
            return reqts

        assert self._data and not self._legacy
        result = LegacyMetadata()
        nmd = self._data
        # import pdb; pdb.set_trace()
        for nk, ok in self.LEGACY_MAPPING.items():
            if not isinstance(nk, tuple):
                if nk in nmd:
                    result[ok] = nmd[nk]
            else:
                d = nmd
                found = True
                for k in nk:
                    try:
                        d = d[k]
                    except (KeyError, IndexError):
                        found = False
                        break
                if found:
                    result[ok] = d
        r1 = process_entries(self.run_requires + self.meta_requires)
        r2 = process_entries(self.build_requires + self.dev_requires)
        if self.extras:
            result['Provides-Extra'] = sorted(self.extras)
        result['Requires-Dist'] = sorted(r1)
        result['Setup-Requires-Dist'] = sorted(r2)
        # TODO: any other fields wanted
        return result

    def write(self, path=None, fileobj=None, legacy=False, skip_unknown=True):
        if [path, fileobj].count(None) != 1:
            raise ValueError('Exactly one of path and fileobj is needed')
        self.validate()
        if legacy:
            if self._legacy:
                legacy_md = self._legacy
            else:
                legacy_md = self._to_legacy()
            if path:
                legacy_md.write(path, skip_unknown=skip_unknown)
            else:
                legacy_md.write_file(fileobj, skip_unknown=skip_unknown)
        else:
            if self._legacy:
                d = self._from_legacy()
            else:
                d = self._data
            if fileobj:
                json.dump(d, fileobj, ensure_ascii=True, indent=2, sort_keys=True)
            else:
                with codecs.open(path, 'w', 'utf-8') as f:
                    json.dump(d, f, ensure_ascii=True, indent=2, sort_keys=True)

    def add_requirements(self, requirements):
        if self._legacy:
            self._legacy.add_requirements(requirements)
        else:
            run_requires = self._data.setdefault('run_requires', [])
            always = None
            for entry in run_requires:
                if 'environment' not in entry and 'extra' not in entry:
                    always = entry
                    break
            if always is None:
                always = {'requires': requirements}
                run_requires.insert(0, always)
            else:
                rset = set(always['requires']) | set(requirements)
                always['requires'] = sorted(rset)

    def __repr__(self):
        name = self.name or '(no name)'
        version = self.version or 'no version'
        return '<%s %s %s (%s)>' % (self.__class__.__name__, self.metadata_version, name, version)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\dialects\postgresql\ranges.py ===
# dialects/postgresql/ranges.py
# Copyright (C) 2013-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

from __future__ import annotations

import dataclasses
from datetime import date
from datetime import datetime
from datetime import timedelta
from decimal import Decimal
from typing import Any
from typing import cast
from typing import Generic
from typing import List
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

from .operators import ADJACENT_TO
from .operators import CONTAINED_BY
from .operators import CONTAINS
from .operators import NOT_EXTEND_LEFT_OF
from .operators import NOT_EXTEND_RIGHT_OF
from .operators import OVERLAP
from .operators import STRICTLY_LEFT_OF
from .operators import STRICTLY_RIGHT_OF
from ... import types as sqltypes
from ...sql import operators
from ...sql.type_api import TypeEngine
from ...util import py310
from ...util.typing import Literal

if TYPE_CHECKING:
    from ...sql.elements import ColumnElement
    from ...sql.type_api import _TE
    from ...sql.type_api import TypeEngineMixin

_T = TypeVar("_T", bound=Any)

_BoundsType = Literal["()", "[)", "(]", "[]"]

if py310:
    dc_slots = {"slots": True}
    dc_kwonly = {"kw_only": True}
else:
    dc_slots = {}
    dc_kwonly = {}


@dataclasses.dataclass(frozen=True, **dc_slots)
class Range(Generic[_T]):
    """Represent a PostgreSQL range.

    E.g.::

        r = Range(10, 50, bounds="()")

    The calling style is similar to that of psycopg and psycopg2, in part
    to allow easier migration from previous SQLAlchemy versions that used
    these objects directly.

    :param lower: Lower bound value, or None
    :param upper: Upper bound value, or None
    :param bounds: keyword-only, optional string value that is one of
     ``"()"``, ``"[)"``, ``"(]"``, ``"[]"``.  Defaults to ``"[)"``.
    :param empty: keyword-only, optional bool indicating this is an "empty"
     range

    .. versionadded:: 2.0

    """

    lower: Optional[_T] = None
    """the lower bound"""

    upper: Optional[_T] = None
    """the upper bound"""

    if TYPE_CHECKING:
        bounds: _BoundsType = dataclasses.field(default="[)")
        empty: bool = dataclasses.field(default=False)
    else:
        bounds: _BoundsType = dataclasses.field(default="[)", **dc_kwonly)
        empty: bool = dataclasses.field(default=False, **dc_kwonly)

    if not py310:

        def __init__(
            self,
            lower: Optional[_T] = None,
            upper: Optional[_T] = None,
            *,
            bounds: _BoundsType = "[)",
            empty: bool = False,
        ):
            # no __slots__ either so we can update dict
            self.__dict__.update(
                {
                    "lower": lower,
                    "upper": upper,
                    "bounds": bounds,
                    "empty": empty,
                }
            )

    def __bool__(self) -> bool:
        return not self.empty

    @property
    def isempty(self) -> bool:
        "A synonym for the 'empty' attribute."

        return self.empty

    @property
    def is_empty(self) -> bool:
        "A synonym for the 'empty' attribute."

        return self.empty

    @property
    def lower_inc(self) -> bool:
        """Return True if the lower bound is inclusive."""

        return self.bounds[0] == "["

    @property
    def lower_inf(self) -> bool:
        """Return True if this range is non-empty and lower bound is
        infinite."""

        return not self.empty and self.lower is None

    @property
    def upper_inc(self) -> bool:
        """Return True if the upper bound is inclusive."""

        return self.bounds[1] == "]"

    @property
    def upper_inf(self) -> bool:
        """Return True if this range is non-empty and the upper bound is
        infinite."""

        return not self.empty and self.upper is None

    @property
    def __sa_type_engine__(self) -> AbstractSingleRange[_T]:
        return AbstractSingleRange()

    def _contains_value(self, value: _T) -> bool:
        """Return True if this range contains the given value."""

        if self.empty:
            return False

        if self.lower is None:
            return self.upper is None or (
                value < self.upper
                if self.bounds[1] == ")"
                else value <= self.upper
            )

        if self.upper is None:
            return (  # type: ignore
                value > self.lower
                if self.bounds[0] == "("
                else value >= self.lower
            )

        return (  # type: ignore
            value > self.lower
            if self.bounds[0] == "("
            else value >= self.lower
        ) and (
            value < self.upper
            if self.bounds[1] == ")"
            else value <= self.upper
        )

    def _get_discrete_step(self) -> Any:
        "Determine the “step” for this range, if it is a discrete one."

        # See
        # https://www.postgresql.org/docs/current/rangetypes.html#RANGETYPES-DISCRETE
        # for the rationale

        if isinstance(self.lower, int) or isinstance(self.upper, int):
            return 1
        elif isinstance(self.lower, datetime) or isinstance(
            self.upper, datetime
        ):
            # This is required, because a `isinstance(datetime.now(), date)`
            # is True
            return None
        elif isinstance(self.lower, date) or isinstance(self.upper, date):
            return timedelta(days=1)
        else:
            return None

    def _compare_edges(
        self,
        value1: Optional[_T],
        bound1: str,
        value2: Optional[_T],
        bound2: str,
        only_values: bool = False,
    ) -> int:
        """Compare two range bounds.

        Return -1, 0 or 1 respectively when `value1` is less than,
        equal to or greater than `value2`.

        When `only_value` is ``True``, do not consider the *inclusivity*
        of the edges, just their values.
        """

        value1_is_lower_bound = bound1 in {"[", "("}
        value2_is_lower_bound = bound2 in {"[", "("}

        # Infinite edges are equal when they are on the same side,
        # otherwise a lower edge is considered less than the upper end
        if value1 is value2 is None:
            if value1_is_lower_bound == value2_is_lower_bound:
                return 0
            else:
                return -1 if value1_is_lower_bound else 1
        elif value1 is None:
            return -1 if value1_is_lower_bound else 1
        elif value2 is None:
            return 1 if value2_is_lower_bound else -1

        # Short path for trivial case
        if bound1 == bound2 and value1 == value2:
            return 0

        value1_inc = bound1 in {"[", "]"}
        value2_inc = bound2 in {"[", "]"}
        step = self._get_discrete_step()

        if step is not None:
            # "Normalize" the two edges as '[)', to simplify successive
            # logic when the range is discrete: otherwise we would need
            # to handle the comparison between ``(0`` and ``[1`` that
            # are equal when dealing with integers while for floats the
            # former is lesser than the latter

            if value1_is_lower_bound:
                if not value1_inc:
                    value1 += step
                    value1_inc = True
            else:
                if value1_inc:
                    value1 += step
                    value1_inc = False
            if value2_is_lower_bound:
                if not value2_inc:
                    value2 += step
                    value2_inc = True
            else:
                if value2_inc:
                    value2 += step
                    value2_inc = False

        if value1 < value2:  # type: ignore
            return -1
        elif value1 > value2:  # type: ignore
            return 1
        elif only_values:
            return 0
        else:
            # Neither one is infinite but are equal, so we
            # need to consider the respective inclusive/exclusive
            # flag

            if value1_inc and value2_inc:
                return 0
            elif not value1_inc and not value2_inc:
                if value1_is_lower_bound == value2_is_lower_bound:
                    return 0
                else:
                    return 1 if value1_is_lower_bound else -1
            elif not value1_inc:
                return 1 if value1_is_lower_bound else -1
            elif not value2_inc:
                return -1 if value2_is_lower_bound else 1
            else:
                return 0

    def __eq__(self, other: Any) -> bool:
        """Compare this range to the `other` taking into account
        bounds inclusivity, returning ``True`` if they are equal.
        """

        if not isinstance(other, Range):
            return NotImplemented

        if self.empty and other.empty:
            return True
        elif self.empty != other.empty:
            return False

        slower = self.lower
        slower_b = self.bounds[0]
        olower = other.lower
        olower_b = other.bounds[0]
        supper = self.upper
        supper_b = self.bounds[1]
        oupper = other.upper
        oupper_b = other.bounds[1]

        return (
            self._compare_edges(slower, slower_b, olower, olower_b) == 0
            and self._compare_edges(supper, supper_b, oupper, oupper_b) == 0
        )

    def contained_by(self, other: Range[_T]) -> bool:
        "Determine whether this range is a contained by `other`."

        # Any range contains the empty one
        if self.empty:
            return True

        # An empty range does not contain any range except the empty one
        if other.empty:
            return False

        slower = self.lower
        slower_b = self.bounds[0]
        olower = other.lower
        olower_b = other.bounds[0]

        if self._compare_edges(slower, slower_b, olower, olower_b) < 0:
            return False

        supper = self.upper
        supper_b = self.bounds[1]
        oupper = other.upper
        oupper_b = other.bounds[1]

        if self._compare_edges(supper, supper_b, oupper, oupper_b) > 0:
            return False

        return True

    def contains(self, value: Union[_T, Range[_T]]) -> bool:
        "Determine whether this range contains `value`."

        if isinstance(value, Range):
            return value.contained_by(self)
        else:
            return self._contains_value(value)

    __contains__ = contains

    def overlaps(self, other: Range[_T]) -> bool:
        "Determine whether this range overlaps with `other`."

        # Empty ranges never overlap with any other range
        if self.empty or other.empty:
            return False

        slower = self.lower
        slower_b = self.bounds[0]
        supper = self.upper
        supper_b = self.bounds[1]
        olower = other.lower
        olower_b = other.bounds[0]
        oupper = other.upper
        oupper_b = other.bounds[1]

        # Check whether this lower bound is contained in the other range
        if (
            self._compare_edges(slower, slower_b, olower, olower_b) >= 0
            and self._compare_edges(slower, slower_b, oupper, oupper_b) <= 0
        ):
            return True

        # Check whether other lower bound is contained in this range
        if (
            self._compare_edges(olower, olower_b, slower, slower_b) >= 0
            and self._compare_edges(olower, olower_b, supper, supper_b) <= 0
        ):
            return True

        return False

    def strictly_left_of(self, other: Range[_T]) -> bool:
        "Determine whether this range is completely to the left of `other`."

        # Empty ranges are neither to left nor to the right of any other range
        if self.empty or other.empty:
            return False

        supper = self.upper
        supper_b = self.bounds[1]
        olower = other.lower
        olower_b = other.bounds[0]

        # Check whether this upper edge is less than other's lower end
        return self._compare_edges(supper, supper_b, olower, olower_b) < 0

    __lshift__ = strictly_left_of

    def strictly_right_of(self, other: Range[_T]) -> bool:
        "Determine whether this range is completely to the right of `other`."

        # Empty ranges are neither to left nor to the right of any other range
        if self.empty or other.empty:
            return False

        slower = self.lower
        slower_b = self.bounds[0]
        oupper = other.upper
        oupper_b = other.bounds[1]

        # Check whether this lower edge is greater than other's upper end
        return self._compare_edges(slower, slower_b, oupper, oupper_b) > 0

    __rshift__ = strictly_right_of

    def not_extend_left_of(self, other: Range[_T]) -> bool:
        "Determine whether this does not extend to the left of `other`."

        # Empty ranges are neither to left nor to the right of any other range
        if self.empty or other.empty:
            return False

        slower = self.lower
        slower_b = self.bounds[0]
        olower = other.lower
        olower_b = other.bounds[0]

        # Check whether this lower edge is not less than other's lower end
        return self._compare_edges(slower, slower_b, olower, olower_b) >= 0

    def not_extend_right_of(self, other: Range[_T]) -> bool:
        "Determine whether this does not extend to the right of `other`."

        # Empty ranges are neither to left nor to the right of any other range
        if self.empty or other.empty:
            return False

        supper = self.upper
        supper_b = self.bounds[1]
        oupper = other.upper
        oupper_b = other.bounds[1]

        # Check whether this upper edge is not greater than other's upper end
        return self._compare_edges(supper, supper_b, oupper, oupper_b) <= 0

    def _upper_edge_adjacent_to_lower(
        self,
        value1: Optional[_T],
        bound1: str,
        value2: Optional[_T],
        bound2: str,
    ) -> bool:
        """Determine whether an upper bound is immediately successive to a
        lower bound."""

        # Since we need a peculiar way to handle the bounds inclusivity,
        # just do a comparison by value here
        res = self._compare_edges(value1, bound1, value2, bound2, True)
        if res == -1:
            step = self._get_discrete_step()
            if step is None:
                return False
            if bound1 == "]":
                if bound2 == "[":
                    return value1 == value2 - step  # type: ignore
                else:
                    return value1 == value2
            else:
                if bound2 == "[":
                    return value1 == value2
                else:
                    return value1 == value2 - step  # type: ignore
        elif res == 0:
            # Cover cases like [0,0] -|- [1,] and [0,2) -|- (1,3]
            if (
                bound1 == "]"
                and bound2 == "["
                or bound1 == ")"
                and bound2 == "("
            ):
                step = self._get_discrete_step()
                if step is not None:
                    return True
            return (
                bound1 == ")"
                and bound2 == "["
                or bound1 == "]"
                and bound2 == "("
            )
        else:
            return False

    def adjacent_to(self, other: Range[_T]) -> bool:
        "Determine whether this range is adjacent to the `other`."

        # Empty ranges are not adjacent to any other range
        if self.empty or other.empty:
            return False

        slower = self.lower
        slower_b = self.bounds[0]
        supper = self.upper
        supper_b = self.bounds[1]
        olower = other.lower
        olower_b = other.bounds[0]
        oupper = other.upper
        oupper_b = other.bounds[1]

        return self._upper_edge_adjacent_to_lower(
            supper, supper_b, olower, olower_b
        ) or self._upper_edge_adjacent_to_lower(
            oupper, oupper_b, slower, slower_b
        )

    def union(self, other: Range[_T]) -> Range[_T]:
        """Compute the union of this range with the `other`.

        This raises a ``ValueError`` exception if the two ranges are
        "disjunct", that is neither adjacent nor overlapping.
        """

        # Empty ranges are "additive identities"
        if self.empty:
            return other
        if other.empty:
            return self

        if not self.overlaps(other) and not self.adjacent_to(other):
            raise ValueError(
                "Adding non-overlapping and non-adjacent"
                " ranges is not implemented"
            )

        slower = self.lower
        slower_b = self.bounds[0]
        supper = self.upper
        supper_b = self.bounds[1]
        olower = other.lower
        olower_b = other.bounds[0]
        oupper = other.upper
        oupper_b = other.bounds[1]

        if self._compare_edges(slower, slower_b, olower, olower_b) < 0:
            rlower = slower
            rlower_b = slower_b
        else:
            rlower = olower
            rlower_b = olower_b

        if self._compare_edges(supper, supper_b, oupper, oupper_b) > 0:
            rupper = supper
            rupper_b = supper_b
        else:
            rupper = oupper
            rupper_b = oupper_b

        return Range(
            rlower, rupper, bounds=cast(_BoundsType, rlower_b + rupper_b)
        )

    def __add__(self, other: Range[_T]) -> Range[_T]:
        return self.union(other)

    def difference(self, other: Range[_T]) -> Range[_T]:
        """Compute the difference between this range and the `other`.

        This raises a ``ValueError`` exception if the two ranges are
        "disjunct", that is neither adjacent nor overlapping.
        """

        # Subtracting an empty range is a no-op
        if self.empty or other.empty:
            return self

        slower = self.lower
        slower_b = self.bounds[0]
        supper = self.upper
        supper_b = self.bounds[1]
        olower = other.lower
        olower_b = other.bounds[0]
        oupper = other.upper
        oupper_b = other.bounds[1]

        sl_vs_ol = self._compare_edges(slower, slower_b, olower, olower_b)
        su_vs_ou = self._compare_edges(supper, supper_b, oupper, oupper_b)
        if sl_vs_ol < 0 and su_vs_ou > 0:
            raise ValueError(
                "Subtracting a strictly inner range is not implemented"
            )

        sl_vs_ou = self._compare_edges(slower, slower_b, oupper, oupper_b)
        su_vs_ol = self._compare_edges(supper, supper_b, olower, olower_b)

        # If the ranges do not overlap, result is simply the first
        if sl_vs_ou > 0 or su_vs_ol < 0:
            return self

        # If this range is completely contained by the other, result is empty
        if sl_vs_ol >= 0 and su_vs_ou <= 0:
            return Range(None, None, empty=True)

        # If this range extends to the left of the other and ends in its
        # middle
        if sl_vs_ol <= 0 and su_vs_ol >= 0 and su_vs_ou <= 0:
            rupper_b = ")" if olower_b == "[" else "]"
            if (
                slower_b != "["
                and rupper_b != "]"
                and self._compare_edges(slower, slower_b, olower, rupper_b)
                == 0
            ):
                return Range(None, None, empty=True)
            else:
                return Range(
                    slower,
                    olower,
                    bounds=cast(_BoundsType, slower_b + rupper_b),
                )

        # If this range starts in the middle of the other and extends to its
        # right
        if sl_vs_ol >= 0 and su_vs_ou >= 0 and sl_vs_ou <= 0:
            rlower_b = "(" if oupper_b == "]" else "["
            if (
                rlower_b != "["
                and supper_b != "]"
                and self._compare_edges(oupper, rlower_b, supper, supper_b)
                == 0
            ):
                return Range(None, None, empty=True)
            else:
                return Range(
                    oupper,
                    supper,
                    bounds=cast(_BoundsType, rlower_b + supper_b),
                )

        assert False, f"Unhandled case computing {self} - {other}"

    def __sub__(self, other: Range[_T]) -> Range[_T]:
        return self.difference(other)

    def intersection(self, other: Range[_T]) -> Range[_T]:
        """Compute the intersection of this range with the `other`.

        .. versionadded:: 2.0.10

        """
        if self.empty or other.empty or not self.overlaps(other):
            return Range(None, None, empty=True)

        slower = self.lower
        slower_b = self.bounds[0]
        supper = self.upper
        supper_b = self.bounds[1]
        olower = other.lower
        olower_b = other.bounds[0]
        oupper = other.upper
        oupper_b = other.bounds[1]

        if self._compare_edges(slower, slower_b, olower, olower_b) < 0:
            rlower = olower
            rlower_b = olower_b
        else:
            rlower = slower
            rlower_b = slower_b

        if self._compare_edges(supper, supper_b, oupper, oupper_b) > 0:
            rupper = oupper
            rupper_b = oupper_b
        else:
            rupper = supper
            rupper_b = supper_b

        return Range(
            rlower,
            rupper,
            bounds=cast(_BoundsType, rlower_b + rupper_b),
        )

    def __mul__(self, other: Range[_T]) -> Range[_T]:
        return self.intersection(other)

    def __str__(self) -> str:
        return self._stringify()

    def _stringify(self) -> str:
        if self.empty:
            return "empty"

        l, r = self.lower, self.upper
        l = "" if l is None else l  # type: ignore
        r = "" if r is None else r  # type: ignore

        b0, b1 = cast("Tuple[str, str]", self.bounds)

        return f"{b0}{l},{r}{b1}"


class MultiRange(List[Range[_T]]):
    """Represents a multirange sequence.

    This list subclass is an utility to allow automatic type inference of
    the proper multi-range SQL type depending on the single range values.
    This is useful when operating on literal multi-ranges::

        import sqlalchemy as sa
        from sqlalchemy.dialects.postgresql import MultiRange, Range

        value = literal(MultiRange([Range(2, 4)]))

        select(tbl).where(tbl.c.value.op("@")(MultiRange([Range(-3, 7)])))

    .. versionadded:: 2.0.26

    .. seealso::

        - :ref:`postgresql_multirange_list_use`.
    """

    @property
    def __sa_type_engine__(self) -> AbstractMultiRange[_T]:
        return AbstractMultiRange()


class AbstractRange(sqltypes.TypeEngine[_T]):
    """Base class for single and multi Range SQL types."""

    render_bind_cast = True

    __abstract__ = True

    @overload
    def adapt(self, cls: Type[_TE], **kw: Any) -> _TE: ...

    @overload
    def adapt(
        self, cls: Type[TypeEngineMixin], **kw: Any
    ) -> TypeEngine[Any]: ...

    def adapt(
        self,
        cls: Type[Union[TypeEngine[Any], TypeEngineMixin]],
        **kw: Any,
    ) -> TypeEngine[Any]:
        """Dynamically adapt a range type to an abstract impl.

        For example ``INT4RANGE().adapt(_Psycopg2NumericRange)`` should
        produce a type that will have ``_Psycopg2NumericRange`` behaviors
        and also render as ``INT4RANGE`` in SQL and DDL.

        """
        if (
            issubclass(cls, (AbstractSingleRangeImpl, AbstractMultiRangeImpl))
            and cls is not self.__class__
        ):
            # two ways to do this are:  1. create a new type on the fly
            # or 2. have AbstractRangeImpl(visit_name) constructor and a
            # visit_abstract_range_impl() method in the PG compiler.
            # I'm choosing #1 as the resulting type object
            # will then make use of the same mechanics
            # as if we had made all these sub-types explicitly, and will
            # also look more obvious under pdb etc.
            # The adapt() operation here is cached per type-class-per-dialect,
            # so is not much of a performance concern
            visit_name = self.__visit_name__
            return type(  # type: ignore
                f"{visit_name}RangeImpl",
                (cls, self.__class__),
                {"__visit_name__": visit_name},
            )()
        else:
            return super().adapt(cls)

    class comparator_factory(TypeEngine.Comparator[Range[Any]]):
        """Define comparison operations for range types."""

        def contains(self, other: Any, **kw: Any) -> ColumnElement[bool]:
            """Boolean expression. Returns true if the right hand operand,
            which can be an element or a range, is contained within the
            column.

            kwargs may be ignored by this operator but are required for API
            conformance.
            """
            return self.expr.operate(CONTAINS, other)

        def contained_by(self, other: Any) -> ColumnElement[bool]:
            """Boolean expression. Returns true if the column is contained
            within the right hand operand.
            """
            return self.expr.operate(CONTAINED_BY, other)

        def overlaps(self, other: Any) -> ColumnElement[bool]:
            """Boolean expression. Returns true if the column overlaps
            (has points in common with) the right hand operand.
            """
            return self.expr.operate(OVERLAP, other)

        def strictly_left_of(self, other: Any) -> ColumnElement[bool]:
            """Boolean expression. Returns true if the column is strictly
            left of the right hand operand.
            """
            return self.expr.operate(STRICTLY_LEFT_OF, other)

        __lshift__ = strictly_left_of

        def strictly_right_of(self, other: Any) -> ColumnElement[bool]:
            """Boolean expression. Returns true if the column is strictly
            right of the right hand operand.
            """
            return self.expr.operate(STRICTLY_RIGHT_OF, other)

        __rshift__ = strictly_right_of

        def not_extend_right_of(self, other: Any) -> ColumnElement[bool]:
            """Boolean expression. Returns true if the range in the column
            does not extend right of the range in the operand.
            """
            return self.expr.operate(NOT_EXTEND_RIGHT_OF, other)

        def not_extend_left_of(self, other: Any) -> ColumnElement[bool]:
            """Boolean expression. Returns true if the range in the column
            does not extend left of the range in the operand.
            """
            return self.expr.operate(NOT_EXTEND_LEFT_OF, other)

        def adjacent_to(self, other: Any) -> ColumnElement[bool]:
            """Boolean expression. Returns true if the range in the column
            is adjacent to the range in the operand.
            """
            return self.expr.operate(ADJACENT_TO, other)

        def union(self, other: Any) -> ColumnElement[bool]:
            """Range expression. Returns the union of the two ranges.
            Will raise an exception if the resulting range is not
            contiguous.
            """
            return self.expr.operate(operators.add, other)

        def difference(self, other: Any) -> ColumnElement[bool]:
            """Range expression. Returns the union of the two ranges.
            Will raise an exception if the resulting range is not
            contiguous.
            """
            return self.expr.operate(operators.sub, other)

        def intersection(self, other: Any) -> ColumnElement[Range[_T]]:
            """Range expression. Returns the intersection of the two ranges.
            Will raise an exception if the resulting range is not
            contiguous.
            """
            return self.expr.operate(operators.mul, other)


class AbstractSingleRange(AbstractRange[Range[_T]]):
    """Base for PostgreSQL RANGE types.

    These are types that return a single :class:`_postgresql.Range` object.

    .. seealso::

        `PostgreSQL range functions <https://www.postgresql.org/docs/current/static/functions-range.html>`_

    """  # noqa: E501

    __abstract__ = True

    def _resolve_for_literal(self, value: Range[Any]) -> Any:
        spec = value.lower if value.lower is not None else value.upper

        if isinstance(spec, int):
            # pg is unreasonably picky here: the query
            # "select 1::INTEGER <@ '[1, 4)'::INT8RANGE" raises
            # "operator does not exist: integer <@ int8range" as of pg 16
            if _is_int32(value):
                return INT4RANGE()
            else:
                return INT8RANGE()
        elif isinstance(spec, (Decimal, float)):
            return NUMRANGE()
        elif isinstance(spec, datetime):
            return TSRANGE() if not spec.tzinfo else TSTZRANGE()
        elif isinstance(spec, date):
            return DATERANGE()
        else:
            # empty Range, SQL datatype can't be determined here
            return sqltypes.NULLTYPE


class AbstractSingleRangeImpl(AbstractSingleRange[_T]):
    """Marker for AbstractSingleRange that will apply a subclass-specific
    adaptation"""


class AbstractMultiRange(AbstractRange[Sequence[Range[_T]]]):
    """Base for PostgreSQL MULTIRANGE types.

    these are types that return a sequence of :class:`_postgresql.Range`
    objects.

    """

    __abstract__ = True

    def _resolve_for_literal(self, value: Sequence[Range[Any]]) -> Any:
        if not value:
            # empty MultiRange, SQL datatype can't be determined here
            return sqltypes.NULLTYPE
        first = value[0]
        spec = first.lower if first.lower is not None else first.upper

        if isinstance(spec, int):
            # pg is unreasonably picky here: the query
            # "select 1::INTEGER <@ '{[1, 4),[6,19)}'::INT8MULTIRANGE" raises
            # "operator does not exist: integer <@ int8multirange" as of pg 16
            if all(_is_int32(r) for r in value):
                return INT4MULTIRANGE()
            else:
                return INT8MULTIRANGE()
        elif isinstance(spec, (Decimal, float)):
            return NUMMULTIRANGE()
        elif isinstance(spec, datetime):
            return TSMULTIRANGE() if not spec.tzinfo else TSTZMULTIRANGE()
        elif isinstance(spec, date):
            return DATEMULTIRANGE()
        else:
            # empty Range, SQL datatype can't be determined here
            return sqltypes.NULLTYPE


class AbstractMultiRangeImpl(AbstractMultiRange[_T]):
    """Marker for AbstractMultiRange that will apply a subclass-specific
    adaptation"""


class INT4RANGE(AbstractSingleRange[int]):
    """Represent the PostgreSQL INT4RANGE type."""

    __visit_name__ = "INT4RANGE"


class INT8RANGE(AbstractSingleRange[int]):
    """Represent the PostgreSQL INT8RANGE type."""

    __visit_name__ = "INT8RANGE"


class NUMRANGE(AbstractSingleRange[Decimal]):
    """Represent the PostgreSQL NUMRANGE type."""

    __visit_name__ = "NUMRANGE"


class DATERANGE(AbstractSingleRange[date]):
    """Represent the PostgreSQL DATERANGE type."""

    __visit_name__ = "DATERANGE"


class TSRANGE(AbstractSingleRange[datetime]):
    """Represent the PostgreSQL TSRANGE type."""

    __visit_name__ = "TSRANGE"


class TSTZRANGE(AbstractSingleRange[datetime]):
    """Represent the PostgreSQL TSTZRANGE type."""

    __visit_name__ = "TSTZRANGE"


class INT4MULTIRANGE(AbstractMultiRange[int]):
    """Represent the PostgreSQL INT4MULTIRANGE type."""

    __visit_name__ = "INT4MULTIRANGE"


class INT8MULTIRANGE(AbstractMultiRange[int]):
    """Represent the PostgreSQL INT8MULTIRANGE type."""

    __visit_name__ = "INT8MULTIRANGE"


class NUMMULTIRANGE(AbstractMultiRange[Decimal]):
    """Represent the PostgreSQL NUMMULTIRANGE type."""

    __visit_name__ = "NUMMULTIRANGE"


class DATEMULTIRANGE(AbstractMultiRange[date]):
    """Represent the PostgreSQL DATEMULTIRANGE type."""

    __visit_name__ = "DATEMULTIRANGE"


class TSMULTIRANGE(AbstractMultiRange[datetime]):
    """Represent the PostgreSQL TSRANGE type."""

    __visit_name__ = "TSMULTIRANGE"


class TSTZMULTIRANGE(AbstractMultiRange[datetime]):
    """Represent the PostgreSQL TSTZRANGE type."""

    __visit_name__ = "TSTZMULTIRANGE"


_max_int_32 = 2**31 - 1
_min_int_32 = -(2**31)


def _is_int32(r: Range[int]) -> bool:
    return (r.lower is None or _min_int_32 <= r.lower <= _max_int_32) and (
        r.upper is None or _min_int_32 <= r.upper <= _max_int_32
    )