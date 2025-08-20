
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\functions\zetazeros.py ===
"""
The function zetazero(n) computes the n-th nontrivial zero of zeta(s).

The general strategy is to locate a block of Gram intervals B where we
know exactly the number of zeros contained and which of those zeros
is that which we search.

If n <= 400 000 000  we know exactly the Rosser exceptions, contained
in a list in this file. Hence for n<=400 000 000 we simply
look at these list of exceptions. If our zero is implicated in one of
these exceptions we have our block B.  In other case we simply locate
the good Rosser block containing our zero.

For n > 400 000 000 we apply the method of Turing, as complemented by
Lehman, Brent and Trudgian  to find a suitable B.
"""

from .functions import defun, defun_wrapped

def find_rosser_block_zero(ctx, n):
    """for n<400 000 000 determines a block were one find our zero"""
    for k in range(len(_ROSSER_EXCEPTIONS)//2):
        a=_ROSSER_EXCEPTIONS[2*k][0]
        b=_ROSSER_EXCEPTIONS[2*k][1]
        if ((a<= n-2) and (n-1 <= b)):
            t0 = ctx.grampoint(a)
            t1 = ctx.grampoint(b)
            v0 = ctx._fp.siegelz(t0)
            v1 = ctx._fp.siegelz(t1)
            my_zero_number = n-a-1
            zero_number_block = b-a
            pattern = _ROSSER_EXCEPTIONS[2*k+1]
            return (my_zero_number, [a,b], [t0,t1], [v0,v1])
    k = n-2
    t,v,b = compute_triple_tvb(ctx, k)
    T = [t]
    V = [v]
    while b < 0:
        k -= 1
        t,v,b = compute_triple_tvb(ctx, k)
        T.insert(0,t)
        V.insert(0,v)
    my_zero_number = n-k-1
    m = n-1
    t,v,b = compute_triple_tvb(ctx, m)
    T.append(t)
    V.append(v)
    while b < 0:
        m += 1
        t,v,b = compute_triple_tvb(ctx, m)
        T.append(t)
        V.append(v)
    return (my_zero_number, [k,m], T, V)

def wpzeros(t):
    """Precision needed to compute higher zeros"""
    wp = 53
    if t > 3*10**8:
        wp = 63
    if t > 10**11:
        wp = 70
    if t > 10**14:
        wp = 83
    return wp

def separate_zeros_in_block(ctx, zero_number_block, T, V, limitloop=None,
    fp_tolerance=None):
    """Separate the zeros contained in the block T, limitloop
    determines how long one must search"""
    if limitloop is None:
        limitloop = ctx.inf
    loopnumber = 0
    variations = count_variations(V)
    while ((variations < zero_number_block) and (loopnumber <limitloop)):
        a = T[0]
        v = V[0]
        newT = [a]
        newV = [v]
        variations = 0
        for n in range(1,len(T)):
            b2 = T[n]
            u = V[n]
            if (u*v>0):
                alpha = ctx.sqrt(u/v)
                b= (alpha*a+b2)/(alpha+1)
            else:
                b = (a+b2)/2
            if fp_tolerance < 10:
                w = ctx._fp.siegelz(b)
                if abs(w)<fp_tolerance:
                    w = ctx.siegelz(b)
            else:
                w=ctx.siegelz(b)
            if v*w<0:
                variations += 1
            newT.append(b)
            newV.append(w)
            u = V[n]
            if u*w <0:
                variations += 1
            newT.append(b2)
            newV.append(u)
            a = b2
            v = u
        T = newT
        V = newV
        loopnumber +=1
        if (limitloop>ITERATION_LIMIT)and(loopnumber>2)and(variations+2==zero_number_block):
            dtMax=0
            dtSec=0
            kMax = 0
            for k1 in range(1,len(T)):
                dt = T[k1]-T[k1-1]
                if dt > dtMax:
                    kMax=k1
                    dtSec = dtMax
                    dtMax = dt
                elif  (dt<dtMax) and(dt >dtSec):
                    dtSec = dt
            if dtMax>3*dtSec:
                f = lambda x: ctx.rs_z(x,derivative=1)
                t0=T[kMax-1]
                t1 = T[kMax]
                t=ctx.findroot(f,  (t0,t1), solver ='illinois',verify=False, verbose=False)
                v = ctx.siegelz(t)
                if (t0<t) and (t<t1) and (v*V[kMax]<0):
                    T.insert(kMax,t)
                    V.insert(kMax,v)
        variations = count_variations(V)
    if variations == zero_number_block:
        separated = True
    else:
        separated = False
    return (T,V, separated)

def separate_my_zero(ctx, my_zero_number, zero_number_block, T, V, prec):
    """If we know which zero of this block is mine,
    the function separates the zero"""
    variations = 0
    v0 = V[0]
    for k in range(1,len(V)):
        v1 = V[k]
        if v0*v1 < 0:
            variations +=1
            if variations == my_zero_number:
                k0 = k
                leftv = v0
                rightv = v1
        v0 = v1
    t1 = T[k0]
    t0 = T[k0-1]
    ctx.prec = prec
    wpz = wpzeros(my_zero_number*ctx.log(my_zero_number))

    guard = 4*ctx.mag(my_zero_number)
    precs = [ctx.prec+4]
    index=0
    while precs[0] > 2*wpz:
        index +=1
        precs = [precs[0] // 2 +3+2*index] + precs
    ctx.prec = precs[0] + guard
    r = ctx.findroot(lambda x:ctx.siegelz(x), (t0,t1), solver ='illinois', verbose=False)
    #print "first step at", ctx.dps, "digits"
    z=ctx.mpc(0.5,r)
    for prec in precs[1:]:
        ctx.prec = prec + guard
        #print "refining to", ctx.dps, "digits"
        znew = z - ctx.zeta(z) / ctx.zeta(z, derivative=1)
        #print "difference", ctx.nstr(abs(z-znew))
        z=ctx.mpc(0.5,ctx.im(znew))
    return ctx.im(z)

def sure_number_block(ctx, n):
    """The number of good Rosser blocks needed to apply
    Turing method
    References:
    R. P. Brent, On the Zeros of the Riemann Zeta Function
    in the Critical Strip, Math. Comp. 33 (1979) 1361--1372
    T. Trudgian, Improvements to Turing Method, Math. Comp."""
    if n < 9*10**5:
        return(2)
    g = ctx.grampoint(n-100)
    lg = ctx._fp.ln(g)
    brent = 0.0061 * lg**2 +0.08*lg
    trudgian = 0.0031 * lg**2 +0.11*lg
    N = ctx.ceil(min(brent,trudgian))
    N = int(N)
    return N

def compute_triple_tvb(ctx, n):
    t = ctx.grampoint(n)
    v = ctx._fp.siegelz(t)
    if ctx.mag(abs(v))<ctx.mag(t)-45:
        v = ctx.siegelz(t)
    b = v*(-1)**n
    return t,v,b



ITERATION_LIMIT = 4

def search_supergood_block(ctx, n, fp_tolerance):
    """To use for n>400 000 000"""
    sb = sure_number_block(ctx, n)
    number_goodblocks = 0
    m2 = n-1
    t, v, b = compute_triple_tvb(ctx, m2)
    Tf = [t]
    Vf = [v]
    while b < 0:
        m2 += 1
        t,v,b = compute_triple_tvb(ctx, m2)
        Tf.append(t)
        Vf.append(v)
    goodpoints = [m2]
    T = [t]
    V = [v]
    while number_goodblocks < 2*sb:
        m2 += 1
        t, v, b = compute_triple_tvb(ctx, m2)
        T.append(t)
        V.append(v)
        while b < 0:
            m2 += 1
            t,v,b = compute_triple_tvb(ctx, m2)
            T.append(t)
            V.append(v)
        goodpoints.append(m2)
        zn = len(T)-1
        A, B, separated =\
           separate_zeros_in_block(ctx, zn, T, V, limitloop=ITERATION_LIMIT,
                fp_tolerance=fp_tolerance)
        Tf.pop()
        Tf.extend(A)
        Vf.pop()
        Vf.extend(B)
        if separated:
            number_goodblocks += 1
        else:
            number_goodblocks = 0
        T = [t]
        V = [v]
    # Now the same procedure to the left
    number_goodblocks = 0
    m2 = n-2
    t, v, b = compute_triple_tvb(ctx, m2)
    Tf.insert(0,t)
    Vf.insert(0,v)
    while b < 0:
        m2 -= 1
        t,v,b = compute_triple_tvb(ctx, m2)
        Tf.insert(0,t)
        Vf.insert(0,v)
    goodpoints.insert(0,m2)
    T = [t]
    V = [v]
    while number_goodblocks < 2*sb:
        m2 -= 1
        t, v, b = compute_triple_tvb(ctx, m2)
        T.insert(0,t)
        V.insert(0,v)
        while b < 0:
            m2 -= 1
            t,v,b = compute_triple_tvb(ctx, m2)
            T.insert(0,t)
            V.insert(0,v)
        goodpoints.insert(0,m2)
        zn = len(T)-1
        A, B, separated =\
           separate_zeros_in_block(ctx, zn, T, V, limitloop=ITERATION_LIMIT, fp_tolerance=fp_tolerance)
        A.pop()
        Tf = A+Tf
        B.pop()
        Vf = B+Vf
        if separated:
            number_goodblocks += 1
        else:
            number_goodblocks = 0
        T = [t]
        V = [v]
    r = goodpoints[2*sb]
    lg = len(goodpoints)
    s = goodpoints[lg-2*sb-1]
    tr, vr, br = compute_triple_tvb(ctx, r)
    ar = Tf.index(tr)
    ts, vs, bs = compute_triple_tvb(ctx, s)
    as1 = Tf.index(ts)
    T = Tf[ar:as1+1]
    V = Vf[ar:as1+1]
    zn = s-r
    A, B, separated =\
       separate_zeros_in_block(ctx, zn,T,V,limitloop=ITERATION_LIMIT, fp_tolerance=fp_tolerance)
    if separated:
        return (n-r-1,[r,s],A,B)
    q = goodpoints[sb]
    lg = len(goodpoints)
    t = goodpoints[lg-sb-1]
    tq, vq, bq = compute_triple_tvb(ctx, q)
    aq = Tf.index(tq)
    tt, vt, bt = compute_triple_tvb(ctx, t)
    at = Tf.index(tt)
    T = Tf[aq:at+1]
    V = Vf[aq:at+1]
    return (n-q-1,[q,t],T,V)

def count_variations(V):
    count = 0
    vold = V[0]
    for n in range(1, len(V)):
        vnew = V[n]
        if vold*vnew < 0:
            count +=1
        vold = vnew
    return count

def pattern_construct(ctx, block, T, V):
    pattern = '('
    a = block[0]
    b = block[1]
    t0,v0,b0 = compute_triple_tvb(ctx, a)
    k = 0
    k0 = 0
    for n in range(a+1,b+1):
        t1,v1,b1 = compute_triple_tvb(ctx, n)
        lgT =len(T)
        while (k < lgT) and (T[k] <= t1):
            k += 1
        L = V[k0:k]
        L.append(v1)
        L.insert(0,v0)
        count = count_variations(L)
        pattern = pattern + ("%s" % count)
        if b1 > 0:
            pattern = pattern + ')('
        k0 = k
        t0,v0,b0 = t1,v1,b1
    pattern = pattern[:-1]
    return pattern

@defun
def zetazero(ctx, n, info=False, round=True):
    r"""
    Computes the `n`-th nontrivial zero of `\zeta(s)` on the critical line,
    i.e. returns an approximation of the `n`-th largest complex number
    `s = \frac{1}{2} + ti` for which `\zeta(s) = 0`. Equivalently, the
    imaginary part `t` is a zero of the Z-function (:func:`~mpmath.siegelz`).

    **Examples**

    The first few zeros::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> zetazero(1)
        (0.5 + 14.13472514173469379045725j)
        >>> zetazero(2)
        (0.5 + 21.02203963877155499262848j)
        >>> zetazero(20)
        (0.5 + 77.14484006887480537268266j)

    Verifying that the values are zeros::

        >>> for n in range(1,5):
        ...     s = zetazero(n)
        ...     chop(zeta(s)), chop(siegelz(s.imag))
        ...
        (0.0, 0.0)
        (0.0, 0.0)
        (0.0, 0.0)
        (0.0, 0.0)

    Negative indices give the conjugate zeros (`n = 0` is undefined)::

        >>> zetazero(-1)
        (0.5 - 14.13472514173469379045725j)

    :func:`~mpmath.zetazero` supports arbitrarily large `n` and arbitrary precision::

        >>> mp.dps = 15
        >>> zetazero(1234567)
        (0.5 + 727690.906948208j)
        >>> mp.dps = 50
        >>> zetazero(1234567)
        (0.5 + 727690.9069482075392389420041147142092708393819935j)
        >>> chop(zeta(_)/_)
        0.0

    with *info=True*, :func:`~mpmath.zetazero` gives additional information::

        >>> mp.dps = 15
        >>> zetazero(542964976,info=True)
        ((0.5 + 209039046.578535j), [542964969, 542964978], 6, '(013111110)')

    This means that the zero is between Gram points 542964969 and 542964978;
    it is the 6-th zero between them. Finally (01311110) is the pattern
    of zeros in this interval. The numbers indicate the number of zeros
    in each Gram interval (Rosser blocks between parenthesis). In this case
    there is only one Rosser block of length nine.
    """
    n = int(n)
    if n < 0:
        return ctx.zetazero(-n).conjugate()
    if n == 0:
        raise ValueError("n must be nonzero")
    wpinitial = ctx.prec
    try:
        wpz, fp_tolerance = comp_fp_tolerance(ctx, n)
        ctx.prec = wpz
        if n < 400000000:
            my_zero_number, block, T, V =\
             find_rosser_block_zero(ctx, n)
        else:
            my_zero_number, block, T, V =\
             search_supergood_block(ctx, n, fp_tolerance)
        zero_number_block = block[1]-block[0]
        T, V, separated = separate_zeros_in_block(ctx, zero_number_block, T, V,
            limitloop=ctx.inf, fp_tolerance=fp_tolerance)
        if info:
            pattern = pattern_construct(ctx,block,T,V)
        prec = max(wpinitial, wpz)
        t = separate_my_zero(ctx, my_zero_number, zero_number_block,T,V,prec)
        v = ctx.mpc(0.5,t)
    finally:
        ctx.prec = wpinitial
    if round:
        v =+v
    if info:
        return (v,block,my_zero_number,pattern)
    else:
        return v

def gram_index(ctx, t):
    if t > 10**13:
        wp = 3*ctx.log(t, 10)
    else:
        wp = 0
    prec = ctx.prec
    try:
        ctx.prec += wp
        h = int(ctx.siegeltheta(t)/ctx.pi)
    finally:
        ctx.prec = prec
    return(h)

def count_to(ctx, t, T, V):
    count = 0
    vold = V[0]
    told = T[0]
    tnew = T[1]
    k = 1
    while tnew < t:
        vnew = V[k]
        if vold*vnew < 0:
            count += 1
        vold = vnew
        k += 1
        tnew = T[k]
    a = ctx.siegelz(t)
    if a*vold < 0:
        count += 1
    return count

def comp_fp_tolerance(ctx, n):
    wpz = wpzeros(n*ctx.log(n))
    if n < 15*10**8:
        fp_tolerance = 0.0005
    elif n <= 10**14:
        fp_tolerance = 0.1
    else:
        fp_tolerance = 100
    return wpz, fp_tolerance

@defun
def nzeros(ctx, t):
    r"""
    Computes the number of zeros of the Riemann zeta function in
    `(0,1) \times (0,t]`, usually denoted by `N(t)`.

    **Examples**

    The first zero has imaginary part between 14 and 15::

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> nzeros(14)
        0
        >>> nzeros(15)
        1
        >>> zetazero(1)
        (0.5 + 14.1347251417347j)

    Some closely spaced zeros::

        >>> nzeros(10**7)
        21136125
        >>> zetazero(21136125)
        (0.5 + 9999999.32718175j)
        >>> zetazero(21136126)
        (0.5 + 10000000.2400236j)
        >>> nzeros(545439823.215)
        1500000001
        >>> zetazero(1500000001)
        (0.5 + 545439823.201985j)
        >>> zetazero(1500000002)
        (0.5 + 545439823.325697j)

    This confirms the data given by J. van de Lune,
    H. J. J. te Riele and D. T. Winter in 1986.
    """
    if t < 14.1347251417347:
        return 0
    x = gram_index(ctx, t)
    k = int(ctx.floor(x))
    wpinitial = ctx.prec
    wpz, fp_tolerance = comp_fp_tolerance(ctx, k)
    ctx.prec = wpz
    a = ctx.siegelz(t)
    if k == -1 and a < 0:
        return 0
    elif k == -1 and a > 0:
        return 1
    if k+2 < 400000000:
        Rblock = find_rosser_block_zero(ctx, k+2)
    else:
        Rblock = search_supergood_block(ctx, k+2, fp_tolerance)
    n1, n2 = Rblock[1]
    if n2-n1 == 1:
        b = Rblock[3][0]
        if a*b > 0:
            ctx.prec = wpinitial
            return k+1
        else:
            ctx.prec = wpinitial
            return k+2
    my_zero_number,block, T, V = Rblock
    zero_number_block = n2-n1
    T, V, separated = separate_zeros_in_block(ctx,\
                                              zero_number_block, T, V,\
                                              limitloop=ctx.inf,\
                                            fp_tolerance=fp_tolerance)
    n = count_to(ctx, t, T, V)
    ctx.prec = wpinitial
    return n+n1+1

@defun_wrapped
def backlunds(ctx, t):
    r"""
    Computes the function
    `S(t) = \operatorname{arg} \zeta(\frac{1}{2} + it) / \pi`.

    See Titchmarsh Section 9.3 for details of the definition.

    **Examples**

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> backlunds(217.3)
        0.16302205431184

    Generally, the value is a small number. At Gram points it is an integer,
    frequently equal to 0::

        >>> chop(backlunds(grampoint(200)))
        0.0
        >>> backlunds(extraprec(10)(grampoint)(211))
        1.0
        >>> backlunds(extraprec(10)(grampoint)(232))
        -1.0

    The number of zeros of the Riemann zeta function up to height `t`
    satisfies `N(t) = \theta(t)/\pi + 1 + S(t)` (see :func:nzeros` and
    :func:`siegeltheta`)::

        >>> t = 1234.55
        >>> nzeros(t)
        842
        >>> siegeltheta(t)/pi+1+backlunds(t)
        842.0

    """
    return ctx.nzeros(t)-1-ctx.siegeltheta(t)/ctx.pi


"""
_ROSSER_EXCEPTIONS is a list of all  exceptions to
Rosser's rule for n <= 400 000 000.

Alternately the  entry is of type   [n,m], or a string.
The string is the zero pattern of the Block and the relevant
adjacent.  For example (010)3 corresponds to a block
composed of three Gram intervals, the first ant third without
a zero and the intermediate with a zero. The next Gram interval
contain three zeros. So that in total we have 4 zeros in 4 Gram
blocks. n and m are the indices of the Gram points  of this
interval of four Gram intervals. The Rosser exception is therefore
formed by the three Gram intervals that are signaled between
parenthesis.

We have included also some Rosser's exceptions beyond n=400 000 000
that are noted in the literature by some reason.

The list is composed from the data published in the references:

R. P. Brent, J. van de Lune, H. J. J. te Riele, D. T. Winter,
'On the Zeros of the Riemann Zeta Function in the Critical Strip. II',
Math. Comp. 39 (1982) 681--688.
See also Corrigenda in Math. Comp. 46 (1986) 771.

J. van de Lune, H. J. J. te Riele,
'On the Zeros of the Riemann Zeta Function in the Critical Strip. III',
Math. Comp. 41 (1983) 759--767.
See also  Corrigenda in Math. Comp. 46 (1986) 771.

J. van de Lune,
'Sums of Equal Powers of Positive Integers',
Dissertation,
Vrije Universiteit te Amsterdam, Centrum voor Wiskunde en Informatica,
Amsterdam, 1984.

Thanks to the authors all this papers and those others that have
contributed to make this possible.
"""







_ROSSER_EXCEPTIONS = \
[[13999525, 13999528], '(00)3',
[30783329, 30783332], '(00)3',
[30930926, 30930929], '3(00)',
[37592215, 37592218], '(00)3',
[40870156, 40870159], '(00)3',
[43628107, 43628110], '(00)3',
[46082042, 46082045], '(00)3',
[46875667, 46875670], '(00)3',
[49624540, 49624543], '3(00)',
[50799238, 50799241], '(00)3',
[55221453, 55221456], '3(00)',
[56948779, 56948782], '3(00)',
[60515663, 60515666], '(00)3',
[61331766, 61331770], '(00)40',
[69784843, 69784846], '3(00)',
[75052114, 75052117], '(00)3',
[79545240, 79545243], '3(00)',
[79652247, 79652250], '3(00)',
[83088043, 83088046], '(00)3',
[83689522, 83689525], '3(00)',
[85348958, 85348961], '(00)3',
[86513820, 86513823], '(00)3',
[87947596, 87947599], '3(00)',
[88600095, 88600098], '(00)3',
[93681183, 93681186], '(00)3',
[100316551, 100316554], '3(00)',
[100788444, 100788447], '(00)3',
[106236172, 106236175], '(00)3',
[106941327, 106941330], '3(00)',
[107287955, 107287958], '(00)3',
[107532016, 107532019], '3(00)',
[110571044, 110571047], '(00)3',
[111885253, 111885256], '3(00)',
[113239783, 113239786], '(00)3',
[120159903, 120159906], '(00)3',
[121424391, 121424394], '3(00)',
[121692931, 121692934], '3(00)',
[121934170, 121934173], '3(00)',
[122612848, 122612851], '3(00)',
[126116567, 126116570], '(00)3',
[127936513, 127936516], '(00)3',
[128710277, 128710280], '3(00)',
[129398902, 129398905], '3(00)',
[130461096, 130461099], '3(00)',
[131331947, 131331950], '3(00)',
[137334071, 137334074], '3(00)',
[137832603, 137832606], '(00)3',
[138799471, 138799474], '3(00)',
[139027791, 139027794], '(00)3',
[141617806, 141617809], '(00)3',
[144454931, 144454934], '(00)3',
[145402379, 145402382], '3(00)',
[146130245, 146130248], '3(00)',
[147059770, 147059773], '(00)3',
[147896099, 147896102], '3(00)',
[151097113, 151097116], '(00)3',
[152539438, 152539441], '(00)3',
[152863168, 152863171], '3(00)',
[153522726, 153522729], '3(00)',
[155171524, 155171527], '3(00)',
[155366607, 155366610], '(00)3',
[157260686, 157260689], '3(00)',
[157269224, 157269227], '(00)3',
[157755123, 157755126], '(00)3',
[158298484, 158298487], '3(00)',
[160369050, 160369053], '3(00)',
[162962787, 162962790], '(00)3',
[163724709, 163724712], '(00)3',
[164198113, 164198116], '3(00)',
[164689301, 164689305], '(00)40',
[164880228, 164880231], '3(00)',
[166201932, 166201935], '(00)3',
[168573836, 168573839], '(00)3',
[169750763, 169750766], '(00)3',
[170375507, 170375510], '(00)3',
[170704879, 170704882], '3(00)',
[172000992, 172000995], '3(00)',
[173289941, 173289944], '(00)3',
[173737613, 173737616], '3(00)',
[174102513, 174102516], '(00)3',
[174284990, 174284993], '(00)3',
[174500513, 174500516], '(00)3',
[175710609, 175710612], '(00)3',
[176870843, 176870846], '3(00)',
[177332732, 177332735], '3(00)',
[177902861, 177902864], '3(00)',
[179979095, 179979098], '(00)3',
[181233726, 181233729], '3(00)',
[181625435, 181625438], '(00)3',
[182105255, 182105259], '22(00)',
[182223559, 182223562], '3(00)',
[191116404, 191116407], '3(00)',
[191165599, 191165602], '3(00)',
[191297535, 191297539], '(00)22',
[192485616, 192485619], '(00)3',
[193264634, 193264638], '22(00)',
[194696968, 194696971], '(00)3',
[195876805, 195876808], '(00)3',
[195916548, 195916551], '3(00)',
[196395160, 196395163], '3(00)',
[196676303, 196676306], '(00)3',
[197889882, 197889885], '3(00)',
[198014122, 198014125], '(00)3',
[199235289, 199235292], '(00)3',
[201007375, 201007378], '(00)3',
[201030605, 201030608], '3(00)',
[201184290, 201184293], '3(00)',
[201685414, 201685418], '(00)22',
[202762875, 202762878], '3(00)',
[202860957, 202860960], '3(00)',
[203832577, 203832580], '3(00)',
[205880544, 205880547], '(00)3',
[206357111, 206357114], '(00)3',
[207159767, 207159770], '3(00)',
[207167343, 207167346], '3(00)',
[207482539, 207482543], '3(010)',
[207669540, 207669543], '3(00)',
[208053426, 208053429], '(00)3',
[208110027, 208110030], '3(00)',
[209513826, 209513829], '3(00)',
[212623522, 212623525], '(00)3',
[213841715, 213841718], '(00)3',
[214012333, 214012336], '(00)3',
[214073567, 214073570], '(00)3',
[215170600, 215170603], '3(00)',
[215881039, 215881042], '3(00)',
[216274604, 216274607], '3(00)',
[216957120, 216957123], '3(00)',
[217323208, 217323211], '(00)3',
[218799264, 218799267], '(00)3',
[218803557, 218803560], '3(00)',
[219735146, 219735149], '(00)3',
[219830062, 219830065], '3(00)',
[219897904, 219897907], '(00)3',
[221205545, 221205548], '(00)3',
[223601929, 223601932], '(00)3',
[223907076, 223907079], '3(00)',
[223970397, 223970400], '(00)3',
[224874044, 224874048], '22(00)',
[225291157, 225291160], '(00)3',
[227481734, 227481737], '(00)3',
[228006442, 228006445], '3(00)',
[228357900, 228357903], '(00)3',
[228386399, 228386402], '(00)3',
[228907446, 228907449], '(00)3',
[228984552, 228984555], '3(00)',
[229140285, 229140288], '3(00)',
[231810024, 231810027], '(00)3',
[232838062, 232838065], '3(00)',
[234389088, 234389091], '3(00)',
[235588194, 235588197], '(00)3',
[236645695, 236645698], '(00)3',
[236962876, 236962879], '3(00)',
[237516723, 237516727], '04(00)',
[240004911, 240004914], '(00)3',
[240221306, 240221309], '3(00)',
[241389213, 241389217], '(010)3',
[241549003, 241549006], '(00)3',
[241729717, 241729720], '(00)3',
[241743684, 241743687], '3(00)',
[243780200, 243780203], '3(00)',
[243801317, 243801320], '(00)3',
[244122072, 244122075], '(00)3',
[244691224, 244691227], '3(00)',
[244841577, 244841580], '(00)3',
[245813461, 245813464], '(00)3',
[246299475, 246299478], '(00)3',
[246450176, 246450179], '3(00)',
[249069349, 249069352], '(00)3',
[250076378, 250076381], '(00)3',
[252442157, 252442160], '3(00)',
[252904231, 252904234], '3(00)',
[255145220, 255145223], '(00)3',
[255285971, 255285974], '3(00)',
[256713230, 256713233], '(00)3',
[257992082, 257992085], '(00)3',
[258447955, 258447959], '22(00)',
[259298045, 259298048], '3(00)',
[262141503, 262141506], '(00)3',
[263681743, 263681746], '3(00)',
[266527881, 266527885], '(010)3',
[266617122, 266617125], '(00)3',
[266628044, 266628047], '3(00)',
[267305763, 267305766], '(00)3',
[267388404, 267388407], '3(00)',
[267441672, 267441675], '3(00)',
[267464886, 267464889], '(00)3',
[267554907, 267554910], '3(00)',
[269787480, 269787483], '(00)3',
[270881434, 270881437], '(00)3',
[270997583, 270997586], '3(00)',
[272096378, 272096381], '3(00)',
[272583009, 272583012], '(00)3',
[274190881, 274190884], '3(00)',
[274268747, 274268750], '(00)3',
[275297429, 275297432], '3(00)',
[275545476, 275545479], '3(00)',
[275898479, 275898482], '3(00)',
[275953000, 275953003], '(00)3',
[277117197, 277117201], '(00)22',
[277447310, 277447313], '3(00)',
[279059657, 279059660], '3(00)',
[279259144, 279259147], '3(00)',
[279513636, 279513639], '3(00)',
[279849069, 279849072], '3(00)',
[280291419, 280291422], '(00)3',
[281449425, 281449428], '3(00)',
[281507953, 281507956], '3(00)',
[281825600, 281825603], '(00)3',
[282547093, 282547096], '3(00)',
[283120963, 283120966], '3(00)',
[283323493, 283323496], '(00)3',
[284764535, 284764538], '3(00)',
[286172639, 286172642], '3(00)',
[286688824, 286688827], '(00)3',
[287222172, 287222175], '3(00)',
[287235534, 287235537], '3(00)',
[287304861, 287304864], '3(00)',
[287433571, 287433574], '(00)3',
[287823551, 287823554], '(00)3',
[287872422, 287872425], '3(00)',
[288766615, 288766618], '3(00)',
[290122963, 290122966], '3(00)',
[290450849, 290450853], '(00)22',
[291426141, 291426144], '3(00)',
[292810353, 292810356], '3(00)',
[293109861, 293109864], '3(00)',
[293398054, 293398057], '3(00)',
[294134426, 294134429], '3(00)',
[294216438, 294216441], '(00)3',
[295367141, 295367144], '3(00)',
[297834111, 297834114], '3(00)',
[299099969, 299099972], '3(00)',
[300746958, 300746961], '3(00)',
[301097423, 301097426], '(00)3',
[301834209, 301834212], '(00)3',
[302554791, 302554794], '(00)3',
[303497445, 303497448], '3(00)',
[304165344, 304165347], '3(00)',
[304790218, 304790222], '3(010)',
[305302352, 305302355], '(00)3',
[306785996, 306785999], '3(00)',
[307051443, 307051446], '3(00)',
[307481539, 307481542], '3(00)',
[308605569, 308605572], '3(00)',
[309237610, 309237613], '3(00)',
[310509287, 310509290], '(00)3',
[310554057, 310554060], '3(00)',
[310646345, 310646348], '3(00)',
[311274896, 311274899], '(00)3',
[311894272, 311894275], '3(00)',
[312269470, 312269473], '(00)3',
[312306601, 312306605], '(00)40',
[312683193, 312683196], '3(00)',
[314499804, 314499807], '3(00)',
[314636802, 314636805], '(00)3',
[314689897, 314689900], '3(00)',
[314721319, 314721322], '3(00)',
[316132890, 316132893], '3(00)',
[316217470, 316217474], '(010)3',
[316465705, 316465708], '3(00)',
[316542790, 316542793], '(00)3',
[320822347, 320822350], '3(00)',
[321733242, 321733245], '3(00)',
[324413970, 324413973], '(00)3',
[325950140, 325950143], '(00)3',
[326675884, 326675887], '(00)3',
[326704208, 326704211], '3(00)',
[327596247, 327596250], '3(00)',
[328123172, 328123175], '3(00)',
[328182212, 328182215], '(00)3',
[328257498, 328257501], '3(00)',
[328315836, 328315839], '(00)3',
[328800974, 328800977], '(00)3',
[328998509, 328998512], '3(00)',
[329725370, 329725373], '(00)3',
[332080601, 332080604], '(00)3',
[332221246, 332221249], '(00)3',
[332299899, 332299902], '(00)3',
[332532822, 332532825], '(00)3',
[333334544, 333334548], '(00)22',
[333881266, 333881269], '3(00)',
[334703267, 334703270], '3(00)',
[334875138, 334875141], '3(00)',
[336531451, 336531454], '3(00)',
[336825907, 336825910], '(00)3',
[336993167, 336993170], '(00)3',
[337493998, 337494001], '3(00)',
[337861034, 337861037], '3(00)',
[337899191, 337899194], '(00)3',
[337958123, 337958126], '(00)3',
[342331982, 342331985], '3(00)',
[342676068, 342676071], '3(00)',
[347063781, 347063784], '3(00)',
[347697348, 347697351], '3(00)',
[347954319, 347954322], '3(00)',
[348162775, 348162778], '3(00)',
[349210702, 349210705], '(00)3',
[349212913, 349212916], '3(00)',
[349248650, 349248653], '(00)3',
[349913500, 349913503], '3(00)',
[350891529, 350891532], '3(00)',
[351089323, 351089326], '3(00)',
[351826158, 351826161], '3(00)',
[352228580, 352228583], '(00)3',
[352376244, 352376247], '3(00)',
[352853758, 352853761], '(00)3',
[355110439, 355110442], '(00)3',
[355808090, 355808094], '(00)40',
[355941556, 355941559], '3(00)',
[356360231, 356360234], '(00)3',
[356586657, 356586660], '3(00)',
[356892926, 356892929], '(00)3',
[356908232, 356908235], '3(00)',
[357912730, 357912733], '3(00)',
[358120344, 358120347], '3(00)',
[359044096, 359044099], '(00)3',
[360819357, 360819360], '3(00)',
[361399662, 361399666], '(010)3',
[362361315, 362361318], '(00)3',
[363610112, 363610115], '(00)3',
[363964804, 363964807], '3(00)',
[364527375, 364527378], '(00)3',
[365090327, 365090330], '(00)3',
[365414539, 365414542], '3(00)',
[366738474, 366738477], '3(00)',
[368714778, 368714783], '04(010)',
[368831545, 368831548], '(00)3',
[368902387, 368902390], '(00)3',
[370109769, 370109772], '3(00)',
[370963333, 370963336], '3(00)',
[372541136, 372541140], '3(010)',
[372681562, 372681565], '(00)3',
[373009410, 373009413], '(00)3',
[373458970, 373458973], '3(00)',
[375648658, 375648661], '3(00)',
[376834728, 376834731], '3(00)',
[377119945, 377119948], '(00)3',
[377335703, 377335706], '(00)3',
[378091745, 378091748], '3(00)',
[379139522, 379139525], '3(00)',
[380279160, 380279163], '(00)3',
[380619442, 380619445], '3(00)',
[381244231, 381244234], '3(00)',
[382327446, 382327450], '(010)3',
[382357073, 382357076], '3(00)',
[383545479, 383545482], '3(00)',
[384363766, 384363769], '(00)3',
[384401786, 384401790], '22(00)',
[385198212, 385198215], '3(00)',
[385824476, 385824479], '(00)3',
[385908194, 385908197], '3(00)',
[386946806, 386946809], '3(00)',
[387592175, 387592179], '22(00)',
[388329293, 388329296], '(00)3',
[388679566, 388679569], '3(00)',
[388832142, 388832145], '3(00)',
[390087103, 390087106], '(00)3',
[390190926, 390190930], '(00)22',
[390331207, 390331210], '3(00)',
[391674495, 391674498], '3(00)',
[391937831, 391937834], '3(00)',
[391951632, 391951636], '(00)22',
[392963986, 392963989], '(00)3',
[393007921, 393007924], '3(00)',
[393373210, 393373213], '3(00)',
[393759572, 393759575], '(00)3',
[394036662, 394036665], '(00)3',
[395813866, 395813869], '(00)3',
[395956690, 395956693], '3(00)',
[396031670, 396031673], '3(00)',
[397076433, 397076436], '3(00)',
[397470601, 397470604], '3(00)',
[398289458, 398289461], '3(00)',
#
[368714778, 368714783], '04(010)',
[437953499, 437953504], '04(010)',
[526196233, 526196238], '032(00)',
[744719566, 744719571], '(010)40',
[750375857, 750375862], '032(00)',
[958241932, 958241937], '04(010)',
[983377342, 983377347], '(00)410',
[1003780080, 1003780085], '04(010)',
[1070232754, 1070232759], '(00)230',
[1209834865, 1209834870], '032(00)',
[1257209100, 1257209105], '(00)410',
[1368002233, 1368002238], '(00)230'
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\support\expected_conditions.py ===
# Licensed to the Software Freedom Conservancy (SFC) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The SFC licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import re
from collections.abc import Iterable
from typing import Any, Callable, List, Literal, Tuple, TypeVar, Union

from selenium.common.exceptions import (
    NoAlertPresentException,
    NoSuchElementException,
    NoSuchFrameException,
    StaleElementReferenceException,
    WebDriverException,
)
from selenium.webdriver.common.alert import Alert
from selenium.webdriver.remote.webdriver import WebDriver, WebElement

"""
 * Canned "Expected Conditions" which are generally useful within webdriver
 * tests.
"""

D = TypeVar("D")
T = TypeVar("T")

WebDriverOrWebElement = Union[WebDriver, WebElement]


def title_is(title: str) -> Callable[[WebDriver], bool]:
    """An expectation for checking the title of a page.

    Parameters:
    -----------
    title : str
        The expected title, which must be an exact match.

    Returns:
    -------
    boolean : True if the title matches, False otherwise.
    """

    def _predicate(driver: WebDriver):
        return driver.title == title

    return _predicate


def title_contains(title: str) -> Callable[[WebDriver], bool]:
    """An expectation for checking that the title contains a case-sensitive
    substring.

    Parameters:
    -----------
    title : str
        The fragment of title expected.

    Returns:
    -------
    boolean : True when the title matches, False otherwise.
    """

    def _predicate(driver: WebDriver):
        return title in driver.title

    return _predicate


def presence_of_element_located(locator: Tuple[str, str]) -> Callable[[WebDriverOrWebElement], WebElement]:
    """An expectation for checking that an element is present on the DOM of a
    page. This does not necessarily mean that the element is visible.

    Parameters:
    -----------
    locator : Tuple[str, str]
        Used to find the element.

    Returns:
    -------
    WebElement : The WebElement once it is located.

    Example:
    --------
    >>> from selenium.webdriver.common.by import By
    >>> from selenium.webdriver.support.ui import WebDriverWait
    >>> from selenium.webdriver.support import expected_conditions as EC
    >>> element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.NAME, "q")))
    """

    def _predicate(driver: WebDriverOrWebElement):
        return driver.find_element(*locator)

    return _predicate


def url_contains(url: str) -> Callable[[WebDriver], bool]:
    """An expectation for checking that the current url contains a case-
    sensitive substring.

    Parameters:
    -----------
    url : str
        The fragment of url expected.

    Returns:
    -------
    boolean : True when the url matches, False otherwise.
    """

    def _predicate(driver: WebDriver):
        return url in driver.current_url

    return _predicate


def url_matches(pattern: str) -> Callable[[WebDriver], bool]:
    """An expectation for checking the current url.

    Parameters:
    -----------
    pattern : str
        The pattern to match with the current url.

    Returns:
    -------
    boolean : True when the pattern matches, False otherwise.

    Notes:
    ------
    More powerful than url_contains, as it allows for regular expressions.
    """

    def _predicate(driver: WebDriver):
        return re.search(pattern, driver.current_url) is not None

    return _predicate


def url_to_be(url: str) -> Callable[[WebDriver], bool]:
    """An expectation for checking the current url.

    Parameters:
    -----------
    url : str
        The expected url, which must be an exact match.

    Returns:
    -------
    boolean : True when the url matches, False otherwise.
    """

    def _predicate(driver: WebDriver):
        return url == driver.current_url

    return _predicate


def url_changes(url: str) -> Callable[[WebDriver], bool]:
    """An expectation for checking the current url is different than a given
    string.

    Parameters:
    -----------
    url : str
        The expected url, which must not be an exact match.

    Returns:
    -------
    boolean : True when the url does not match, False otherwise
    """

    def _predicate(driver: WebDriver):
        return url != driver.current_url

    return _predicate


def visibility_of_element_located(
    locator: Tuple[str, str],
) -> Callable[[WebDriverOrWebElement], Union[Literal[False], WebElement]]:
    """An expectation for checking that an element is present on the DOM of a
    page and visible. Visibility means that the element is not only displayed
    but also has a height and width that is greater than 0.

    Parameters:
    -----------
    locator : Tuple[str, str]
        Used to find the element.

    Returns:
    -------
    WebElement : The WebElement once it is located and visible.

    Example:
    --------
    >>> from selenium.webdriver.common.by import By
    >>> from selenium.webdriver.support.ui import WebDriverWait
    >>> from selenium.webdriver.support import expected_conditions as EC
    >>> element = WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.NAME, "q")))
    """

    def _predicate(driver: WebDriverOrWebElement):
        try:
            return _element_if_visible(driver.find_element(*locator))
        except StaleElementReferenceException:
            return False

    return _predicate


def visibility_of(element: WebElement) -> Callable[[Any], Union[Literal[False], WebElement]]:
    """An expectation for checking that an element, known to be present on the
    DOM of a page, is visible.

    Parameters:
    -----------
    element : WebElement
        The WebElement to check.

    Returns:
    -------
    WebElement : The WebElement once it is visible.

    Example:
    --------
    >>> from selenium.webdriver.common.by import By
    >>> from selenium.webdriver.support.ui import WebDriverWait
    >>> from selenium.webdriver.support import expected_conditions as EC
    >>> element = WebDriverWait(driver, 10).until(EC.visibility_of(driver.find_element(By.NAME, "q")))

    Notes:
    ------
    Visibility means that the element is not only displayed but also has
    a height and width that is greater than 0. element is the WebElement
    returns the (same) WebElement once it is visible
    """

    def _predicate(_):
        return _element_if_visible(element)

    return _predicate


def _element_if_visible(element: WebElement, visibility: bool = True) -> Union[Literal[False], WebElement]:
    """An expectation for checking that an element, known to be present on the
    DOM of a page, is of the expected visibility.

    Parameters:
    -----------
    element : WebElement
        The WebElement to check.
    visibility : bool
        The expected visibility of the element.

    Returns:
    -------
    WebElement : The WebElement once it is visible or not visible.
    """
    return element if element.is_displayed() == visibility else False


def presence_of_all_elements_located(locator: Tuple[str, str]) -> Callable[[WebDriverOrWebElement], List[WebElement]]:
    """An expectation for checking that there is at least one element present
    on a web page.

    Parameters:
    -----------
    locator : Tuple[str, str]
        Used to find the element.

    Returns:
    -------
    List[WebElement] : The list of WebElements once they are located.

    Example:
    --------
    >>> from selenium.webdriver.common.by import By
    >>> from selenium.webdriver.support.ui import WebDriverWait
    >>> from selenium.webdriver.support import expected_conditions as EC
    >>> elements = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.CLASS_NAME, "foo")))
    """

    def _predicate(driver: WebDriverOrWebElement):
        return driver.find_elements(*locator)

    return _predicate


def visibility_of_any_elements_located(locator: Tuple[str, str]) -> Callable[[WebDriverOrWebElement], List[WebElement]]:
    """An expectation for checking that there is at least one element visible
    on a web page.

    Parameters:
    -----------
    locator : Tuple[str, str]
        Used to find the element.

    Returns:
    -------
    List[WebElement] : The list of WebElements once they are located and visible.

    Example:
    --------
    >>> from selenium.webdriver.common.by import By
    >>> from selenium.webdriver.support.ui import WebDriverWait
    >>> from selenium.webdriver.support import expected_conditions as EC
    >>> elements = WebDriverWait(driver, 10).until(EC.visibility_of_any_elements_located((By.CLASS_NAME, "foo")))
    """

    def _predicate(driver: WebDriverOrWebElement):
        return [element for element in driver.find_elements(*locator) if _element_if_visible(element)]

    return _predicate


def visibility_of_all_elements_located(
    locator: Tuple[str, str],
) -> Callable[[WebDriverOrWebElement], Union[List[WebElement], Literal[False]]]:
    """An expectation for checking that all elements are present on the DOM of
    a page and visible. Visibility means that the elements are not only
    displayed but also has a height and width that is greater than 0.

    Parameters:
    -----------
    locator : Tuple[str, str]
        Used to find the elements.

    Returns:
    -------
    List[WebElement] : The list of WebElements once they are located and visible.

    Example:
    --------
    >>> from selenium.webdriver.common.by import By
    >>> from selenium.webdriver.support.ui import WebDriverWait
    >>> from selenium.webdriver.support import expected_conditions as EC
    >>> elements = WebDriverWait(driver, 10).until(EC.visibility_of_all_elements_located((By.CLASS_NAME, "foo")))
    """

    def _predicate(driver: WebDriverOrWebElement):
        try:
            elements = driver.find_elements(*locator)
            for element in elements:
                if _element_if_visible(element, visibility=False):
                    return False
            return elements
        except StaleElementReferenceException:
            return False

    return _predicate


def text_to_be_present_in_element(locator: Tuple[str, str], text_: str) -> Callable[[WebDriverOrWebElement], bool]:
    """An expectation for checking if the given text is present in the
    specified element.

    Parameters:
    -----------
    locator : Tuple[str, str]
        Used to find the element.
    text_ : str
        The text to be present in the element.

    Returns:
    -------
    boolean : True when the text is present, False otherwise.

    Example:
    --------
    >>> from selenium.webdriver.common.by import By
    >>> from selenium.webdriver.support.ui import WebDriverWait
    >>> from selenium.webdriver.support import expected_conditions as EC
    >>> is_text_in_element = WebDriverWait(driver, 10).until(
            EC.text_to_be_present_in_element((By.CLASS_NAME, "foo"), "bar")
        )
    """

    def _predicate(driver: WebDriverOrWebElement):
        try:
            element_text = driver.find_element(*locator).text
            return text_ in element_text
        except StaleElementReferenceException:
            return False

    return _predicate


def text_to_be_present_in_element_value(
    locator: Tuple[str, str], text_: str
) -> Callable[[WebDriverOrWebElement], bool]:
    """An expectation for checking if the given text is present in the
    element's value.

    Parameters:
    -----------
    locator : Tuple[str, str]
        Used to find the element.
    text_ : str
        The text to be present in the element's value.

    Returns:
    -------
    boolean : True when the text is present, False otherwise.

    Example:
    --------
    >>> from selenium.webdriver.common.by import By
    >>> from selenium.webdriver.support.ui import WebDriverWait
    >>> from selenium.webdriver.support import expected_conditions as EC
    >>> is_text_in_element_value = WebDriverWait(driver, 10).until(
    ...     EC.text_to_be_present_in_element_value((By.CLASS_NAME, "foo"), "bar")
    ... )
    """

    def _predicate(driver: WebDriverOrWebElement):
        try:
            element_text = driver.find_element(*locator).get_attribute("value")
            return text_ in element_text
        except StaleElementReferenceException:
            return False

    return _predicate


def text_to_be_present_in_element_attribute(
    locator: Tuple[str, str], attribute_: str, text_: str
) -> Callable[[WebDriverOrWebElement], bool]:
    """An expectation for checking if the given text is present in the
    element's attribute.

    Parameters:
    -----------
    locator : Tuple[str, str]
        Used to find the element.
    attribute_ : str
        The attribute to check the text in.
    text_ : str
        The text to be present in the element's attribute.

    Returns:
    -------
    boolean : True when the text is present, False otherwise.

    Example:
    --------
    >>> from selenium.webdriver.common.by import By
    >>> from selenium.webdriver.support.ui import WebDriverWait
    >>> from selenium.webdriver.support import expected_conditions as EC
    >>> is_text_in_element_attribute = WebDriverWait(driver, 10).until(
    ...     EC.text_to_be_present_in_element_attribute((By.CLASS_NAME, "foo"), "bar", "baz")
    ... )
    """

    def _predicate(driver: WebDriverOrWebElement):
        try:
            element_text = driver.find_element(*locator).get_attribute(attribute_)
            if element_text is None:
                return False
            return text_ in element_text
        except StaleElementReferenceException:
            return False

    return _predicate


def frame_to_be_available_and_switch_to_it(
    locator: Union[Tuple[str, str], str, WebElement],
) -> Callable[[WebDriver], bool]:
    """An expectation for checking whether the given frame is available to
    switch to.

    Parameters:
    -----------
    locator : Union[Tuple[str, str], str, WebElement]
        Used to find the frame.

    Returns:
    -------
    boolean : True when the frame is available, False otherwise.

    Example:
    --------
    >>> from selenium.webdriver.support.ui import WebDriverWait
    >>> from selenium.webdriver.support import expected_conditions as EC
    >>> WebDriverWait(driver, 10).until(EC.frame_to_be_available_and_switch_to_it("frame_name"))

    Notes:
    ------
    If the frame is available it switches the given driver to the
    specified frame.
    """

    def _predicate(driver: WebDriver):
        try:
            if isinstance(locator, Iterable) and not isinstance(locator, str):
                driver.switch_to.frame(driver.find_element(*locator))
            else:
                driver.switch_to.frame(locator)
            return True
        except NoSuchFrameException:
            return False

    return _predicate


def invisibility_of_element_located(
    locator: Union[WebElement, Tuple[str, str]],
) -> Callable[[WebDriverOrWebElement], Union[WebElement, bool]]:
    """An Expectation for checking that an element is either invisible or not
    present on the DOM.

    Parameters:
    -----------
    locator : Union[WebElement, Tuple[str, str]]
        Used to find the element.

    Returns:
    -------
    boolean : True when the element is invisible or not present, False otherwise.

    Example:
    --------
    >>> from selenium.webdriver.common.by import By
    >>> from selenium.webdriver.support.ui import WebDriverWait
    >>> from selenium.webdriver.support import expected_conditions as EC
    >>> is_invisible = WebDriverWait(driver, 10).until(EC.invisibility_of_element_located((By.CLASS_NAME, "foo")))

    Notes:
    ------
    - In the case of NoSuchElement, returns true because the element is not
    present in DOM. The try block checks if the element is present but is
    invisible.
    - In the case of StaleElementReference, returns true because stale element
    reference implies that element is no longer visible.
    """

    def _predicate(driver: WebDriverOrWebElement):
        try:
            target = locator
            if not isinstance(target, WebElement):
                target = driver.find_element(*target)
            return _element_if_visible(target, visibility=False)
        except (NoSuchElementException, StaleElementReferenceException):
            # In the case of NoSuchElement, returns true because the element is
            # not present in DOM. The try block checks if the element is present
            # but is invisible.
            # In the case of StaleElementReference, returns true because stale
            # element reference implies that element is no longer visible.
            return True

    return _predicate


def invisibility_of_element(
    element: Union[WebElement, Tuple[str, str]],
) -> Callable[[WebDriverOrWebElement], Union[WebElement, bool]]:
    """An Expectation for checking that an element is either invisible or not
    present on the DOM.

    Parameters:
    -----------
    element : Union[WebElement, Tuple[str, str]]
        Used to find the element.

    Returns:
    -------
    boolean : True when the element is invisible or not present, False otherwise.

    Example:
    --------
    >>> from selenium.webdriver.common.by import By
    >>> from selenium.webdriver.support.ui import WebDriverWait
    >>> from selenium.webdriver.support import expected_conditions as EC
    >>> is_invisible_or_not_present = WebDriverWait(driver, 10).until(
    ...     EC.invisibility_of_element(driver.find_element(By.CLASS_NAME, "foo"))
    ... )
    """
    return invisibility_of_element_located(element)


def element_to_be_clickable(
    mark: Union[WebElement, Tuple[str, str]],
) -> Callable[[WebDriverOrWebElement], Union[Literal[False], WebElement]]:
    """An Expectation for checking an element is visible and enabled such that
    you can click it.

    Parameters:
    -----------
    mark : Union[WebElement, Tuple[str, str]]
        Used to find the element.

    Returns:
    -------
    WebElement : The WebElement once it is located and clickable.

    Example:
    --------
    >>> from selenium.webdriver.common.by import By
    >>> from selenium.webdriver.support.ui import WebDriverWait
    >>> from selenium.webdriver.support import expected_conditions as EC
    >>> element = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CLASS_NAME, "foo")))
    """

    # renamed argument to 'mark', to indicate that both locator
    # and WebElement args are valid
    def _predicate(driver: WebDriverOrWebElement):
        target = mark
        if not isinstance(target, WebElement):  # if given locator instead of WebElement
            target = driver.find_element(*target)  # grab element at locator
        element = visibility_of(target)(driver)
        if element and element.is_enabled():
            return element
        return False

    return _predicate


def staleness_of(element: WebElement) -> Callable[[Any], bool]:
    """Wait until an element is no longer attached to the DOM.

    Parameters:
    -----------
    element : WebElement
        The element to wait for.

    Returns:
    -------
    boolean : False if the element is still attached to the DOM, true otherwise.

    Example:
    --------
    >>> from selenium.webdriver.common.by import By
    >>> from selenium.webdriver.support.ui import WebDriverWait
    >>> from selenium.webdriver.support import expected_conditions as EC
    >>> is_element_stale = WebDriverWait(driver, 10).until(EC.staleness_of(driver.find_element(By.CLASS_NAME, "foo")))
    """

    def _predicate(_):
        try:
            # Calling any method forces a staleness check
            element.is_enabled()
            return False
        except StaleElementReferenceException:
            return True

    return _predicate


def element_to_be_selected(element: WebElement) -> Callable[[Any], bool]:
    """An expectation for checking the selection is selected.

    Parameters:
    -----------
    element : WebElement
        The WebElement to check.

    Returns:
    -------
    boolean : True if the element is selected, False otherwise.

    Example:
    --------
    >>> from selenium.webdriver.common.by import By
    >>> from selenium.webdriver.support.ui import WebDriverWait
    >>> from selenium.webdriver.support import expected_conditions as EC
    >>> is_selected = WebDriverWait(driver, 10).until(EC.element_to_be_selected(driver.find_element(
            By.CLASS_NAME, "foo"))
        )
    """

    def _predicate(_):
        return element.is_selected()

    return _predicate


def element_located_to_be_selected(locator: Tuple[str, str]) -> Callable[[WebDriverOrWebElement], bool]:
    """An expectation for the element to be located is selected.

    Parameters:
    -----------
    locator : Tuple[str, str]
        Used to find the element.

    Returns:
    -------
    boolean : True if the element is selected, False otherwise.

    Example:
    --------
    >>> from selenium.webdriver.common.by import By
    >>> from selenium.webdriver.support.ui import WebDriverWait
    >>> from selenium.webdriver.support import expected_conditions as EC
    >>> is_selected = WebDriverWait(driver, 10).until(EC.element_located_to_be_selected((By.CLASS_NAME, "foo")))
    """

    def _predicate(driver: WebDriverOrWebElement):
        return driver.find_element(*locator).is_selected()

    return _predicate


def element_selection_state_to_be(element: WebElement, is_selected: bool) -> Callable[[Any], bool]:
    """An expectation for checking if the given element is selected.

    Parameters:
    -----------
    element : WebElement
        The WebElement to check.
    is_selected : bool

    Returns:
    -------
    boolean : True if the element's selection state is the same as is_selected

    Example:
    --------
    >>> from selenium.webdriver.common.by import By
    >>> from selenium.webdriver.support.ui import WebDriverWait
    >>> from selenium.webdriver.support import expected_conditions as EC
    >>> is_selected = WebDriverWait(driver, 10).until(
    ...     EC.element_selection_state_to_be(driver.find_element(By.CLASS_NAME, "foo"), True)
    ... )
    """

    def _predicate(_):
        return element.is_selected() == is_selected

    return _predicate


def element_located_selection_state_to_be(
    locator: Tuple[str, str], is_selected: bool
) -> Callable[[WebDriverOrWebElement], bool]:
    """An expectation to locate an element and check if the selection state
    specified is in that state.

    Parameters:
    -----------
    locator : Tuple[str, str]
        Used to find the element.
    is_selected : bool

    Returns:
    -------
    boolean : True if the element's selection state is the same as is_selected

    Example:
    --------
    >>> from selenium.webdriver.common.by import By
    >>> from selenium.webdriver.support.ui import WebDriverWait
    >>> from selenium.webdriver.support import expected_conditions as EC
    >>> is_selected = WebDriverWait(driver, 10).until(EC.element_located_selection_state_to_be(
            (By.CLASS_NAME, "foo"), True)
        )
    """

    def _predicate(driver: WebDriverOrWebElement):
        try:
            element = driver.find_element(*locator)
            return element.is_selected() == is_selected
        except StaleElementReferenceException:
            return False

    return _predicate


def number_of_windows_to_be(num_windows: int) -> Callable[[WebDriver], bool]:
    """An expectation for the number of windows to be a certain value.

    Parameters:
    -----------
    num_windows : int
        The expected number of windows.

    Returns:
    -------
    boolean : True when the number of windows matches, False otherwise.

    Example:
    --------
    >>> from selenium.webdriver.support.ui import WebDriverWait
    >>> from selenium.webdriver.support import expected_conditions as EC
    >>> is_number_of_windows = WebDriverWait(driver, 10).until(EC.number_of_windows_to_be(2))
    """

    def _predicate(driver: WebDriver):
        return len(driver.window_handles) == num_windows

    return _predicate


def new_window_is_opened(current_handles: List[str]) -> Callable[[WebDriver], bool]:
    """An expectation that a new window will be opened and have the number of
    windows handles increase.

    Parameters:
    -----------
    current_handles : List[str]
        The current window handles.

    Returns:
    -------
    boolean : True when a new window is opened, False otherwise.

    Example:
    --------
    >>> from selenium.webdriver.support.ui import By
    >>> from selenium.webdriver.support.ui import WebDriverWait
    >>> from selenium.webdriver.support import expected_conditions as EC
    >>> is_new_window_opened = WebDriverWait(driver, 10).until(EC.new_window_is_opened(driver.window_handles))
    """

    def _predicate(driver: WebDriver):
        return len(driver.window_handles) > len(current_handles)

    return _predicate


def alert_is_present() -> Callable[[WebDriver], Union[Alert, Literal[False]]]:
    """An expectation for checking if an alert is currently present and
    switching to it.

    Returns:
    -------
    Alert : The Alert once it is located.

    Example:
    --------
    >>> from selenium.webdriver.support.ui import WebDriverWait
    >>> from selenium.webdriver.support import expected_conditions as EC
    >>> alert = WebDriverWait(driver, 10).until(EC.alert_is_present())

    Notes:
    ------
    If the alert is present it switches the given driver to it.
    """

    def _predicate(driver: WebDriver):
        try:
            return driver.switch_to.alert
        except NoAlertPresentException:
            return False

    return _predicate


def element_attribute_to_include(locator: Tuple[str, str], attribute_: str) -> Callable[[WebDriverOrWebElement], bool]:
    """An expectation for checking if the given attribute is included in the
    specified element.

    Parameters:
    -----------
    locator : Tuple[str, str]
        Used to find the element.
    attribute_ : str
        The attribute to check.

    Returns:
    -------
    boolean : True when the attribute is included, False otherwise.

    Example:
    --------
    >>> from selenium.webdriver.common.by import By
    >>> from selenium.webdriver.support.ui import WebDriverWait
    >>> from selenium.webdriver.support import expected_conditions as EC
    >>> is_attribute_in_element = WebDriverWait(driver, 10).until(
    ...     EC.element_attribute_to_include((By.CLASS_NAME, "foo"), "bar")
    ... )
    """

    def _predicate(driver: WebDriverOrWebElement):
        try:
            element_attribute = driver.find_element(*locator).get_attribute(attribute_)
            return element_attribute is not None
        except StaleElementReferenceException:
            return False

    return _predicate


def any_of(*expected_conditions: Callable[[D], T]) -> Callable[[D], Union[Literal[False], T]]:
    """An expectation that any of multiple expected conditions is true.

    Parameters:
    -----------
    expected_conditions : Callable[[D], T]
        The list of expected conditions to check.

    Returns:
    -------
    T : The result of the first matching condition, or False if none do.

    Example:
    --------
    >>> from selenium.webdriver.common.by import By
    >>> from selenium.webdriver.support.ui import WebDriverWait
    >>> from selenium.webdriver.support import expected_conditions as EC
    >>> element = WebDriverWait(driver, 10).until(
    ... EC.any_of(EC.presence_of_element_located((By.NAME, "q"),
    ... EC.visibility_of_element_located((By.NAME, "q"))))

    Notes:
    ------
    Equivalent to a logical 'OR'. Returns results of the first matching
    condition, or False if none do.
    """

    def any_of_condition(driver: D):
        for expected_condition in expected_conditions:
            try:
                result = expected_condition(driver)
                if result:
                    return result
            except WebDriverException:
                pass
        return False

    return any_of_condition


def all_of(
    *expected_conditions: Callable[[D], Union[T, Literal[False]]],
) -> Callable[[D], Union[List[T], Literal[False]]]:
    """An expectation that all of multiple expected conditions is true.

    Parameters:
    -----------
    expected_conditions : Callable[[D], Union[T, Literal[False]]]
        The list of expected conditions to check.

    Returns:
    -------
    List[T] : The results of all the matching conditions, or False if any do not.

    Example:
    --------
    >>> from selenium.webdriver.common.by import By
    >>> from selenium.webdriver.support.ui import WebDriverWait
    >>> from selenium.webdriver.support import expected_conditions as EC
    >>> elements = WebDriverWait(driver, 10).until(
    ... EC.all_of(EC.presence_of_element_located((By.NAME, "q"),
    ... EC.visibility_of_element_located((By.NAME, "q"))))

    Notes:
    ------
    Equivalent to a logical 'AND'.
    Returns: When any ExpectedCondition is not met: False.
    When all ExpectedConditions are met: A List with each ExpectedCondition's return value.
    """

    def all_of_condition(driver: D):
        results: List[T] = []
        for expected_condition in expected_conditions:
            try:
                result = expected_condition(driver)
                if not result:
                    return False
                results.append(result)
            except WebDriverException:
                return False
        return results

    return all_of_condition


def none_of(*expected_conditions: Callable[[D], Any]) -> Callable[[D], bool]:
    """An expectation that none of 1 or multiple expected conditions is true.

    Parameters:
    -----------
    expected_conditions : Callable[[D], Any]
        The list of expected conditions to check.

    Returns:
    -------
    boolean : True if none of the conditions are true, False otherwise.

    Example:
    --------
    >>> from selenium.webdriver.common.by import By
    >>> from selenium.webdriver.support.ui import WebDriverWait
    >>> from selenium.webdriver.support import expected_conditions as EC
    >>> element = WebDriverWait(driver, 10).until(
    ... EC.none_of(EC.presence_of_element_located((By.NAME, "q"),
    ... EC.visibility_of_element_located((By.NAME, "q"))))

    Notes:
    ------
    Equivalent to a logical 'NOT-OR'. Returns a Boolean
    """

    def none_of_condition(driver: D):
        for expected_condition in expected_conditions:
            try:
                result = expected_condition(driver)
                if result:
                    return False
            except WebDriverException:
                pass
        return True

    return none_of_condition

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\pretty.py ===
import builtins
import collections
import dataclasses
import inspect
import os
import reprlib
import sys
from array import array
from collections import Counter, UserDict, UserList, defaultdict, deque
from dataclasses import dataclass, fields, is_dataclass
from inspect import isclass
from itertools import islice
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    DefaultDict,
    Deque,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from pip._vendor.rich.repr import RichReprResult

try:
    import attr as _attr_module

    _has_attrs = hasattr(_attr_module, "ib")
except ImportError:  # pragma: no cover
    _has_attrs = False

from . import get_console
from ._loop import loop_last
from ._pick import pick_bool
from .abc import RichRenderable
from .cells import cell_len
from .highlighter import ReprHighlighter
from .jupyter import JupyterMixin, JupyterRenderable
from .measure import Measurement
from .text import Text

if TYPE_CHECKING:
    from .console import (
        Console,
        ConsoleOptions,
        HighlighterType,
        JustifyMethod,
        OverflowMethod,
        RenderResult,
    )


def _is_attr_object(obj: Any) -> bool:
    """Check if an object was created with attrs module."""
    return _has_attrs and _attr_module.has(type(obj))


def _get_attr_fields(obj: Any) -> Sequence["_attr_module.Attribute[Any]"]:
    """Get fields for an attrs object."""
    return _attr_module.fields(type(obj)) if _has_attrs else []


def _is_dataclass_repr(obj: object) -> bool:
    """Check if an instance of a dataclass contains the default repr.

    Args:
        obj (object): A dataclass instance.

    Returns:
        bool: True if the default repr is used, False if there is a custom repr.
    """
    # Digging in to a lot of internals here
    # Catching all exceptions in case something is missing on a non CPython implementation
    try:
        return obj.__repr__.__code__.co_filename in (
            dataclasses.__file__,
            reprlib.__file__,
        )
    except Exception:  # pragma: no coverage
        return False


_dummy_namedtuple = collections.namedtuple("_dummy_namedtuple", [])


def _has_default_namedtuple_repr(obj: object) -> bool:
    """Check if an instance of namedtuple contains the default repr

    Args:
        obj (object): A namedtuple

    Returns:
        bool: True if the default repr is used, False if there's a custom repr.
    """
    obj_file = None
    try:
        obj_file = inspect.getfile(obj.__repr__)
    except (OSError, TypeError):
        # OSError handles case where object is defined in __main__ scope, e.g. REPL - no filename available.
        # TypeError trapped defensively, in case of object without filename slips through.
        pass
    default_repr_file = inspect.getfile(_dummy_namedtuple.__repr__)
    return obj_file == default_repr_file


def _ipy_display_hook(
    value: Any,
    console: Optional["Console"] = None,
    overflow: "OverflowMethod" = "ignore",
    crop: bool = False,
    indent_guides: bool = False,
    max_length: Optional[int] = None,
    max_string: Optional[int] = None,
    max_depth: Optional[int] = None,
    expand_all: bool = False,
) -> Union[str, None]:
    # needed here to prevent circular import:
    from .console import ConsoleRenderable

    # always skip rich generated jupyter renderables or None values
    if _safe_isinstance(value, JupyterRenderable) or value is None:
        return None

    console = console or get_console()

    with console.capture() as capture:
        # certain renderables should start on a new line
        if _safe_isinstance(value, ConsoleRenderable):
            console.line()
        console.print(
            (
                value
                if _safe_isinstance(value, RichRenderable)
                else Pretty(
                    value,
                    overflow=overflow,
                    indent_guides=indent_guides,
                    max_length=max_length,
                    max_string=max_string,
                    max_depth=max_depth,
                    expand_all=expand_all,
                    margin=12,
                )
            ),
            crop=crop,
            new_line_start=True,
            end="",
        )
    # strip trailing newline, not usually part of a text repr
    # I'm not sure if this should be prevented at a lower level
    return capture.get().rstrip("\n")


def _safe_isinstance(
    obj: object, class_or_tuple: Union[type, Tuple[type, ...]]
) -> bool:
    """isinstance can fail in rare cases, for example types with no __class__"""
    try:
        return isinstance(obj, class_or_tuple)
    except Exception:
        return False


def install(
    console: Optional["Console"] = None,
    overflow: "OverflowMethod" = "ignore",
    crop: bool = False,
    indent_guides: bool = False,
    max_length: Optional[int] = None,
    max_string: Optional[int] = None,
    max_depth: Optional[int] = None,
    expand_all: bool = False,
) -> None:
    """Install automatic pretty printing in the Python REPL.

    Args:
        console (Console, optional): Console instance or ``None`` to use global console. Defaults to None.
        overflow (Optional[OverflowMethod], optional): Overflow method. Defaults to "ignore".
        crop (Optional[bool], optional): Enable cropping of long lines. Defaults to False.
        indent_guides (bool, optional): Enable indentation guides. Defaults to False.
        max_length (int, optional): Maximum length of containers before abbreviating, or None for no abbreviation.
            Defaults to None.
        max_string (int, optional): Maximum length of string before truncating, or None to disable. Defaults to None.
        max_depth (int, optional): Maximum depth of nested data structures, or None for no maximum. Defaults to None.
        expand_all (bool, optional): Expand all containers. Defaults to False.
        max_frames (int): Maximum number of frames to show in a traceback, 0 for no maximum. Defaults to 100.
    """
    from pip._vendor.rich import get_console

    console = console or get_console()
    assert console is not None

    def display_hook(value: Any) -> None:
        """Replacement sys.displayhook which prettifies objects with Rich."""
        if value is not None:
            assert console is not None
            builtins._ = None  # type: ignore[attr-defined]
            console.print(
                (
                    value
                    if _safe_isinstance(value, RichRenderable)
                    else Pretty(
                        value,
                        overflow=overflow,
                        indent_guides=indent_guides,
                        max_length=max_length,
                        max_string=max_string,
                        max_depth=max_depth,
                        expand_all=expand_all,
                    )
                ),
                crop=crop,
            )
            builtins._ = value  # type: ignore[attr-defined]

    try:
        ip = get_ipython()  # type: ignore[name-defined]
    except NameError:
        sys.displayhook = display_hook
    else:
        from IPython.core.formatters import BaseFormatter

        class RichFormatter(BaseFormatter):  # type: ignore[misc]
            pprint: bool = True

            def __call__(self, value: Any) -> Any:
                if self.pprint:
                    return _ipy_display_hook(
                        value,
                        console=get_console(),
                        overflow=overflow,
                        indent_guides=indent_guides,
                        max_length=max_length,
                        max_string=max_string,
                        max_depth=max_depth,
                        expand_all=expand_all,
                    )
                else:
                    return repr(value)

        # replace plain text formatter with rich formatter
        rich_formatter = RichFormatter()
        ip.display_formatter.formatters["text/plain"] = rich_formatter


class Pretty(JupyterMixin):
    """A rich renderable that pretty prints an object.

    Args:
        _object (Any): An object to pretty print.
        highlighter (HighlighterType, optional): Highlighter object to apply to result, or None for ReprHighlighter. Defaults to None.
        indent_size (int, optional): Number of spaces in indent. Defaults to 4.
        justify (JustifyMethod, optional): Justify method, or None for default. Defaults to None.
        overflow (OverflowMethod, optional): Overflow method, or None for default. Defaults to None.
        no_wrap (Optional[bool], optional): Disable word wrapping. Defaults to False.
        indent_guides (bool, optional): Enable indentation guides. Defaults to False.
        max_length (int, optional): Maximum length of containers before abbreviating, or None for no abbreviation.
            Defaults to None.
        max_string (int, optional): Maximum length of string before truncating, or None to disable. Defaults to None.
        max_depth (int, optional): Maximum depth of nested data structures, or None for no maximum. Defaults to None.
        expand_all (bool, optional): Expand all containers. Defaults to False.
        margin (int, optional): Subtrace a margin from width to force containers to expand earlier. Defaults to 0.
        insert_line (bool, optional): Insert a new line if the output has multiple new lines. Defaults to False.
    """

    def __init__(
        self,
        _object: Any,
        highlighter: Optional["HighlighterType"] = None,
        *,
        indent_size: int = 4,
        justify: Optional["JustifyMethod"] = None,
        overflow: Optional["OverflowMethod"] = None,
        no_wrap: Optional[bool] = False,
        indent_guides: bool = False,
        max_length: Optional[int] = None,
        max_string: Optional[int] = None,
        max_depth: Optional[int] = None,
        expand_all: bool = False,
        margin: int = 0,
        insert_line: bool = False,
    ) -> None:
        self._object = _object
        self.highlighter = highlighter or ReprHighlighter()
        self.indent_size = indent_size
        self.justify: Optional["JustifyMethod"] = justify
        self.overflow: Optional["OverflowMethod"] = overflow
        self.no_wrap = no_wrap
        self.indent_guides = indent_guides
        self.max_length = max_length
        self.max_string = max_string
        self.max_depth = max_depth
        self.expand_all = expand_all
        self.margin = margin
        self.insert_line = insert_line

    def __rich_console__(
        self, console: "Console", options: "ConsoleOptions"
    ) -> "RenderResult":
        pretty_str = pretty_repr(
            self._object,
            max_width=options.max_width - self.margin,
            indent_size=self.indent_size,
            max_length=self.max_length,
            max_string=self.max_string,
            max_depth=self.max_depth,
            expand_all=self.expand_all,
        )
        pretty_text = Text.from_ansi(
            pretty_str,
            justify=self.justify or options.justify,
            overflow=self.overflow or options.overflow,
            no_wrap=pick_bool(self.no_wrap, options.no_wrap),
            style="pretty",
        )
        pretty_text = (
            self.highlighter(pretty_text)
            if pretty_text
            else Text(
                f"{type(self._object)}.__repr__ returned empty string",
                style="dim italic",
            )
        )
        if self.indent_guides and not options.ascii_only:
            pretty_text = pretty_text.with_indent_guides(
                self.indent_size, style="repr.indent"
            )
        if self.insert_line and "\n" in pretty_text:
            yield ""
        yield pretty_text

    def __rich_measure__(
        self, console: "Console", options: "ConsoleOptions"
    ) -> "Measurement":
        pretty_str = pretty_repr(
            self._object,
            max_width=options.max_width,
            indent_size=self.indent_size,
            max_length=self.max_length,
            max_string=self.max_string,
            max_depth=self.max_depth,
            expand_all=self.expand_all,
        )
        text_width = (
            max(cell_len(line) for line in pretty_str.splitlines()) if pretty_str else 0
        )
        return Measurement(text_width, text_width)


def _get_braces_for_defaultdict(_object: DefaultDict[Any, Any]) -> Tuple[str, str, str]:
    return (
        f"defaultdict({_object.default_factory!r}, {{",
        "})",
        f"defaultdict({_object.default_factory!r}, {{}})",
    )


def _get_braces_for_deque(_object: Deque[Any]) -> Tuple[str, str, str]:
    if _object.maxlen is None:
        return ("deque([", "])", "deque()")
    return (
        "deque([",
        f"], maxlen={_object.maxlen})",
        f"deque(maxlen={_object.maxlen})",
    )


def _get_braces_for_array(_object: "array[Any]") -> Tuple[str, str, str]:
    return (f"array({_object.typecode!r}, [", "])", f"array({_object.typecode!r})")


_BRACES: Dict[type, Callable[[Any], Tuple[str, str, str]]] = {
    os._Environ: lambda _object: ("environ({", "})", "environ({})"),
    array: _get_braces_for_array,
    defaultdict: _get_braces_for_defaultdict,
    Counter: lambda _object: ("Counter({", "})", "Counter()"),
    deque: _get_braces_for_deque,
    dict: lambda _object: ("{", "}", "{}"),
    UserDict: lambda _object: ("{", "}", "{}"),
    frozenset: lambda _object: ("frozenset({", "})", "frozenset()"),
    list: lambda _object: ("[", "]", "[]"),
    UserList: lambda _object: ("[", "]", "[]"),
    set: lambda _object: ("{", "}", "set()"),
    tuple: lambda _object: ("(", ")", "()"),
    MappingProxyType: lambda _object: ("mappingproxy({", "})", "mappingproxy({})"),
}
_CONTAINERS = tuple(_BRACES.keys())
_MAPPING_CONTAINERS = (dict, os._Environ, MappingProxyType, UserDict)


def is_expandable(obj: Any) -> bool:
    """Check if an object may be expanded by pretty print."""
    return (
        _safe_isinstance(obj, _CONTAINERS)
        or (is_dataclass(obj))
        or (hasattr(obj, "__rich_repr__"))
        or _is_attr_object(obj)
    ) and not isclass(obj)


@dataclass
class Node:
    """A node in a repr tree. May be atomic or a container."""

    key_repr: str = ""
    value_repr: str = ""
    open_brace: str = ""
    close_brace: str = ""
    empty: str = ""
    last: bool = False
    is_tuple: bool = False
    is_namedtuple: bool = False
    children: Optional[List["Node"]] = None
    key_separator: str = ": "
    separator: str = ", "

    def iter_tokens(self) -> Iterable[str]:
        """Generate tokens for this node."""
        if self.key_repr:
            yield self.key_repr
            yield self.key_separator
        if self.value_repr:
            yield self.value_repr
        elif self.children is not None:
            if self.children:
                yield self.open_brace
                if self.is_tuple and not self.is_namedtuple and len(self.children) == 1:
                    yield from self.children[0].iter_tokens()
                    yield ","
                else:
                    for child in self.children:
                        yield from child.iter_tokens()
                        if not child.last:
                            yield self.separator
                yield self.close_brace
            else:
                yield self.empty

    def check_length(self, start_length: int, max_length: int) -> bool:
        """Check the length fits within a limit.

        Args:
            start_length (int): Starting length of the line (indent, prefix, suffix).
            max_length (int): Maximum length.

        Returns:
            bool: True if the node can be rendered within max length, otherwise False.
        """
        total_length = start_length
        for token in self.iter_tokens():
            total_length += cell_len(token)
            if total_length > max_length:
                return False
        return True

    def __str__(self) -> str:
        repr_text = "".join(self.iter_tokens())
        return repr_text

    def render(
        self, max_width: int = 80, indent_size: int = 4, expand_all: bool = False
    ) -> str:
        """Render the node to a pretty repr.

        Args:
            max_width (int, optional): Maximum width of the repr. Defaults to 80.
            indent_size (int, optional): Size of indents. Defaults to 4.
            expand_all (bool, optional): Expand all levels. Defaults to False.

        Returns:
            str: A repr string of the original object.
        """
        lines = [_Line(node=self, is_root=True)]
        line_no = 0
        while line_no < len(lines):
            line = lines[line_no]
            if line.expandable and not line.expanded:
                if expand_all or not line.check_length(max_width):
                    lines[line_no : line_no + 1] = line.expand(indent_size)
            line_no += 1

        repr_str = "\n".join(str(line) for line in lines)
        return repr_str


@dataclass
class _Line:
    """A line in repr output."""

    parent: Optional["_Line"] = None
    is_root: bool = False
    node: Optional[Node] = None
    text: str = ""
    suffix: str = ""
    whitespace: str = ""
    expanded: bool = False
    last: bool = False

    @property
    def expandable(self) -> bool:
        """Check if the line may be expanded."""
        return bool(self.node is not None and self.node.children)

    def check_length(self, max_length: int) -> bool:
        """Check this line fits within a given number of cells."""
        start_length = (
            len(self.whitespace) + cell_len(self.text) + cell_len(self.suffix)
        )
        assert self.node is not None
        return self.node.check_length(start_length, max_length)

    def expand(self, indent_size: int) -> Iterable["_Line"]:
        """Expand this line by adding children on their own line."""
        node = self.node
        assert node is not None
        whitespace = self.whitespace
        assert node.children
        if node.key_repr:
            new_line = yield _Line(
                text=f"{node.key_repr}{node.key_separator}{node.open_brace}",
                whitespace=whitespace,
            )
        else:
            new_line = yield _Line(text=node.open_brace, whitespace=whitespace)
        child_whitespace = self.whitespace + " " * indent_size
        tuple_of_one = node.is_tuple and len(node.children) == 1
        for last, child in loop_last(node.children):
            separator = "," if tuple_of_one else node.separator
            line = _Line(
                parent=new_line,
                node=child,
                whitespace=child_whitespace,
                suffix=separator,
                last=last and not tuple_of_one,
            )
            yield line

        yield _Line(
            text=node.close_brace,
            whitespace=whitespace,
            suffix=self.suffix,
            last=self.last,
        )

    def __str__(self) -> str:
        if self.last:
            return f"{self.whitespace}{self.text}{self.node or ''}"
        else:
            return (
                f"{self.whitespace}{self.text}{self.node or ''}{self.suffix.rstrip()}"
            )


def _is_namedtuple(obj: Any) -> bool:
    """Checks if an object is most likely a namedtuple. It is possible
    to craft an object that passes this check and isn't a namedtuple, but
    there is only a minuscule chance of this happening unintentionally.

    Args:
        obj (Any): The object to test

    Returns:
        bool: True if the object is a namedtuple. False otherwise.
    """
    try:
        fields = getattr(obj, "_fields", None)
    except Exception:
        # Being very defensive - if we cannot get the attr then its not a namedtuple
        return False
    return isinstance(obj, tuple) and isinstance(fields, tuple)


def traverse(
    _object: Any,
    max_length: Optional[int] = None,
    max_string: Optional[int] = None,
    max_depth: Optional[int] = None,
) -> Node:
    """Traverse object and generate a tree.

    Args:
        _object (Any): Object to be traversed.
        max_length (int, optional): Maximum length of containers before abbreviating, or None for no abbreviation.
            Defaults to None.
        max_string (int, optional): Maximum length of string before truncating, or None to disable truncating.
            Defaults to None.
        max_depth (int, optional): Maximum depth of data structures, or None for no maximum.
            Defaults to None.

    Returns:
        Node: The root of a tree structure which can be used to render a pretty repr.
    """

    def to_repr(obj: Any) -> str:
        """Get repr string for an object, but catch errors."""
        if (
            max_string is not None
            and _safe_isinstance(obj, (bytes, str))
            and len(obj) > max_string
        ):
            truncated = len(obj) - max_string
            obj_repr = f"{obj[:max_string]!r}+{truncated}"
        else:
            try:
                obj_repr = repr(obj)
            except Exception as error:
                obj_repr = f"<repr-error {str(error)!r}>"
        return obj_repr

    visited_ids: Set[int] = set()
    push_visited = visited_ids.add
    pop_visited = visited_ids.remove

    def _traverse(obj: Any, root: bool = False, depth: int = 0) -> Node:
        """Walk the object depth first."""

        obj_id = id(obj)
        if obj_id in visited_ids:
            # Recursion detected
            return Node(value_repr="...")

        obj_type = type(obj)
        children: List[Node]
        reached_max_depth = max_depth is not None and depth >= max_depth

        def iter_rich_args(rich_args: Any) -> Iterable[Union[Any, Tuple[str, Any]]]:
            for arg in rich_args:
                if _safe_isinstance(arg, tuple):
                    if len(arg) == 3:
                        key, child, default = arg
                        if default == child:
                            continue
                        yield key, child
                    elif len(arg) == 2:
                        key, child = arg
                        yield key, child
                    elif len(arg) == 1:
                        yield arg[0]
                else:
                    yield arg

        try:
            fake_attributes = hasattr(
                obj, "awehoi234_wdfjwljet234_234wdfoijsdfmmnxpi492"
            )
        except Exception:
            fake_attributes = False

        rich_repr_result: Optional[RichReprResult] = None
        if not fake_attributes:
            try:
                if hasattr(obj, "__rich_repr__") and not isclass(obj):
                    rich_repr_result = obj.__rich_repr__()
            except Exception:
                pass

        if rich_repr_result is not None:
            push_visited(obj_id)
            angular = getattr(obj.__rich_repr__, "angular", False)
            args = list(iter_rich_args(rich_repr_result))
            class_name = obj.__class__.__name__

            if args:
                children = []
                append = children.append

                if reached_max_depth:
                    if angular:
                        node = Node(value_repr=f"<{class_name}...>")
                    else:
                        node = Node(value_repr=f"{class_name}(...)")
                else:
                    if angular:
                        node = Node(
                            open_brace=f"<{class_name} ",
                            close_brace=">",
                            children=children,
                            last=root,
                            separator=" ",
                        )
                    else:
                        node = Node(
                            open_brace=f"{class_name}(",
                            close_brace=")",
                            children=children,
                            last=root,
                        )
                    for last, arg in loop_last(args):
                        if _safe_isinstance(arg, tuple):
                            key, child = arg
                            child_node = _traverse(child, depth=depth + 1)
                            child_node.last = last
                            child_node.key_repr = key
                            child_node.key_separator = "="
                            append(child_node)
                        else:
                            child_node = _traverse(arg, depth=depth + 1)
                            child_node.last = last
                            append(child_node)
            else:
                node = Node(
                    value_repr=f"<{class_name}>" if angular else f"{class_name}()",
                    children=[],
                    last=root,
                )
            pop_visited(obj_id)
        elif _is_attr_object(obj) and not fake_attributes:
            push_visited(obj_id)
            children = []
            append = children.append

            attr_fields = _get_attr_fields(obj)
            if attr_fields:
                if reached_max_depth:
                    node = Node(value_repr=f"{obj.__class__.__name__}(...)")
                else:
                    node = Node(
                        open_brace=f"{obj.__class__.__name__}(",
                        close_brace=")",
                        children=children,
                        last=root,
                    )

                    def iter_attrs() -> (
                        Iterable[Tuple[str, Any, Optional[Callable[[Any], str]]]]
                    ):
                        """Iterate over attr fields and values."""
                        for attr in attr_fields:
                            if attr.repr:
                                try:
                                    value = getattr(obj, attr.name)
                                except Exception as error:
                                    # Can happen, albeit rarely
                                    yield (attr.name, error, None)
                                else:
                                    yield (
                                        attr.name,
                                        value,
                                        attr.repr if callable(attr.repr) else None,
                                    )

                    for last, (name, value, repr_callable) in loop_last(iter_attrs()):
                        if repr_callable:
                            child_node = Node(value_repr=str(repr_callable(value)))
                        else:
                            child_node = _traverse(value, depth=depth + 1)
                        child_node.last = last
                        child_node.key_repr = name
                        child_node.key_separator = "="
                        append(child_node)
            else:
                node = Node(
                    value_repr=f"{obj.__class__.__name__}()", children=[], last=root
                )
            pop_visited(obj_id)
        elif (
            is_dataclass(obj)
            and not _safe_isinstance(obj, type)
            and not fake_attributes
            and _is_dataclass_repr(obj)
        ):
            push_visited(obj_id)
            children = []
            append = children.append
            if reached_max_depth:
                node = Node(value_repr=f"{obj.__class__.__name__}(...)")
            else:
                node = Node(
                    open_brace=f"{obj.__class__.__name__}(",
                    close_brace=")",
                    children=children,
                    last=root,
                    empty=f"{obj.__class__.__name__}()",
                )

                for last, field in loop_last(
                    field
                    for field in fields(obj)
                    if field.repr and hasattr(obj, field.name)
                ):
                    child_node = _traverse(getattr(obj, field.name), depth=depth + 1)
                    child_node.key_repr = field.name
                    child_node.last = last
                    child_node.key_separator = "="
                    append(child_node)

            pop_visited(obj_id)
        elif _is_namedtuple(obj) and _has_default_namedtuple_repr(obj):
            push_visited(obj_id)
            class_name = obj.__class__.__name__
            if reached_max_depth:
                # If we've reached the max depth, we still show the class name, but not its contents
                node = Node(
                    value_repr=f"{class_name}(...)",
                )
            else:
                children = []
                append = children.append
                node = Node(
                    open_brace=f"{class_name}(",
                    close_brace=")",
                    children=children,
                    empty=f"{class_name}()",
                )
                for last, (key, value) in loop_last(obj._asdict().items()):
                    child_node = _traverse(value, depth=depth + 1)
                    child_node.key_repr = key
                    child_node.last = last
                    child_node.key_separator = "="
                    append(child_node)
            pop_visited(obj_id)
        elif _safe_isinstance(obj, _CONTAINERS):
            for container_type in _CONTAINERS:
                if _safe_isinstance(obj, container_type):
                    obj_type = container_type
                    break

            push_visited(obj_id)

            open_brace, close_brace, empty = _BRACES[obj_type](obj)

            if reached_max_depth:
                node = Node(value_repr=f"{open_brace}...{close_brace}")
            elif obj_type.__repr__ != type(obj).__repr__:
                node = Node(value_repr=to_repr(obj), last=root)
            elif obj:
                children = []
                node = Node(
                    open_brace=open_brace,
                    close_brace=close_brace,
                    children=children,
                    last=root,
                )
                append = children.append
                num_items = len(obj)
                last_item_index = num_items - 1

                if _safe_isinstance(obj, _MAPPING_CONTAINERS):
                    iter_items = iter(obj.items())
                    if max_length is not None:
                        iter_items = islice(iter_items, max_length)
                    for index, (key, child) in enumerate(iter_items):
                        child_node = _traverse(child, depth=depth + 1)
                        child_node.key_repr = to_repr(key)
                        child_node.last = index == last_item_index
                        append(child_node)
                else:
                    iter_values = iter(obj)
                    if max_length is not None:
                        iter_values = islice(iter_values, max_length)
                    for index, child in enumerate(iter_values):
                        child_node = _traverse(child, depth=depth + 1)
                        child_node.last = index == last_item_index
                        append(child_node)
                if max_length is not None and num_items > max_length:
                    append(Node(value_repr=f"... +{num_items - max_length}", last=True))
            else:
                node = Node(empty=empty, children=[], last=root)

            pop_visited(obj_id)
        else:
            node = Node(value_repr=to_repr(obj), last=root)
        node.is_tuple = type(obj) == tuple
        node.is_namedtuple = _is_namedtuple(obj)
        return node

    node = _traverse(_object, root=True)
    return node


def pretty_repr(
    _object: Any,
    *,
    max_width: int = 80,
    indent_size: int = 4,
    max_length: Optional[int] = None,
    max_string: Optional[int] = None,
    max_depth: Optional[int] = None,
    expand_all: bool = False,
) -> str:
    """Prettify repr string by expanding on to new lines to fit within a given width.

    Args:
        _object (Any): Object to repr.
        max_width (int, optional): Desired maximum width of repr string. Defaults to 80.
        indent_size (int, optional): Number of spaces to indent. Defaults to 4.
        max_length (int, optional): Maximum length of containers before abbreviating, or None for no abbreviation.
            Defaults to None.
        max_string (int, optional): Maximum length of string before truncating, or None to disable truncating.
            Defaults to None.
        max_depth (int, optional): Maximum depth of nested data structure, or None for no depth.
            Defaults to None.
        expand_all (bool, optional): Expand all containers regardless of available width. Defaults to False.

    Returns:
        str: A possibly multi-line representation of the object.
    """

    if _safe_isinstance(_object, Node):
        node = _object
    else:
        node = traverse(
            _object, max_length=max_length, max_string=max_string, max_depth=max_depth
        )
    repr_str: str = node.render(
        max_width=max_width, indent_size=indent_size, expand_all=expand_all
    )
    return repr_str


def pprint(
    _object: Any,
    *,
    console: Optional["Console"] = None,
    indent_guides: bool = True,
    max_length: Optional[int] = None,
    max_string: Optional[int] = None,
    max_depth: Optional[int] = None,
    expand_all: bool = False,
) -> None:
    """A convenience function for pretty printing.

    Args:
        _object (Any): Object to pretty print.
        console (Console, optional): Console instance, or None to use default. Defaults to None.
        max_length (int, optional): Maximum length of containers before abbreviating, or None for no abbreviation.
            Defaults to None.
        max_string (int, optional): Maximum length of strings before truncating, or None to disable. Defaults to None.
        max_depth (int, optional): Maximum depth for nested data structures, or None for unlimited depth. Defaults to None.
        indent_guides (bool, optional): Enable indentation guides. Defaults to True.
        expand_all (bool, optional): Expand all containers. Defaults to False.
    """
    _console = get_console() if console is None else console
    _console.print(
        Pretty(
            _object,
            max_length=max_length,
            max_string=max_string,
            max_depth=max_depth,
            indent_guides=indent_guides,
            expand_all=expand_all,
            overflow="ignore",
        ),
        soft_wrap=True,
    )


if __name__ == "__main__":  # pragma: no cover

    class BrokenRepr:
        def __repr__(self) -> str:
            1 / 0
            return "this will fail"

    from typing import NamedTuple

    class StockKeepingUnit(NamedTuple):
        name: str
        description: str
        price: float
        category: str
        reviews: List[str]

    d = defaultdict(int)
    d["foo"] = 5
    data = {
        "foo": [
            1,
            "Hello World!",
            100.123,
            323.232,
            432324.0,
            {5, 6, 7, (1, 2, 3, 4), 8},
        ],
        "bar": frozenset({1, 2, 3}),
        "defaultdict": defaultdict(
            list, {"crumble": ["apple", "rhubarb", "butter", "sugar", "flour"]}
        ),
        "counter": Counter(
            [
                "apple",
                "orange",
                "pear",
                "kumquat",
                "kumquat",
                "durian" * 100,
            ]
        ),
        "atomic": (False, True, None),
        "namedtuple": StockKeepingUnit(
            "Sparkling British Spring Water",
            "Carbonated spring water",
            0.9,
            "water",
            ["its amazing!", "its terrible!"],
        ),
        "Broken": BrokenRepr(),
    }
    data["foo"].append(data)  # type: ignore[attr-defined]

    from pip._vendor.rich import print

    print(Pretty(data, indent_guides=True, max_string=20))

    class Thing:
        def __repr__(self) -> str:
            return "Hello\x1b[38;5;239m World!"

    print(Pretty(Thing()))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\cffi\cparser.py ===
from . import model
from .commontypes import COMMON_TYPES, resolve_common_type
from .error import FFIError, CDefError
try:
    from . import _pycparser as pycparser
except ImportError:
    import pycparser
import weakref, re, sys

try:
    if sys.version_info < (3,):
        import thread as _thread
    else:
        import _thread
    lock = _thread.allocate_lock()
except ImportError:
    lock = None

def _workaround_for_static_import_finders():
    # Issue #392: packaging tools like cx_Freeze can not find these
    # because pycparser uses exec dynamic import.  This is an obscure
    # workaround.  This function is never called.
    import pycparser.yacctab
    import pycparser.lextab

CDEF_SOURCE_STRING = "<cdef source string>"
_r_comment = re.compile(r"/\*.*?\*/|//([^\n\\]|\\.)*?$",
                        re.DOTALL | re.MULTILINE)
_r_define  = re.compile(r"^\s*#\s*define\s+([A-Za-z_][A-Za-z_0-9]*)"
                        r"\b((?:[^\n\\]|\\.)*?)$",
                        re.DOTALL | re.MULTILINE)
_r_line_directive = re.compile(r"^[ \t]*#[ \t]*(?:line|\d+)\b.*$", re.MULTILINE)
_r_partial_enum = re.compile(r"=\s*\.\.\.\s*[,}]|\.\.\.\s*\}")
_r_enum_dotdotdot = re.compile(r"__dotdotdot\d+__$")
_r_partial_array = re.compile(r"\[\s*\.\.\.\s*\]")
_r_words = re.compile(r"\w+|\S")
_parser_cache = None
_r_int_literal = re.compile(r"-?0?x?[0-9a-f]+[lu]*$", re.IGNORECASE)
_r_stdcall1 = re.compile(r"\b(__stdcall|WINAPI)\b")
_r_stdcall2 = re.compile(r"[(]\s*(__stdcall|WINAPI)\b")
_r_cdecl = re.compile(r"\b__cdecl\b")
_r_extern_python = re.compile(r'\bextern\s*"'
                              r'(Python|Python\s*\+\s*C|C\s*\+\s*Python)"\s*.')
_r_star_const_space = re.compile(       # matches "* const "
    r"[*]\s*((const|volatile|restrict)\b\s*)+")
_r_int_dotdotdot = re.compile(r"(\b(int|long|short|signed|unsigned|char)\s*)+"
                              r"\.\.\.")
_r_float_dotdotdot = re.compile(r"\b(double|float)\s*\.\.\.")

def _get_parser():
    global _parser_cache
    if _parser_cache is None:
        _parser_cache = pycparser.CParser()
    return _parser_cache

def _workaround_for_old_pycparser(csource):
    # Workaround for a pycparser issue (fixed between pycparser 2.10 and
    # 2.14): "char*const***" gives us a wrong syntax tree, the same as
    # for "char***(*const)".  This means we can't tell the difference
    # afterwards.  But "char(*const(***))" gives us the right syntax
    # tree.  The issue only occurs if there are several stars in
    # sequence with no parenthesis inbetween, just possibly qualifiers.
    # Attempt to fix it by adding some parentheses in the source: each
    # time we see "* const" or "* const *", we add an opening
    # parenthesis before each star---the hard part is figuring out where
    # to close them.
    parts = []
    while True:
        match = _r_star_const_space.search(csource)
        if not match:
            break
        #print repr(''.join(parts)+csource), '=>',
        parts.append(csource[:match.start()])
        parts.append('('); closing = ')'
        parts.append(match.group())   # e.g. "* const "
        endpos = match.end()
        if csource.startswith('*', endpos):
            parts.append('('); closing += ')'
        level = 0
        i = endpos
        while i < len(csource):
            c = csource[i]
            if c == '(':
                level += 1
            elif c == ')':
                if level == 0:
                    break
                level -= 1
            elif c in ',;=':
                if level == 0:
                    break
            i += 1
        csource = csource[endpos:i] + closing + csource[i:]
        #print repr(''.join(parts)+csource)
    parts.append(csource)
    return ''.join(parts)

def _preprocess_extern_python(csource):
    # input: `extern "Python" int foo(int);` or
    #        `extern "Python" { int foo(int); }`
    # output:
    #     void __cffi_extern_python_start;
    #     int foo(int);
    #     void __cffi_extern_python_stop;
    #
    # input: `extern "Python+C" int foo(int);`
    # output:
    #     void __cffi_extern_python_plus_c_start;
    #     int foo(int);
    #     void __cffi_extern_python_stop;
    parts = []
    while True:
        match = _r_extern_python.search(csource)
        if not match:
            break
        endpos = match.end() - 1
        #print
        #print ''.join(parts)+csource
        #print '=>'
        parts.append(csource[:match.start()])
        if 'C' in match.group(1):
            parts.append('void __cffi_extern_python_plus_c_start; ')
        else:
            parts.append('void __cffi_extern_python_start; ')
        if csource[endpos] == '{':
            # grouping variant
            closing = csource.find('}', endpos)
            if closing < 0:
                raise CDefError("'extern \"Python\" {': no '}' found")
            if csource.find('{', endpos + 1, closing) >= 0:
                raise NotImplementedError("cannot use { } inside a block "
                                          "'extern \"Python\" { ... }'")
            parts.append(csource[endpos+1:closing])
            csource = csource[closing+1:]
        else:
            # non-grouping variant
            semicolon = csource.find(';', endpos)
            if semicolon < 0:
                raise CDefError("'extern \"Python\": no ';' found")
            parts.append(csource[endpos:semicolon+1])
            csource = csource[semicolon+1:]
        parts.append(' void __cffi_extern_python_stop;')
        #print ''.join(parts)+csource
        #print
    parts.append(csource)
    return ''.join(parts)

def _warn_for_string_literal(csource):
    if '"' not in csource:
        return
    for line in csource.splitlines():
        if '"' in line and not line.lstrip().startswith('#'):
            import warnings
            warnings.warn("String literal found in cdef() or type source. "
                          "String literals are ignored here, but you should "
                          "remove them anyway because some character sequences "
                          "confuse pre-parsing.")
            break

def _warn_for_non_extern_non_static_global_variable(decl):
    if not decl.storage:
        import warnings
        warnings.warn("Global variable '%s' in cdef(): for consistency "
                      "with C it should have a storage class specifier "
                      "(usually 'extern')" % (decl.name,))

def _remove_line_directives(csource):
    # _r_line_directive matches whole lines, without the final \n, if they
    # start with '#line' with some spacing allowed, or '#NUMBER'.  This
    # function stores them away and replaces them with exactly the string
    # '#line@N', where N is the index in the list 'line_directives'.
    line_directives = []
    def replace(m):
        i = len(line_directives)
        line_directives.append(m.group())
        return '#line@%d' % i
    csource = _r_line_directive.sub(replace, csource)
    return csource, line_directives

def _put_back_line_directives(csource, line_directives):
    def replace(m):
        s = m.group()
        if not s.startswith('#line@'):
            raise AssertionError("unexpected #line directive "
                                 "(should have been processed and removed")
        return line_directives[int(s[6:])]
    return _r_line_directive.sub(replace, csource)

def _preprocess(csource):
    # First, remove the lines of the form '#line N "filename"' because
    # the "filename" part could confuse the rest
    csource, line_directives = _remove_line_directives(csource)
    # Remove comments.  NOTE: this only work because the cdef() section
    # should not contain any string literals (except in line directives)!
    def replace_keeping_newlines(m):
        return ' ' + m.group().count('\n') * '\n'
    csource = _r_comment.sub(replace_keeping_newlines, csource)
    # Remove the "#define FOO x" lines
    macros = {}
    for match in _r_define.finditer(csource):
        macroname, macrovalue = match.groups()
        macrovalue = macrovalue.replace('\\\n', '').strip()
        macros[macroname] = macrovalue
    csource = _r_define.sub('', csource)
    #
    if pycparser.__version__ < '2.14':
        csource = _workaround_for_old_pycparser(csource)
    #
    # BIG HACK: replace WINAPI or __stdcall with "volatile const".
    # It doesn't make sense for the return type of a function to be
    # "volatile volatile const", so we abuse it to detect __stdcall...
    # Hack number 2 is that "int(volatile *fptr)();" is not valid C
    # syntax, so we place the "volatile" before the opening parenthesis.
    csource = _r_stdcall2.sub(' volatile volatile const(', csource)
    csource = _r_stdcall1.sub(' volatile volatile const ', csource)
    csource = _r_cdecl.sub(' ', csource)
    #
    # Replace `extern "Python"` with start/end markers
    csource = _preprocess_extern_python(csource)
    #
    # Now there should not be any string literal left; warn if we get one
    _warn_for_string_literal(csource)
    #
    # Replace "[...]" with "[__dotdotdotarray__]"
    csource = _r_partial_array.sub('[__dotdotdotarray__]', csource)
    #
    # Replace "...}" with "__dotdotdotNUM__}".  This construction should
    # occur only at the end of enums; at the end of structs we have "...;}"
    # and at the end of vararg functions "...);".  Also replace "=...[,}]"
    # with ",__dotdotdotNUM__[,}]": this occurs in the enums too, when
    # giving an unknown value.
    matches = list(_r_partial_enum.finditer(csource))
    for number, match in enumerate(reversed(matches)):
        p = match.start()
        if csource[p] == '=':
            p2 = csource.find('...', p, match.end())
            assert p2 > p
            csource = '%s,__dotdotdot%d__ %s' % (csource[:p], number,
                                                 csource[p2+3:])
        else:
            assert csource[p:p+3] == '...'
            csource = '%s __dotdotdot%d__ %s' % (csource[:p], number,
                                                 csource[p+3:])
    # Replace "int ..." or "unsigned long int..." with "__dotdotdotint__"
    csource = _r_int_dotdotdot.sub(' __dotdotdotint__ ', csource)
    # Replace "float ..." or "double..." with "__dotdotdotfloat__"
    csource = _r_float_dotdotdot.sub(' __dotdotdotfloat__ ', csource)
    # Replace all remaining "..." with the same name, "__dotdotdot__",
    # which is declared with a typedef for the purpose of C parsing.
    csource = csource.replace('...', ' __dotdotdot__ ')
    # Finally, put back the line directives
    csource = _put_back_line_directives(csource, line_directives)
    return csource, macros

def _common_type_names(csource):
    # Look in the source for what looks like usages of types from the
    # list of common types.  A "usage" is approximated here as the
    # appearance of the word, minus a "definition" of the type, which
    # is the last word in a "typedef" statement.  Approximative only
    # but should be fine for all the common types.
    look_for_words = set(COMMON_TYPES)
    look_for_words.add(';')
    look_for_words.add(',')
    look_for_words.add('(')
    look_for_words.add(')')
    look_for_words.add('typedef')
    words_used = set()
    is_typedef = False
    paren = 0
    previous_word = ''
    for word in _r_words.findall(csource):
        if word in look_for_words:
            if word == ';':
                if is_typedef:
                    words_used.discard(previous_word)
                    look_for_words.discard(previous_word)
                    is_typedef = False
            elif word == 'typedef':
                is_typedef = True
                paren = 0
            elif word == '(':
                paren += 1
            elif word == ')':
                paren -= 1
            elif word == ',':
                if is_typedef and paren == 0:
                    words_used.discard(previous_word)
                    look_for_words.discard(previous_word)
            else:   # word in COMMON_TYPES
                words_used.add(word)
        previous_word = word
    return words_used


class Parser(object):

    def __init__(self):
        self._declarations = {}
        self._included_declarations = set()
        self._anonymous_counter = 0
        self._structnode2type = weakref.WeakKeyDictionary()
        self._options = {}
        self._int_constants = {}
        self._recomplete = []
        self._uses_new_feature = None

    def _parse(self, csource):
        csource, macros = _preprocess(csource)
        # XXX: for more efficiency we would need to poke into the
        # internals of CParser...  the following registers the
        # typedefs, because their presence or absence influences the
        # parsing itself (but what they are typedef'ed to plays no role)
        ctn = _common_type_names(csource)
        typenames = []
        for name in sorted(self._declarations):
            if name.startswith('typedef '):
                name = name[8:]
                typenames.append(name)
                ctn.discard(name)
        typenames += sorted(ctn)
        #
        csourcelines = []
        csourcelines.append('# 1 "<cdef automatic initialization code>"')
        for typename in typenames:
            csourcelines.append('typedef int %s;' % typename)
        csourcelines.append('typedef int __dotdotdotint__, __dotdotdotfloat__,'
                            ' __dotdotdot__;')
        # this forces pycparser to consider the following in the file
        # called <cdef source string> from line 1
        csourcelines.append('# 1 "%s"' % (CDEF_SOURCE_STRING,))
        csourcelines.append(csource)
        csourcelines.append('')   # see test_missing_newline_bug
        fullcsource = '\n'.join(csourcelines)
        if lock is not None:
            lock.acquire()     # pycparser is not thread-safe...
        try:
            ast = _get_parser().parse(fullcsource)
        except pycparser.c_parser.ParseError as e:
            self.convert_pycparser_error(e, csource)
        finally:
            if lock is not None:
                lock.release()
        # csource will be used to find buggy source text
        return ast, macros, csource

    def _convert_pycparser_error(self, e, csource):
        # xxx look for "<cdef source string>:NUM:" at the start of str(e)
        # and interpret that as a line number.  This will not work if
        # the user gives explicit ``# NUM "FILE"`` directives.
        line = None
        msg = str(e)
        match = re.match(r"%s:(\d+):" % (CDEF_SOURCE_STRING,), msg)
        if match:
            linenum = int(match.group(1), 10)
            csourcelines = csource.splitlines()
            if 1 <= linenum <= len(csourcelines):
                line = csourcelines[linenum-1]
        return line

    def convert_pycparser_error(self, e, csource):
        line = self._convert_pycparser_error(e, csource)

        msg = str(e)
        if line:
            msg = 'cannot parse "%s"\n%s' % (line.strip(), msg)
        else:
            msg = 'parse error\n%s' % (msg,)
        raise CDefError(msg)

    def parse(self, csource, override=False, packed=False, pack=None,
                    dllexport=False):
        if packed:
            if packed != True:
                raise ValueError("'packed' should be False or True; use "
                                 "'pack' to give another value")
            if pack:
                raise ValueError("cannot give both 'pack' and 'packed'")
            pack = 1
        elif pack:
            if pack & (pack - 1):
                raise ValueError("'pack' must be a power of two, not %r" %
                    (pack,))
        else:
            pack = 0
        prev_options = self._options
        try:
            self._options = {'override': override,
                             'packed': pack,
                             'dllexport': dllexport}
            self._internal_parse(csource)
        finally:
            self._options = prev_options

    def _internal_parse(self, csource):
        ast, macros, csource = self._parse(csource)
        # add the macros
        self._process_macros(macros)
        # find the first "__dotdotdot__" and use that as a separator
        # between the repeated typedefs and the real csource
        iterator = iter(ast.ext)
        for decl in iterator:
            if decl.name == '__dotdotdot__':
                break
        else:
            assert 0
        current_decl = None
        #
        try:
            self._inside_extern_python = '__cffi_extern_python_stop'
            for decl in iterator:
                current_decl = decl
                if isinstance(decl, pycparser.c_ast.Decl):
                    self._parse_decl(decl)
                elif isinstance(decl, pycparser.c_ast.Typedef):
                    if not decl.name:
                        raise CDefError("typedef does not declare any name",
                                        decl)
                    quals = 0
                    if (isinstance(decl.type.type, pycparser.c_ast.IdentifierType) and
                            decl.type.type.names[-1].startswith('__dotdotdot')):
                        realtype = self._get_unknown_type(decl)
                    elif (isinstance(decl.type, pycparser.c_ast.PtrDecl) and
                          isinstance(decl.type.type, pycparser.c_ast.TypeDecl) and
                          isinstance(decl.type.type.type,
                                     pycparser.c_ast.IdentifierType) and
                          decl.type.type.type.names[-1].startswith('__dotdotdot')):
                        realtype = self._get_unknown_ptr_type(decl)
                    else:
                        realtype, quals = self._get_type_and_quals(
                            decl.type, name=decl.name, partial_length_ok=True,
                            typedef_example="*(%s *)0" % (decl.name,))
                    self._declare('typedef ' + decl.name, realtype, quals=quals)
                elif decl.__class__.__name__ == 'Pragma':
                    # skip pragma, only in pycparser 2.15
                    import warnings
                    warnings.warn(
                        "#pragma in cdef() are entirely ignored. "
                        "They should be removed for now, otherwise your "
                        "code might behave differently in a future version "
                        "of CFFI if #pragma support gets added. Note that "
                        "'#pragma pack' needs to be replaced with the "
                        "'packed' keyword argument to cdef().")
                else:
                    raise CDefError("unexpected <%s>: this construct is valid "
                                    "C but not valid in cdef()" %
                                    decl.__class__.__name__, decl)
        except CDefError as e:
            if len(e.args) == 1:
                e.args = e.args + (current_decl,)
            raise
        except FFIError as e:
            msg = self._convert_pycparser_error(e, csource)
            if msg:
                e.args = (e.args[0] + "\n    *** Err: %s" % msg,)
            raise

    def _add_constants(self, key, val):
        if key in self._int_constants:
            if self._int_constants[key] == val:
                return     # ignore identical double declarations
            raise FFIError(
                "multiple declarations of constant: %s" % (key,))
        self._int_constants[key] = val

    def _add_integer_constant(self, name, int_str):
        int_str = int_str.lower().rstrip("ul")
        neg = int_str.startswith('-')
        if neg:
            int_str = int_str[1:]
        # "010" is not valid oct in py3
        if (int_str.startswith("0") and int_str != '0'
                and not int_str.startswith("0x")):
            int_str = "0o" + int_str[1:]
        pyvalue = int(int_str, 0)
        if neg:
            pyvalue = -pyvalue
        self._add_constants(name, pyvalue)
        self._declare('macro ' + name, pyvalue)

    def _process_macros(self, macros):
        for key, value in macros.items():
            value = value.strip()
            if _r_int_literal.match(value):
                self._add_integer_constant(key, value)
            elif value == '...':
                self._declare('macro ' + key, value)
            else:
                raise CDefError(
                    'only supports one of the following syntax:\n'
                    '  #define %s ...     (literally dot-dot-dot)\n'
                    '  #define %s NUMBER  (with NUMBER an integer'
                                    ' constant, decimal/hex/octal)\n'
                    'got:\n'
                    '  #define %s %s'
                    % (key, key, key, value))

    def _declare_function(self, tp, quals, decl):
        tp = self._get_type_pointer(tp, quals)
        if self._options.get('dllexport'):
            tag = 'dllexport_python '
        elif self._inside_extern_python == '__cffi_extern_python_start':
            tag = 'extern_python '
        elif self._inside_extern_python == '__cffi_extern_python_plus_c_start':
            tag = 'extern_python_plus_c '
        else:
            tag = 'function '
        self._declare(tag + decl.name, tp)

    def _parse_decl(self, decl):
        node = decl.type
        if isinstance(node, pycparser.c_ast.FuncDecl):
            tp, quals = self._get_type_and_quals(node, name=decl.name)
            assert isinstance(tp, model.RawFunctionType)
            self._declare_function(tp, quals, decl)
        else:
            if isinstance(node, pycparser.c_ast.Struct):
                self._get_struct_union_enum_type('struct', node)
            elif isinstance(node, pycparser.c_ast.Union):
                self._get_struct_union_enum_type('union', node)
            elif isinstance(node, pycparser.c_ast.Enum):
                self._get_struct_union_enum_type('enum', node)
            elif not decl.name:
                raise CDefError("construct does not declare any variable",
                                decl)
            #
            if decl.name:
                tp, quals = self._get_type_and_quals(node,
                                                     partial_length_ok=True)
                if tp.is_raw_function:
                    self._declare_function(tp, quals, decl)
                elif (tp.is_integer_type() and
                        hasattr(decl, 'init') and
                        hasattr(decl.init, 'value') and
                        _r_int_literal.match(decl.init.value)):
                    self._add_integer_constant(decl.name, decl.init.value)
                elif (tp.is_integer_type() and
                        isinstance(decl.init, pycparser.c_ast.UnaryOp) and
                        decl.init.op == '-' and
                        hasattr(decl.init.expr, 'value') and
                        _r_int_literal.match(decl.init.expr.value)):
                    self._add_integer_constant(decl.name,
                                               '-' + decl.init.expr.value)
                elif (tp is model.void_type and
                      decl.name.startswith('__cffi_extern_python_')):
                    # hack: `extern "Python"` in the C source is replaced
                    # with "void __cffi_extern_python_start;" and
                    # "void __cffi_extern_python_stop;"
                    self._inside_extern_python = decl.name
                else:
                    if self._inside_extern_python !='__cffi_extern_python_stop':
                        raise CDefError(
                            "cannot declare constants or "
                            "variables with 'extern \"Python\"'")
                    if (quals & model.Q_CONST) and not tp.is_array_type:
                        self._declare('constant ' + decl.name, tp, quals=quals)
                    else:
                        _warn_for_non_extern_non_static_global_variable(decl)
                        self._declare('variable ' + decl.name, tp, quals=quals)

    def parse_type(self, cdecl):
        return self.parse_type_and_quals(cdecl)[0]

    def parse_type_and_quals(self, cdecl):
        ast, macros = self._parse('void __dummy(\n%s\n);' % cdecl)[:2]
        assert not macros
        exprnode = ast.ext[-1].type.args.params[0]
        if isinstance(exprnode, pycparser.c_ast.ID):
            raise CDefError("unknown identifier '%s'" % (exprnode.name,))
        return self._get_type_and_quals(exprnode.type)

    def _declare(self, name, obj, included=False, quals=0):
        if name in self._declarations:
            prevobj, prevquals = self._declarations[name]
            if prevobj is obj and prevquals == quals:
                return
            if not self._options.get('override'):
                raise FFIError(
                    "multiple declarations of %s (for interactive usage, "
                    "try cdef(xx, override=True))" % (name,))
        assert '__dotdotdot__' not in name.split()
        self._declarations[name] = (obj, quals)
        if included:
            self._included_declarations.add(obj)

    def _extract_quals(self, type):
        quals = 0
        if isinstance(type, (pycparser.c_ast.TypeDecl,
                             pycparser.c_ast.PtrDecl)):
            if 'const' in type.quals:
                quals |= model.Q_CONST
            if 'volatile' in type.quals:
                quals |= model.Q_VOLATILE
            if 'restrict' in type.quals:
                quals |= model.Q_RESTRICT
        return quals

    def _get_type_pointer(self, type, quals, declname=None):
        if isinstance(type, model.RawFunctionType):
            return type.as_function_pointer()
        if (isinstance(type, model.StructOrUnionOrEnum) and
                type.name.startswith('$') and type.name[1:].isdigit() and
                type.forcename is None and declname is not None):
            return model.NamedPointerType(type, declname, quals)
        return model.PointerType(type, quals)

    def _get_type_and_quals(self, typenode, name=None, partial_length_ok=False,
                            typedef_example=None):
        # first, dereference typedefs, if we have it already parsed, we're good
        if (isinstance(typenode, pycparser.c_ast.TypeDecl) and
            isinstance(typenode.type, pycparser.c_ast.IdentifierType) and
            len(typenode.type.names) == 1 and
            ('typedef ' + typenode.type.names[0]) in self._declarations):
            tp, quals = self._declarations['typedef ' + typenode.type.names[0]]
            quals |= self._extract_quals(typenode)
            return tp, quals
        #
        if isinstance(typenode, pycparser.c_ast.ArrayDecl):
            # array type
            if typenode.dim is None:
                length = None
            else:
                length = self._parse_constant(
                    typenode.dim, partial_length_ok=partial_length_ok)
            # a hack: in 'typedef int foo_t[...][...];', don't use '...' as
            # the length but use directly the C expression that would be
            # generated by recompiler.py.  This lets the typedef be used in
            # many more places within recompiler.py
            if typedef_example is not None:
                if length == '...':
                    length = '_cffi_array_len(%s)' % (typedef_example,)
                typedef_example = "*" + typedef_example
            #
            tp, quals = self._get_type_and_quals(typenode.type,
                                partial_length_ok=partial_length_ok,
                                typedef_example=typedef_example)
            return model.ArrayType(tp, length), quals
        #
        if isinstance(typenode, pycparser.c_ast.PtrDecl):
            # pointer type
            itemtype, itemquals = self._get_type_and_quals(typenode.type)
            tp = self._get_type_pointer(itemtype, itemquals, declname=name)
            quals = self._extract_quals(typenode)
            return tp, quals
        #
        if isinstance(typenode, pycparser.c_ast.TypeDecl):
            quals = self._extract_quals(typenode)
            type = typenode.type
            if isinstance(type, pycparser.c_ast.IdentifierType):
                # assume a primitive type.  get it from .names, but reduce
                # synonyms to a single chosen combination
                names = list(type.names)
                if names != ['signed', 'char']:    # keep this unmodified
                    prefixes = {}
                    while names:
                        name = names[0]
                        if name in ('short', 'long', 'signed', 'unsigned'):
                            prefixes[name] = prefixes.get(name, 0) + 1
                            del names[0]
                        else:
                            break
                    # ignore the 'signed' prefix below, and reorder the others
                    newnames = []
                    for prefix in ('unsigned', 'short', 'long'):
                        for i in range(prefixes.get(prefix, 0)):
                            newnames.append(prefix)
                    if not names:
                        names = ['int']    # implicitly
                    if names == ['int']:   # but kill it if 'short' or 'long'
                        if 'short' in prefixes or 'long' in prefixes:
                            names = []
                    names = newnames + names
                ident = ' '.join(names)
                if ident == 'void':
                    return model.void_type, quals
                if ident == '__dotdotdot__':
                    raise FFIError(':%d: bad usage of "..."' %
                            typenode.coord.line)
                tp0, quals0 = resolve_common_type(self, ident)
                return tp0, (quals | quals0)
            #
            if isinstance(type, pycparser.c_ast.Struct):
                # 'struct foobar'
                tp = self._get_struct_union_enum_type('struct', type, name)
                return tp, quals
            #
            if isinstance(type, pycparser.c_ast.Union):
                # 'union foobar'
                tp = self._get_struct_union_enum_type('union', type, name)
                return tp, quals
            #
            if isinstance(type, pycparser.c_ast.Enum):
                # 'enum foobar'
                tp = self._get_struct_union_enum_type('enum', type, name)
                return tp, quals
        #
        if isinstance(typenode, pycparser.c_ast.FuncDecl):
            # a function type
            return self._parse_function_type(typenode, name), 0
        #
        # nested anonymous structs or unions end up here
        if isinstance(typenode, pycparser.c_ast.Struct):
            return self._get_struct_union_enum_type('struct', typenode, name,
                                                    nested=True), 0
        if isinstance(typenode, pycparser.c_ast.Union):
            return self._get_struct_union_enum_type('union', typenode, name,
                                                    nested=True), 0
        #
        raise FFIError(":%d: bad or unsupported type declaration" %
                typenode.coord.line)

    def _parse_function_type(self, typenode, funcname=None):
        params = list(getattr(typenode.args, 'params', []))
        for i, arg in enumerate(params):
            if not hasattr(arg, 'type'):
                raise CDefError("%s arg %d: unknown type '%s'"
                    " (if you meant to use the old C syntax of giving"
                    " untyped arguments, it is not supported)"
                    % (funcname or 'in expression', i + 1,
                       getattr(arg, 'name', '?')))
        ellipsis = (
            len(params) > 0 and
            isinstance(params[-1].type, pycparser.c_ast.TypeDecl) and
            isinstance(params[-1].type.type,
                       pycparser.c_ast.IdentifierType) and
            params[-1].type.type.names == ['__dotdotdot__'])
        if ellipsis:
            params.pop()
            if not params:
                raise CDefError(
                    "%s: a function with only '(...)' as argument"
                    " is not correct C" % (funcname or 'in expression'))
        args = [self._as_func_arg(*self._get_type_and_quals(argdeclnode.type))
                for argdeclnode in params]
        if not ellipsis and args == [model.void_type]:
            args = []
        result, quals = self._get_type_and_quals(typenode.type)
        # the 'quals' on the result type are ignored.  HACK: we absure them
        # to detect __stdcall functions: we textually replace "__stdcall"
        # with "volatile volatile const" above.
        abi = None
        if hasattr(typenode.type, 'quals'): # else, probable syntax error anyway
            if typenode.type.quals[-3:] == ['volatile', 'volatile', 'const']:
                abi = '__stdcall'
        return model.RawFunctionType(tuple(args), result, ellipsis, abi)

    def _as_func_arg(self, type, quals):
        if isinstance(type, model.ArrayType):
            return model.PointerType(type.item, quals)
        elif isinstance(type, model.RawFunctionType):
            return type.as_function_pointer()
        else:
            return type

    def _get_struct_union_enum_type(self, kind, type, name=None, nested=False):
        # First, a level of caching on the exact 'type' node of the AST.
        # This is obscure, but needed because pycparser "unrolls" declarations
        # such as "typedef struct { } foo_t, *foo_p" and we end up with
        # an AST that is not a tree, but a DAG, with the "type" node of the
        # two branches foo_t and foo_p of the trees being the same node.
        # It's a bit silly but detecting "DAG-ness" in the AST tree seems
        # to be the only way to distinguish this case from two independent
        # structs.  See test_struct_with_two_usages.
        try:
            return self._structnode2type[type]
        except KeyError:
            pass
        #
        # Note that this must handle parsing "struct foo" any number of
        # times and always return the same StructType object.  Additionally,
        # one of these times (not necessarily the first), the fields of
        # the struct can be specified with "struct foo { ...fields... }".
        # If no name is given, then we have to create a new anonymous struct
        # with no caching; in this case, the fields are either specified
        # right now or never.
        #
        force_name = name
        name = type.name
        #
        # get the type or create it if needed
        if name is None:
            # 'force_name' is used to guess a more readable name for
            # anonymous structs, for the common case "typedef struct { } foo".
            if force_name is not None:
                explicit_name = '$%s' % force_name
            else:
                self._anonymous_counter += 1
                explicit_name = '$%d' % self._anonymous_counter
            tp = None
        else:
            explicit_name = name
            key = '%s %s' % (kind, name)
            tp, _ = self._declarations.get(key, (None, None))
        #
        if tp is None:
            if kind == 'struct':
                tp = model.StructType(explicit_name, None, None, None)
            elif kind == 'union':
                tp = model.UnionType(explicit_name, None, None, None)
            elif kind == 'enum':
                if explicit_name == '__dotdotdot__':
                    raise CDefError("Enums cannot be declared with ...")
                tp = self._build_enum_type(explicit_name, type.values)
            else:
                raise AssertionError("kind = %r" % (kind,))
            if name is not None:
                self._declare(key, tp)
        else:
            if kind == 'enum' and type.values is not None:
                raise NotImplementedError(
                    "enum %s: the '{}' declaration should appear on the first "
                    "time the enum is mentioned, not later" % explicit_name)
        if not tp.forcename:
            tp.force_the_name(force_name)
        if tp.forcename and '$' in tp.name:
            self._declare('anonymous %s' % tp.forcename, tp)
        #
        self._structnode2type[type] = tp
        #
        # enums: done here
        if kind == 'enum':
            return tp
        #
        # is there a 'type.decls'?  If yes, then this is the place in the
        # C sources that declare the fields.  If no, then just return the
        # existing type, possibly still incomplete.
        if type.decls is None:
            return tp
        #
        if tp.fldnames is not None:
            raise CDefError("duplicate declaration of struct %s" % name)
        fldnames = []
        fldtypes = []
        fldbitsize = []
        fldquals = []
        for decl in type.decls:
            if (isinstance(decl.type, pycparser.c_ast.IdentifierType) and
                    ''.join(decl.type.names) == '__dotdotdot__'):
                # XXX pycparser is inconsistent: 'names' should be a list
                # of strings, but is sometimes just one string.  Use
                # str.join() as a way to cope with both.
                self._make_partial(tp, nested)
                continue
            if decl.bitsize is None:
                bitsize = -1
            else:
                bitsize = self._parse_constant(decl.bitsize)
            self._partial_length = False
            type, fqual = self._get_type_and_quals(decl.type,
                                                   partial_length_ok=True)
            if self._partial_length:
                self._make_partial(tp, nested)
            if isinstance(type, model.StructType) and type.partial:
                self._make_partial(tp, nested)
            fldnames.append(decl.name or '')
            fldtypes.append(type)
            fldbitsize.append(bitsize)
            fldquals.append(fqual)
        tp.fldnames = tuple(fldnames)
        tp.fldtypes = tuple(fldtypes)
        tp.fldbitsize = tuple(fldbitsize)
        tp.fldquals = tuple(fldquals)
        if fldbitsize != [-1] * len(fldbitsize):
            if isinstance(tp, model.StructType) and tp.partial:
                raise NotImplementedError("%s: using both bitfields and '...;'"
                                          % (tp,))
        tp.packed = self._options.get('packed')
        if tp.completed:    # must be re-completed: it is not opaque any more
            tp.completed = 0
            self._recomplete.append(tp)
        return tp

    def _make_partial(self, tp, nested):
        if not isinstance(tp, model.StructOrUnion):
            raise CDefError("%s cannot be partial" % (tp,))
        if not tp.has_c_name() and not nested:
            raise NotImplementedError("%s is partial but has no C name" %(tp,))
        tp.partial = True

    def _parse_constant(self, exprnode, partial_length_ok=False):
        # for now, limited to expressions that are an immediate number
        # or positive/negative number
        if isinstance(exprnode, pycparser.c_ast.Constant):
            s = exprnode.value
            if '0' <= s[0] <= '9':
                s = s.rstrip('uUlL')
                try:
                    if s.startswith('0'):
                        return int(s, 8)
                    else:
                        return int(s, 10)
                except ValueError:
                    if len(s) > 1:
                        if s.lower()[0:2] == '0x':
                            return int(s, 16)
                        elif s.lower()[0:2] == '0b':
                            return int(s, 2)
                raise CDefError("invalid constant %r" % (s,))
            elif s[0] == "'" and s[-1] == "'" and (
                    len(s) == 3 or (len(s) == 4 and s[1] == "\\")):
                return ord(s[-2])
            else:
                raise CDefError("invalid constant %r" % (s,))
        #
        if (isinstance(exprnode, pycparser.c_ast.UnaryOp) and
                exprnode.op == '+'):
            return self._parse_constant(exprnode.expr)
        #
        if (isinstance(exprnode, pycparser.c_ast.UnaryOp) and
                exprnode.op == '-'):
            return -self._parse_constant(exprnode.expr)
        # load previously defined int constant
        if (isinstance(exprnode, pycparser.c_ast.ID) and
                exprnode.name in self._int_constants):
            return self._int_constants[exprnode.name]
        #
        if (isinstance(exprnode, pycparser.c_ast.ID) and
                    exprnode.name == '__dotdotdotarray__'):
            if partial_length_ok:
                self._partial_length = True
                return '...'
            raise FFIError(":%d: unsupported '[...]' here, cannot derive "
                           "the actual array length in this context"
                           % exprnode.coord.line)
        #
        if isinstance(exprnode, pycparser.c_ast.BinaryOp):
            left = self._parse_constant(exprnode.left)
            right = self._parse_constant(exprnode.right)
            if exprnode.op == '+':
                return left + right
            elif exprnode.op == '-':
                return left - right
            elif exprnode.op == '*':
                return left * right
            elif exprnode.op == '/':
                return self._c_div(left, right)
            elif exprnode.op == '%':
                return left - self._c_div(left, right) * right
            elif exprnode.op == '<<':
                return left << right
            elif exprnode.op == '>>':
                return left >> right
            elif exprnode.op == '&':
                return left & right
            elif exprnode.op == '|':
                return left | right
            elif exprnode.op == '^':
                return left ^ right
        #
        raise FFIError(":%d: unsupported expression: expected a "
                       "simple numeric constant" % exprnode.coord.line)

    def _c_div(self, a, b):
        result = a // b
        if ((a < 0) ^ (b < 0)) and (a % b) != 0:
            result += 1
        return result

    def _build_enum_type(self, explicit_name, decls):
        if decls is not None:
            partial = False
            enumerators = []
            enumvalues = []
            nextenumvalue = 0
            for enum in decls.enumerators:
                if _r_enum_dotdotdot.match(enum.name):
                    partial = True
                    continue
                if enum.value is not None:
                    nextenumvalue = self._parse_constant(enum.value)
                enumerators.append(enum.name)
                enumvalues.append(nextenumvalue)
                self._add_constants(enum.name, nextenumvalue)
                nextenumvalue += 1
            enumerators = tuple(enumerators)
            enumvalues = tuple(enumvalues)
            tp = model.EnumType(explicit_name, enumerators, enumvalues)
            tp.partial = partial
        else:   # opaque enum
            tp = model.EnumType(explicit_name, (), ())
        return tp

    def include(self, other):
        for name, (tp, quals) in other._declarations.items():
            if name.startswith('anonymous $enum_$'):
                continue   # fix for test_anonymous_enum_include
            kind = name.split(' ', 1)[0]
            if kind in ('struct', 'union', 'enum', 'anonymous', 'typedef'):
                self._declare(name, tp, included=True, quals=quals)
        for k, v in other._int_constants.items():
            self._add_constants(k, v)

    def _get_unknown_type(self, decl):
        typenames = decl.type.type.names
        if typenames == ['__dotdotdot__']:
            return model.unknown_type(decl.name)

        if typenames == ['__dotdotdotint__']:
            if self._uses_new_feature is None:
                self._uses_new_feature = "'typedef int... %s'" % decl.name
            return model.UnknownIntegerType(decl.name)

        if typenames == ['__dotdotdotfloat__']:
            # note: not for 'long double' so far
            if self._uses_new_feature is None:
                self._uses_new_feature = "'typedef float... %s'" % decl.name
            return model.UnknownFloatType(decl.name)

        raise FFIError(':%d: unsupported usage of "..." in typedef'
                       % decl.coord.line)

    def _get_unknown_ptr_type(self, decl):
        if decl.type.type.type.names == ['__dotdotdot__']:
            return model.unknown_ptr_type(decl.name)
        raise FFIError(':%d: unsupported usage of "..." in typedef'
                       % decl.coord.line)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\flask_sqlalchemy\extension.py ===
from __future__ import annotations

import os
import types
import typing as t
import warnings
from weakref import WeakKeyDictionary

import sqlalchemy as sa
import sqlalchemy.event as sa_event
import sqlalchemy.exc as sa_exc
import sqlalchemy.orm as sa_orm
from flask import abort
from flask import current_app
from flask import Flask
from flask import has_app_context

from .model import _QueryProperty
from .model import BindMixin
from .model import DefaultMeta
from .model import DefaultMetaNoName
from .model import Model
from .model import NameMixin
from .pagination import Pagination
from .pagination import SelectPagination
from .query import Query
from .session import _app_ctx_id
from .session import Session
from .table import _Table

_O = t.TypeVar("_O", bound=object)  # Based on sqlalchemy.orm._typing.py


# Type accepted for model_class argument
_FSA_MCT = t.TypeVar(
    "_FSA_MCT",
    bound=t.Union[
        t.Type[Model],
        sa_orm.DeclarativeMeta,
        t.Type[sa_orm.DeclarativeBase],
        t.Type[sa_orm.DeclarativeBaseNoMeta],
    ],
)


# Type returned by make_declarative_base
class _FSAModel(Model):
    metadata: sa.MetaData


def _get_2x_declarative_bases(
    model_class: _FSA_MCT,
) -> list[t.Type[t.Union[sa_orm.DeclarativeBase, sa_orm.DeclarativeBaseNoMeta]]]:
    return [
        b
        for b in model_class.__bases__
        if issubclass(b, (sa_orm.DeclarativeBase, sa_orm.DeclarativeBaseNoMeta))
    ]


class SQLAlchemy:
    """Integrates SQLAlchemy with Flask. This handles setting up one or more engines,
    associating tables and models with specific engines, and cleaning up connections and
    sessions after each request.

    Only the engine configuration is specific to each application, other things like
    the model, table, metadata, and session are shared for all applications using that
    extension instance. Call :meth:`init_app` to configure the extension on an
    application.

    After creating the extension, create model classes by subclassing :attr:`Model`, and
    table classes with :attr:`Table`. These can be accessed before :meth:`init_app` is
    called, making it possible to define the models separately from the application.

    Accessing :attr:`session` and :attr:`engine` requires an active Flask application
    context. This includes methods like :meth:`create_all` which use the engine.

    This class also provides access to names in SQLAlchemy's ``sqlalchemy`` and
    ``sqlalchemy.orm`` modules. For example, you can use ``db.Column`` and
    ``db.relationship`` instead of importing ``sqlalchemy.Column`` and
    ``sqlalchemy.orm.relationship``. This can be convenient when defining models.

    :param app: Call :meth:`init_app` on this Flask application now.
    :param metadata: Use this as the default :class:`sqlalchemy.schema.MetaData`. Useful
        for setting a naming convention.
    :param session_options: Arguments used by :attr:`session` to create each session
        instance. A ``scopefunc`` key will be passed to the scoped session, not the
        session instance. See :class:`sqlalchemy.orm.sessionmaker` for a list of
        arguments.
    :param query_class: Use this as the default query class for models and dynamic
        relationships. The query interface is considered legacy in SQLAlchemy.
    :param model_class: Use this as the model base class when creating the declarative
        model class :attr:`Model`. Can also be a fully created declarative model class
        for further customization.
    :param engine_options: Default arguments used when creating every engine. These are
        lower precedence than application config. See :func:`sqlalchemy.create_engine`
        for a list of arguments.
    :param add_models_to_shell: Add the ``db`` instance and all model classes to
        ``flask shell``.

    .. versionchanged:: 3.1.0
        The ``metadata`` parameter can still be used with SQLAlchemy 1.x classes,
        but is ignored when using SQLAlchemy 2.x style of declarative classes.
        Instead, specify metadata on your Base class.

    .. versionchanged:: 3.1.0
        Added the ``disable_autonaming`` parameter.

    .. versionchanged:: 3.1.0
        Changed ``model_class`` parameter to accepta SQLAlchemy 2.x
        declarative base subclass.

    .. versionchanged:: 3.0
        An active Flask application context is always required to access ``session`` and
        ``engine``.

    .. versionchanged:: 3.0
        Separate ``metadata`` are used for each bind key.

    .. versionchanged:: 3.0
        The ``engine_options`` parameter is applied as defaults before per-engine
        configuration.

    .. versionchanged:: 3.0
        The session class can be customized in ``session_options``.

    .. versionchanged:: 3.0
        Added the ``add_models_to_shell`` parameter.

    .. versionchanged:: 3.0
        Engines are created when calling ``init_app`` rather than the first time they
        are accessed.

    .. versionchanged:: 3.0
        All parameters except ``app`` are keyword-only.

    .. versionchanged:: 3.0
        The extension instance is stored directly as ``app.extensions["sqlalchemy"]``.

    .. versionchanged:: 3.0
        Setup methods are renamed with a leading underscore. They are considered
        internal interfaces which may change at any time.

    .. versionchanged:: 3.0
        Removed the ``use_native_unicode`` parameter and config.

    .. versionchanged:: 2.4
        Added the ``engine_options`` parameter.

    .. versionchanged:: 2.1
        Added the ``metadata``, ``query_class``, and ``model_class`` parameters.

    .. versionchanged:: 2.1
        Use the same query class across ``session``, ``Model.query`` and
        ``Query``.

    .. versionchanged:: 0.16
        ``scopefunc`` is accepted in ``session_options``.

    .. versionchanged:: 0.10
        Added the ``session_options`` parameter.
    """

    def __init__(
        self,
        app: Flask | None = None,
        *,
        metadata: sa.MetaData | None = None,
        session_options: dict[str, t.Any] | None = None,
        query_class: type[Query] = Query,
        model_class: _FSA_MCT = Model,  # type: ignore[assignment]
        engine_options: dict[str, t.Any] | None = None,
        add_models_to_shell: bool = True,
        disable_autonaming: bool = False,
    ):
        if session_options is None:
            session_options = {}

        self.Query = query_class
        """The default query class used by ``Model.query`` and ``lazy="dynamic"``
        relationships.

        .. warning::
            The query interface is considered legacy in SQLAlchemy.

        Customize this by passing the ``query_class`` parameter to the extension.
        """

        self.session = self._make_scoped_session(session_options)
        """A :class:`sqlalchemy.orm.scoping.scoped_session` that creates instances of
        :class:`.Session` scoped to the current Flask application context. The session
        will be removed, returning the engine connection to the pool, when the
        application context exits.

        Customize this by passing ``session_options`` to the extension.

        This requires that a Flask application context is active.

        .. versionchanged:: 3.0
            The session is scoped to the current app context.
        """

        self.metadatas: dict[str | None, sa.MetaData] = {}
        """Map of bind keys to :class:`sqlalchemy.schema.MetaData` instances. The
        ``None`` key refers to the default metadata, and is available as
        :attr:`metadata`.

        Customize the default metadata by passing the ``metadata`` parameter to the
        extension. This can be used to set a naming convention. When metadata for
        another bind key is created, it copies the default's naming convention.

        .. versionadded:: 3.0
        """

        if metadata is not None:
            if len(_get_2x_declarative_bases(model_class)) > 0:
                warnings.warn(
                    "When using SQLAlchemy 2.x style of declarative classes,"
                    " the `metadata` should be an attribute of the base class."
                    "The metadata passed into SQLAlchemy() is ignored.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            else:
                metadata.info["bind_key"] = None
                self.metadatas[None] = metadata

        self.Table = self._make_table_class()
        """A :class:`sqlalchemy.schema.Table` class that chooses a metadata
        automatically.

        Unlike the base ``Table``, the ``metadata`` argument is not required. If it is
        not given, it is selected based on the ``bind_key`` argument.

        :param bind_key: Used to select a different metadata.
        :param args: Arguments passed to the base class. These are typically the table's
            name, columns, and constraints.
        :param kwargs: Arguments passed to the base class.

        .. versionchanged:: 3.0
            This is a subclass of SQLAlchemy's ``Table`` rather than a function.
        """

        self.Model = self._make_declarative_base(
            model_class, disable_autonaming=disable_autonaming
        )
        """A SQLAlchemy declarative model class. Subclass this to define database
        models.

        If a model does not set ``__tablename__``, it will be generated by converting
        the class name from ``CamelCase`` to ``snake_case``. It will not be generated
        if the model looks like it uses single-table inheritance.

        If a model or parent class sets ``__bind_key__``, it will use that metadata and
        database engine. Otherwise, it will use the default :attr:`metadata` and
        :attr:`engine`. This is ignored if the model sets ``metadata`` or ``__table__``.

        For code using the SQLAlchemy 1.x API, customize this model by subclassing
        :class:`.Model` and passing the ``model_class`` parameter to the extension.
        A fully created declarative model class can be
        passed as well, to use a custom metaclass.

        For code using the SQLAlchemy 2.x API, customize this model by subclassing
        :class:`sqlalchemy.orm.DeclarativeBase` or
        :class:`sqlalchemy.orm.DeclarativeBaseNoMeta`
        and passing the ``model_class`` parameter to the extension.
        """

        if engine_options is None:
            engine_options = {}

        self._engine_options = engine_options
        self._app_engines: WeakKeyDictionary[Flask, dict[str | None, sa.engine.Engine]]
        self._app_engines = WeakKeyDictionary()
        self._add_models_to_shell = add_models_to_shell

        if app is not None:
            self.init_app(app)

    def __repr__(self) -> str:
        if not has_app_context():
            return f"<{type(self).__name__}>"

        message = f"{type(self).__name__} {self.engine.url}"

        if len(self.engines) > 1:
            message = f"{message} +{len(self.engines) - 1}"

        return f"<{message}>"

    def init_app(self, app: Flask) -> None:
        """Initialize a Flask application for use with this extension instance. This
        must be called before accessing the database engine or session with the app.

        This sets default configuration values, then configures the extension on the
        application and creates the engines for each bind key. Therefore, this must be
        called after the application has been configured. Changes to application config
        after this call will not be reflected.

        The following keys from ``app.config`` are used:

        - :data:`.SQLALCHEMY_DATABASE_URI`
        - :data:`.SQLALCHEMY_ENGINE_OPTIONS`
        - :data:`.SQLALCHEMY_ECHO`
        - :data:`.SQLALCHEMY_BINDS`
        - :data:`.SQLALCHEMY_RECORD_QUERIES`
        - :data:`.SQLALCHEMY_TRACK_MODIFICATIONS`

        :param app: The Flask application to initialize.
        """
        if "sqlalchemy" in app.extensions:
            raise RuntimeError(
                "A 'SQLAlchemy' instance has already been registered on this Flask app."
                " Import and use that instance instead."
            )

        app.extensions["sqlalchemy"] = self
        app.teardown_appcontext(self._teardown_session)

        if self._add_models_to_shell:
            from .cli import add_models_to_shell

            app.shell_context_processor(add_models_to_shell)

        basic_uri: str | sa.engine.URL | None = app.config.setdefault(
            "SQLALCHEMY_DATABASE_URI", None
        )
        basic_engine_options = self._engine_options.copy()
        basic_engine_options.update(
            app.config.setdefault("SQLALCHEMY_ENGINE_OPTIONS", {})
        )
        echo: bool = app.config.setdefault("SQLALCHEMY_ECHO", False)
        config_binds: dict[
            str | None, str | sa.engine.URL | dict[str, t.Any]
        ] = app.config.setdefault("SQLALCHEMY_BINDS", {})
        engine_options: dict[str | None, dict[str, t.Any]] = {}

        # Build the engine config for each bind key.
        for key, value in config_binds.items():
            engine_options[key] = self._engine_options.copy()

            if isinstance(value, (str, sa.engine.URL)):
                engine_options[key]["url"] = value
            else:
                engine_options[key].update(value)

        # Build the engine config for the default bind key.
        if basic_uri is not None:
            basic_engine_options["url"] = basic_uri

        if "url" in basic_engine_options:
            engine_options.setdefault(None, {}).update(basic_engine_options)

        if not engine_options:
            raise RuntimeError(
                "Either 'SQLALCHEMY_DATABASE_URI' or 'SQLALCHEMY_BINDS' must be set."
            )

        engines = self._app_engines.setdefault(app, {})

        # Dispose existing engines in case init_app is called again.
        if engines:
            for engine in engines.values():
                engine.dispose()

            engines.clear()

        # Create the metadata and engine for each bind key.
        for key, options in engine_options.items():
            self._make_metadata(key)
            options.setdefault("echo", echo)
            options.setdefault("echo_pool", echo)
            self._apply_driver_defaults(options, app)
            engines[key] = self._make_engine(key, options, app)

        if app.config.setdefault("SQLALCHEMY_RECORD_QUERIES", False):
            from . import record_queries

            for engine in engines.values():
                record_queries._listen(engine)

        if app.config.setdefault("SQLALCHEMY_TRACK_MODIFICATIONS", False):
            from . import track_modifications

            track_modifications._listen(self.session)

    def _make_scoped_session(
        self, options: dict[str, t.Any]
    ) -> sa_orm.scoped_session[Session]:
        """Create a :class:`sqlalchemy.orm.scoping.scoped_session` around the factory
        from :meth:`_make_session_factory`. The result is available as :attr:`session`.

        The scope function can be customized using the ``scopefunc`` key in the
        ``session_options`` parameter to the extension. By default it uses the current
        thread or greenlet id.

        This method is used for internal setup. Its signature may change at any time.

        :meta private:

        :param options: The ``session_options`` parameter from ``__init__``. Keyword
            arguments passed to the session factory. A ``scopefunc`` key is popped.

        .. versionchanged:: 3.0
            The session is scoped to the current app context.

        .. versionchanged:: 3.0
            Renamed from ``create_scoped_session``, this method is internal.
        """
        scope = options.pop("scopefunc", _app_ctx_id)
        factory = self._make_session_factory(options)
        return sa_orm.scoped_session(factory, scope)

    def _make_session_factory(
        self, options: dict[str, t.Any]
    ) -> sa_orm.sessionmaker[Session]:
        """Create the SQLAlchemy :class:`sqlalchemy.orm.sessionmaker` used by
        :meth:`_make_scoped_session`.

        To customize, pass the ``session_options`` parameter to :class:`SQLAlchemy`. To
        customize the session class, subclass :class:`.Session` and pass it as the
        ``class_`` key.

        This method is used for internal setup. Its signature may change at any time.

        :meta private:

        :param options: The ``session_options`` parameter from ``__init__``. Keyword
            arguments passed to the session factory.

        .. versionchanged:: 3.0
            The session class can be customized.

        .. versionchanged:: 3.0
            Renamed from ``create_session``, this method is internal.
        """
        options.setdefault("class_", Session)
        options.setdefault("query_cls", self.Query)
        return sa_orm.sessionmaker(db=self, **options)

    def _teardown_session(self, exc: BaseException | None) -> None:
        """Remove the current session at the end of the request.

        :meta private:

        .. versionadded:: 3.0
        """
        self.session.remove()

    def _make_metadata(self, bind_key: str | None) -> sa.MetaData:
        """Get or create a :class:`sqlalchemy.schema.MetaData` for the given bind key.

        This method is used for internal setup. Its signature may change at any time.

        :meta private:

        :param bind_key: The name of the metadata being created.

        .. versionadded:: 3.0
        """
        if bind_key in self.metadatas:
            return self.metadatas[bind_key]

        if bind_key is not None:
            # Copy the naming convention from the default metadata.
            naming_convention = self._make_metadata(None).naming_convention
        else:
            naming_convention = None

        # Set the bind key in info to be used by session.get_bind.
        metadata = sa.MetaData(
            naming_convention=naming_convention, info={"bind_key": bind_key}
        )
        self.metadatas[bind_key] = metadata
        return metadata

    def _make_table_class(self) -> type[_Table]:
        """Create a SQLAlchemy :class:`sqlalchemy.schema.Table` class that chooses a
        metadata automatically based on the ``bind_key``. The result is available as
        :attr:`Table`.

        This method is used for internal setup. Its signature may change at any time.

        :meta private:

        .. versionadded:: 3.0
        """

        class Table(_Table):
            def __new__(
                cls, *args: t.Any, bind_key: str | None = None, **kwargs: t.Any
            ) -> Table:
                # If a metadata arg is passed, go directly to the base Table. Also do
                # this for no args so the correct error is shown.
                if not args or (len(args) >= 2 and isinstance(args[1], sa.MetaData)):
                    return super().__new__(cls, *args, **kwargs)

                metadata = self._make_metadata(bind_key)
                return super().__new__(cls, *[args[0], metadata, *args[1:]], **kwargs)

        return Table

    def _make_declarative_base(
        self,
        model_class: _FSA_MCT,
        disable_autonaming: bool = False,
    ) -> t.Type[_FSAModel]:
        """Create a SQLAlchemy declarative model class. The result is available as
        :attr:`Model`.

        To customize, subclass :class:`.Model` and pass it as ``model_class`` to
        :class:`SQLAlchemy`. To customize at the metaclass level, pass an already
        created declarative model class as ``model_class``.

        This method is used for internal setup. Its signature may change at any time.

        :meta private:

        :param model_class: A model base class, or an already created declarative model
        class.

        :param disable_autonaming: Turns off automatic tablename generation in models.

        .. versionchanged:: 3.1.0
            Added support for passing SQLAlchemy 2.x base class as model class.
            Added optional ``disable_autonaming`` parameter.

        .. versionchanged:: 3.0
            Renamed with a leading underscore, this method is internal.

        .. versionchanged:: 2.3
            ``model`` can be an already created declarative model class.
        """
        model: t.Type[_FSAModel]
        declarative_bases = _get_2x_declarative_bases(model_class)
        if len(declarative_bases) > 1:
            # raise error if more than one declarative base is found
            raise ValueError(
                "Only one declarative base can be passed to SQLAlchemy."
                " Got: {}".format(model_class.__bases__)
            )
        elif len(declarative_bases) == 1:
            body = dict(model_class.__dict__)
            body["__fsa__"] = self
            mixin_classes = [BindMixin, NameMixin, Model]
            if disable_autonaming:
                mixin_classes.remove(NameMixin)
            model = types.new_class(
                "FlaskSQLAlchemyBase",
                (*mixin_classes, *model_class.__bases__),
                {"metaclass": type(declarative_bases[0])},
                lambda ns: ns.update(body),
            )
        elif not isinstance(model_class, sa_orm.DeclarativeMeta):
            metadata = self._make_metadata(None)
            metaclass = DefaultMetaNoName if disable_autonaming else DefaultMeta
            model = sa_orm.declarative_base(
                metadata=metadata, cls=model_class, name="Model", metaclass=metaclass
            )
        else:
            model = model_class  # type: ignore[assignment]

        if None not in self.metadatas:
            # Use the model's metadata as the default metadata.
            model.metadata.info["bind_key"] = None
            self.metadatas[None] = model.metadata
        else:
            # Use the passed in default metadata as the model's metadata.
            model.metadata = self.metadatas[None]

        model.query_class = self.Query
        model.query = _QueryProperty()  # type: ignore[assignment]
        model.__fsa__ = self
        return model

    def _apply_driver_defaults(self, options: dict[str, t.Any], app: Flask) -> None:
        """Apply driver-specific configuration to an engine.

        SQLite in-memory databases use ``StaticPool`` and disable ``check_same_thread``.
        File paths are relative to the app's :attr:`~flask.Flask.instance_path`,
        which is created if it doesn't exist.

        MySQL sets ``charset="utf8mb4"``, and ``pool_timeout`` defaults to 2 hours.

        This method is used for internal setup. Its signature may change at any time.

        :meta private:

        :param options: Arguments passed to the engine.
        :param app: The application that the engine configuration belongs to.

        .. versionchanged:: 3.0
            SQLite paths are relative to ``app.instance_path``. It does not use
            ``NullPool`` if ``pool_size`` is 0. Driver-level URIs are supported.

        .. versionchanged:: 3.0
            MySQL sets ``charset="utf8mb4". It does not set ``pool_size`` to 10. It
            does not set ``pool_recycle`` if not using a queue pool.

        .. versionchanged:: 3.0
            Renamed from ``apply_driver_hacks``, this method is internal. It does not
            return anything.

        .. versionchanged:: 2.5
            Returns ``(sa_url, options)``.
        """
        url = sa.engine.make_url(options["url"])

        if url.drivername in {"sqlite", "sqlite+pysqlite"}:
            if url.database is None or url.database in {"", ":memory:"}:
                options["poolclass"] = sa.pool.StaticPool

                if "connect_args" not in options:
                    options["connect_args"] = {}

                options["connect_args"]["check_same_thread"] = False
            else:
                # the url might look like sqlite:///file:path?uri=true
                is_uri = url.query.get("uri", False)

                if is_uri:
                    db_str = url.database[5:]
                else:
                    db_str = url.database

                if not os.path.isabs(db_str):
                    os.makedirs(app.instance_path, exist_ok=True)
                    db_str = os.path.join(app.instance_path, db_str)

                    if is_uri:
                        db_str = f"file:{db_str}"

                    options["url"] = url.set(database=db_str)
        elif url.drivername.startswith("mysql"):
            # set queue defaults only when using queue pool
            if (
                "pool_class" not in options
                or options["pool_class"] is sa.pool.QueuePool
            ):
                options.setdefault("pool_recycle", 7200)

            if "charset" not in url.query:
                options["url"] = url.update_query_dict({"charset": "utf8mb4"})

    def _make_engine(
        self, bind_key: str | None, options: dict[str, t.Any], app: Flask
    ) -> sa.engine.Engine:
        """Create the :class:`sqlalchemy.engine.Engine` for the given bind key and app.

        To customize, use :data:`.SQLALCHEMY_ENGINE_OPTIONS` or
        :data:`.SQLALCHEMY_BINDS` config. Pass ``engine_options`` to :class:`SQLAlchemy`
        to set defaults for all engines.

        This method is used for internal setup. Its signature may change at any time.

        :meta private:

        :param bind_key: The name of the engine being created.
        :param options: Arguments passed to the engine.
        :param app: The application that the engine configuration belongs to.

        .. versionchanged:: 3.0
            Renamed from ``create_engine``, this method is internal.
        """
        return sa.engine_from_config(options, prefix="")

    @property
    def metadata(self) -> sa.MetaData:
        """The default metadata used by :attr:`Model` and :attr:`Table` if no bind key
        is set.
        """
        return self.metadatas[None]

    @property
    def engines(self) -> t.Mapping[str | None, sa.engine.Engine]:
        """Map of bind keys to :class:`sqlalchemy.engine.Engine` instances for current
        application. The ``None`` key refers to the default engine, and is available as
        :attr:`engine`.

        To customize, set the :data:`.SQLALCHEMY_BINDS` config, and set defaults by
        passing the ``engine_options`` parameter to the extension.

        This requires that a Flask application context is active.

        .. versionadded:: 3.0
        """
        app = current_app._get_current_object()  # type: ignore[attr-defined]

        if app not in self._app_engines:
            raise RuntimeError(
                "The current Flask app is not registered with this 'SQLAlchemy'"
                " instance. Did you forget to call 'init_app', or did you create"
                " multiple 'SQLAlchemy' instances?"
            )

        return self._app_engines[app]

    @property
    def engine(self) -> sa.engine.Engine:
        """The default :class:`~sqlalchemy.engine.Engine` for the current application,
        used by :attr:`session` if the :attr:`Model` or :attr:`Table` being queried does
        not set a bind key.

        To customize, set the :data:`.SQLALCHEMY_ENGINE_OPTIONS` config, and set
        defaults by passing the ``engine_options`` parameter to the extension.

        This requires that a Flask application context is active.
        """
        return self.engines[None]

    def get_engine(
        self, bind_key: str | None = None, **kwargs: t.Any
    ) -> sa.engine.Engine:
        """Get the engine for the given bind key for the current application.
        This requires that a Flask application context is active.

        :param bind_key: The name of the engine.

        .. deprecated:: 3.0
            Will be removed in Flask-SQLAlchemy 3.2. Use ``engines[key]`` instead.

        .. versionchanged:: 3.0
            Renamed the ``bind`` parameter to ``bind_key``. Removed the ``app``
            parameter.
        """
        warnings.warn(
            "'get_engine' is deprecated and will be removed in Flask-SQLAlchemy"
            " 3.2. Use 'engine' or 'engines[key]' instead. If you're using"
            " Flask-Migrate or Alembic, you'll need to update your 'env.py' file.",
            DeprecationWarning,
            stacklevel=2,
        )

        if "bind" in kwargs:
            bind_key = kwargs.pop("bind")

        return self.engines[bind_key]

    def get_or_404(
        self,
        entity: type[_O],
        ident: t.Any,
        *,
        description: str | None = None,
        **kwargs: t.Any,
    ) -> _O:
        """Like :meth:`session.get() <sqlalchemy.orm.Session.get>` but aborts with a
        ``404 Not Found`` error instead of returning ``None``.

        :param entity: The model class to query.
        :param ident: The primary key to query.
        :param description: A custom message to show on the error page.
        :param kwargs: Extra arguments passed to ``session.get()``.

        .. versionchanged:: 3.1
            Pass extra keyword arguments to ``session.get()``.

        .. versionadded:: 3.0
        """
        value = self.session.get(entity, ident, **kwargs)

        if value is None:
            abort(404, description=description)

        return value

    def first_or_404(
        self, statement: sa.sql.Select[t.Any], *, description: str | None = None
    ) -> t.Any:
        """Like :meth:`Result.scalar() <sqlalchemy.engine.Result.scalar>`, but aborts
        with a ``404 Not Found`` error instead of returning ``None``.

        :param statement: The ``select`` statement to execute.
        :param description: A custom message to show on the error page.

        .. versionadded:: 3.0
        """
        value = self.session.execute(statement).scalar()

        if value is None:
            abort(404, description=description)

        return value

    def one_or_404(
        self, statement: sa.sql.Select[t.Any], *, description: str | None = None
    ) -> t.Any:
        """Like :meth:`Result.scalar_one() <sqlalchemy.engine.Result.scalar_one>`,
        but aborts with a ``404 Not Found`` error instead of raising ``NoResultFound``
        or ``MultipleResultsFound``.

        :param statement: The ``select`` statement to execute.
        :param description: A custom message to show on the error page.

        .. versionadded:: 3.0
        """
        try:
            return self.session.execute(statement).scalar_one()
        except (sa_exc.NoResultFound, sa_exc.MultipleResultsFound):
            abort(404, description=description)

    def paginate(
        self,
        select: sa.sql.Select[t.Any],
        *,
        page: int | None = None,
        per_page: int | None = None,
        max_per_page: int | None = None,
        error_out: bool = True,
        count: bool = True,
    ) -> Pagination:
        """Apply an offset and limit to a select statment based on the current page and
        number of items per page, returning a :class:`.Pagination` object.

        The statement should select a model class, like ``select(User)``. This applies
        ``unique()`` and ``scalars()`` modifiers to the result, so compound selects will
        not return the expected results.

        :param select: The ``select`` statement to paginate.
        :param page: The current page, used to calculate the offset. Defaults to the
            ``page`` query arg during a request, or 1 otherwise.
        :param per_page: The maximum number of items on a page, used to calculate the
            offset and limit. Defaults to the ``per_page`` query arg during a request,
            or 20 otherwise.
        :param max_per_page: The maximum allowed value for ``per_page``, to limit a
            user-provided value. Use ``None`` for no limit. Defaults to 100.
        :param error_out: Abort with a ``404 Not Found`` error if no items are returned
            and ``page`` is not 1, or if ``page`` or ``per_page`` is less than 1, or if
            either are not ints.
        :param count: Calculate the total number of values by issuing an extra count
            query. For very complex queries this may be inaccurate or slow, so it can be
            disabled and set manually if necessary.

        .. versionchanged:: 3.0
            The ``count`` query is more efficient.

        .. versionadded:: 3.0
        """
        return SelectPagination(
            select=select,
            session=self.session(),
            page=page,
            per_page=per_page,
            max_per_page=max_per_page,
            error_out=error_out,
            count=count,
        )

    def _call_for_binds(
        self, bind_key: str | None | list[str | None], op_name: str
    ) -> None:
        """Call a method on each metadata.

        :meta private:

        :param bind_key: A bind key or list of keys. Defaults to all binds.
        :param op_name: The name of the method to call.

        .. versionchanged:: 3.0
            Renamed from ``_execute_for_all_tables``.
        """
        if bind_key == "__all__":
            keys: list[str | None] = list(self.metadatas)
        elif bind_key is None or isinstance(bind_key, str):
            keys = [bind_key]
        else:
            keys = bind_key

        for key in keys:
            try:
                engine = self.engines[key]
            except KeyError:
                message = f"Bind key '{key}' is not in 'SQLALCHEMY_BINDS' config."

                if key is None:
                    message = f"'SQLALCHEMY_DATABASE_URI' config is not set. {message}"

                raise sa_exc.UnboundExecutionError(message) from None

            metadata = self.metadatas[key]
            getattr(metadata, op_name)(bind=engine)

    def create_all(self, bind_key: str | None | list[str | None] = "__all__") -> None:
        """Create tables that do not exist in the database by calling
        ``metadata.create_all()`` for all or some bind keys. This does not
        update existing tables, use a migration library for that.

        This requires that a Flask application context is active.

        :param bind_key: A bind key or list of keys to create the tables for. Defaults
            to all binds.

        .. versionchanged:: 3.0
            Renamed the ``bind`` parameter to ``bind_key``. Removed the ``app``
            parameter.

        .. versionchanged:: 0.12
            Added the ``bind`` and ``app`` parameters.
        """
        self._call_for_binds(bind_key, "create_all")

    def drop_all(self, bind_key: str | None | list[str | None] = "__all__") -> None:
        """Drop tables by calling ``metadata.drop_all()`` for all or some bind keys.

        This requires that a Flask application context is active.

        :param bind_key: A bind key or list of keys to drop the tables from. Defaults to
            all binds.

        .. versionchanged:: 3.0
            Renamed the ``bind`` parameter to ``bind_key``. Removed the ``app``
            parameter.

        .. versionchanged:: 0.12
            Added the ``bind`` and ``app`` parameters.
        """
        self._call_for_binds(bind_key, "drop_all")

    def reflect(self, bind_key: str | None | list[str | None] = "__all__") -> None:
        """Load table definitions from the database by calling ``metadata.reflect()``
        for all or some bind keys.

        This requires that a Flask application context is active.

        :param bind_key: A bind key or list of keys to reflect the tables from. Defaults
            to all binds.

        .. versionchanged:: 3.0
            Renamed the ``bind`` parameter to ``bind_key``. Removed the ``app``
            parameter.

        .. versionchanged:: 0.12
            Added the ``bind`` and ``app`` parameters.
        """
        self._call_for_binds(bind_key, "reflect")

    def _set_rel_query(self, kwargs: dict[str, t.Any]) -> None:
        """Apply the extension's :attr:`Query` class as the default for relationships
        and backrefs.

        :meta private:
        """
        kwargs.setdefault("query_class", self.Query)

        if "backref" in kwargs:
            backref = kwargs["backref"]

            if isinstance(backref, str):
                backref = (backref, {})

            backref[1].setdefault("query_class", self.Query)

    def relationship(
        self, *args: t.Any, **kwargs: t.Any
    ) -> sa_orm.RelationshipProperty[t.Any]:
        """A :func:`sqlalchemy.orm.relationship` that applies this extension's
        :attr:`Query` class for dynamic relationships and backrefs.

        .. versionchanged:: 3.0
            The :attr:`Query` class is set on ``backref``.
        """
        self._set_rel_query(kwargs)
        return sa_orm.relationship(*args, **kwargs)

    def dynamic_loader(
        self, argument: t.Any, **kwargs: t.Any
    ) -> sa_orm.RelationshipProperty[t.Any]:
        """A :func:`sqlalchemy.orm.dynamic_loader` that applies this extension's
        :attr:`Query` class for relationships and backrefs.

        .. versionchanged:: 3.0
            The :attr:`Query` class is set on ``backref``.
        """
        self._set_rel_query(kwargs)
        return sa_orm.dynamic_loader(argument, **kwargs)

    def _relation(
        self, *args: t.Any, **kwargs: t.Any
    ) -> sa_orm.RelationshipProperty[t.Any]:
        """A :func:`sqlalchemy.orm.relationship` that applies this extension's
        :attr:`Query` class for dynamic relationships and backrefs.

        SQLAlchemy 2.0 removes this name, use ``relationship`` instead.

        :meta private:

        .. versionchanged:: 3.0
            The :attr:`Query` class is set on ``backref``.
        """
        self._set_rel_query(kwargs)
        f = sa_orm.relationship
        return f(*args, **kwargs)

    def __getattr__(self, name: str) -> t.Any:
        if name == "relation":
            return self._relation

        if name == "event":
            return sa_event

        if name.startswith("_"):
            raise AttributeError(name)

        for mod in (sa, sa_orm):
            if hasattr(mod, name):
                return getattr(mod, name)

        raise AttributeError(name)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\quantization\onnx_quantizer.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import logging

import numpy as np
import onnx
import onnx.numpy_helper
from onnx import onnx_pb as onnx_proto

from .base_quantizer import BaseQuantizer, QuantizationParams
from .calibrate import TensorData
from .onnx_model import ONNXModel
from .quant_utils import (
    TENSOR_NAME_QUANT_SUFFIX,
    QuantizationMode,
    QuantizedValue,
    QuantizedValueType,
    __producer__,
    __version__,
    add_infer_metadata,
    attribute_to_kwarg,
    compute_scale_zp,
    compute_scale_zp_float8,
    find_by_name,
    get_qmin_qmax_for_qType,
    get_qrange_for_qType,
    ms_domain,
    save_and_reload_model_with_shape_infer,
    tensor_proto_to_array,
)
from .registry import CreateOpQuantizer


class ONNXQuantizer(BaseQuantizer):
    def __init__(
        self,
        model,
        per_channel,
        reduce_range,
        mode,
        static,
        weight_qType,
        activation_qType,
        tensors_range,
        nodes_to_quantize,
        nodes_to_exclude,
        op_types_to_quantize,
        extra_options=None,
    ):
        BaseQuantizer.__init__(
            self,
            model,
            per_channel,
            reduce_range,
            weight_qType,
            activation_qType,
            tensors_range,
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            extra_options,
        )

        if not static:
            self.model.replace_gemm_with_matmul()
            # We need to update value_infos.
            model = save_and_reload_model_with_shape_infer(self.model.model)
            self.value_infos = {vi.name: vi for vi in model.graph.value_info}
            self.value_infos.update({ot.name: ot for ot in model.graph.output})
            self.value_infos.update({it.name: it for it in model.graph.input})
            self.model = ONNXModel(model)

        self.mode = mode  # QuantizationMode.Value
        self.static = static  # use static quantization for inputs.
        self.fuse_dynamic_quant = self.opset_version > 10

        self.q_matmul_const_b_only = "MatMulConstBOnly" in self.extra_options and self.extra_options["MatMulConstBOnly"]

        self.new_nodes = []
        self.graph_scope = "/"  # for human readable debug information
        self.tensor_names = {}  # in case the shape inference not totally working
        self.tensor_names.update({ot.name: 1 for ot in model.graph.output})
        self.tensor_names.update({it.name: 1 for it in model.graph.input})
        for node in self.model.model.graph.node:
            self.tensor_names.update({output_name: 1 for output_name in node.output})

        if self.mode not in QuantizationMode:
            raise ValueError(f"unsupported quantization mode {self.mode}")

        self.quantization_params = self.calculate_quantization_params()

        # QuantizeRange tensor name and zero tensor name for scale and zero point calculation.
        # Used when static is False
        self.fixed_qrange_uint8_name = "fixed_quantization_range_uint8"
        self.fixed_qrange_int8_name = "fixed_quantization_range_int8"
        # For uint8 data-type, to compute zero point, we subtract rmin from 0 (represented by fixed_zero_name tensor)
        self.fixed_zero_name = "fixed_zero"
        # For int8 data-type, zero point is always zero (respresented by fixed_zero_point_name tensor)
        self.fixed_zero_zp_name = "fixed_zero_zp"

        # Map of all original value names to quantized value names
        self.quantized_value_map = {}
        # some output from nodes will be quantized, yet itself should be treat as existing so
        # no dequantized will be applied when needed later
        self.generated_value_names = self.model.get_non_initializer_inputs()

    # routines for subgraph support
    def quantize_subgraph(self, subgraph, graph_key):
        """
        generate submodel for the subgraph, so that we re-utilize current quantization implementation.
        quantize the submodel
        update subgraph and set it back to node
        """
        warped_model = onnx.helper.make_model(
            subgraph,
            producer_name="onnx-quantizer",
            opset_imports=self.model.model.opset_import,
        )
        add_infer_metadata(warped_model)
        sub_quantizer = ONNXQuantizer(
            warped_model,
            self.per_channel,
            self.reduce_range,
            self.mode,
            self.static,
            self.weight_qType,
            self.activation_qType,
            self.tensors_range,
            self.nodes_to_quantize,
            self.nodes_to_exclude,
            self.op_types_to_quantize,
            self.extra_options,
        )
        sub_quantizer.parent = self
        sub_quantizer.graph_scope = f"{self.graph_scope}{graph_key}/"
        sub_quantizer.quantize_model()
        return sub_quantizer.model.model.graph

    def quantize_node_with_sub_graph(self, node):
        """
        Check subgraph, if any, quantize it and replace it.
        return new_nodes added for quantizing subgraph
        """
        graph_attrs = [
            attr
            for attr in node.attribute
            if attr.type == onnx.AttributeProto.GRAPH or attr.type == onnx.AttributeProto.GRAPHS
        ]
        if len(graph_attrs) == 0:
            return node
        node_name = node.name if node.name else f"{node.op_type}_node_count_{len(self.new_nodes)}"
        kwargs = {}
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                kv = {attr.name: self.quantize_subgraph(attr.g, f"{node_name}:{attr.name}")}
            elif attr.type == onnx.AttributeProto.GRAPHS:
                value = []
                for subgraph in attr.graphs:
                    value.extend(
                        [
                            self.quantize_subgraph(
                                subgraph,
                                f"{node_name}:{attr.name}:{len(value)}",
                            )
                        ]
                    )
                kv = {attr.name: value}
            else:
                kv = attribute_to_kwarg(attr)
            kwargs.update(kv)
        return onnx.helper.make_node(node.op_type, node.input, node.output, name=node.name, **kwargs)

    def has_QDQ_nodes(self):  # noqa: N802
        """
        Detect if model already has QuantizeLinear or DequantizeLinear.
        """
        return any(
            node.op_type == "QuantizeLinear" or node.op_type == "DequantizeLinear" for node in self.model.nodes()
        )

    def find_initializer_in_path(self, initializer_name):
        if find_by_name(initializer_name, self.model.initializer()) is not None:
            return True
        if self.parent is not None:
            return self.parent.find_initializer_in_path(initializer_name)
        return False

    def add_new_nodes(self, nodes):
        self.new_nodes.extend(nodes)
        for node in nodes:
            for output_name in node.output:
                self.generated_value_names.add(output_name)

    def quantize_model(self):
        if self.has_QDQ_nodes():
            logging.warning(
                "Please check if the model is already quantized. "
                "Note you don't need to quantize a QAT model. OnnxRuntime support to run QAT model directly."
            )

        for node in self.model.nodes():
            # quantize subgraphes if have
            if self.enable_subgraph_quantization:
                node = self.quantize_node_with_sub_graph(node)  # noqa: PLW2901

            number_of_existing_new_nodes = len(self.new_nodes)
            op_quantizer = CreateOpQuantizer(self, node)
            op_quantizer.quantize()
            for i in range(number_of_existing_new_nodes, len(self.new_nodes)):
                for output_name in self.new_nodes[i].output:
                    self.generated_value_names.add(output_name)

        self._dequantize_outputs()

        # extend is used to append to the list for a protobuf fields
        # https://developers.google.com/protocol-buffers/docs/reference/python-generated?csw=1#fields
        self.model.graph().ClearField("node")
        self.model.graph().node.extend(self.new_nodes)

        # Remove ununsed initializers from graph, starting from the top level graph.
        if self.parent is None:
            _, initializers_not_found = self.model.clean_initializers()
            if len(initializers_not_found) > 0:
                raise RuntimeError("Invalid model with unknown initializers/tensors." + str(initializers_not_found))

        self.model.model.producer_name = __producer__
        self.model.model.producer_version = __version__
        # Add ms domain if needed
        ms_opset = [opset for opset in self.model.model.opset_import if opset.domain == ms_domain]
        if not ms_opset:
            ms_nodes = [node for node in self.new_nodes if node.domain == "com.microsoft"]
            if ms_nodes:
                opset = self.model.model.opset_import.add()
                opset.version = 1
                opset.domain = ms_domain

        return self.model.model

    def _get_default_tensor_type(self, tensor_name):
        if "DefaultTensorType" in self.extra_options:
            logging.info(
                "get_tensor_type returns DefaultTensorType for tensor name %r, use %d",
                tensor_name,
                self.extra_options["DefaultTensorType"],
            )
            return self.extra_options["DefaultTensorType"]
        raise RuntimeError(
            f"Unable to find data type for weight_name={tensor_name!r}. "
            f"shape_inference failed to return a type probably this node is "
            f"from a different domain or using an input produced by such an operator. "
            f"This may happen if you quantize a model already quantized. "
            f"You may use extra_options `DefaultTensorType` to indicate "
            f"the default weight type, usually `onnx.TensorProto.FLOAT`."
        )

    def get_tensor_type(self, tensor_name, mandatory=False):
        weight = find_by_name(tensor_name, self.model.initializer())
        if weight is not None:
            return weight.data_type
        if tensor_name in self.value_infos:
            vi = self.value_infos[tensor_name]
            if vi.type.HasField("tensor_type"):
                if mandatory and vi.type.tensor_type.elem_type == 0:
                    return self._get_default_tensor_type(tensor_name)
                return vi.type.tensor_type.elem_type
        if (not self.enable_subgraph_quantization) or (self.parent is None):
            if mandatory:
                return self._get_default_tensor_type(tensor_name)
            return None
        otype = self.parent.is_valid_quantize_weight(tensor_name)
        if otype is not None:
            return otype
        if self.enable_subgraph_quantization and self.parent:
            res = self.parent.get_tensor_type(tensor_name)
            if res is not None:
                return res
        if mandatory:
            return self._get_default_tensor_type(tensor_name)
        return None

    def is_float_tensor(self, tensor_name):
        if self.is_input_a_initializer(tensor_name):
            return self.is_valid_quantize_weight(tensor_name)

        if tensor_name in self.value_infos:
            vi = self.value_infos[tensor_name]
            if vi.type.HasField("tensor_type") and vi.type.tensor_type.elem_type in (
                onnx_proto.TensorProto.FLOAT,
                onnx_proto.TensorProto.FLOAT16,
            ):
                return True
            logging.warning(
                f"Inference failed or unsupported type to quantize for tensor {tensor_name!r}, type is {vi.type}."
            )
            return False

        if self.enable_subgraph_quantization and self.parent:
            return self.parent.is_float_tensor(tensor_name)

        logging.warning(
            f"Failed to infer data type of tensor: {tensor_name!r}. Please add data type info for this tensor "
            f"if your model has customized operators."
        )
        return False

    def _get_dynamic_input_quantization_params(self, input_name, nodes_list, qType, initial_type):
        """
        Create nodes for dynamic quantization of input and add them to nodes_list.
            parameter input_name: Name of the input.
            parameter nodes_list: new nodes are appended to this list.
            parameter qType: type to quantize to.
            parameter initial_type: type to quantize from
            return: scale_name, zero_point_name, scale_shape, zero_point_shape.
        """
        if qType == onnx_proto.TensorProto.INT8:
            return self._get_dynamic_input_quantization_params_int8(input_name, nodes_list, initial_type)
        if qType == onnx_proto.TensorProto.UINT8:
            return self._get_dynamic_input_quantization_params_uint8(input_name, nodes_list, initial_type)
        raise ValueError(f"Unexpected value for qType={qType}.")

    def _get_dynamic_input_quantization_params_int8(self, input_name, nodes_list, initial_type):
        """
        Create nodes for dynamic quantization of input to int8 and add them to nodes_list
            parameter input_name: Name of the input.
            parameter nodes_list: new nodes are appended to this list.
            parameter initial_type: initial weight type (FLOAT or FLOAT16)
            return: scale_name, zero_point_name, scale_shape, zero_point_shape.
        """
        qType = onnx_proto.TensorProto.INT8  # noqa: N806

        # Reduce min and Reduce max
        input_scale_name = input_name + "_scale"

        reduce_min_name = input_name + "_ReduceMin"
        reduce_min_node = onnx.helper.make_node(
            "ReduceMin",
            [input_name],
            [reduce_min_name + ":0"],
            reduce_min_name,
            keepdims=0,
        )
        nodes_list.append(reduce_min_node)

        reduce_max_name = input_name + "_ReduceMax"
        reduce_max_node = onnx.helper.make_node(
            "ReduceMax",
            [input_name],
            [reduce_max_name + ":0"],
            reduce_max_name,
            keepdims=0,
        )
        nodes_list.append(reduce_max_node)

        # Compute scale
        #   Find abs(rmin)
        reduce_min_abs_name = reduce_min_name + "_Abs"
        reduce_min_abs_node = onnx.helper.make_node(
            "Abs",
            [reduce_min_node.output[0]],
            [reduce_min_abs_name + ":0"],
            reduce_min_abs_name,
        )
        nodes_list.append(reduce_min_abs_node)
        #   Find abs(rmax)
        reduce_max_abs_name = reduce_max_name + "_Abs"
        reduce_max_abs_node = onnx.helper.make_node(
            "Abs",
            [reduce_max_node.output[0]],
            [reduce_max_abs_name + ":0"],
            reduce_max_abs_name,
        )
        nodes_list.append(reduce_max_abs_node)
        #   Compute max of abs(rmin) and abs(rmax)
        abs_max_name = input_name + "_Abs_Max"
        abs_max_node = onnx.helper.make_node(
            "Max",
            [reduce_min_abs_node.output[0], reduce_max_abs_node.output[0]],
            [abs_max_name + ":0"],
            abs_max_name,
        )
        nodes_list.append(abs_max_node)
        #   and divide by (quantize_range/2.0) which will be equal to max(...)*2.0/quantize_range
        initializer_div = onnx.helper.make_tensor(
            self.fixed_qrange_int8_name,
            initial_type,
            [],
            [get_qrange_for_qType(qType) / 2.0],
        )
        self.model.add_initializer(initializer_div)
        scale_div_name = input_name + "scale_Div"
        scale_div_node = onnx.helper.make_node(
            "Div",
            [abs_max_node.output[0], self.fixed_qrange_int8_name],
            [input_scale_name],
            scale_div_name,
        )
        nodes_list.append(scale_div_node)

        # Zero point
        initializer_zp = onnx.helper.make_tensor(self.fixed_zero_zp_name, qType, [], [0])
        self.model.add_initializer(initializer_zp)

        return input_scale_name, self.fixed_zero_zp_name, [], []

    def _get_dynamic_input_quantization_params_uint8(self, input_name, nodes_list, initial_type):
        """
        Create nodes for dynamic quantization of input to uint8 and add them to nodes_list
            parameter input_name: Name of the input.
            parameter nodes_list: new nodes are appended to this list.
            parameter initial_type: initial weight type (FLAOT or FLOAT16)
            return: scale_name, zero_point_name, scale_shape, zero_point_shape.
        """
        qType = onnx_proto.TensorProto.UINT8  # noqa: N806
        # Reduce min and Reduce max
        input_scale_name = input_name + "_scale"
        input_zp_name = input_name + "_zero_point"

        reduce_min_name = input_name + "_ReduceMin"
        reduce_min_node = onnx.helper.make_node(
            "ReduceMin",
            [input_name],
            [reduce_min_name + ":0"],
            reduce_min_name,
            keepdims=0,
        )
        nodes_list.append(reduce_min_node)

        reduce_max_name = input_name + "_ReduceMax"
        reduce_max_node = onnx.helper.make_node(
            "ReduceMax",
            [input_name],
            [reduce_max_name + ":0"],
            reduce_max_name,
            keepdims=0,
        )
        nodes_list.append(reduce_max_node)

        # Add tensors for quantize range and zero value.
        initializer_qrange = onnx.helper.make_tensor(
            self.fixed_qrange_uint8_name,
            initial_type,
            [],
            [get_qrange_for_qType(qType)],
        )
        self.model.add_initializer(initializer_qrange)
        initializer_qvalue = onnx.helper.make_tensor(self.fixed_zero_name, initial_type, [], [0.0])
        self.model.add_initializer(initializer_qvalue)

        # Compute Scale
        #   Subtract rmax and rmin
        scale_sub_name = input_name + "_scale_Sub"
        scale_sub_node = onnx.helper.make_node(
            "Sub",
            [reduce_max_node.output[0], reduce_min_node.output[0]],
            [scale_sub_name + ":0"],
            scale_sub_name,
        )
        nodes_list.append(scale_sub_node)
        #   and divide by quantize range
        scale_div_name = input_name + "_scale_Div"
        scale_div_node = onnx.helper.make_node(
            "Div",
            [scale_sub_node.output[0], self.fixed_qrange_uint8_name],
            [input_scale_name],
            scale_div_name,
        )
        nodes_list.append(scale_div_node)

        # Compute zero point
        #   Subtract zero and rmin
        zp_sub_name = input_name + "_zero_point_Sub"
        zp_sub_node = onnx.helper.make_node(
            "Sub",
            [self.fixed_zero_name, reduce_min_node.output[0]],
            [zp_sub_name + ":0"],
            zp_sub_name,
        )
        nodes_list.append(zp_sub_node)
        #   Divide by scale
        zp_div_name = input_name + "_zero_point_Div"
        zp_div_node = onnx.helper.make_node(
            "Div",
            [zp_sub_node.output[0], input_scale_name],
            [zp_div_name + ":0"],
            zp_div_name,
        )
        nodes_list.append(zp_div_node)
        #   Compute floor
        zp_floor_name = input_name + "_zero_point_Floor"
        zp_floor_node = onnx.helper.make_node("Floor", zp_div_node.output, [zp_floor_name + ":0"], zp_floor_name)
        nodes_list.append(zp_floor_node)
        #   Cast to integer
        zp_cast_name = input_name + "_zero_point_Cast"
        zp_cast_node = onnx.helper.make_node("Cast", zp_floor_node.output, [input_zp_name], zp_cast_name, to=qType)
        nodes_list.append(zp_cast_node)

        return input_scale_name, input_zp_name, [], []

    def _get_quantization_params(self, param_name, use_scale=None, use_zeropoint=None):
        """
        Create initializers and inputs in the graph for zero point and scale of output.
        Zero point and scale values are obtained from self.quantization_params if specified.
            parameter param_name: Name of the quantization parameter.
            return: result, scale_name, zero_point_name, scale_shape, zero_point_shape.
        """
        zero_point_type = self.activation_qType

        if use_scale is None or use_zeropoint is None:
            if self.quantization_params is None or param_name not in self.quantization_params:
                logging.info(f'Quantization parameters for tensor:"{param_name}" not specified')
                return False, "", "", "", ""

            params = self.quantization_params[param_name]
            if not isinstance(params, QuantizationParams):
                raise TypeError(f"Unexpected type {type(params)} for {param_name!r}.")
            if params is None or len(params) != 3:
                raise ValueError(
                    "Quantization parameters should contain zero point, scale, quant type. "
                    f"Specified values for output {param_name}: {params}"
                )

            zero_point_values = np.array([params["zero_point"]])
            if not hasattr(params["scale"], "dtype") or params["scale"].dtype not in (np.float32, np.float16):
                raise ValueError(f"Unexpected type {type(params['scale'])} and param_name={param_name!r}")
            scale_values = np.array([params["scale"]])
            assert scale_values.dtype != np.float64
            zero_point_type = params["quant_type"]
        else:
            zero_point_values = np.array([use_zeropoint])
            scale_values = np.array([use_scale])
            params = self.quantization_params[param_name]
            if "scale" in params:
                dtype = params["scale"].dtype
                scale_values = scale_values.astype(dtype)
            assert scale_values.dtype != np.float64

        zero_point_shape = []
        zero_point_name = param_name + "_zero_point"
        scale_shape = []
        scale_name = param_name + "_scale"

        # Add initializers
        init_zp = onnx.helper.make_tensor(
            zero_point_name, zero_point_type, zero_point_shape, zero_point_values.ravel().tolist()
        )
        self.model.add_initializer(init_zp)
        if scale_values.dtype == np.float32:
            scale_type = onnx_proto.TensorProto.FLOAT
        elif scale_values.dtype == np.float16:
            scale_type = onnx_proto.TensorProto.FLOAT16
        else:
            raise ValueError(f"Unexpected dtype={scale_values.dtype} for param_name={param_name!r}")
        init_scale = onnx.helper.make_tensor(scale_name, scale_type, scale_shape, scale_values.reshape((-1,)).tolist())
        self.model.add_initializer(init_scale)

        return True, scale_name, zero_point_name, scale_shape, zero_point_shape

    def _get_quantize_input_nodes(
        self, node, input_index, qType, given_scale_name=None, given_zp_name=None, initial_type=None
    ):
        """
        Given an input for a node (which is not a initializer), this function

        - add nodes to compute zero point and scale for this input if they don't exist.
        - add new QuantizeLinear node to quantize the input.

        :param node: node being quantized in NodeProto format.
        :param input_index: index of input in node.input.
        :param qType: type to quantize to.
        :param given_scale_name: if those inputs need to be quanitzed using this scale tensor.
        :param given_zp_name: if those inputs to be quantized using this zeropoint tensor.
        :param initial_type: type of the weight to quantize
        :return: List of newly created nodes in NodeProto format.
        """
        input_name = node.input[input_index]
        assert input_name != "", "Cannot access undefined variable in graph."
        output_name = input_name + TENSOR_NAME_QUANT_SUFFIX
        ql_node_name = input_name + "_QuantizeLinear"

        if (given_scale_name is not None) and (given_zp_name is not None):
            data_found, scale_name, zp_name = (True, given_scale_name, given_zp_name)
        else:
            data_found, scale_name, zp_name, _, _ = self._get_quantization_params(input_name)

        nodes = []
        if data_found:
            qlinear_node = onnx.helper.make_node(
                "QuantizeLinear",
                [input_name, scale_name, zp_name],
                [output_name],
                ql_node_name,
            )
        else:
            if self.static:
                return None
            # dynamic mode
            # Scale and Zero Points not available for this input. Add nodes to dynamically compute it
            if self.fuse_dynamic_quant and qType == onnx_proto.TensorProto.UINT8:
                scale_name = input_name + "_scale"
                zp_name = input_name + "_zero_point"
                qlinear_node = onnx.helper.make_node(
                    "DynamicQuantizeLinear",
                    [input_name],
                    [output_name, scale_name, zp_name],
                    ql_node_name,
                )
            else:
                assert initial_type is not None, (
                    f"Cannot quantize input without knowing the initial type, "
                    f"input_name={input_name!r}, input_index={input_index}, qType={qType}, node={node}"
                )
                (
                    scale_name,
                    zp_name,
                    scale_shape,
                    zp_shape,
                ) = self._get_dynamic_input_quantization_params(input_name, nodes, qType, initial_type=initial_type)
                qlinear_node = onnx.helper.make_node(
                    "QuantizeLinear",
                    [input_name, scale_name, zp_name],
                    [output_name],
                    ql_node_name,
                )

        self.quantized_value_map[input_name] = QuantizedValue(input_name, output_name, scale_name, zp_name, qType)
        return [*nodes, qlinear_node]

    def find_quantized_value(self, input_name):
        if input_name in self.quantized_value_map:
            return self.quantized_value_map[input_name]
        if self.parent is not None:
            return self.parent.find_quantized_value(input_name)
        return None

    def quantize_bias_static(self, bias_name, input_name, weight_name, beta=1.0):
        """
        Quantized the bias. Zero Point == 0 and Scale == Input_Scale * Weight_Scale
        """

        # Handle case where bias already in quantization map
        if bias_name in self.quantized_value_map:
            return self.quantized_value_map[bias_name].q_name

        # get scale for weight
        weight_scale_name = self.quantized_value_map[weight_name].scale_name
        weight_initializer = find_by_name(weight_scale_name, self.model.initializer())
        weight_scale = tensor_proto_to_array(weight_initializer)

        # get scale for input
        if input_name in self.quantized_value_map:
            input_scale_name = self.quantized_value_map[input_name].scale_name
        elif input_name in self.quantization_params:
            _, input_scale_name, _, _, _ = self._get_quantization_params(input_name)
        else:
            raise ValueError(f"Expected {input_name} to be in quantized value map for static quantization")

        inputscale_initializer = find_by_name(input_scale_name, self.model.initializer())
        input_scale = tensor_proto_to_array(inputscale_initializer)

        (
            quantized_bias_name,
            quantized_bias_scale_name,
            quantized_bias_zp_name,
            bias_scale_data,
            node_type,
            node_qtype,
        ) = self.quantize_bias_static_impl(bias_name, input_scale, weight_scale, beta)

        assert bias_name not in self.quantized_value_map
        quantized_value = QuantizedValue(
            bias_name,
            quantized_bias_name,
            quantized_bias_scale_name,
            quantized_bias_zp_name,
            QuantizedValueType.Initializer,
            0 if bias_scale_data.size > 1 else None,
            node_type=node_type,
            node_qtype=node_qtype,
        )
        self.quantized_value_map[bias_name] = quantized_value

        return quantized_bias_name

    def contains_tensor(self, tensor_name):
        """
        only check for value info and newly generated tensor names, initializers are checked separately
        """
        return (
            (tensor_name in self.value_infos)
            or (tensor_name in self.tensor_names)
            or (tensor_name in self.generated_value_names)
        )

    def quantize_activation(self, node, indices, from_subgraph=False):
        return self.__quantize_inputs(
            node=node,
            indices=indices,
            initializer_use_weight_qType=False,
            reduce_range=False,
            op_level_per_channel=False,
            axis=-1,
            from_subgraph=from_subgraph,
        )

    # In some circumstances a weight is not an initializer, for example of MatMul, if both A and B are not
    # initializer, B can still be considered as Weight
    def quantize_weight(
        self,
        node,
        indices,
        reduce_range=False,
        op_level_per_channel=False,
        axis=-1,
        from_subgraph=False,
    ):
        return self.__quantize_inputs(
            node=node,
            indices=indices,
            initializer_use_weight_qType=True,
            reduce_range=reduce_range,
            op_level_per_channel=op_level_per_channel,
            axis=axis,
            from_subgraph=from_subgraph,
        )

    def __quantize_inputs(
        self,
        node,
        indices,
        initializer_use_weight_qType=True,
        reduce_range=False,
        op_level_per_channel=False,
        axis=-1,
        from_subgraph=False,
    ):
        """
        Given a node, this function quantizes the inputs as follows:
            - If input is an initializer, quantize the initializer data, replace old initializer
              with new initializer
            - Else, add QuantizeLinear nodes to perform quantization
            parameter node: node being quantized in NodeProto format.
            parameter indices: input indices to quantize.
            return: (List of quantized input names,
                     List of zero point names used for input quantization,
                     List of scale names used for input quantization,
                     List of new QuantizeLinear nodes created)
        """

        scale_names = []
        zero_point_names = []
        quantized_input_names = []
        nodes = []

        for input_index in indices:
            node_input = node.input[input_index]

            # Find if this input is already quantized
            if node_input in self.quantized_value_map:
                quantized_value = self.quantized_value_map[node_input]
                scale_names.append(quantized_value.scale_name)
                zero_point_names.append(quantized_value.zp_name)
                quantized_input_names.append(quantized_value.q_name)
                continue
            # adding this for case embed_layernorm.py has optional segment_embedding
            if not node_input:
                quantized_input_names.append("")
                scale_names.append("")
                zero_point_names.append("")
                continue
            # Quantize the input
            initializer = find_by_name(node_input, self.model.initializer())
            if initializer is not None:
                if self.per_channel and op_level_per_channel:
                    (
                        q_weight_name,
                        zp_name,
                        scale_name,
                    ) = self.quantize_weight_per_channel(
                        initializer.name,
                        self.weight_qType if initializer_use_weight_qType else self.activation_qType,
                        axis,
                        reduce_range,
                    )
                else:
                    q_weight_name, zp_name, scale_name = self.quantize_initializer(
                        initializer,
                        self.weight_qType if initializer_use_weight_qType else self.activation_qType,
                        reduce_range,
                    )

                quantized_input_names.append(q_weight_name)
                zero_point_names.append(zp_name)
                scale_names.append(scale_name)
            elif self.contains_tensor(node_input):
                # Add QuantizeLinear node.
                qlinear_node = self.model.find_node_by_name(
                    node_input + "_QuantizeLinear", self.new_nodes, self.model.graph()
                )
                if qlinear_node is None:
                    input_name = node.input[input_index]
                    if input_name in self.value_infos:
                        value_info = self.value_infos[input_name]
                        assert value_info.HasField("type"), f"value_info={value_info} has no type."
                        assert value_info.type.HasField("tensor_type"), f"value_info={value_info} is not a tensor."
                        initial_type = value_info.type.tensor_type.elem_type
                    else:
                        # Shape inference failed. Fallback to self.tensor_names.
                        assert input_name in self.tensor_names, (
                            f"shape inference failed for {input_name!r} and "
                            f"attribute 'tensor_names' does not have any value for "
                            f"this tensor."
                        )
                        initial_type = self.tensor_names[input_name]
                    quantize_input_nodes = self._get_quantize_input_nodes(
                        node, input_index, self.activation_qType, initial_type=initial_type
                    )
                    if quantize_input_nodes is None:
                        return (None, None, None, None)
                    if from_subgraph:
                        self.add_new_nodes(quantize_input_nodes)
                    else:
                        nodes.extend(quantize_input_nodes)
                    qlinear_node = quantize_input_nodes[-1]

                if qlinear_node.op_type == "QuantizeLinear":
                    quantized_input_names.extend(qlinear_node.output)
                    scale_names.append(qlinear_node.input[1])
                    zero_point_names.append(qlinear_node.input[2])
                else:
                    quantized_input_names.append(qlinear_node.output[0])
                    scale_names.append(qlinear_node.output[1])
                    zero_point_names.append(qlinear_node.output[2])
            elif self.parent is not None:
                (
                    parent_quantized_input_names,
                    parent_zero_point_names,
                    parent_scale_names,
                    _,
                ) = self.parent.__quantize_inputs(
                    node,
                    [input_index],
                    initializer_use_weight_qType=initializer_use_weight_qType,
                    reduce_range=reduce_range,
                    op_level_per_channel=op_level_per_channel,
                    axis=axis,
                    from_subgraph=True,
                )
                quantized_input_names.append(parent_quantized_input_names[0])
                scale_names.append(parent_scale_names[0])
                zero_point_names.append(parent_zero_point_names[0])
                # node should not be add this child level here
            else:
                raise ValueError(f"Invalid tensor name to quantize: {node_input} @graph scope{self.graph_scope}")

        return quantized_input_names, zero_point_names, scale_names, nodes

    def quantize_initializer(self, weight, qType, reduce_range=False, keep_float_weight=False):
        """
        :param weight: TensorProto initializer
        :param qType: type to quantize to
        :param keep_float_weight: Whether to quantize the weight. In some cases, we only want to qunatize scale and zero point.
                                  If keep_float_weight is False, quantize the weight, or don't quantize the weight.
        :return: quantized weight name, zero point name, scale name
        """
        # Find if this input is already quantized
        if weight.name in self.quantized_value_map:
            quantized_value = self.quantized_value_map[weight.name]
            return (
                quantized_value.q_name,
                quantized_value.zp_name,
                quantized_value.scale_name,
            )

        q_weight_name, zp_name, scale_name = self.quantize_initializer_impl(
            weight, qType, reduce_range, keep_float_weight
        )

        # Log entry for this quantized weight
        quantized_value = QuantizedValue(
            weight.name,
            q_weight_name,
            scale_name,
            zp_name,
            QuantizedValueType.Initializer,
            None,
        )
        self.quantized_value_map[weight.name] = quantized_value
        return q_weight_name, zp_name, scale_name

    def quantize_weight_per_channel(
        self,
        weight_name,
        weight_qType,
        channel_axis,
        reduce_range=True,
        keep_float_weight=False,
    ):
        # Find if this input is already quantized
        if weight_name in self.quantized_value_map:
            quantized_value = self.quantized_value_map[weight_name]
            return (
                quantized_value.q_name,
                quantized_value.zp_name,
                quantized_value.scale_name,
            )

        q_weight_name, zp_name, scale_name = self.quantize_weight_per_channel_impl(
            weight_name, weight_qType, channel_axis, reduce_range, keep_float_weight
        )
        quantized_value = QuantizedValue(
            weight_name,
            q_weight_name,
            scale_name,
            zp_name,
            QuantizedValueType.Initializer,
            None,
        )
        self.quantized_value_map[weight_name] = quantized_value

        return q_weight_name, zp_name, scale_name

    def _dequantize_value(self, value_name):
        """
        Given a value (input/output) which is quantized, add a DequantizeLinear node to dequantize
        it back to float32 or float16
            parameter value_name: value to dequantize
            parameter new_nodes_list: List of new nodes created before processing current node
            return: None if there is already a DequantizeLinear node that dequantizes it
                    A DequantizeLinear node otherwise
        """
        if (value_name in self.quantized_value_map) and (value_name not in self.generated_value_names):
            quantized_value = self.quantized_value_map[value_name]
            # Add DequantizeLinear Node for this input

            scale_init = find_by_name(quantized_value.scale_name, self.model.initializer())

            # In case we are working with subgraphs, the graph `producer_name` is set to `"onnx-quantizer"` in the `quantize_subgraph` method. In this case, the scale initializer may be on the top level graph, so the check below can not be done.
            if self.model.model.producer_name != "onnx-quantizer" or (
                self.model.model.producer_name == "onnx-quantizer" and scale_init is not None
            ):
                # axis is not specified so scale_init must be a scalar.
                assert scale_init is None or onnx.numpy_helper.to_array(scale_init).size == 1

            dqlinear_name = value_name + "_DequantizeLinear"
            dqlinear_node = self.model.find_node_by_name(dqlinear_name, self.new_nodes, self.model.graph())
            if dqlinear_node is None:
                dqlinear_inputs = [
                    quantized_value.q_name,
                    quantized_value.scale_name,
                    quantized_value.zp_name,
                ]
                dequantize_node = onnx.helper.make_node(
                    "DequantizeLinear", dqlinear_inputs, [value_name], dqlinear_name
                )
                return dequantize_node
            else:
                # DQ op is already present, assert it's output matches the input of current node
                assert value_name == dqlinear_node.output[0]
        return None

    def _dequantize_outputs(self):
        """
        Dequantize output if it is quantized
            parameter new_nodes_list: List of new nodes created before processing current node
            return: List of new nodes created
        """

        for output in self.model.graph().output:
            dequantize_node = self._dequantize_value(output.name)
            if dequantize_node is not None:
                self.new_nodes.append(dequantize_node)

    def calculate_quantization_params(self):
        if self.tensors_range is None:
            return None

        self.adjust_tensor_ranges()

        quantization_params = {}
        for tensor_name in self.tensors_range:
            td = self.tensors_range[tensor_name]
            if not isinstance(td, TensorData):
                raise TypeError(f"Unexpected type {type(td)} for {tensor_name!r}.")

            quant_overrides = self.tensor_quant_overrides.get_per_tensor_overrides(tensor_name, default_val={})

            quant_type = self.activation_qType
            if "quant_type" in quant_overrides:
                quant_type = quant_overrides["quant_type"].tensor_type

            if "scale" in quant_overrides and "zero_point" in quant_overrides:
                zero, scale = quant_overrides["zero_point"], quant_overrides["scale"]
            elif quant_type == onnx.TensorProto.FLOAT8E4M3FN:
                zero, scale = compute_scale_zp_float8(quant_type, td.avg_std[1])
            else:
                rmin = quant_overrides.get("rmin", td.range_value[0])
                rmax = quant_overrides.get("rmax", td.range_value[1])
                symmetric = quant_overrides.get("symmetric", self.is_activation_symmetric)
                reduce_range = quant_overrides.get("reduce_range", False)
                qmin, qmax = get_qmin_qmax_for_qType(quant_type, reduce_range=reduce_range, symmetric=symmetric)
                zero, scale = compute_scale_zp(rmin, rmax, qmin, qmax, symmetric, self.min_real_range)

            quantization_params[tensor_name] = QuantizationParams(zero_point=zero, scale=scale, quant_type=quant_type)

        return quantization_params

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\table.py ===
from dataclasses import dataclass, field, replace
from typing import (
    TYPE_CHECKING,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from . import box, errors
from ._loop import loop_first_last, loop_last
from ._pick import pick_bool
from ._ratio import ratio_distribute, ratio_reduce
from .align import VerticalAlignMethod
from .jupyter import JupyterMixin
from .measure import Measurement
from .padding import Padding, PaddingDimensions
from .protocol import is_renderable
from .segment import Segment
from .style import Style, StyleType
from .text import Text, TextType

if TYPE_CHECKING:
    from .console import (
        Console,
        ConsoleOptions,
        JustifyMethod,
        OverflowMethod,
        RenderableType,
        RenderResult,
    )


@dataclass
class Column:
    """Defines a column within a ~Table.

    Args:
        title (Union[str, Text], optional): The title of the table rendered at the top. Defaults to None.
        caption (Union[str, Text], optional): The table caption rendered below. Defaults to None.
        width (int, optional): The width in characters of the table, or ``None`` to automatically fit. Defaults to None.
        min_width (Optional[int], optional): The minimum width of the table, or ``None`` for no minimum. Defaults to None.
        box (box.Box, optional): One of the constants in box.py used to draw the edges (see :ref:`appendix_box`), or ``None`` for no box lines. Defaults to box.HEAVY_HEAD.
        safe_box (Optional[bool], optional): Disable box characters that don't display on windows legacy terminal with *raster* fonts. Defaults to True.
        padding (PaddingDimensions, optional): Padding for cells (top, right, bottom, left). Defaults to (0, 1).
        collapse_padding (bool, optional): Enable collapsing of padding around cells. Defaults to False.
        pad_edge (bool, optional): Enable padding of edge cells. Defaults to True.
        expand (bool, optional): Expand the table to fit the available space if ``True``, otherwise the table width will be auto-calculated. Defaults to False.
        show_header (bool, optional): Show a header row. Defaults to True.
        show_footer (bool, optional): Show a footer row. Defaults to False.
        show_edge (bool, optional): Draw a box around the outside of the table. Defaults to True.
        show_lines (bool, optional): Draw lines between every row. Defaults to False.
        leading (int, optional): Number of blank lines between rows (precludes ``show_lines``). Defaults to 0.
        style (Union[str, Style], optional): Default style for the table. Defaults to "none".
        row_styles (List[Union, str], optional): Optional list of row styles, if more than one style is given then the styles will alternate. Defaults to None.
        header_style (Union[str, Style], optional): Style of the header. Defaults to "table.header".
        footer_style (Union[str, Style], optional): Style of the footer. Defaults to "table.footer".
        border_style (Union[str, Style], optional): Style of the border. Defaults to None.
        title_style (Union[str, Style], optional): Style of the title. Defaults to None.
        caption_style (Union[str, Style], optional): Style of the caption. Defaults to None.
        title_justify (str, optional): Justify method for title. Defaults to "center".
        caption_justify (str, optional): Justify method for caption. Defaults to "center".
        highlight (bool, optional): Highlight cell contents (if str). Defaults to False.
    """

    header: "RenderableType" = ""
    """RenderableType: Renderable for the header (typically a string)"""

    footer: "RenderableType" = ""
    """RenderableType: Renderable for the footer (typically a string)"""

    header_style: StyleType = ""
    """StyleType: The style of the header."""

    footer_style: StyleType = ""
    """StyleType: The style of the footer."""

    style: StyleType = ""
    """StyleType: The style of the column."""

    justify: "JustifyMethod" = "left"
    """str: How to justify text within the column ("left", "center", "right", or "full")"""

    vertical: "VerticalAlignMethod" = "top"
    """str: How to vertically align content ("top", "middle", or "bottom")"""

    overflow: "OverflowMethod" = "ellipsis"
    """str: Overflow method."""

    width: Optional[int] = None
    """Optional[int]: Width of the column, or ``None`` (default) to auto calculate width."""

    min_width: Optional[int] = None
    """Optional[int]: Minimum width of column, or ``None`` for no minimum. Defaults to None."""

    max_width: Optional[int] = None
    """Optional[int]: Maximum width of column, or ``None`` for no maximum. Defaults to None."""

    ratio: Optional[int] = None
    """Optional[int]: Ratio to use when calculating column width, or ``None`` (default) to adapt to column contents."""

    no_wrap: bool = False
    """bool: Prevent wrapping of text within the column. Defaults to ``False``."""

    highlight: bool = False
    """bool: Apply highlighter to column. Defaults to ``False``."""

    _index: int = 0
    """Index of column."""

    _cells: List["RenderableType"] = field(default_factory=list)

    def copy(self) -> "Column":
        """Return a copy of this Column."""
        return replace(self, _cells=[])

    @property
    def cells(self) -> Iterable["RenderableType"]:
        """Get all cells in the column, not including header."""
        yield from self._cells

    @property
    def flexible(self) -> bool:
        """Check if this column is flexible."""
        return self.ratio is not None


@dataclass
class Row:
    """Information regarding a row."""

    style: Optional[StyleType] = None
    """Style to apply to row."""

    end_section: bool = False
    """Indicated end of section, which will force a line beneath the row."""


class _Cell(NamedTuple):
    """A single cell in a table."""

    style: StyleType
    """Style to apply to cell."""
    renderable: "RenderableType"
    """Cell renderable."""
    vertical: VerticalAlignMethod
    """Cell vertical alignment."""


class Table(JupyterMixin):
    """A console renderable to draw a table.

    Args:
        *headers (Union[Column, str]): Column headers, either as a string, or :class:`~rich.table.Column` instance.
        title (Union[str, Text], optional): The title of the table rendered at the top. Defaults to None.
        caption (Union[str, Text], optional): The table caption rendered below. Defaults to None.
        width (int, optional): The width in characters of the table, or ``None`` to automatically fit. Defaults to None.
        min_width (Optional[int], optional): The minimum width of the table, or ``None`` for no minimum. Defaults to None.
        box (box.Box, optional): One of the constants in box.py used to draw the edges (see :ref:`appendix_box`), or ``None`` for no box lines. Defaults to box.HEAVY_HEAD.
        safe_box (Optional[bool], optional): Disable box characters that don't display on windows legacy terminal with *raster* fonts. Defaults to True.
        padding (PaddingDimensions, optional): Padding for cells (top, right, bottom, left). Defaults to (0, 1).
        collapse_padding (bool, optional): Enable collapsing of padding around cells. Defaults to False.
        pad_edge (bool, optional): Enable padding of edge cells. Defaults to True.
        expand (bool, optional): Expand the table to fit the available space if ``True``, otherwise the table width will be auto-calculated. Defaults to False.
        show_header (bool, optional): Show a header row. Defaults to True.
        show_footer (bool, optional): Show a footer row. Defaults to False.
        show_edge (bool, optional): Draw a box around the outside of the table. Defaults to True.
        show_lines (bool, optional): Draw lines between every row. Defaults to False.
        leading (int, optional): Number of blank lines between rows (precludes ``show_lines``). Defaults to 0.
        style (Union[str, Style], optional): Default style for the table. Defaults to "none".
        row_styles (List[Union, str], optional): Optional list of row styles, if more than one style is given then the styles will alternate. Defaults to None.
        header_style (Union[str, Style], optional): Style of the header. Defaults to "table.header".
        footer_style (Union[str, Style], optional): Style of the footer. Defaults to "table.footer".
        border_style (Union[str, Style], optional): Style of the border. Defaults to None.
        title_style (Union[str, Style], optional): Style of the title. Defaults to None.
        caption_style (Union[str, Style], optional): Style of the caption. Defaults to None.
        title_justify (str, optional): Justify method for title. Defaults to "center".
        caption_justify (str, optional): Justify method for caption. Defaults to "center".
        highlight (bool, optional): Highlight cell contents (if str). Defaults to False.
    """

    columns: List[Column]
    rows: List[Row]

    def __init__(
        self,
        *headers: Union[Column, str],
        title: Optional[TextType] = None,
        caption: Optional[TextType] = None,
        width: Optional[int] = None,
        min_width: Optional[int] = None,
        box: Optional[box.Box] = box.HEAVY_HEAD,
        safe_box: Optional[bool] = None,
        padding: PaddingDimensions = (0, 1),
        collapse_padding: bool = False,
        pad_edge: bool = True,
        expand: bool = False,
        show_header: bool = True,
        show_footer: bool = False,
        show_edge: bool = True,
        show_lines: bool = False,
        leading: int = 0,
        style: StyleType = "none",
        row_styles: Optional[Iterable[StyleType]] = None,
        header_style: Optional[StyleType] = "table.header",
        footer_style: Optional[StyleType] = "table.footer",
        border_style: Optional[StyleType] = None,
        title_style: Optional[StyleType] = None,
        caption_style: Optional[StyleType] = None,
        title_justify: "JustifyMethod" = "center",
        caption_justify: "JustifyMethod" = "center",
        highlight: bool = False,
    ) -> None:
        self.columns: List[Column] = []
        self.rows: List[Row] = []
        self.title = title
        self.caption = caption
        self.width = width
        self.min_width = min_width
        self.box = box
        self.safe_box = safe_box
        self._padding = Padding.unpack(padding)
        self.pad_edge = pad_edge
        self._expand = expand
        self.show_header = show_header
        self.show_footer = show_footer
        self.show_edge = show_edge
        self.show_lines = show_lines
        self.leading = leading
        self.collapse_padding = collapse_padding
        self.style = style
        self.header_style = header_style or ""
        self.footer_style = footer_style or ""
        self.border_style = border_style
        self.title_style = title_style
        self.caption_style = caption_style
        self.title_justify: "JustifyMethod" = title_justify
        self.caption_justify: "JustifyMethod" = caption_justify
        self.highlight = highlight
        self.row_styles: Sequence[StyleType] = list(row_styles or [])
        append_column = self.columns.append
        for header in headers:
            if isinstance(header, str):
                self.add_column(header=header)
            else:
                header._index = len(self.columns)
                append_column(header)

    @classmethod
    def grid(
        cls,
        *headers: Union[Column, str],
        padding: PaddingDimensions = 0,
        collapse_padding: bool = True,
        pad_edge: bool = False,
        expand: bool = False,
    ) -> "Table":
        """Get a table with no lines, headers, or footer.

        Args:
            *headers (Union[Column, str]): Column headers, either as a string, or :class:`~rich.table.Column` instance.
            padding (PaddingDimensions, optional): Get padding around cells. Defaults to 0.
            collapse_padding (bool, optional): Enable collapsing of padding around cells. Defaults to True.
            pad_edge (bool, optional): Enable padding around edges of table. Defaults to False.
            expand (bool, optional): Expand the table to fit the available space if ``True``, otherwise the table width will be auto-calculated. Defaults to False.

        Returns:
            Table: A table instance.
        """
        return cls(
            *headers,
            box=None,
            padding=padding,
            collapse_padding=collapse_padding,
            show_header=False,
            show_footer=False,
            show_edge=False,
            pad_edge=pad_edge,
            expand=expand,
        )

    @property
    def expand(self) -> bool:
        """Setting a non-None self.width implies expand."""
        return self._expand or self.width is not None

    @expand.setter
    def expand(self, expand: bool) -> None:
        """Set expand."""
        self._expand = expand

    @property
    def _extra_width(self) -> int:
        """Get extra width to add to cell content."""
        width = 0
        if self.box and self.show_edge:
            width += 2
        if self.box:
            width += len(self.columns) - 1
        return width

    @property
    def row_count(self) -> int:
        """Get the current number of rows."""
        return len(self.rows)

    def get_row_style(self, console: "Console", index: int) -> StyleType:
        """Get the current row style."""
        style = Style.null()
        if self.row_styles:
            style += console.get_style(self.row_styles[index % len(self.row_styles)])
        row_style = self.rows[index].style
        if row_style is not None:
            style += console.get_style(row_style)
        return style

    def __rich_measure__(
        self, console: "Console", options: "ConsoleOptions"
    ) -> Measurement:
        max_width = options.max_width
        if self.width is not None:
            max_width = self.width
        if max_width < 0:
            return Measurement(0, 0)

        extra_width = self._extra_width
        max_width = sum(
            self._calculate_column_widths(
                console, options.update_width(max_width - extra_width)
            )
        )
        _measure_column = self._measure_column

        measurements = [
            _measure_column(console, options.update_width(max_width), column)
            for column in self.columns
        ]
        minimum_width = (
            sum(measurement.minimum for measurement in measurements) + extra_width
        )
        maximum_width = (
            sum(measurement.maximum for measurement in measurements) + extra_width
            if (self.width is None)
            else self.width
        )
        measurement = Measurement(minimum_width, maximum_width)
        measurement = measurement.clamp(self.min_width)
        return measurement

    @property
    def padding(self) -> Tuple[int, int, int, int]:
        """Get cell padding."""
        return self._padding

    @padding.setter
    def padding(self, padding: PaddingDimensions) -> "Table":
        """Set cell padding."""
        self._padding = Padding.unpack(padding)
        return self

    def add_column(
        self,
        header: "RenderableType" = "",
        footer: "RenderableType" = "",
        *,
        header_style: Optional[StyleType] = None,
        highlight: Optional[bool] = None,
        footer_style: Optional[StyleType] = None,
        style: Optional[StyleType] = None,
        justify: "JustifyMethod" = "left",
        vertical: "VerticalAlignMethod" = "top",
        overflow: "OverflowMethod" = "ellipsis",
        width: Optional[int] = None,
        min_width: Optional[int] = None,
        max_width: Optional[int] = None,
        ratio: Optional[int] = None,
        no_wrap: bool = False,
    ) -> None:
        """Add a column to the table.

        Args:
            header (RenderableType, optional): Text or renderable for the header.
                Defaults to "".
            footer (RenderableType, optional): Text or renderable for the footer.
                Defaults to "".
            header_style (Union[str, Style], optional): Style for the header, or None for default. Defaults to None.
            highlight (bool, optional): Whether to highlight the text. The default of None uses the value of the table (self) object.
            footer_style (Union[str, Style], optional): Style for the footer, or None for default. Defaults to None.
            style (Union[str, Style], optional): Style for the column cells, or None for default. Defaults to None.
            justify (JustifyMethod, optional): Alignment for cells. Defaults to "left".
            vertical (VerticalAlignMethod, optional): Vertical alignment, one of "top", "middle", or "bottom". Defaults to "top".
            overflow (OverflowMethod): Overflow method: "crop", "fold", "ellipsis". Defaults to "ellipsis".
            width (int, optional): Desired width of column in characters, or None to fit to contents. Defaults to None.
            min_width (Optional[int], optional): Minimum width of column, or ``None`` for no minimum. Defaults to None.
            max_width (Optional[int], optional): Maximum width of column, or ``None`` for no maximum. Defaults to None.
            ratio (int, optional): Flexible ratio for the column (requires ``Table.expand`` or ``Table.width``). Defaults to None.
            no_wrap (bool, optional): Set to ``True`` to disable wrapping of this column.
        """

        column = Column(
            _index=len(self.columns),
            header=header,
            footer=footer,
            header_style=header_style or "",
            highlight=highlight if highlight is not None else self.highlight,
            footer_style=footer_style or "",
            style=style or "",
            justify=justify,
            vertical=vertical,
            overflow=overflow,
            width=width,
            min_width=min_width,
            max_width=max_width,
            ratio=ratio,
            no_wrap=no_wrap,
        )
        self.columns.append(column)

    def add_row(
        self,
        *renderables: Optional["RenderableType"],
        style: Optional[StyleType] = None,
        end_section: bool = False,
    ) -> None:
        """Add a row of renderables.

        Args:
            *renderables (None or renderable): Each cell in a row must be a renderable object (including str),
                or ``None`` for a blank cell.
            style (StyleType, optional): An optional style to apply to the entire row. Defaults to None.
            end_section (bool, optional): End a section and draw a line. Defaults to False.

        Raises:
            errors.NotRenderableError: If you add something that can't be rendered.
        """

        def add_cell(column: Column, renderable: "RenderableType") -> None:
            column._cells.append(renderable)

        cell_renderables: List[Optional["RenderableType"]] = list(renderables)

        columns = self.columns
        if len(cell_renderables) < len(columns):
            cell_renderables = [
                *cell_renderables,
                *[None] * (len(columns) - len(cell_renderables)),
            ]
        for index, renderable in enumerate(cell_renderables):
            if index == len(columns):
                column = Column(_index=index, highlight=self.highlight)
                for _ in self.rows:
                    add_cell(column, Text(""))
                self.columns.append(column)
            else:
                column = columns[index]
            if renderable is None:
                add_cell(column, "")
            elif is_renderable(renderable):
                add_cell(column, renderable)
            else:
                raise errors.NotRenderableError(
                    f"unable to render {type(renderable).__name__}; a string or other renderable object is required"
                )
        self.rows.append(Row(style=style, end_section=end_section))

    def add_section(self) -> None:
        """Add a new section (draw a line after current row)."""

        if self.rows:
            self.rows[-1].end_section = True

    def __rich_console__(
        self, console: "Console", options: "ConsoleOptions"
    ) -> "RenderResult":
        if not self.columns:
            yield Segment("\n")
            return

        max_width = options.max_width
        if self.width is not None:
            max_width = self.width

        extra_width = self._extra_width
        widths = self._calculate_column_widths(
            console, options.update_width(max_width - extra_width)
        )
        table_width = sum(widths) + extra_width

        render_options = options.update(
            width=table_width, highlight=self.highlight, height=None
        )

        def render_annotation(
            text: TextType, style: StyleType, justify: "JustifyMethod" = "center"
        ) -> "RenderResult":
            render_text = (
                console.render_str(text, style=style, highlight=False)
                if isinstance(text, str)
                else text
            )
            return console.render(
                render_text, options=render_options.update(justify=justify)
            )

        if self.title:
            yield from render_annotation(
                self.title,
                style=Style.pick_first(self.title_style, "table.title"),
                justify=self.title_justify,
            )
        yield from self._render(console, render_options, widths)
        if self.caption:
            yield from render_annotation(
                self.caption,
                style=Style.pick_first(self.caption_style, "table.caption"),
                justify=self.caption_justify,
            )

    def _calculate_column_widths(
        self, console: "Console", options: "ConsoleOptions"
    ) -> List[int]:
        """Calculate the widths of each column, including padding, not including borders."""
        max_width = options.max_width
        columns = self.columns
        width_ranges = [
            self._measure_column(console, options, column) for column in columns
        ]
        widths = [_range.maximum or 1 for _range in width_ranges]
        get_padding_width = self._get_padding_width
        extra_width = self._extra_width
        if self.expand:
            ratios = [col.ratio or 0 for col in columns if col.flexible]
            if any(ratios):
                fixed_widths = [
                    0 if column.flexible else _range.maximum
                    for _range, column in zip(width_ranges, columns)
                ]
                flex_minimum = [
                    (column.width or 1) + get_padding_width(column._index)
                    for column in columns
                    if column.flexible
                ]
                flexible_width = max_width - sum(fixed_widths)
                flex_widths = ratio_distribute(flexible_width, ratios, flex_minimum)
                iter_flex_widths = iter(flex_widths)
                for index, column in enumerate(columns):
                    if column.flexible:
                        widths[index] = fixed_widths[index] + next(iter_flex_widths)
        table_width = sum(widths)

        if table_width > max_width:
            widths = self._collapse_widths(
                widths,
                [(column.width is None and not column.no_wrap) for column in columns],
                max_width,
            )
            table_width = sum(widths)
            # last resort, reduce columns evenly
            if table_width > max_width:
                excess_width = table_width - max_width
                widths = ratio_reduce(excess_width, [1] * len(widths), widths, widths)
                table_width = sum(widths)

            width_ranges = [
                self._measure_column(console, options.update_width(width), column)
                for width, column in zip(widths, columns)
            ]
            widths = [_range.maximum or 0 for _range in width_ranges]

        if (table_width < max_width and self.expand) or (
            self.min_width is not None and table_width < (self.min_width - extra_width)
        ):
            _max_width = (
                max_width
                if self.min_width is None
                else min(self.min_width - extra_width, max_width)
            )
            pad_widths = ratio_distribute(_max_width - table_width, widths)
            widths = [_width + pad for _width, pad in zip(widths, pad_widths)]

        return widths

    @classmethod
    def _collapse_widths(
        cls, widths: List[int], wrapable: List[bool], max_width: int
    ) -> List[int]:
        """Reduce widths so that the total is under max_width.

        Args:
            widths (List[int]): List of widths.
            wrapable (List[bool]): List of booleans that indicate if a column may shrink.
            max_width (int): Maximum width to reduce to.

        Returns:
            List[int]: A new list of widths.
        """
        total_width = sum(widths)
        excess_width = total_width - max_width
        if any(wrapable):
            while total_width and excess_width > 0:
                max_column = max(
                    width for width, allow_wrap in zip(widths, wrapable) if allow_wrap
                )
                second_max_column = max(
                    width if allow_wrap and width != max_column else 0
                    for width, allow_wrap in zip(widths, wrapable)
                )
                column_difference = max_column - second_max_column
                ratios = [
                    (1 if (width == max_column and allow_wrap) else 0)
                    for width, allow_wrap in zip(widths, wrapable)
                ]
                if not any(ratios) or not column_difference:
                    break
                max_reduce = [min(excess_width, column_difference)] * len(widths)
                widths = ratio_reduce(excess_width, ratios, max_reduce, widths)

                total_width = sum(widths)
                excess_width = total_width - max_width
        return widths

    def _get_cells(
        self, console: "Console", column_index: int, column: Column
    ) -> Iterable[_Cell]:
        """Get all the cells with padding and optional header."""

        collapse_padding = self.collapse_padding
        pad_edge = self.pad_edge
        padding = self.padding
        any_padding = any(padding)

        first_column = column_index == 0
        last_column = column_index == len(self.columns) - 1

        _padding_cache: Dict[Tuple[bool, bool], Tuple[int, int, int, int]] = {}

        def get_padding(first_row: bool, last_row: bool) -> Tuple[int, int, int, int]:
            cached = _padding_cache.get((first_row, last_row))
            if cached:
                return cached
            top, right, bottom, left = padding

            if collapse_padding:
                if not first_column:
                    left = max(0, left - right)
                if not last_row:
                    bottom = max(0, top - bottom)

            if not pad_edge:
                if first_column:
                    left = 0
                if last_column:
                    right = 0
                if first_row:
                    top = 0
                if last_row:
                    bottom = 0
            _padding = (top, right, bottom, left)
            _padding_cache[(first_row, last_row)] = _padding
            return _padding

        raw_cells: List[Tuple[StyleType, "RenderableType"]] = []
        _append = raw_cells.append
        get_style = console.get_style
        if self.show_header:
            header_style = get_style(self.header_style or "") + get_style(
                column.header_style
            )
            _append((header_style, column.header))
        cell_style = get_style(column.style or "")
        for cell in column.cells:
            _append((cell_style, cell))
        if self.show_footer:
            footer_style = get_style(self.footer_style or "") + get_style(
                column.footer_style
            )
            _append((footer_style, column.footer))

        if any_padding:
            _Padding = Padding
            for first, last, (style, renderable) in loop_first_last(raw_cells):
                yield _Cell(
                    style,
                    _Padding(renderable, get_padding(first, last)),
                    getattr(renderable, "vertical", None) or column.vertical,
                )
        else:
            for style, renderable in raw_cells:
                yield _Cell(
                    style,
                    renderable,
                    getattr(renderable, "vertical", None) or column.vertical,
                )

    def _get_padding_width(self, column_index: int) -> int:
        """Get extra width from padding."""
        _, pad_right, _, pad_left = self.padding
        if self.collapse_padding:
            if column_index > 0:
                pad_left = max(0, pad_left - pad_right)
        return pad_left + pad_right

    def _measure_column(
        self,
        console: "Console",
        options: "ConsoleOptions",
        column: Column,
    ) -> Measurement:
        """Get the minimum and maximum width of the column."""

        max_width = options.max_width
        if max_width < 1:
            return Measurement(0, 0)

        padding_width = self._get_padding_width(column._index)

        if column.width is not None:
            # Fixed width column
            return Measurement(
                column.width + padding_width, column.width + padding_width
            ).with_maximum(max_width)
        # Flexible column, we need to measure contents
        min_widths: List[int] = []
        max_widths: List[int] = []
        append_min = min_widths.append
        append_max = max_widths.append
        get_render_width = Measurement.get
        for cell in self._get_cells(console, column._index, column):
            _min, _max = get_render_width(console, options, cell.renderable)
            append_min(_min)
            append_max(_max)

        measurement = Measurement(
            max(min_widths) if min_widths else 1,
            max(max_widths) if max_widths else max_width,
        ).with_maximum(max_width)
        measurement = measurement.clamp(
            None if column.min_width is None else column.min_width + padding_width,
            None if column.max_width is None else column.max_width + padding_width,
        )
        return measurement

    def _render(
        self, console: "Console", options: "ConsoleOptions", widths: List[int]
    ) -> "RenderResult":
        table_style = console.get_style(self.style or "")

        border_style = table_style + console.get_style(self.border_style or "")
        _column_cells = (
            self._get_cells(console, column_index, column)
            for column_index, column in enumerate(self.columns)
        )
        row_cells: List[Tuple[_Cell, ...]] = list(zip(*_column_cells))
        _box = (
            self.box.substitute(
                options, safe=pick_bool(self.safe_box, console.safe_box)
            )
            if self.box
            else None
        )
        _box = _box.get_plain_headed_box() if _box and not self.show_header else _box

        new_line = Segment.line()

        columns = self.columns
        show_header = self.show_header
        show_footer = self.show_footer
        show_edge = self.show_edge
        show_lines = self.show_lines
        leading = self.leading

        _Segment = Segment
        if _box:
            box_segments = [
                (
                    _Segment(_box.head_left, border_style),
                    _Segment(_box.head_right, border_style),
                    _Segment(_box.head_vertical, border_style),
                ),
                (
                    _Segment(_box.mid_left, border_style),
                    _Segment(_box.mid_right, border_style),
                    _Segment(_box.mid_vertical, border_style),
                ),
                (
                    _Segment(_box.foot_left, border_style),
                    _Segment(_box.foot_right, border_style),
                    _Segment(_box.foot_vertical, border_style),
                ),
            ]
            if show_edge:
                yield _Segment(_box.get_top(widths), border_style)
                yield new_line
        else:
            box_segments = []

        get_row_style = self.get_row_style
        get_style = console.get_style

        for index, (first, last, row_cell) in enumerate(loop_first_last(row_cells)):
            header_row = first and show_header
            footer_row = last and show_footer
            row = (
                self.rows[index - show_header]
                if (not header_row and not footer_row)
                else None
            )
            max_height = 1
            cells: List[List[List[Segment]]] = []
            if header_row or footer_row:
                row_style = Style.null()
            else:
                row_style = get_style(
                    get_row_style(console, index - 1 if show_header else index)
                )
            for width, cell, column in zip(widths, row_cell, columns):
                render_options = options.update(
                    width=width,
                    justify=column.justify,
                    no_wrap=column.no_wrap,
                    overflow=column.overflow,
                    height=None,
                    highlight=column.highlight,
                )
                lines = console.render_lines(
                    cell.renderable,
                    render_options,
                    style=get_style(cell.style) + row_style,
                )
                max_height = max(max_height, len(lines))
                cells.append(lines)

            row_height = max(len(cell) for cell in cells)

            def align_cell(
                cell: List[List[Segment]],
                vertical: "VerticalAlignMethod",
                width: int,
                style: Style,
            ) -> List[List[Segment]]:
                if header_row:
                    vertical = "bottom"
                elif footer_row:
                    vertical = "top"

                if vertical == "top":
                    return _Segment.align_top(cell, width, row_height, style)
                elif vertical == "middle":
                    return _Segment.align_middle(cell, width, row_height, style)
                return _Segment.align_bottom(cell, width, row_height, style)

            cells[:] = [
                _Segment.set_shape(
                    align_cell(
                        cell,
                        _cell.vertical,
                        width,
                        get_style(_cell.style) + row_style,
                    ),
                    width,
                    max_height,
                )
                for width, _cell, cell, column in zip(widths, row_cell, cells, columns)
            ]

            if _box:
                if last and show_footer:
                    yield _Segment(
                        _box.get_row(widths, "foot", edge=show_edge), border_style
                    )
                    yield new_line
                left, right, _divider = box_segments[0 if first else (2 if last else 1)]

                # If the column divider is whitespace also style it with the row background
                divider = (
                    _divider
                    if _divider.text.strip()
                    else _Segment(
                        _divider.text, row_style.background_style + _divider.style
                    )
                )
                for line_no in range(max_height):
                    if show_edge:
                        yield left
                    for last_cell, rendered_cell in loop_last(cells):
                        yield from rendered_cell[line_no]
                        if not last_cell:
                            yield divider
                    if show_edge:
                        yield right
                    yield new_line
            else:
                for line_no in range(max_height):
                    for rendered_cell in cells:
                        yield from rendered_cell[line_no]
                    yield new_line
            if _box and first and show_header:
                yield _Segment(
                    _box.get_row(widths, "head", edge=show_edge), border_style
                )
                yield new_line
            end_section = row and row.end_section
            if _box and (show_lines or leading or end_section):
                if (
                    not last
                    and not (show_footer and index >= len(row_cells) - 2)
                    and not (show_header and header_row)
                ):
                    if leading:
                        yield _Segment(
                            _box.get_row(widths, "mid", edge=show_edge) * leading,
                            border_style,
                        )
                    else:
                        yield _Segment(
                            _box.get_row(widths, "row", edge=show_edge), border_style
                        )
                    yield new_line

        if _box and show_edge:
            yield _Segment(_box.get_bottom(widths), border_style)
            yield new_line


if __name__ == "__main__":  # pragma: no cover
    from pip._vendor.rich.console import Console
    from pip._vendor.rich.highlighter import ReprHighlighter

    from ._timer import timer

    with timer("Table render"):
        table = Table(
            title="Star Wars Movies",
            caption="Rich example table",
            caption_justify="right",
        )

        table.add_column(
            "Released", header_style="bright_cyan", style="cyan", no_wrap=True
        )
        table.add_column("Title", style="magenta")
        table.add_column("Box Office", justify="right", style="green")

        table.add_row(
            "Dec 20, 2019",
            "Star Wars: The Rise of Skywalker",
            "$952,110,690",
        )
        table.add_row("May 25, 2018", "Solo: A Star Wars Story", "$393,151,347")
        table.add_row(
            "Dec 15, 2017",
            "Star Wars Ep. V111: The Last Jedi",
            "$1,332,539,889",
            style="on black",
            end_section=True,
        )
        table.add_row(
            "Dec 16, 2016",
            "Rogue One: A Star Wars Story",
            "$1,332,439,889",
        )

        def header(text: str) -> None:
            console.print()
            console.rule(highlight(text))
            console.print()

        console = Console()
        highlight = ReprHighlighter()
        header("Example Table")
        console.print(table, justify="center")

        table.expand = True
        header("expand=True")
        console.print(table)

        table.width = 50
        header("width=50")

        console.print(table, justify="center")

        table.width = None
        table.expand = False
        table.row_styles = ["dim", "none"]
        header("row_styles=['dim', 'none']")

        console.print(table, justify="center")

        table.width = None
        table.expand = False
        table.row_styles = ["dim", "none"]
        table.leading = 1
        header("leading=1, row_styles=['dim', 'none']")
        console.print(table, justify="center")

        table.width = None
        table.expand = False
        table.row_styles = ["dim", "none"]
        table.show_lines = True
        table.leading = 0
        header("show_lines=True, row_styles=['dim', 'none']")
        console.print(table, justify="center")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\matrices\matrices.py ===
from ..libmp.backend import xrange
import warnings

# TODO: interpret list as vectors (for multiplication)

rowsep = '\n'
colsep = '  '

class _matrix(object):
    """
    Numerical matrix.

    Specify the dimensions or the data as a nested list.
    Elements default to zero.
    Use a flat list to create a column vector easily.

    The datatype of the context (mpf for mp, mpi for iv, and float for fp) is used to store the data.

    Creating matrices
    -----------------

    Matrices in mpmath are implemented using dictionaries. Only non-zero values
    are stored, so it is cheap to represent sparse matrices.

    The most basic way to create one is to use the ``matrix`` class directly.
    You can create an empty matrix specifying the dimensions:

        >>> from mpmath import *
        >>> mp.dps = 15
        >>> matrix(2)
        matrix(
        [['0.0', '0.0'],
         ['0.0', '0.0']])
        >>> matrix(2, 3)
        matrix(
        [['0.0', '0.0', '0.0'],
         ['0.0', '0.0', '0.0']])

    Calling ``matrix`` with one dimension will create a square matrix.

    To access the dimensions of a matrix, use the ``rows`` or ``cols`` keyword:

        >>> A = matrix(3, 2)
        >>> A
        matrix(
        [['0.0', '0.0'],
         ['0.0', '0.0'],
         ['0.0', '0.0']])
        >>> A.rows
        3
        >>> A.cols
        2

    You can also change the dimension of an existing matrix. This will set the
    new elements to 0. If the new dimension is smaller than before, the
    concerning elements are discarded:

        >>> A.rows = 2
        >>> A
        matrix(
        [['0.0', '0.0'],
         ['0.0', '0.0']])

    Internally ``mpmathify`` is used every time an element is set. This
    is done using the syntax A[row,column], counting from 0:

        >>> A = matrix(2)
        >>> A[1,1] = 1 + 1j
        >>> A
        matrix(
        [['0.0', '0.0'],
         ['0.0', mpc(real='1.0', imag='1.0')]])

    A more comfortable way to create a matrix lets you use nested lists:

        >>> matrix([[1, 2], [3, 4]])
        matrix(
        [['1.0', '2.0'],
         ['3.0', '4.0']])

    Convenient advanced functions are available for creating various standard
    matrices, see ``zeros``, ``ones``, ``diag``, ``eye``, ``randmatrix`` and
    ``hilbert``.

    Vectors
    .......

    Vectors may also be represented by the ``matrix`` class (with rows = 1 or cols = 1).
    For vectors there are some things which make life easier. A column vector can
    be created using a flat list, a row vectors using an almost flat nested list::

        >>> matrix([1, 2, 3])
        matrix(
        [['1.0'],
         ['2.0'],
         ['3.0']])
        >>> matrix([[1, 2, 3]])
        matrix(
        [['1.0', '2.0', '3.0']])

    Optionally vectors can be accessed like lists, using only a single index::

        >>> x = matrix([1, 2, 3])
        >>> x[1]
        mpf('2.0')
        >>> x[1,0]
        mpf('2.0')

    Other
    .....

    Like you probably expected, matrices can be printed::

        >>> print randmatrix(3) # doctest:+SKIP
        [ 0.782963853573023  0.802057689719883  0.427895717335467]
        [0.0541876859348597  0.708243266653103  0.615134039977379]
        [ 0.856151514955773  0.544759264818486  0.686210904770947]

    Use ``nstr`` or ``nprint`` to specify the number of digits to print::

        >>> nprint(randmatrix(5), 3) # doctest:+SKIP
        [2.07e-1  1.66e-1  5.06e-1  1.89e-1  8.29e-1]
        [6.62e-1  6.55e-1  4.47e-1  4.82e-1  2.06e-2]
        [4.33e-1  7.75e-1  6.93e-2  2.86e-1  5.71e-1]
        [1.01e-1  2.53e-1  6.13e-1  3.32e-1  2.59e-1]
        [1.56e-1  7.27e-2  6.05e-1  6.67e-2  2.79e-1]

    As matrices are mutable, you will need to copy them sometimes::

        >>> A = matrix(2)
        >>> A
        matrix(
        [['0.0', '0.0'],
         ['0.0', '0.0']])
        >>> B = A.copy()
        >>> B[0,0] = 1
        >>> B
        matrix(
        [['1.0', '0.0'],
         ['0.0', '0.0']])
        >>> A
        matrix(
        [['0.0', '0.0'],
         ['0.0', '0.0']])

    Finally, it is possible to convert a matrix to a nested list. This is very useful,
    as most Python libraries involving matrices or arrays (namely NumPy or SymPy)
    support this format::

        >>> B.tolist()
        [[mpf('1.0'), mpf('0.0')], [mpf('0.0'), mpf('0.0')]]


    Matrix operations
    -----------------

    You can add and subtract matrices of compatible dimensions::

        >>> A = matrix([[1, 2], [3, 4]])
        >>> B = matrix([[-2, 4], [5, 9]])
        >>> A + B
        matrix(
        [['-1.0', '6.0'],
         ['8.0', '13.0']])
        >>> A - B
        matrix(
        [['3.0', '-2.0'],
         ['-2.0', '-5.0']])
        >>> A + ones(3) # doctest:+ELLIPSIS
        Traceback (most recent call last):
          ...
        ValueError: incompatible dimensions for addition

    It is possible to multiply or add matrices and scalars. In the latter case the
    operation will be done element-wise::

        >>> A * 2
        matrix(
        [['2.0', '4.0'],
         ['6.0', '8.0']])
        >>> A / 4
        matrix(
        [['0.25', '0.5'],
         ['0.75', '1.0']])
        >>> A - 1
        matrix(
        [['0.0', '1.0'],
         ['2.0', '3.0']])

    Of course you can perform matrix multiplication, if the dimensions are
    compatible, using ``@`` (for Python >= 3.5) or ``*``. For clarity, ``@`` is
    recommended (`PEP 465 <https://www.python.org/dev/peps/pep-0465/>`), because
    the meaning of ``*`` is different in many other Python libraries such as NumPy.

        >>> A @ B # doctest:+SKIP
        matrix(
        [['8.0', '22.0'],
         ['14.0', '48.0']])
        >>> A * B # same as A @ B
        matrix(
        [['8.0', '22.0'],
         ['14.0', '48.0']])
        >>> matrix([[1, 2, 3]]) * matrix([[-6], [7], [-2]])
        matrix(
        [['2.0']])

    ..
        COMMENT: TODO: the above "doctest:+SKIP" may be removed as soon as we
        have dropped support for Python 3.5 and below.

    You can raise powers of square matrices::

        >>> A**2
        matrix(
        [['7.0', '10.0'],
         ['15.0', '22.0']])

    Negative powers will calculate the inverse::

        >>> A**-1
        matrix(
        [['-2.0', '1.0'],
         ['1.5', '-0.5']])
        >>> A * A**-1
        matrix(
        [['1.0', '1.0842021724855e-19'],
         ['-2.16840434497101e-19', '1.0']])



    Matrix transposition is straightforward::

        >>> A = ones(2, 3)
        >>> A
        matrix(
        [['1.0', '1.0', '1.0'],
         ['1.0', '1.0', '1.0']])
        >>> A.T
        matrix(
        [['1.0', '1.0'],
         ['1.0', '1.0'],
         ['1.0', '1.0']])

    Norms
    .....

    Sometimes you need to know how "large" a matrix or vector is. Due to their
    multidimensional nature it's not possible to compare them, but there are
    several functions to map a matrix or a vector to a positive real number, the
    so called norms.

    For vectors the p-norm is intended, usually the 1-, the 2- and the oo-norm are
    used.

        >>> x = matrix([-10, 2, 100])
        >>> norm(x, 1)
        mpf('112.0')
        >>> norm(x, 2)
        mpf('100.5186549850325')
        >>> norm(x, inf)
        mpf('100.0')

    Please note that the 2-norm is the most used one, though it is more expensive
    to calculate than the 1- or oo-norm.

    It is possible to generalize some vector norms to matrix norm::

        >>> A = matrix([[1, -1000], [100, 50]])
        >>> mnorm(A, 1)
        mpf('1050.0')
        >>> mnorm(A, inf)
        mpf('1001.0')
        >>> mnorm(A, 'F')
        mpf('1006.2310867787777')

    The last norm (the "Frobenius-norm") is an approximation for the 2-norm, which
    is hard to calculate and not available. The Frobenius-norm lacks some
    mathematical properties you might expect from a norm.
    """

    def __init__(self, *args, **kwargs):
        self.__data = {}
        # LU decompostion cache, this is useful when solving the same system
        # multiple times, when calculating the inverse and when calculating the
        # determinant
        self._LU = None
        if "force_type" in kwargs:
            warnings.warn("The force_type argument was removed, it did not work"
                " properly anyway. If you want to force floating-point or"
                " interval computations, use the respective methods from `fp`"
                " or `mp` instead, e.g., `fp.matrix()` or `iv.matrix()`."
                " If you want to truncate values to integer, use .apply(int) instead.")
        if isinstance(args[0], (list, tuple)):
            if isinstance(args[0][0], (list, tuple)):
                # interpret nested list as matrix
                A = args[0]
                self.__rows = len(A)
                self.__cols = len(A[0])
                for i, row in enumerate(A):
                    for j, a in enumerate(row):
                        # note: this will call __setitem__ which will call self.ctx.convert() to convert the datatype.
                        self[i, j] = a
            else:
                # interpret list as row vector
                v = args[0]
                self.__rows = len(v)
                self.__cols = 1
                for i, e in enumerate(v):
                    self[i, 0] = e
        elif isinstance(args[0], int):
            # create empty matrix of given dimensions
            if len(args) == 1:
                self.__rows = self.__cols = args[0]
            else:
                if not isinstance(args[1], int):
                    raise TypeError("expected int")
                self.__rows = args[0]
                self.__cols = args[1]
        elif isinstance(args[0], _matrix):
            A = args[0]
            self.__rows = A._matrix__rows
            self.__cols = A._matrix__cols
            for i in xrange(A.__rows):
                for j in xrange(A.__cols):
                    self[i, j] = A[i, j]
        elif hasattr(args[0], 'tolist'):
            A = self.ctx.matrix(args[0].tolist())
            self.__data = A._matrix__data
            self.__rows = A._matrix__rows
            self.__cols = A._matrix__cols
        else:
            raise TypeError('could not interpret given arguments')

    def apply(self, f):
        """
        Return a copy of self with the function `f` applied elementwise.
        """
        new = self.ctx.matrix(self.__rows, self.__cols)
        for i in xrange(self.__rows):
            for j in xrange(self.__cols):
                new[i,j] = f(self[i,j])
        return new

    def __nstr__(self, n=None, **kwargs):
        # Build table of string representations of the elements
        res = []
        # Track per-column max lengths for pretty alignment
        maxlen = [0] * self.cols
        for i in range(self.rows):
            res.append([])
            for j in range(self.cols):
                if n:
                    string = self.ctx.nstr(self[i,j], n, **kwargs)
                else:
                    string = str(self[i,j])
                res[-1].append(string)
                maxlen[j] = max(len(string), maxlen[j])
        # Patch strings together
        for i, row in enumerate(res):
            for j, elem in enumerate(row):
                # Pad each element up to maxlen so the columns line up
                row[j] = elem.rjust(maxlen[j])
            res[i] = "[" + colsep.join(row) + "]"
        return rowsep.join(res)

    def __str__(self):
        return self.__nstr__()

    def _toliststr(self, avoid_type=False):
        """
        Create a list string from a matrix.

        If avoid_type: avoid multiple 'mpf's.
        """
        # XXX: should be something like self.ctx._types
        typ = self.ctx.mpf
        s = '['
        for i in xrange(self.__rows):
            s += '['
            for j in xrange(self.__cols):
                if not avoid_type or not isinstance(self[i,j], typ):
                    a = repr(self[i,j])
                else:
                    a = "'" + str(self[i,j]) + "'"
                s += a + ', '
            s = s[:-2]
            s += '],\n '
        s = s[:-3]
        s += ']'
        return s

    def tolist(self):
        """
        Convert the matrix to a nested list.
        """
        return [[self[i,j] for j in range(self.__cols)] for i in range(self.__rows)]

    def __repr__(self):
        if self.ctx.pretty:
            return self.__str__()
        s = 'matrix(\n'
        s += self._toliststr(avoid_type=True) + ')'
        return s

    def __get_element(self, key):
        '''
        Fast extraction of the i,j element from the matrix
            This function is for private use only because is unsafe:
                1. Does not check on the value of key it expects key to be a integer tuple (i,j)
                2. Does not check bounds
        '''
        if key in self.__data:
            return self.__data[key]
        else:
            return self.ctx.zero

    def __set_element(self, key, value):
        '''
        Fast assignment of the i,j element in the matrix
            This function is unsafe:
                1. Does not check on the value of key it expects key to be a integer tuple (i,j)
                2. Does not check bounds
                3. Does not check the value type
                4. Does not reset the LU cache
        '''
        if value: # only store non-zeros
            self.__data[key] = value
        elif key in self.__data:
            del self.__data[key]


    def __getitem__(self, key):
        '''
            Getitem function for mp matrix class with slice index enabled
            it allows the following assingments
            scalar to a slice of the matrix
         B = A[:,2:6]
        '''
        # Convert vector to matrix indexing
        if isinstance(key, int) or isinstance(key,slice):
            # only sufficent for vectors
            if self.__rows == 1:
                key = (0, key)
            elif self.__cols == 1:
                key = (key, 0)
            else:
                raise IndexError('insufficient indices for matrix')

        if isinstance(key[0],slice) or isinstance(key[1],slice):

            #Rows
            if isinstance(key[0],slice):
                #Check bounds
                if (key[0].start is None or key[0].start >= 0) and \
                    (key[0].stop is None or key[0].stop <= self.__rows+1):
                    # Generate indices
                    rows = xrange(*key[0].indices(self.__rows))
                else:
                    raise IndexError('Row index out of bounds')
            else:
                # Single row
                rows = [key[0]]

            # Columns
            if isinstance(key[1],slice):
                # Check bounds
                if (key[1].start is None or key[1].start >= 0) and \
                    (key[1].stop is None or key[1].stop <= self.__cols+1):
                    # Generate indices
                    columns = xrange(*key[1].indices(self.__cols))
                else:
                    raise IndexError('Column index out of bounds')

            else:
                # Single column
                columns = [key[1]]

            # Create matrix slice
            m = self.ctx.matrix(len(rows),len(columns))

            # Assign elements to the output matrix
            for i,x in enumerate(rows):
                for j,y in enumerate(columns):
                    m.__set_element((i,j),self.__get_element((x,y)))

            return m

        else:
            # single element extraction
            if key[0] >= self.__rows or key[1] >= self.__cols:
                raise IndexError('matrix index out of range')
            if key in self.__data:
                return self.__data[key]
            else:
                return self.ctx.zero

    def __setitem__(self, key, value):
        # setitem function for mp matrix class with slice index enabled
        # it allows the following assingments
        #  scalar to a slice of the matrix
        # A[:,2:6] = 2.5
        #  submatrix to matrix (the value matrix should be the same size as the slice size)
        # A[3,:] = B   where A is n x m  and B is n x 1
        # Convert vector to matrix indexing
        if isinstance(key, int) or isinstance(key,slice):
            # only sufficent for vectors
            if self.__rows == 1:
                key = (0, key)
            elif self.__cols == 1:
                key = (key, 0)
            else:
                raise IndexError('insufficient indices for matrix')
        # Slice indexing
        if isinstance(key[0],slice) or isinstance(key[1],slice):
            # Rows
            if isinstance(key[0],slice):
                # Check bounds
                if (key[0].start is None or key[0].start >= 0) and \
                    (key[0].stop is None or key[0].stop <= self.__rows+1):
                    # generate row indices
                    rows = xrange(*key[0].indices(self.__rows))
                else:
                    raise IndexError('Row index out of bounds')
            else:
                # Single row
                rows = [key[0]]
            # Columns
            if isinstance(key[1],slice):
                # Check bounds
                if (key[1].start is None or key[1].start >= 0) and \
                    (key[1].stop is None or key[1].stop <= self.__cols+1):
                    # Generate column indices
                    columns = xrange(*key[1].indices(self.__cols))
                else:
                    raise IndexError('Column index out of bounds')
            else:
                # Single column
                columns = [key[1]]
            # Assign slice with a scalar
            if isinstance(value,self.ctx.matrix):
                # Assign elements to matrix if input and output dimensions match
                if len(rows) == value.rows and len(columns) == value.cols:
                    for i,x in enumerate(rows):
                        for j,y in enumerate(columns):
                            self.__set_element((x,y), value.__get_element((i,j)))
                else:
                    raise ValueError('Dimensions do not match')
            else:
                # Assign slice with scalars
                value = self.ctx.convert(value)
                for i in rows:
                    for j in columns:
                        self.__set_element((i,j), value)
        else:
            # Single element assingment
            # Check bounds
            if key[0] >= self.__rows or key[1] >= self.__cols:
                raise IndexError('matrix index out of range')
            # Convert and store value
            value = self.ctx.convert(value)
            if value: # only store non-zeros
                self.__data[key] = value
            elif key in self.__data:
                del self.__data[key]

        if self._LU:
            self._LU = None
        return

    def __iter__(self):
        for i in xrange(self.__rows):
            for j in xrange(self.__cols):
                yield self[i,j]

    def __mul__(self, other):
        if isinstance(other, self.ctx.matrix):
            # dot multiplication
            if self.__cols != other.__rows:
                raise ValueError('dimensions not compatible for multiplication')
            new = self.ctx.matrix(self.__rows, other.__cols)
            self_zero = self.ctx.zero
            self_get = self.__data.get
            other_zero = other.ctx.zero
            other_get = other.__data.get
            for i in xrange(self.__rows):
                for j in xrange(other.__cols):
                    new[i, j] = self.ctx.fdot((self_get((i,k), self_zero), other_get((k,j), other_zero))
                                     for k in xrange(other.__rows))
            return new
        else:
            # try scalar multiplication
            new = self.ctx.matrix(self.__rows, self.__cols)
            for i in xrange(self.__rows):
                for j in xrange(self.__cols):
                    new[i, j] = other * self[i, j]
            return new

    def __matmul__(self, other):
        return self.__mul__(other)

    def __rmul__(self, other):
        # assume other is scalar and thus commutative
        if isinstance(other, self.ctx.matrix):
            raise TypeError("other should not be type of ctx.matrix")
        return self.__mul__(other)

    def __pow__(self, other):
        # avoid cyclic import problems
        #from linalg import inverse
        if not isinstance(other, int):
            raise ValueError('only integer exponents are supported')
        if not self.__rows == self.__cols:
            raise ValueError('only powers of square matrices are defined')
        n = other
        if n == 0:
            return self.ctx.eye(self.__rows)
        if n < 0:
            n = -n
            neg = True
        else:
            neg = False
        i = n
        y = 1
        z = self.copy()
        while i != 0:
            if i % 2 == 1:
                y = y * z
            z = z*z
            i = i // 2
        if neg:
            y = self.ctx.inverse(y)
        return y

    def __div__(self, other):
        # assume other is scalar and do element-wise divison
        assert not isinstance(other, self.ctx.matrix)
        new = self.ctx.matrix(self.__rows, self.__cols)
        for i in xrange(self.__rows):
            for j in xrange(self.__cols):
                new[i,j] = self[i,j] / other
        return new

    __truediv__ = __div__

    def __add__(self, other):
        if isinstance(other, self.ctx.matrix):
            if not (self.__rows == other.__rows and self.__cols == other.__cols):
                raise ValueError('incompatible dimensions for addition')
            new = self.ctx.matrix(self.__rows, self.__cols)
            for i in xrange(self.__rows):
                for j in xrange(self.__cols):
                    new[i,j] = self[i,j] + other[i,j]
            return new
        else:
            # assume other is scalar and add element-wise
            new = self.ctx.matrix(self.__rows, self.__cols)
            for i in xrange(self.__rows):
                for j in xrange(self.__cols):
                    new[i,j] += self[i,j] + other
            return new

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, self.ctx.matrix) and not (self.__rows == other.__rows
                                              and self.__cols == other.__cols):
            raise ValueError('incompatible dimensions for subtraction')
        return self.__add__(other * (-1))

    def __pos__(self):
        """
        +M returns a copy of M, rounded to current working precision.
        """
        return (+1) * self

    def __neg__(self):
        return (-1) * self

    def __rsub__(self, other):
        return -self + other

    def __eq__(self, other):
        return self.__rows == other.__rows and self.__cols == other.__cols \
               and self.__data == other.__data

    def __len__(self):
        if self.rows == 1:
            return self.cols
        elif self.cols == 1:
            return self.rows
        else:
            return self.rows # do it like numpy

    def __getrows(self):
        return self.__rows

    def __setrows(self, value):
        for key in self.__data.copy():
            if key[0] >= value:
                del self.__data[key]
        self.__rows = value

    rows = property(__getrows, __setrows, doc='number of rows')

    def __getcols(self):
        return self.__cols

    def __setcols(self, value):
        for key in self.__data.copy():
            if key[1] >= value:
                del self.__data[key]
        self.__cols = value

    cols = property(__getcols, __setcols, doc='number of columns')

    def transpose(self):
        new = self.ctx.matrix(self.__cols, self.__rows)
        for i in xrange(self.__rows):
            for j in xrange(self.__cols):
                new[j,i] = self[i,j]
        return new

    T = property(transpose)

    def conjugate(self):
        return self.apply(self.ctx.conj)

    def transpose_conj(self):
        return self.conjugate().transpose()

    H = property(transpose_conj)

    def copy(self):
        new = self.ctx.matrix(self.__rows, self.__cols)
        new.__data = self.__data.copy()
        return new

    __copy__ = copy

    def column(self, n):
        m = self.ctx.matrix(self.rows, 1)
        for i in range(self.rows):
            m[i] = self[i,n]
        return m

class MatrixMethods(object):

    def __init__(ctx):
        # XXX: subclass
        ctx.matrix = type('matrix', (_matrix,), {})
        ctx.matrix.ctx = ctx
        ctx.matrix.convert = ctx.convert

    def eye(ctx, n, **kwargs):
        """
        Create square identity matrix n x n.
        """
        A = ctx.matrix(n, **kwargs)
        for i in xrange(n):
            A[i,i] = 1
        return A

    def diag(ctx, diagonal, **kwargs):
        """
        Create square diagonal matrix using given list.

        Example:
        >>> from mpmath import diag, mp
        >>> mp.pretty = False
        >>> diag([1, 2, 3])
        matrix(
        [['1.0', '0.0', '0.0'],
         ['0.0', '2.0', '0.0'],
         ['0.0', '0.0', '3.0']])
        """
        A = ctx.matrix(len(diagonal), **kwargs)
        for i in xrange(len(diagonal)):
            A[i,i] = diagonal[i]
        return A

    def zeros(ctx, *args, **kwargs):
        """
        Create matrix m x n filled with zeros.
        One given dimension will create square matrix n x n.

        Example:
        >>> from mpmath import zeros, mp
        >>> mp.pretty = False
        >>> zeros(2)
        matrix(
        [['0.0', '0.0'],
         ['0.0', '0.0']])
        """
        if len(args) == 1:
            m = n = args[0]
        elif len(args) == 2:
            m = args[0]
            n = args[1]
        else:
            raise TypeError('zeros expected at most 2 arguments, got %i' % len(args))
        A = ctx.matrix(m, n, **kwargs)
        for i in xrange(m):
            for j in xrange(n):
                A[i,j] = 0
        return A

    def ones(ctx, *args, **kwargs):
        """
        Create matrix m x n filled with ones.
        One given dimension will create square matrix n x n.

        Example:
        >>> from mpmath import ones, mp
        >>> mp.pretty = False
        >>> ones(2)
        matrix(
        [['1.0', '1.0'],
         ['1.0', '1.0']])
        """
        if len(args) == 1:
            m = n = args[0]
        elif len(args) == 2:
            m = args[0]
            n = args[1]
        else:
            raise TypeError('ones expected at most 2 arguments, got %i' % len(args))
        A = ctx.matrix(m, n, **kwargs)
        for i in xrange(m):
            for j in xrange(n):
                A[i,j] = 1
        return A

    def hilbert(ctx, m, n=None):
        """
        Create (pseudo) hilbert matrix m x n.
        One given dimension will create hilbert matrix n x n.

        The matrix is very ill-conditioned and symmetric, positive definite if
        square.
        """
        if n is None:
            n = m
        A = ctx.matrix(m, n)
        for i in xrange(m):
            for j in xrange(n):
                A[i,j] = ctx.one / (i + j + 1)
        return A

    def randmatrix(ctx, m, n=None, min=0, max=1, **kwargs):
        """
        Create a random m x n matrix.

        All values are >= min and <max.
        n defaults to m.

        Example:
        >>> from mpmath import randmatrix
        >>> randmatrix(2) # doctest:+SKIP
        matrix(
        [['0.53491598236191806', '0.57195669543302752'],
         ['0.85589992269513615', '0.82444367501382143']])
        """
        if not n:
            n = m
        A = ctx.matrix(m, n, **kwargs)
        for i in xrange(m):
            for j in xrange(n):
                A[i,j] = ctx.rand() * (max - min) + min
        return A

    def swap_row(ctx, A, i, j):
        """
        Swap row i with row j.
        """
        if i == j:
            return
        if isinstance(A, ctx.matrix):
            for k in xrange(A.cols):
                A[i,k], A[j,k] = A[j,k], A[i,k]
        elif isinstance(A, list):
            A[i], A[j] = A[j], A[i]
        else:
            raise TypeError('could not interpret type')

    def extend(ctx, A, b):
        """
        Extend matrix A with column b and return result.
        """
        if not isinstance(A, ctx.matrix):
            raise TypeError("A should be a type of ctx.matrix")
        if A.rows != len(b):
            raise ValueError("Value should be equal to len(b)")
        A = A.copy()
        A.cols += 1
        for i in xrange(A.rows):
            A[i, A.cols-1] = b[i]
        return A

    def norm(ctx, x, p=2):
        r"""
        Gives the entrywise `p`-norm of an iterable *x*, i.e. the vector norm
        `\left(\sum_k |x_k|^p\right)^{1/p}`, for any given `1 \le p \le \infty`.

        Special cases:

        If *x* is not iterable, this just returns ``absmax(x)``.

        ``p=1`` gives the sum of absolute values.

        ``p=2`` is the standard Euclidean vector norm.

        ``p=inf`` gives the magnitude of the largest element.

        For *x* a matrix, ``p=2`` is the Frobenius norm.
        For operator matrix norms, use :func:`~mpmath.mnorm` instead.

        You can use the string 'inf' as well as float('inf') or mpf('inf')
        to specify the infinity norm.

        **Examples**

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = False
            >>> x = matrix([-10, 2, 100])
            >>> norm(x, 1)
            mpf('112.0')
            >>> norm(x, 2)
            mpf('100.5186549850325')
            >>> norm(x, inf)
            mpf('100.0')

        """
        try:
            iter(x)
        except TypeError:
            return ctx.absmax(x)
        if type(p) is not int:
            p = ctx.convert(p)
        if p == ctx.inf:
            return max(ctx.absmax(i) for i in x)
        elif p == 1:
            return ctx.fsum(x, absolute=1)
        elif p == 2:
            return ctx.sqrt(ctx.fsum(x, absolute=1, squared=1))
        elif p > 1:
            return ctx.nthroot(ctx.fsum(abs(i)**p for i in x), p)
        else:
            raise ValueError('p has to be >= 1')

    def mnorm(ctx, A, p=1):
        r"""
        Gives the matrix (operator) `p`-norm of A. Currently ``p=1`` and ``p=inf``
        are supported:

        ``p=1`` gives the 1-norm (maximal column sum)

        ``p=inf`` gives the `\infty`-norm (maximal row sum).
        You can use the string 'inf' as well as float('inf') or mpf('inf')

        ``p=2`` (not implemented) for a square matrix is the usual spectral
        matrix norm, i.e. the largest singular value.

        ``p='f'`` (or 'F', 'fro', 'Frobenius, 'frobenius') gives the
        Frobenius norm, which is the elementwise 2-norm. The Frobenius norm is an
        approximation of the spectral norm and satisfies

        .. math ::

            \frac{1}{\sqrt{\mathrm{rank}(A)}} \|A\|_F \le \|A\|_2 \le \|A\|_F

        The Frobenius norm lacks some mathematical properties that might
        be expected of a norm.

        For general elementwise `p`-norms, use :func:`~mpmath.norm` instead.

        **Examples**

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = False
            >>> A = matrix([[1, -1000], [100, 50]])
            >>> mnorm(A, 1)
            mpf('1050.0')
            >>> mnorm(A, inf)
            mpf('1001.0')
            >>> mnorm(A, 'F')
            mpf('1006.2310867787777')

        """
        A = ctx.matrix(A)
        if type(p) is not int:
            if type(p) is str and 'frobenius'.startswith(p.lower()):
                return ctx.norm(A, 2)
            p = ctx.convert(p)
        m, n = A.rows, A.cols
        if p == 1:
            return max(ctx.fsum((A[i,j] for i in xrange(m)), absolute=1) for j in xrange(n))
        elif p == ctx.inf:
            return max(ctx.fsum((A[i,j] for j in xrange(n)), absolute=1) for i in xrange(m))
        else:
            raise NotImplementedError("matrix p-norm for arbitrary p")

if __name__ == '__main__':
    import doctest
    doctest.testmod()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\f2py\auxfuncs.py ===
"""
Auxiliary functions for f2py2e.

Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy (BSD style) LICENSE.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""
import pprint
import re
import sys
import types
from functools import reduce

from . import __version__, cfuncs
from .cfuncs import errmess

__all__ = [
    'applyrules', 'debugcapi', 'dictappend', 'errmess', 'gentitle',
    'getargs2', 'getcallprotoargument', 'getcallstatement',
    'getfortranname', 'getpymethoddef', 'getrestdoc', 'getusercode',
    'getusercode1', 'getdimension', 'hasbody', 'hascallstatement', 'hascommon',
    'hasexternals', 'hasinitvalue', 'hasnote', 'hasresultnote',
    'isallocatable', 'isarray', 'isarrayofstrings',
    'ischaracter', 'ischaracterarray', 'ischaracter_or_characterarray',
    'iscomplex', 'iscstyledirective',
    'iscomplexarray', 'iscomplexfunction', 'iscomplexfunction_warn',
    'isdouble', 'isdummyroutine', 'isexternal', 'isfunction',
    'isfunction_wrap', 'isint1', 'isint1array', 'isinteger', 'isintent_aux',
    'isintent_c', 'isintent_callback', 'isintent_copy', 'isintent_dict',
    'isintent_hide', 'isintent_in', 'isintent_inout', 'isintent_inplace',
    'isintent_nothide', 'isintent_out', 'isintent_overwrite', 'islogical',
    'islogicalfunction', 'islong_complex', 'islong_double',
    'islong_doublefunction', 'islong_long', 'islong_longfunction',
    'ismodule', 'ismoduleroutine', 'isoptional', 'isprivate', 'isvariable',
    'isrequired', 'isroutine', 'isscalar', 'issigned_long_longarray',
    'isstring', 'isstringarray', 'isstring_or_stringarray', 'isstringfunction',
    'issubroutine', 'get_f2py_modulename', 'issubroutine_wrap', 'isthreadsafe',
    'isunsigned', 'isunsigned_char', 'isunsigned_chararray',
    'isunsigned_long_long', 'isunsigned_long_longarray', 'isunsigned_short',
    'isunsigned_shortarray', 'l_and', 'l_not', 'l_or', 'outmess', 'replace',
    'show', 'stripcomma', 'throw_error', 'isattr_value', 'getuseblocks',
    'process_f2cmap_dict', 'containscommon', 'containsderivedtypes'
]


f2py_version = __version__.version


show = pprint.pprint

options = {}
debugoptions = []
wrapfuncs = 1


def outmess(t):
    if options.get('verbose', 1):
        sys.stdout.write(t)


def debugcapi(var):
    return 'capi' in debugoptions


def _ischaracter(var):
    return 'typespec' in var and var['typespec'] == 'character' and \
           not isexternal(var)


def _isstring(var):
    return 'typespec' in var and var['typespec'] == 'character' and \
           not isexternal(var)


def ischaracter_or_characterarray(var):
    return _ischaracter(var) and 'charselector' not in var


def ischaracter(var):
    return ischaracter_or_characterarray(var) and not isarray(var)


def ischaracterarray(var):
    return ischaracter_or_characterarray(var) and isarray(var)


def isstring_or_stringarray(var):
    return _ischaracter(var) and 'charselector' in var


def isstring(var):
    return isstring_or_stringarray(var) and not isarray(var)


def isstringarray(var):
    return isstring_or_stringarray(var) and isarray(var)


def isarrayofstrings(var):  # obsolete?
    # leaving out '*' for now so that `character*(*) a(m)` and `character
    # a(m,*)` are treated differently. Luckily `character**` is illegal.
    return isstringarray(var) and var['dimension'][-1] == '(*)'


def isarray(var):
    return 'dimension' in var and not isexternal(var)


def isscalar(var):
    return not (isarray(var) or isstring(var) or isexternal(var))


def iscomplex(var):
    return isscalar(var) and \
           var.get('typespec') in ['complex', 'double complex']


def islogical(var):
    return isscalar(var) and var.get('typespec') == 'logical'


def isinteger(var):
    return isscalar(var) and var.get('typespec') == 'integer'


def isreal(var):
    return isscalar(var) and var.get('typespec') == 'real'


def get_kind(var):
    try:
        return var['kindselector']['*']
    except KeyError:
        try:
            return var['kindselector']['kind']
        except KeyError:
            pass


def isint1(var):
    return var.get('typespec') == 'integer' \
        and get_kind(var) == '1' and not isarray(var)


def islong_long(var):
    if not isscalar(var):
        return 0
    if var.get('typespec') not in ['integer', 'logical']:
        return 0
    return get_kind(var) == '8'


def isunsigned_char(var):
    if not isscalar(var):
        return 0
    if var.get('typespec') != 'integer':
        return 0
    return get_kind(var) == '-1'


def isunsigned_short(var):
    if not isscalar(var):
        return 0
    if var.get('typespec') != 'integer':
        return 0
    return get_kind(var) == '-2'


def isunsigned(var):
    if not isscalar(var):
        return 0
    if var.get('typespec') != 'integer':
        return 0
    return get_kind(var) == '-4'


def isunsigned_long_long(var):
    if not isscalar(var):
        return 0
    if var.get('typespec') != 'integer':
        return 0
    return get_kind(var) == '-8'


def isdouble(var):
    if not isscalar(var):
        return 0
    if not var.get('typespec') == 'real':
        return 0
    return get_kind(var) == '8'


def islong_double(var):
    if not isscalar(var):
        return 0
    if not var.get('typespec') == 'real':
        return 0
    return get_kind(var) == '16'


def islong_complex(var):
    if not iscomplex(var):
        return 0
    return get_kind(var) == '32'


def iscomplexarray(var):
    return isarray(var) and \
           var.get('typespec') in ['complex', 'double complex']


def isint1array(var):
    return isarray(var) and var.get('typespec') == 'integer' \
        and get_kind(var) == '1'


def isunsigned_chararray(var):
    return isarray(var) and var.get('typespec') in ['integer', 'logical']\
        and get_kind(var) == '-1'


def isunsigned_shortarray(var):
    return isarray(var) and var.get('typespec') in ['integer', 'logical']\
        and get_kind(var) == '-2'


def isunsignedarray(var):
    return isarray(var) and var.get('typespec') in ['integer', 'logical']\
        and get_kind(var) == '-4'


def isunsigned_long_longarray(var):
    return isarray(var) and var.get('typespec') in ['integer', 'logical']\
        and get_kind(var) == '-8'


def issigned_chararray(var):
    return isarray(var) and var.get('typespec') in ['integer', 'logical']\
        and get_kind(var) == '1'


def issigned_shortarray(var):
    return isarray(var) and var.get('typespec') in ['integer', 'logical']\
        and get_kind(var) == '2'


def issigned_array(var):
    return isarray(var) and var.get('typespec') in ['integer', 'logical']\
        and get_kind(var) == '4'


def issigned_long_longarray(var):
    return isarray(var) and var.get('typespec') in ['integer', 'logical']\
        and get_kind(var) == '8'


def isallocatable(var):
    return 'attrspec' in var and 'allocatable' in var['attrspec']


def ismutable(var):
    return not ('dimension' not in var or isstring(var))


def ismoduleroutine(rout):
    return 'modulename' in rout


def ismodule(rout):
    return 'block' in rout and 'module' == rout['block']


def isfunction(rout):
    return 'block' in rout and 'function' == rout['block']


def isfunction_wrap(rout):
    if isintent_c(rout):
        return 0
    return wrapfuncs and isfunction(rout) and (not isexternal(rout))


def issubroutine(rout):
    return 'block' in rout and 'subroutine' == rout['block']


def issubroutine_wrap(rout):
    if isintent_c(rout):
        return 0
    return issubroutine(rout) and hasassumedshape(rout)

def isattr_value(var):
    return 'value' in var.get('attrspec', [])


def hasassumedshape(rout):
    if rout.get('hasassumedshape'):
        return True
    for a in rout['args']:
        for d in rout['vars'].get(a, {}).get('dimension', []):
            if d == ':':
                rout['hasassumedshape'] = True
                return True
    return False


def requiresf90wrapper(rout):
    return ismoduleroutine(rout) or hasassumedshape(rout)


def isroutine(rout):
    return isfunction(rout) or issubroutine(rout)


def islogicalfunction(rout):
    if not isfunction(rout):
        return 0
    if 'result' in rout:
        a = rout['result']
    else:
        a = rout['name']
    if a in rout['vars']:
        return islogical(rout['vars'][a])
    return 0


def islong_longfunction(rout):
    if not isfunction(rout):
        return 0
    if 'result' in rout:
        a = rout['result']
    else:
        a = rout['name']
    if a in rout['vars']:
        return islong_long(rout['vars'][a])
    return 0


def islong_doublefunction(rout):
    if not isfunction(rout):
        return 0
    if 'result' in rout:
        a = rout['result']
    else:
        a = rout['name']
    if a in rout['vars']:
        return islong_double(rout['vars'][a])
    return 0


def iscomplexfunction(rout):
    if not isfunction(rout):
        return 0
    if 'result' in rout:
        a = rout['result']
    else:
        a = rout['name']
    if a in rout['vars']:
        return iscomplex(rout['vars'][a])
    return 0


def iscomplexfunction_warn(rout):
    if iscomplexfunction(rout):
        outmess("""\
    **************************************************************
        Warning: code with a function returning complex value
        may not work correctly with your Fortran compiler.
        When using GNU gcc/g77 compilers, codes should work
        correctly for callbacks with:
        f2py -c -DF2PY_CB_RETURNCOMPLEX
    **************************************************************\n""")
        return 1
    return 0


def isstringfunction(rout):
    if not isfunction(rout):
        return 0
    if 'result' in rout:
        a = rout['result']
    else:
        a = rout['name']
    if a in rout['vars']:
        return isstring(rout['vars'][a])
    return 0


def hasexternals(rout):
    return 'externals' in rout and rout['externals']


def isthreadsafe(rout):
    return 'f2pyenhancements' in rout and \
           'threadsafe' in rout['f2pyenhancements']


def hasvariables(rout):
    return 'vars' in rout and rout['vars']


def isoptional(var):
    return ('attrspec' in var and 'optional' in var['attrspec'] and
            'required' not in var['attrspec']) and isintent_nothide(var)


def isexternal(var):
    return 'attrspec' in var and 'external' in var['attrspec']


def getdimension(var):
    dimpattern = r"\((.*?)\)"
    if 'attrspec' in var.keys():
        if any('dimension' in s for s in var['attrspec']):
            return next(re.findall(dimpattern, v) for v in var['attrspec'])


def isrequired(var):
    return not isoptional(var) and isintent_nothide(var)


def iscstyledirective(f2py_line):
    directives = {"callstatement", "callprotoargument", "pymethoddef"}
    return any(directive in f2py_line.lower() for directive in directives)


def isintent_in(var):
    if 'intent' not in var:
        return 1
    if 'hide' in var['intent']:
        return 0
    if 'inplace' in var['intent']:
        return 0
    if 'in' in var['intent']:
        return 1
    if 'out' in var['intent']:
        return 0
    if 'inout' in var['intent']:
        return 0
    if 'outin' in var['intent']:
        return 0
    return 1


def isintent_inout(var):
    return ('intent' in var and ('inout' in var['intent'] or
            'outin' in var['intent']) and 'in' not in var['intent'] and
            'hide' not in var['intent'] and 'inplace' not in var['intent'])


def isintent_out(var):
    return 'out' in var.get('intent', [])


def isintent_hide(var):
    return ('intent' in var and ('hide' in var['intent'] or
            ('out' in var['intent'] and 'in' not in var['intent'] and
                (not l_or(isintent_inout, isintent_inplace)(var)))))


def isintent_nothide(var):
    return not isintent_hide(var)


def isintent_c(var):
    return 'c' in var.get('intent', [])


def isintent_cache(var):
    return 'cache' in var.get('intent', [])


def isintent_copy(var):
    return 'copy' in var.get('intent', [])


def isintent_overwrite(var):
    return 'overwrite' in var.get('intent', [])


def isintent_callback(var):
    return 'callback' in var.get('intent', [])


def isintent_inplace(var):
    return 'inplace' in var.get('intent', [])


def isintent_aux(var):
    return 'aux' in var.get('intent', [])


def isintent_aligned4(var):
    return 'aligned4' in var.get('intent', [])


def isintent_aligned8(var):
    return 'aligned8' in var.get('intent', [])


def isintent_aligned16(var):
    return 'aligned16' in var.get('intent', [])


isintent_dict = {isintent_in: 'INTENT_IN', isintent_inout: 'INTENT_INOUT',
                 isintent_out: 'INTENT_OUT', isintent_hide: 'INTENT_HIDE',
                 isintent_cache: 'INTENT_CACHE',
                 isintent_c: 'INTENT_C', isoptional: 'OPTIONAL',
                 isintent_inplace: 'INTENT_INPLACE',
                 isintent_aligned4: 'INTENT_ALIGNED4',
                 isintent_aligned8: 'INTENT_ALIGNED8',
                 isintent_aligned16: 'INTENT_ALIGNED16',
                 }


def isprivate(var):
    return 'attrspec' in var and 'private' in var['attrspec']


def isvariable(var):
    # heuristic to find public/private declarations of filtered subroutines
    if len(var) == 1 and 'attrspec' in var and \
            var['attrspec'][0] in ('public', 'private'):
        is_var = False
    else:
        is_var = True
    return is_var

def hasinitvalue(var):
    return '=' in var


def hasinitvalueasstring(var):
    if not hasinitvalue(var):
        return 0
    return var['='][0] in ['"', "'"]


def hasnote(var):
    return 'note' in var


def hasresultnote(rout):
    if not isfunction(rout):
        return 0
    if 'result' in rout:
        a = rout['result']
    else:
        a = rout['name']
    if a in rout['vars']:
        return hasnote(rout['vars'][a])
    return 0


def hascommon(rout):
    return 'common' in rout


def containscommon(rout):
    if hascommon(rout):
        return 1
    if hasbody(rout):
        for b in rout['body']:
            if containscommon(b):
                return 1
    return 0


def hasderivedtypes(rout):
    return ('block' in rout) and rout['block'] == 'type'


def containsderivedtypes(rout):
    if hasderivedtypes(rout):
        return 1
    if hasbody(rout):
        for b in rout['body']:
            if hasderivedtypes(b):
                return 1
    return 0


def containsmodule(block):
    if ismodule(block):
        return 1
    if not hasbody(block):
        return 0
    for b in block['body']:
        if containsmodule(b):
            return 1
    return 0


def hasbody(rout):
    return 'body' in rout


def hascallstatement(rout):
    return getcallstatement(rout) is not None


def istrue(var):
    return 1


def isfalse(var):
    return 0


class F2PYError(Exception):
    pass


class throw_error:

    def __init__(self, mess):
        self.mess = mess

    def __call__(self, var):
        mess = f'\n\n  var = {var}\n  Message: {self.mess}\n'
        raise F2PYError(mess)


def l_and(*f):
    l1, l2 = 'lambda v', []
    for i in range(len(f)):
        l1 = '%s,f%d=f[%d]' % (l1, i, i)
        l2.append('f%d(v)' % (i))
    return eval(f"{l1}:{' and '.join(l2)}")


def l_or(*f):
    l1, l2 = 'lambda v', []
    for i in range(len(f)):
        l1 = '%s,f%d=f[%d]' % (l1, i, i)
        l2.append('f%d(v)' % (i))
    return eval(f"{l1}:{' or '.join(l2)}")


def l_not(f):
    return eval('lambda v,f=f:not f(v)')


def isdummyroutine(rout):
    try:
        return rout['f2pyenhancements']['fortranname'] == ''
    except KeyError:
        return 0


def getfortranname(rout):
    try:
        name = rout['f2pyenhancements']['fortranname']
        if name == '':
            raise KeyError
        if not name:
            errmess(f"Failed to use fortranname from {rout['f2pyenhancements']}\n")
            raise KeyError
    except KeyError:
        name = rout['name']
    return name


def getmultilineblock(rout, blockname, comment=1, counter=0):
    try:
        r = rout['f2pyenhancements'].get(blockname)
    except KeyError:
        return
    if not r:
        return
    if counter > 0 and isinstance(r, str):
        return
    if isinstance(r, list):
        if counter >= len(r):
            return
        r = r[counter]
    if r[:3] == "'''":
        if comment:
            r = '\t/* start ' + blockname + \
                ' multiline (' + repr(counter) + ') */\n' + r[3:]
        else:
            r = r[3:]
        if r[-3:] == "'''":
            if comment:
                r = r[:-3] + '\n\t/* end multiline (' + repr(counter) + ')*/'
            else:
                r = r[:-3]
        else:
            errmess(f"{blockname} multiline block should end with `'''`: {repr(r)}\n")
    return r


def getcallstatement(rout):
    return getmultilineblock(rout, 'callstatement')


def getcallprotoargument(rout, cb_map={}):
    r = getmultilineblock(rout, 'callprotoargument', comment=0)
    if r:
        return r
    if hascallstatement(rout):
        outmess(
            'warning: callstatement is defined without callprotoargument\n')
        return
    from .capi_maps import getctype
    arg_types, arg_types2 = [], []
    if l_and(isstringfunction, l_not(isfunction_wrap))(rout):
        arg_types.extend(['char*', 'size_t'])
    for n in rout['args']:
        var = rout['vars'][n]
        if isintent_callback(var):
            continue
        if n in cb_map:
            ctype = cb_map[n] + '_typedef'
        else:
            ctype = getctype(var)
            if l_and(isintent_c, l_or(isscalar, iscomplex))(var):
                pass
            elif isstring(var):
                pass
            elif not isattr_value(var):
                ctype = ctype + '*'
            if (isstring(var)
                 or isarrayofstrings(var)  # obsolete?
                 or isstringarray(var)):
                arg_types2.append('size_t')
        arg_types.append(ctype)

    proto_args = ','.join(arg_types + arg_types2)
    if not proto_args:
        proto_args = 'void'
    return proto_args


def getusercode(rout):
    return getmultilineblock(rout, 'usercode')


def getusercode1(rout):
    return getmultilineblock(rout, 'usercode', counter=1)


def getpymethoddef(rout):
    return getmultilineblock(rout, 'pymethoddef')


def getargs(rout):
    sortargs, args = [], []
    if 'args' in rout:
        args = rout['args']
        if 'sortvars' in rout:
            for a in rout['sortvars']:
                if a in args:
                    sortargs.append(a)
            for a in args:
                if a not in sortargs:
                    sortargs.append(a)
        else:
            sortargs = rout['args']
    return args, sortargs


def getargs2(rout):
    sortargs, args = [], rout.get('args', [])
    auxvars = [a for a in rout['vars'].keys() if isintent_aux(rout['vars'][a])
               and a not in args]
    args = auxvars + args
    if 'sortvars' in rout:
        for a in rout['sortvars']:
            if a in args:
                sortargs.append(a)
        for a in args:
            if a not in sortargs:
                sortargs.append(a)
    else:
        sortargs = auxvars + rout['args']
    return args, sortargs


def getrestdoc(rout):
    if 'f2pymultilines' not in rout:
        return None
    k = None
    if rout['block'] == 'python module':
        k = rout['block'], rout['name']
    return rout['f2pymultilines'].get(k, None)


def gentitle(name):
    ln = (80 - len(name) - 6) // 2
    return f"/*{ln * '*'} {name} {ln * '*'}*/"


def flatlist(lst):
    if isinstance(lst, list):
        return reduce(lambda x, y, f=flatlist: x + f(y), lst, [])
    return [lst]


def stripcomma(s):
    if s and s[-1] == ',':
        return s[:-1]
    return s


def replace(str, d, defaultsep=''):
    if isinstance(d, list):
        return [replace(str, _m, defaultsep) for _m in d]
    if isinstance(str, list):
        return [replace(_m, d, defaultsep) for _m in str]
    for k in 2 * list(d.keys()):
        if k == 'separatorsfor':
            continue
        if 'separatorsfor' in d and k in d['separatorsfor']:
            sep = d['separatorsfor'][k]
        else:
            sep = defaultsep
        if isinstance(d[k], list):
            str = str.replace(f'#{k}#', sep.join(flatlist(d[k])))
        else:
            str = str.replace(f'#{k}#', d[k])
    return str


def dictappend(rd, ar):
    if isinstance(ar, list):
        for a in ar:
            rd = dictappend(rd, a)
        return rd
    for k in ar.keys():
        if k[0] == '_':
            continue
        if k in rd:
            if isinstance(rd[k], str):
                rd[k] = [rd[k]]
            if isinstance(rd[k], list):
                if isinstance(ar[k], list):
                    rd[k] = rd[k] + ar[k]
                else:
                    rd[k].append(ar[k])
            elif isinstance(rd[k], dict):
                if isinstance(ar[k], dict):
                    if k == 'separatorsfor':
                        for k1 in ar[k].keys():
                            if k1 not in rd[k]:
                                rd[k][k1] = ar[k][k1]
                    else:
                        rd[k] = dictappend(rd[k], ar[k])
        else:
            rd[k] = ar[k]
    return rd


def applyrules(rules, d, var={}):
    ret = {}
    if isinstance(rules, list):
        for r in rules:
            rr = applyrules(r, d, var)
            ret = dictappend(ret, rr)
            if '_break' in rr:
                break
        return ret
    if '_check' in rules and (not rules['_check'](var)):
        return ret
    if 'need' in rules:
        res = applyrules({'needs': rules['need']}, d, var)
        if 'needs' in res:
            cfuncs.append_needs(res['needs'])

    for k in rules.keys():
        if k == 'separatorsfor':
            ret[k] = rules[k]
            continue
        if isinstance(rules[k], str):
            ret[k] = replace(rules[k], d)
        elif isinstance(rules[k], list):
            ret[k] = []
            for i in rules[k]:
                ar = applyrules({k: i}, d, var)
                if k in ar:
                    ret[k].append(ar[k])
        elif k[0] == '_':
            continue
        elif isinstance(rules[k], dict):
            ret[k] = []
            for k1 in rules[k].keys():
                if isinstance(k1, types.FunctionType) and k1(var):
                    if isinstance(rules[k][k1], list):
                        for i in rules[k][k1]:
                            if isinstance(i, dict):
                                res = applyrules({'supertext': i}, d, var)
                                i = res.get('supertext', '')
                            ret[k].append(replace(i, d))
                    else:
                        i = rules[k][k1]
                        if isinstance(i, dict):
                            res = applyrules({'supertext': i}, d)
                            i = res.get('supertext', '')
                        ret[k].append(replace(i, d))
        else:
            errmess(f'applyrules: ignoring rule {repr(rules[k])}.\n')
        if isinstance(ret[k], list):
            if len(ret[k]) == 1:
                ret[k] = ret[k][0]
            if ret[k] == []:
                del ret[k]
    return ret


_f2py_module_name_match = re.compile(r'\s*python\s*module\s*(?P<name>[\w_]+)',
                                     re.I).match
_f2py_user_module_name_match = re.compile(r'\s*python\s*module\s*(?P<name>[\w_]*?'
                                          r'__user__[\w_]*)', re.I).match

def get_f2py_modulename(source):
    name = None
    with open(source) as f:
        for line in f:
            m = _f2py_module_name_match(line)
            if m:
                if _f2py_user_module_name_match(line):  # skip *__user__* names
                    continue
                name = m.group('name')
                break
    return name

def getuseblocks(pymod):
    all_uses = []
    for inner in pymod['body']:
        for modblock in inner['body']:
            if modblock.get('use'):
                all_uses.extend([x for x in modblock.get("use").keys() if "__" not in x])
    return all_uses

def process_f2cmap_dict(f2cmap_all, new_map, c2py_map, verbose=False):
    """
    Update the Fortran-to-C type mapping dictionary with new mappings and
    return a list of successfully mapped C types.

    This function integrates a new mapping dictionary into an existing
    Fortran-to-C type mapping dictionary. It ensures that all keys are in
    lowercase and validates new entries against a given C-to-Python mapping
    dictionary. Redefinitions and invalid entries are reported with a warning.

    Parameters
    ----------
    f2cmap_all : dict
        The existing Fortran-to-C type mapping dictionary that will be updated.
        It should be a dictionary of dictionaries where the main keys represent
        Fortran types and the nested dictionaries map Fortran type specifiers
        to corresponding C types.

    new_map : dict
        A dictionary containing new type mappings to be added to `f2cmap_all`.
        The structure should be similar to `f2cmap_all`, with keys representing
        Fortran types and values being dictionaries of type specifiers and their
        C type equivalents.

    c2py_map : dict
        A dictionary used for validating the C types in `new_map`. It maps C
        types to corresponding Python types and is used to ensure that the C
        types specified in `new_map` are valid.

    verbose : boolean
        A flag used to provide information about the types mapped

    Returns
    -------
    tuple of (dict, list)
        The updated Fortran-to-C type mapping dictionary and a list of
        successfully mapped C types.
    """
    f2cmap_mapped = []

    new_map_lower = {}
    for k, d1 in new_map.items():
        d1_lower = {k1.lower(): v1 for k1, v1 in d1.items()}
        new_map_lower[k.lower()] = d1_lower

    for k, d1 in new_map_lower.items():
        if k not in f2cmap_all:
            f2cmap_all[k] = {}

        for k1, v1 in d1.items():
            if v1 in c2py_map:
                if k1 in f2cmap_all[k]:
                    outmess(
                        "\tWarning: redefinition of {'%s':{'%s':'%s'->'%s'}}\n"
                        % (k, k1, f2cmap_all[k][k1], v1)
                    )
                f2cmap_all[k][k1] = v1
                if verbose:
                    outmess(f'\tMapping "{k}(kind={k1})" to "{v1}\"\n')
                f2cmap_mapped.append(v1)
            elif verbose:
                errmess(
                    "\tIgnoring map {'%s':{'%s':'%s'}}: '%s' must be in %s\n"
                    % (k, k1, v1, v1, list(c2py_map.keys()))
                )

    return f2cmap_all, f2cmap_mapped

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\tensor\array\expressions\from_array_to_matrix.py ===
from __future__ import annotations
import itertools
from collections import defaultdict
from typing import FrozenSet
from functools import singledispatch
from itertools import accumulate

from sympy import MatMul, Basic, Wild, KroneckerProduct
from sympy.assumptions.ask import (Q, ask)
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.matrices.expressions.diagonal import DiagMatrix
from sympy.matrices.expressions.hadamard import hadamard_product, HadamardPower
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.matrices.expressions.special import (Identity, ZeroMatrix, OneMatrix)
from sympy.matrices.expressions.trace import Trace
from sympy.matrices.expressions.transpose import Transpose
from sympy.combinatorics.permutations import _af_invert, Permutation
from sympy.matrices.matrixbase import MatrixBase
from sympy.matrices.expressions.applyfunc import ElementwiseApplyFunction
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.tensor.array.expressions.array_expressions import PermuteDims, ArrayDiagonal, \
    ArrayTensorProduct, OneArray, get_rank, _get_subrank, ZeroArray, ArrayContraction, \
    ArrayAdd, _CodegenArrayAbstract, get_shape, ArrayElementwiseApplyFunc, _ArrayExpr, _EditArrayContraction, _ArgE, \
    ArrayElement, _array_tensor_product, _array_contraction, _array_diagonal, _array_add, _permute_dims
from sympy.tensor.array.expressions.utils import _get_mapping_from_subranks


def _get_candidate_for_matmul_from_contraction(scan_indices: list[int | None], remaining_args: list[_ArgE]) -> tuple[_ArgE | None, bool, int]:

    scan_indices_int: list[int] = [i for i in scan_indices if i is not None]
    if len(scan_indices_int) == 0:
        return None, False, -1

    transpose: bool = False
    candidate: _ArgE | None = None
    candidate_index: int = -1
    for arg_with_ind2 in remaining_args:
        if not isinstance(arg_with_ind2.element, MatrixExpr):
            continue
        for index in scan_indices_int:
            if candidate_index != -1 and candidate_index != index:
                # A candidate index has already been selected, check
                # repetitions only for that index:
                continue
            if index in arg_with_ind2.indices:
                if set(arg_with_ind2.indices) == {index}:
                    # Index repeated twice in arg_with_ind2
                    candidate = None
                    break
                if candidate is None:
                    candidate = arg_with_ind2
                    candidate_index = index
                    transpose = (index == arg_with_ind2.indices[1])
                else:
                    # Index repeated more than twice, break
                    candidate = None
                    break
    return candidate, transpose, candidate_index


def _insert_candidate_into_editor(editor: _EditArrayContraction, arg_with_ind: _ArgE, candidate: _ArgE, transpose1: bool, transpose2: bool):
    other = candidate.element
    other_index: int | None
    if transpose2:
        other = Transpose(other)
        other_index = candidate.indices[0]
    else:
        other_index = candidate.indices[1]
    new_element = (Transpose(arg_with_ind.element) if transpose1 else arg_with_ind.element) * other
    editor.args_with_ind.remove(candidate)
    new_arge = _ArgE(new_element)
    return new_arge, other_index


def _support_function_tp1_recognize(contraction_indices, args):
    if len(contraction_indices) == 0:
        return _a2m_tensor_product(*args)

    ac = _array_contraction(_array_tensor_product(*args), *contraction_indices)
    editor = _EditArrayContraction(ac)
    editor.track_permutation_start()

    while True:
        flag_stop = True
        for i, arg_with_ind in enumerate(editor.args_with_ind):
            if not isinstance(arg_with_ind.element, MatrixExpr):
                continue

            first_index = arg_with_ind.indices[0]
            second_index = arg_with_ind.indices[1]

            first_frequency = editor.count_args_with_index(first_index)
            second_frequency = editor.count_args_with_index(second_index)

            if first_index is not None and first_frequency == 1 and first_index == second_index:
                flag_stop = False
                arg_with_ind.element = Trace(arg_with_ind.element)._normalize()
                arg_with_ind.indices = []
                break

            scan_indices = []
            if first_frequency == 2:
                scan_indices.append(first_index)
            if second_frequency == 2:
                scan_indices.append(second_index)

            candidate, transpose, found_index = _get_candidate_for_matmul_from_contraction(scan_indices, editor.args_with_ind[i+1:])
            if candidate is not None:
                flag_stop = False
                editor.track_permutation_merge(arg_with_ind, candidate)
                transpose1 = found_index == first_index
                new_arge, other_index = _insert_candidate_into_editor(editor, arg_with_ind, candidate, transpose1, transpose)
                if found_index == first_index:
                    new_arge.indices = [second_index, other_index]
                else:
                    new_arge.indices = [first_index, other_index]
                set_indices = set(new_arge.indices)
                if len(set_indices) == 1 and set_indices != {None}:
                    # This is a trace:
                    new_arge.element = Trace(new_arge.element)._normalize()
                    new_arge.indices = []
                editor.args_with_ind[i] = new_arge
                # TODO: is this break necessary?
                break

        if flag_stop:
            break

    editor.refresh_indices()
    return editor.to_array_contraction()


def _find_trivial_matrices_rewrite(expr: ArrayTensorProduct):
    # If there are matrices of trivial shape in the tensor product (i.e. shape
    # (1, 1)), try to check if there is a suitable non-trivial MatMul where the
    # expression can be inserted.

    # For example, if "a" has shape (1, 1) and "b" has shape (k, 1), the
    # expressions "_array_tensor_product(a, b*b.T)" can be rewritten as
    # "b*a*b.T"

    trivial_matrices = []
    pos: int | None = None  # must be initialized else causes UnboundLocalError
    first: MatrixExpr | None = None  # may cause UnboundLocalError if not initialized
    second: MatrixExpr | None = None  # may cause UnboundLocalError if not initialized
    removed: list[int] = []
    counter: int = 0
    args: list[Basic | None] = list(expr.args)
    for i, arg in enumerate(expr.args):
        if isinstance(arg, MatrixExpr):
            if arg.shape == (1, 1):
                trivial_matrices.append(arg)
                args[i] = None
                removed.extend([counter, counter+1])
            elif pos is None and isinstance(arg, MatMul):
                margs = arg.args
                for j, e in enumerate(margs):
                    if isinstance(e, MatrixExpr) and e.shape[1] == 1:
                        pos = i
                        first = MatMul.fromiter(margs[:j+1])
                        second = MatMul.fromiter(margs[j+1:])
                        break
        counter += get_rank(arg)
    if pos is None:
        return expr, []
    args[pos] = (first*MatMul.fromiter(i for i in trivial_matrices)*second).doit()
    return _array_tensor_product(*[i for i in args if i is not None]), removed


def _find_trivial_kronecker_products_broadcast(expr: ArrayTensorProduct):
    newargs: list[Basic] = []
    removed = []
    count_dims = 0
    for arg in expr.args:
        count_dims += get_rank(arg)
        shape = get_shape(arg)
        current_range = [count_dims-i for i in range(len(shape), 0, -1)]
        if (shape == (1, 1) and len(newargs) > 0 and 1 not in get_shape(newargs[-1]) and
            isinstance(newargs[-1], MatrixExpr) and isinstance(arg, MatrixExpr)):
            # KroneckerProduct object allows the trick of broadcasting:
            newargs[-1] = KroneckerProduct(newargs[-1], arg)
            removed.extend(current_range)
        elif 1 not in shape and len(newargs) > 0 and get_shape(newargs[-1]) == (1, 1):
            # Broadcast:
            newargs[-1] = KroneckerProduct(newargs[-1], arg)
            prev_range = [i for i in range(min(current_range)) if i not in removed]
            removed.extend(prev_range[-2:])
        else:
            newargs.append(arg)
    return _array_tensor_product(*newargs), removed


@singledispatch
def _array2matrix(expr):
    return expr


@_array2matrix.register(ZeroArray)
def _(expr: ZeroArray):
    if get_rank(expr) == 2:
        return ZeroMatrix(*expr.shape)
    else:
        return expr


@_array2matrix.register(ArrayTensorProduct)
def _(expr: ArrayTensorProduct):
    return _a2m_tensor_product(*[_array2matrix(arg) for arg in expr.args])


@_array2matrix.register(ArrayContraction)
def _(expr: ArrayContraction):
    expr = expr.flatten_contraction_of_diagonal()
    expr = identify_removable_identity_matrices(expr)
    expr = expr.split_multiple_contractions()
    expr = identify_hadamard_products(expr)
    if not isinstance(expr, ArrayContraction):
        return _array2matrix(expr)
    subexpr = expr.expr
    contraction_indices: tuple[tuple[int]] = expr.contraction_indices
    if contraction_indices == ((0,), (1,)) or (
        contraction_indices == ((0,),) and subexpr.shape[1] == 1
    ) or (
        contraction_indices == ((1,),) and subexpr.shape[0] == 1
    ):
        shape = subexpr.shape
        subexpr = _array2matrix(subexpr)
        if isinstance(subexpr, MatrixExpr):
            return OneMatrix(1, shape[0])*subexpr*OneMatrix(shape[1], 1)
    if isinstance(subexpr, ArrayTensorProduct):
        newexpr = _array_contraction(_array2matrix(subexpr), *contraction_indices)
        contraction_indices = newexpr.contraction_indices
        if any(i > 2 for i in newexpr.subranks):
            addends = _array_add(*[_a2m_tensor_product(*j) for j in itertools.product(*[i.args if isinstance(i,
                                                                                                                             ArrayAdd) else [i] for i in expr.expr.args])])
            newexpr = _array_contraction(addends, *contraction_indices)
        if isinstance(newexpr, ArrayAdd):
            ret = _array2matrix(newexpr)
            return ret
        assert isinstance(newexpr, ArrayContraction)
        ret = _support_function_tp1_recognize(contraction_indices, list(newexpr.expr.args))
        return ret
    elif not isinstance(subexpr, _CodegenArrayAbstract):
        ret = _array2matrix(subexpr)
        if isinstance(ret, MatrixExpr):
            assert expr.contraction_indices == ((0, 1),)
            return _a2m_trace(ret)
        else:
            return _array_contraction(ret, *expr.contraction_indices)


@_array2matrix.register(ArrayDiagonal)
def _(expr: ArrayDiagonal):
    pexpr = _array_diagonal(_array2matrix(expr.expr), *expr.diagonal_indices)
    pexpr = identify_hadamard_products(pexpr)
    if isinstance(pexpr, ArrayDiagonal):
        pexpr = _array_diag2contr_diagmatrix(pexpr)
    if expr == pexpr:
        return expr
    return _array2matrix(pexpr)


@_array2matrix.register(PermuteDims)
def _(expr: PermuteDims):
    if expr.permutation.array_form == [1, 0]:
        return _a2m_transpose(_array2matrix(expr.expr))
    elif isinstance(expr.expr, ArrayTensorProduct):
        ranks = expr.expr.subranks
        inv_permutation = expr.permutation**(-1)
        newrange = [inv_permutation(i) for i in range(sum(ranks))]
        newpos = []
        counter = 0
        for rank in ranks:
            newpos.append(newrange[counter:counter+rank])
            counter += rank
        newargs = []
        newperm = []
        scalars = []
        for pos, arg in zip(newpos, expr.expr.args):
            if len(pos) == 0:
                scalars.append(_array2matrix(arg))
            elif pos == sorted(pos):
                newargs.append((_array2matrix(arg), pos[0]))
                newperm.extend(pos)
            elif len(pos) == 2:
                newargs.append((_a2m_transpose(_array2matrix(arg)), pos[0]))
                newperm.extend(reversed(pos))
            else:
                raise NotImplementedError()
        newargs = [i[0] for i in newargs]
        return _permute_dims(_a2m_tensor_product(*scalars, *newargs), _af_invert(newperm))
    elif isinstance(expr.expr, ArrayContraction):
        mat_mul_lines = _array2matrix(expr.expr)
        if not isinstance(mat_mul_lines, ArrayTensorProduct):
            return _permute_dims(mat_mul_lines, expr.permutation)
        # TODO: this assumes that all arguments are matrices, it may not be the case:
        permutation = Permutation(2*len(mat_mul_lines.args)-1)*expr.permutation
        permuted = [permutation(i) for i in range(2*len(mat_mul_lines.args))]
        args_array = [None for i in mat_mul_lines.args]
        for i in range(len(mat_mul_lines.args)):
            p1 = permuted[2*i]
            p2 = permuted[2*i+1]
            if p1 // 2 != p2 // 2:
                return _permute_dims(mat_mul_lines, permutation)
            if p1 > p2:
                args_array[i] = _a2m_transpose(mat_mul_lines.args[p1 // 2])
            else:
                args_array[i] = mat_mul_lines.args[p1 // 2]
        return _a2m_tensor_product(*args_array)
    else:
        return expr


@_array2matrix.register(ArrayAdd)
def _(expr: ArrayAdd):
    addends = [_array2matrix(arg) for arg in expr.args]
    return _a2m_add(*addends)


@_array2matrix.register(ArrayElementwiseApplyFunc)
def _(expr: ArrayElementwiseApplyFunc):
    subexpr = _array2matrix(expr.expr)
    if isinstance(subexpr, MatrixExpr):
        if subexpr.shape != (1, 1):
            d = expr.function.bound_symbols[0]
            w = Wild("w", exclude=[d])
            p = Wild("p", exclude=[d])
            m = expr.function.expr.match(w*d**p)
            if m is not None:
                return m[w]*HadamardPower(subexpr, m[p])
        return ElementwiseApplyFunction(expr.function, subexpr)
    else:
        return ArrayElementwiseApplyFunc(expr.function, subexpr)


@_array2matrix.register(ArrayElement)
def _(expr: ArrayElement):
    ret = _array2matrix(expr.name)
    if isinstance(ret, MatrixExpr):
        return MatrixElement(ret, *expr.indices)
    return ArrayElement(ret, expr.indices)


@singledispatch
def _remove_trivial_dims(expr):
    return expr, []


@_remove_trivial_dims.register(ArrayTensorProduct)
def _(expr: ArrayTensorProduct):
    # Recognize expressions like [x, y] with shape (k, 1, k, 1) as `x*y.T`.
    # The matrix expression has to be equivalent to the tensor product of the
    # matrices, with trivial dimensions (i.e. dim=1) dropped.
    # That is, add contractions over trivial dimensions:

    removed = []
    newargs = []
    cumul = list(accumulate([0] + [get_rank(arg) for arg in expr.args]))
    pending = None
    prev_i = None
    for i, arg in enumerate(expr.args):
        current_range = list(range(cumul[i], cumul[i+1]))
        if isinstance(arg, OneArray):
            removed.extend(current_range)
            continue
        if not isinstance(arg, (MatrixExpr, MatrixBase)):
            rarg, rem = _remove_trivial_dims(arg)
            removed.extend(rem)
            newargs.append(rarg)
            continue
        elif getattr(arg, "is_Identity", False) and arg.shape == (1, 1):
            if arg.shape == (1, 1):
                # Ignore identity matrices of shape (1, 1) - they are equivalent to scalar 1.
                removed.extend(current_range)
            continue
        elif arg.shape == (1, 1):
            arg, _ = _remove_trivial_dims(arg)
            # Matrix is equivalent to scalar:
            if len(newargs) == 0:
                newargs.append(arg)
            elif 1 in get_shape(newargs[-1]):
                if newargs[-1].shape[1] == 1:
                    newargs[-1] = newargs[-1]*arg
                else:
                    newargs[-1] = arg*newargs[-1]
                removed.extend(current_range)
            else:
                newargs.append(arg)
        elif 1 in arg.shape:
            k = [i for i in arg.shape if i != 1][0]
            if pending is None:
                pending = k
                prev_i = i
                newargs.append(arg)
            elif pending == k:
                prev = newargs[-1]
                if prev.shape[0] == 1:
                    d1 = cumul[prev_i]  # type: ignore
                    prev = _a2m_transpose(prev)
                else:
                    d1 = cumul[prev_i] + 1  # type: ignore
                if arg.shape[1] == 1:
                    d2 = cumul[i] + 1
                    arg = _a2m_transpose(arg)
                else:
                    d2 = cumul[i]
                newargs[-1] = prev*arg
                pending = None
                removed.extend([d1, d2])
            else:
                newargs.append(arg)
                pending = k
                prev_i = i
        else:
            newargs.append(arg)
            pending = None
    newexpr, newremoved = _a2m_tensor_product(*newargs), sorted(removed)
    if isinstance(newexpr, ArrayTensorProduct):
        newexpr, newremoved2 = _find_trivial_matrices_rewrite(newexpr)
        newremoved = _combine_removed(-1, newremoved, newremoved2)
    if isinstance(newexpr, ArrayTensorProduct):
        newexpr, newremoved2 = _find_trivial_kronecker_products_broadcast(newexpr)
        newremoved = _combine_removed(-1, newremoved, newremoved2)
    return newexpr, newremoved


@_remove_trivial_dims.register(ArrayAdd)
def _(expr: ArrayAdd):
    rec = [_remove_trivial_dims(arg) for arg in expr.args]
    newargs, removed = zip(*rec)
    if len({get_shape(i) for i in newargs}) > 1:
        return expr, []
    if len(removed) == 0:
        return expr, removed
    removed1 = removed[0]
    return _a2m_add(*newargs), removed1


@_remove_trivial_dims.register(PermuteDims)
def _(expr: PermuteDims):
    subexpr, subremoved = _remove_trivial_dims(expr.expr)
    p = expr.permutation.array_form
    pinv = _af_invert(expr.permutation.array_form)
    shift = list(accumulate([1 if i in subremoved else 0 for i in range(len(p))]))
    premoved = [pinv[i] for i in subremoved]
    p2 = [e - shift[e] for e in p if e not in subremoved]
    # TODO: check if subremoved should be permuted as well...
    newexpr = _permute_dims(subexpr, p2)
    premoved = sorted(premoved)
    if newexpr != expr:
        newexpr, removed2 = _remove_trivial_dims(_array2matrix(newexpr))
        premoved = _combine_removed(-1, premoved, removed2)
    return newexpr, premoved


@_remove_trivial_dims.register(ArrayContraction)
def _(expr: ArrayContraction):
    new_expr, removed0 = _array_contraction_to_diagonal_multiple_identity(expr)
    if new_expr != expr:
        new_expr2, removed1 = _remove_trivial_dims(_array2matrix(new_expr))
        removed = _combine_removed(-1, removed0, removed1)
        return new_expr2, removed
    rank1 = get_rank(expr)
    expr, removed1 = remove_identity_matrices(expr)
    if not isinstance(expr, ArrayContraction):
        expr2, removed2 = _remove_trivial_dims(expr)
        return expr2, _combine_removed(rank1, removed1, removed2)
    newexpr, removed2 = _remove_trivial_dims(expr.expr)
    shifts = list(accumulate([1 if i in removed2 else 0 for i in range(get_rank(expr.expr))]))
    new_contraction_indices = [tuple(j for j in i if j not in removed2) for i in expr.contraction_indices]
    # Remove possible empty tuples "()":
    new_contraction_indices = [i for i in new_contraction_indices if len(i) > 0]
    contraction_indices_flat = [j for i in expr.contraction_indices for j in i]
    removed2 = [i for i in removed2 if i not in contraction_indices_flat]
    new_contraction_indices = [tuple(j - shifts[j] for j in i) for i in new_contraction_indices]
    # Shift removed2:
    removed2 = ArrayContraction._push_indices_up(expr.contraction_indices, removed2)
    removed = _combine_removed(rank1, removed1, removed2)
    return _array_contraction(newexpr, *new_contraction_indices), list(removed)


def _remove_diagonalized_identity_matrices(expr: ArrayDiagonal):
    assert isinstance(expr, ArrayDiagonal)
    editor = _EditArrayContraction(expr)
    mapping = {i: {j for j in editor.args_with_ind if i in j.indices} for i in range(-1, -1-editor.number_of_diagonal_indices, -1)}
    removed = []
    counter: int = 0
    for i, arg_with_ind in enumerate(editor.args_with_ind):
        counter += len(arg_with_ind.indices)
        if isinstance(arg_with_ind.element, Identity):
            if None in arg_with_ind.indices and any(i is not None and (i < 0) == True for i in arg_with_ind.indices):
                diag_ind = [j for j in arg_with_ind.indices if j is not None][0]
                other = [j for j in mapping[diag_ind] if j != arg_with_ind][0]
                if not isinstance(other.element, MatrixExpr):
                    continue
                if 1 not in other.element.shape:
                    continue
                if None not in other.indices:
                    continue
                editor.args_with_ind[i].element = None
                none_index = other.indices.index(None)
                other.element = DiagMatrix(other.element)
                other_range = editor.get_absolute_range(other)
                removed.extend([other_range[0] + none_index])
    editor.args_with_ind = [i for i in editor.args_with_ind if i.element is not None]
    removed = ArrayDiagonal._push_indices_up(expr.diagonal_indices, removed, get_rank(expr.expr))
    return editor.to_array_contraction(), removed


@_remove_trivial_dims.register(ArrayDiagonal)
def _(expr: ArrayDiagonal):
    newexpr, removed = _remove_trivial_dims(expr.expr)
    shifts = list(accumulate([0] + [1 if i in removed else 0 for i in range(get_rank(expr.expr))]))
    new_diag_indices_map = {i: tuple(j for j in i if j not in removed) for i in expr.diagonal_indices}
    for old_diag_tuple, new_diag_tuple in new_diag_indices_map.items():
        if len(new_diag_tuple) == 1:
            removed = [i for i in removed if i not in old_diag_tuple]
    new_diag_indices = [tuple(j - shifts[j] for j in i) for i in new_diag_indices_map.values()]
    rank = get_rank(expr.expr)
    removed = ArrayDiagonal._push_indices_up(expr.diagonal_indices, removed, rank)
    removed = sorted(set(removed))
    # If there are single axes to diagonalize remaining, it means that their
    # corresponding dimension has been removed, they no longer need diagonalization:
    new_diag_indices = [i for i in new_diag_indices if len(i) > 0]
    if len(new_diag_indices) > 0:
        newexpr2 = _array_diagonal(newexpr, *new_diag_indices, allow_trivial_diags=True)
    else:
        newexpr2 = newexpr
    if isinstance(newexpr2, ArrayDiagonal):
        newexpr3, removed2 = _remove_diagonalized_identity_matrices(newexpr2)
        removed = _combine_removed(-1, removed, removed2)
        return newexpr3, removed
    else:
        return newexpr2, removed


@_remove_trivial_dims.register(ElementwiseApplyFunction)
def _(expr: ElementwiseApplyFunction):
    subexpr, removed = _remove_trivial_dims(expr.expr)
    if subexpr.shape == (1, 1):
        # TODO: move this to ElementwiseApplyFunction
        return expr.function(subexpr), removed + [0, 1]
    return ElementwiseApplyFunction(expr.function, subexpr), []


@_remove_trivial_dims.register(ArrayElementwiseApplyFunc)
def _(expr: ArrayElementwiseApplyFunc):
    subexpr, removed = _remove_trivial_dims(expr.expr)
    return ArrayElementwiseApplyFunc(expr.function, subexpr), removed


def convert_array_to_matrix(expr):
    r"""
    Recognize matrix expressions in codegen objects.

    If more than one matrix multiplication line have been detected, return a
    list with the matrix expressions.

    Examples
    ========

    >>> from sympy.tensor.array.expressions.from_indexed_to_array import convert_indexed_to_array
    >>> from sympy.tensor.array import tensorcontraction, tensorproduct
    >>> from sympy import MatrixSymbol, Sum
    >>> from sympy.abc import i, j, k, l, N
    >>> from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array
    >>> from sympy.tensor.array.expressions.from_array_to_matrix import convert_array_to_matrix
    >>> A = MatrixSymbol("A", N, N)
    >>> B = MatrixSymbol("B", N, N)
    >>> C = MatrixSymbol("C", N, N)
    >>> D = MatrixSymbol("D", N, N)

    >>> expr = Sum(A[i, j]*B[j, k], (j, 0, N-1))
    >>> cg = convert_indexed_to_array(expr)
    >>> convert_array_to_matrix(cg)
    A*B
    >>> cg = convert_indexed_to_array(expr, first_indices=[k])
    >>> convert_array_to_matrix(cg)
    B.T*A.T

    Transposition is detected:

    >>> expr = Sum(A[j, i]*B[j, k], (j, 0, N-1))
    >>> cg = convert_indexed_to_array(expr)
    >>> convert_array_to_matrix(cg)
    A.T*B
    >>> cg = convert_indexed_to_array(expr, first_indices=[k])
    >>> convert_array_to_matrix(cg)
    B.T*A

    Detect the trace:

    >>> expr = Sum(A[i, i], (i, 0, N-1))
    >>> cg = convert_indexed_to_array(expr)
    >>> convert_array_to_matrix(cg)
    Trace(A)

    Recognize some more complex traces:

    >>> expr = Sum(A[i, j]*B[j, i], (i, 0, N-1), (j, 0, N-1))
    >>> cg = convert_indexed_to_array(expr)
    >>> convert_array_to_matrix(cg)
    Trace(A*B)

    More complicated expressions:

    >>> expr = Sum(A[i, j]*B[k, j]*A[l, k], (j, 0, N-1), (k, 0, N-1))
    >>> cg = convert_indexed_to_array(expr)
    >>> convert_array_to_matrix(cg)
    A*B.T*A.T

    Expressions constructed from matrix expressions do not contain literal
    indices, the positions of free indices are returned instead:

    >>> expr = A*B
    >>> cg = convert_matrix_to_array(expr)
    >>> convert_array_to_matrix(cg)
    A*B

    If more than one line of matrix multiplications is detected, return
    separate matrix multiplication factors embedded in a tensor product object:

    >>> cg = tensorcontraction(tensorproduct(A, B, C, D), (1, 2), (5, 6))
    >>> convert_array_to_matrix(cg)
    ArrayTensorProduct(A*B, C*D)

    The two lines have free indices at axes 0, 3 and 4, 7, respectively.
    """
    rec = _array2matrix(expr)
    rec, removed = _remove_trivial_dims(rec)
    return rec


def _array_diag2contr_diagmatrix(expr: ArrayDiagonal):
    if isinstance(expr.expr, ArrayTensorProduct):
        args = list(expr.expr.args)
        diag_indices = list(expr.diagonal_indices)
        mapping = _get_mapping_from_subranks([_get_subrank(arg) for arg in args])
        tuple_links = [[mapping[j] for j in i] for i in diag_indices]
        contr_indices = []
        total_rank = get_rank(expr)
        replaced = [False for arg in args]
        for i, (abs_pos, rel_pos) in enumerate(zip(diag_indices, tuple_links)):
            if len(abs_pos) != 2:
                continue
            (pos1_outer, pos1_inner), (pos2_outer, pos2_inner) = rel_pos
            arg1 = args[pos1_outer]
            arg2 = args[pos2_outer]
            if get_rank(arg1) != 2 or get_rank(arg2) != 2:
                if replaced[pos1_outer]:
                    diag_indices[i] = None
                if replaced[pos2_outer]:
                    diag_indices[i] = None
                continue
            pos1_in2 = 1 - pos1_inner
            pos2_in2 = 1 - pos2_inner
            if arg1.shape[pos1_in2] == 1:
                if arg1.shape[pos1_inner] != 1:
                    darg1 = DiagMatrix(arg1)
                else:
                    darg1 = arg1
                args.append(darg1)
                contr_indices.append(((pos2_outer, pos2_inner), (len(args)-1, pos1_inner)))
                total_rank += 1
                diag_indices[i] = None
                args[pos1_outer] = OneArray(arg1.shape[pos1_in2])
                replaced[pos1_outer] = True
            elif arg2.shape[pos2_in2] == 1:
                if arg2.shape[pos2_inner] != 1:
                    darg2 = DiagMatrix(arg2)
                else:
                    darg2 = arg2
                args.append(darg2)
                contr_indices.append(((pos1_outer, pos1_inner), (len(args)-1, pos2_inner)))
                total_rank += 1
                diag_indices[i] = None
                args[pos2_outer] = OneArray(arg2.shape[pos2_in2])
                replaced[pos2_outer] = True
        diag_indices_new = [i for i in diag_indices if i is not None]
        cumul = list(accumulate([0] + [get_rank(arg) for arg in args]))
        contr_indices2 = [tuple(cumul[a] + b for a, b in i) for i in contr_indices]
        tc = _array_contraction(
            _array_tensor_product(*args), *contr_indices2
        )
        td = _array_diagonal(tc, *diag_indices_new)
        return td
    return expr


def _a2m_mul(*args):
    if not any(isinstance(i, _CodegenArrayAbstract) for i in args):
        from sympy.matrices.expressions.matmul import MatMul
        return MatMul(*args).doit()
    else:
        return _array_contraction(
            _array_tensor_product(*args),
            *[(2*i-1, 2*i) for i in range(1, len(args))]
        )


def _a2m_tensor_product(*args):
    scalars = []
    arrays = []
    for arg in args:
        if isinstance(arg, (MatrixExpr, _ArrayExpr, _CodegenArrayAbstract)):
            arrays.append(arg)
        else:
            scalars.append(arg)
    scalar = Mul.fromiter(scalars)
    if len(arrays) == 0:
        return scalar
    if scalar != 1:
        if isinstance(arrays[0], _CodegenArrayAbstract):
            arrays = [scalar] + arrays
        else:
            arrays[0] *= scalar
    return _array_tensor_product(*arrays)


def _a2m_add(*args):
    if not any(isinstance(i, _CodegenArrayAbstract) for i in args):
        from sympy.matrices.expressions.matadd import MatAdd
        return MatAdd(*args).doit()
    else:
        return _array_add(*args)


def _a2m_trace(arg):
    if isinstance(arg, _CodegenArrayAbstract):
        return _array_contraction(arg, (0, 1))
    else:
        from sympy.matrices.expressions.trace import Trace
        return Trace(arg)


def _a2m_transpose(arg):
    if isinstance(arg, _CodegenArrayAbstract):
        return _permute_dims(arg, [1, 0])
    else:
        from sympy.matrices.expressions.transpose import Transpose
        return Transpose(arg).doit()


def identify_hadamard_products(expr: ArrayContraction | ArrayDiagonal):

    editor: _EditArrayContraction = _EditArrayContraction(expr)

    map_contr_to_args: dict[FrozenSet, list[_ArgE]] = defaultdict(list)
    map_ind_to_inds: dict[int | None, int] = defaultdict(int)
    for arg_with_ind in editor.args_with_ind:
        for ind in arg_with_ind.indices:
            map_ind_to_inds[ind] += 1
        if None in arg_with_ind.indices:
            continue
        map_contr_to_args[frozenset(arg_with_ind.indices)].append(arg_with_ind)

    k: FrozenSet[int]
    v: list[_ArgE]
    for k, v in map_contr_to_args.items():
        make_trace: bool = False
        if len(k) == 1 and next(iter(k)) >= 0 and sum(next(iter(k)) in i for i in map_contr_to_args) == 1:
            # This is a trace: the arguments are fully contracted with only one
            # index, and the index isn't used anywhere else:
            make_trace = True
            first_element = S.One
        elif len(k) != 2:
            # Hadamard product only defined for matrices:
            continue
        if len(v) == 1:
            # Hadamard product with a single argument makes no sense:
            continue
        for ind in k:
            if map_ind_to_inds[ind] <= 2:
                # There is no other contraction, skip:
                continue

        def check_transpose(x):
            x = [i if i >= 0 else -1-i for i in x]
            return x == sorted(x)

        # Check if expression is a trace:
        if all(map_ind_to_inds[j] == len(v) and j >= 0 for j in k) and all(j >= 0 for j in k):
            # This is a trace
            make_trace = True
            first_element = v[0].element
            if not check_transpose(v[0].indices):
                first_element = first_element.T # type: ignore
            hadamard_factors = v[1:]
        else:
            hadamard_factors = v

        # This is a Hadamard product:

        hp = hadamard_product(*[i.element if check_transpose(i.indices) else Transpose(i.element) for i in hadamard_factors])
        hp_indices = v[0].indices
        if not check_transpose(hadamard_factors[0].indices):
            hp_indices = list(reversed(hp_indices))
        if make_trace:
            hp = Trace(first_element*hp.T)._normalize()
            hp_indices = []
        editor.insert_after(v[0], _ArgE(hp, hp_indices))
        for i in v:
            editor.args_with_ind.remove(i)

    return editor.to_array_contraction()


def identify_removable_identity_matrices(expr):
    editor = _EditArrayContraction(expr)

    flag = True
    while flag:
        flag = False
        for arg_with_ind in editor.args_with_ind:
            if isinstance(arg_with_ind.element, Identity):
                k = arg_with_ind.element.shape[0]
                # Candidate for removal:
                if arg_with_ind.indices == [None, None]:
                    # Free identity matrix, will be cleared by _remove_trivial_dims:
                    continue
                elif None in arg_with_ind.indices:
                    ind = [j for j in arg_with_ind.indices if j is not None][0]
                    counted = editor.count_args_with_index(ind)
                    if counted == 1:
                        # Identity matrix contracted only on one index with itself,
                        # transform to a OneArray(k) element:
                        editor.insert_after(arg_with_ind, OneArray(k))
                        editor.args_with_ind.remove(arg_with_ind)
                        flag = True
                        break
                    elif counted > 2:
                        # Case counted = 2 is a matrix multiplication by identity matrix, skip it.
                        # Case counted > 2 is a multiple contraction,
                        # this is a case where the contraction becomes a diagonalization if the
                        # identity matrix is dropped.
                        continue
                elif arg_with_ind.indices[0] == arg_with_ind.indices[1]:
                    ind = arg_with_ind.indices[0]
                    counted = editor.count_args_with_index(ind)
                    if counted > 1:
                        editor.args_with_ind.remove(arg_with_ind)
                        flag = True
                        break
                    else:
                        # This is a trace, skip it as it will be recognized somewhere else:
                        pass
            elif ask(Q.diagonal(arg_with_ind.element)):
                if arg_with_ind.indices == [None, None]:
                    continue
                elif None in arg_with_ind.indices:
                    pass
                elif arg_with_ind.indices[0] == arg_with_ind.indices[1]:
                    ind = arg_with_ind.indices[0]
                    counted = editor.count_args_with_index(ind)
                    if counted == 3:
                        # A_ai B_bi D_ii ==> A_ai D_ij B_bj
                        ind_new = editor.get_new_contraction_index()
                        other_args = [j for j in editor.args_with_ind if j != arg_with_ind]
                        other_args[1].indices = [ind_new if j == ind else j for j in other_args[1].indices]
                        arg_with_ind.indices = [ind, ind_new]
                        flag = True
                        break

    return editor.to_array_contraction()


def remove_identity_matrices(expr: ArrayContraction):
    editor = _EditArrayContraction(expr)
    removed: list[int] = []

    permutation_map = {}

    free_indices = list(accumulate([0] + [sum(i is None for i in arg.indices) for arg in editor.args_with_ind]))
    free_map = dict(zip(editor.args_with_ind, free_indices[:-1]))

    update_pairs = {}

    for ind in range(editor.number_of_contraction_indices):
        args = editor.get_args_with_index(ind)
        identity_matrices = [i for i in args if isinstance(i.element, Identity)]
        number_identity_matrices = len(identity_matrices)
        # If the contraction involves a non-identity matrix and multiple identity matrices:
        if number_identity_matrices != len(args) - 1 or number_identity_matrices == 0:
            continue
        # Get the non-identity element:
        non_identity = [i for i in args if not isinstance(i.element, Identity)][0]
        # Check that all identity matrices have at least one free index
        # (otherwise they would be contractions to some other elements)
        if any(None not in i.indices for i in identity_matrices):
            continue
        # Mark the identity matrices for removal:
        for i in identity_matrices:
            i.element = None
            removed.extend(range(free_map[i], free_map[i] + len([j for j in i.indices if j is None])))
        last_removed = removed.pop(-1)
        update_pairs[last_removed, ind] = non_identity.indices[:]
        # Remove the indices from the non-identity matrix, as the contraction
        # no longer exists:
        non_identity.indices = [None if i == ind else i for i in non_identity.indices]

    removed.sort()

    shifts = list(accumulate([1 if i in removed else 0 for i in range(get_rank(expr))]))
    for (last_removed, ind), non_identity_indices in update_pairs.items():
        pos = [free_map[non_identity] + i for i, e in enumerate(non_identity_indices) if e == ind]
        assert len(pos) == 1
        for j in pos:
            permutation_map[j] = last_removed

    editor.args_with_ind = [i for i in editor.args_with_ind if i.element is not None]
    ret_expr = editor.to_array_contraction()
    permutation = []
    counter = 0
    counter2 = 0
    for j in range(get_rank(expr)):
        if j in removed:
            continue
        if counter2 in permutation_map:
            target = permutation_map[counter2]
            permutation.append(target - shifts[target])
            counter2 += 1
        else:
            while counter in permutation_map.values():
                counter += 1
            permutation.append(counter)
            counter += 1
            counter2 += 1
    ret_expr2 = _permute_dims(ret_expr, _af_invert(permutation))
    return ret_expr2, removed


def _combine_removed(dim: int, removed1: list[int], removed2: list[int]) -> list[int]:
    # Concatenate two axis removal operations as performed by
    # _remove_trivial_dims,
    removed1 = sorted(removed1)
    removed2 = sorted(removed2)
    i = 0
    j = 0
    removed = []
    while True:
        if j >= len(removed2):
            while i < len(removed1):
                removed.append(removed1[i])
                i += 1
            break
        elif i < len(removed1) and removed1[i] <= i + removed2[j]:
            removed.append(removed1[i])
            i += 1
        else:
            removed.append(i + removed2[j])
            j += 1
    return removed


def _array_contraction_to_diagonal_multiple_identity(expr: ArrayContraction):
    editor = _EditArrayContraction(expr)
    editor.track_permutation_start()
    removed: list[int] = []
    diag_index_counter: int = 0
    for i in range(editor.number_of_contraction_indices):
        identities = []
        args = []
        for j, arg in enumerate(editor.args_with_ind):
            if i not in arg.indices:
                continue
            if isinstance(arg.element, Identity):
                identities.append(arg)
            else:
                args.append(arg)
        if len(identities) == 0:
            continue
        if len(args) + len(identities) < 3:
            continue
        new_diag_ind = -1 - diag_index_counter
        diag_index_counter += 1
        # Variable "flag" to control whether to skip this contraction set:
        flag: bool = True
        for i1, id1 in enumerate(identities):
            if None not in id1.indices:
                flag = True
                break
            free_pos = list(range(*editor.get_absolute_free_range(id1)))[0]
            editor._track_permutation[-1].append(free_pos) # type: ignore
            id1.element = None
            flag = False
            break
        if flag:
            continue
        for arg in identities[:i1] + identities[i1+1:]:
            arg.element = None
            removed.extend(range(*editor.get_absolute_free_range(arg)))
        for arg in args:
            arg.indices = [new_diag_ind if j == i else j for j in arg.indices]
    for j, e in enumerate(editor.args_with_ind):
        if e.element is None:
            editor._track_permutation[j] = None # type: ignore
    editor._track_permutation = [i for i in editor._track_permutation if i is not None] # type: ignore
    # Renumber permutation array form in order to deal with deleted positions:
    remap = {e: i for i, e in enumerate(sorted({k for j in editor._track_permutation for k in j}))}
    editor._track_permutation = [[remap[j] for j in i] for i in editor._track_permutation]
    editor.args_with_ind = [i for i in editor.args_with_ind if i.element is not None]
    new_expr = editor.to_array_contraction()
    return new_expr, removed