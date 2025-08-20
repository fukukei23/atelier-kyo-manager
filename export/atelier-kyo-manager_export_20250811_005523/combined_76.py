
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\functions\rszeta.py ===
"""
---------------------------------------------------------------------
.. sectionauthor:: Juan Arias de Reyna <arias@us.es>

This module implements zeta-related functions using the Riemann-Siegel
expansion: zeta_offline(s,k=0)

* coef(J, eps): Need in the computation of Rzeta(s,k)

* Rzeta_simul(s, der=0) computes Rzeta^(k)(s) and Rzeta^(k)(1-s) simultaneously
  for  0 <= k <= der. Used by zeta_offline and z_offline

* Rzeta_set(s, derivatives) computes Rzeta^(k)(s) for given derivatives, used by
  z_half(t,k) and zeta_half

* z_offline(w,k): Z(w) and its derivatives of order k <= 4
* z_half(t,k): Z(t) (Riemann Siegel function) and its derivatives of order k <= 4
* zeta_offline(s): zeta(s) and its derivatives of order k<= 4
* zeta_half(1/2+it,k):  zeta(s)  and its derivatives of order k<= 4

* rs_zeta(s,k=0) Computes zeta^(k)(s)   Unifies zeta_half and zeta_offline
* rs_z(w,k=0)    Computes Z^(k)(w)      Unifies z_offline and z_half
----------------------------------------------------------------------

This program uses Riemann-Siegel expansion even to compute
zeta(s) on points s = sigma + i t  with sigma arbitrary not
necessarily equal to 1/2.

It is founded on a new deduction of the formula, with rigorous
and sharp bounds for the  terms and rest of this expansion.

More information on the papers:

 J. Arias de Reyna, High Precision Computation of Riemann's
 Zeta Function by the Riemann-Siegel Formula I, II

 We refer to them as I, II.

 In them we shall find detailed explanation of all the
 procedure.

The program uses Riemann-Siegel expansion.
This  is useful when t is big, ( say  t > 10000 ).
The precision is limited, roughly it can compute zeta(sigma+it)
with an error less than exp(-c t) for some constant c depending
on sigma.  The program gives an error when the Riemann-Siegel
formula can not compute to the wanted precision.

"""

import math

class RSCache(object):
    def __init__(ctx):
        ctx._rs_cache = [0, 10, {}, {}]

from .functions import defun

#-------------------------------------------------------------------------------#
#                                                                               #
#                       coef(ctx, J, eps, _cache=[0, 10, {} ] )                 #
#                                                                               #
#-------------------------------------------------------------------------------#

#  This function computes the coefficients c[n] defined on (I, equation (47))
#  but see also  (II, section 3.14).
#
#  Since these coefficients are very difficult to compute we save the values
#  in a cache. So if we compute several values of the functions Rzeta(s) for
#  near values of s, we do not recompute these coefficients.
#
#  c[n] are the Taylor coefficients of the function:
#
#  F(z):= (exp(pi*j*(z*z/2+3/8))-j* sqrt(2) cos(pi*z/2))/(2*cos(pi *z))
#
#

def _coef(ctx, J, eps):
    r"""
    Computes the coefficients  `c_n`  for `0\le n\le 2J` with error less than eps

    **Definition**

    The coefficients c_n are defined by

    .. math ::

        \begin{equation}
        F(z)=\frac{e^{\pi i
        \bigl(\frac{z^2}{2}+\frac38\bigr)}-i\sqrt{2}\cos\frac{\pi}{2}z}{2\cos\pi
        z}=\sum_{n=0}^\infty c_{2n} z^{2n}
        \end{equation}

    they are computed applying the relation

    .. math ::

        \begin{multline}
        c_{2n}=-\frac{i}{\sqrt{2}}\Bigl(\frac{\pi}{2}\Bigr)^{2n}
        \sum_{k=0}^n\frac{(-1)^k}{(2k)!}
        2^{2n-2k}\frac{(-1)^{n-k}E_{2n-2k}}{(2n-2k)!}+\\
        +e^{3\pi i/8}\sum_{j=0}^n(-1)^j\frac{
        E_{2j}}{(2j)!}\frac{i^{n-j}\pi^{n+j}}{(n-j)!2^{n-j+1}}.
        \end{multline}
    """

    newJ = J+2        # compute more coefficients that are needed
    neweps6 = eps/2.  # compute with a slight more precision that are needed

    #  PREPARATION FOR THE COMPUTATION OF V(N) AND W(N)
    #    See II Section 3.16
    #
    #  Computing the exponent wpvw of the error II equation (81)
    wpvw = max(ctx.mag(10*(newJ+3)), 4*newJ+5-ctx.mag(neweps6))

    #  Preparation of Euler numbers (we need until the 2*RS_NEWJ)
    E = ctx._eulernum(2*newJ)

    #  Now we have in the cache all the needed Euler numbers.
    #
    #  Computing the powers of pi
    #
    # We need to compute the powers pi**n for 1<= n <= 2*J
    # with relative error less than 2**(-wpvw)
    # it is easy to show that this is obtained
    # taking wppi as the least d with
    # 2**d>40*J and 2**d> 4.24 *newJ + 2**wpvw
    # In II Section 3.9 we need also that
    #  wppi > wptcoef[0], and that the powers
    # here computed  0<= k <= 2*newJ are more
    # than those needed there that are 2*L-2.
    # so we need  J >= L this will be checked
    # before computing tcoef[]
    wppi = max(ctx.mag(40*newJ), ctx.mag(newJ)+3 +wpvw)
    ctx.prec = wppi
    pipower = {}
    pipower[0] = ctx.one
    pipower[1] = ctx.pi
    for n in range(2,2*newJ+1):
        pipower[n] = pipower[n-1]*ctx.pi

    # COMPUTING THE COEFFICIENTS v(n) AND w(n)
    #  see II equation (61) and equations (81) and (82)
    ctx.prec = wpvw+2
    v={}
    w={}
    for n in range(0,newJ+1):
        va = (-1)**n * ctx._eulernum(2*n)
        va = ctx.mpf(va)/ctx.fac(2*n)
        v[n]=va*pipower[2*n]
    for n in range(0,2*newJ+1):
        wa = ctx.one/ctx.fac(n)
        wa=wa/(2**n)
        w[n]=wa*pipower[n]

    # COMPUTATION OF THE CONVOLUTIONS RS_P1 AND RS_P2
    #  See II Section 3.16
    ctx.prec = 15
    wpp1a = 9 - ctx.mag(neweps6)
    P1 = {}
    for n in range(0,newJ+1):
        ctx.prec = 15
        wpp1 = max(ctx.mag(10*(n+4)),4*n+wpp1a)
        ctx.prec = wpp1
        sump = 0
        for k in range(0,n+1):
            sump += ((-1)**k) * v[k]*w[2*n-2*k]
        P1[n]=((-1)**(n+1))*ctx.j*sump
    P2={}
    for n in range(0,newJ+1):
        ctx.prec = 15
        wpp2 = max(ctx.mag(10*(n+4)),4*n+wpp1a)
        ctx.prec = wpp2
        sump = 0
        for k in range(0,n+1):
            sump += (ctx.j**(n-k)) * v[k]*w[n-k]
        P2[n]=sump
    # COMPUTING THE COEFFICIENTS c[2n]
    # See II Section 3.14
    ctx.prec = 15
    wpc0 = 5 - ctx.mag(neweps6)
    wpc = max(6,4*newJ+wpc0)
    ctx.prec = wpc
    mu = ctx.sqrt(ctx.mpf('2'))/2
    nu = ctx.expjpi(3./8)/2
    c={}
    for n in range(0,newJ):
        ctx.prec = 15
        wpc = max(6,4*n+wpc0)
        ctx.prec = wpc
        c[2*n] = mu*P1[n]+nu*P2[n]
    for n in range(1,2*newJ,2):
        c[n] = 0
    return [newJ, neweps6, c, pipower]

def coef(ctx, J, eps):
    _cache = ctx._rs_cache
    if J <= _cache[0] and eps >= _cache[1]:
        return _cache[2], _cache[3]
    orig = ctx._mp.prec
    try:
        data = _coef(ctx._mp, J, eps)
    finally:
        ctx._mp.prec = orig
    if ctx is not ctx._mp:
        data[2] = dict((k,ctx.convert(v)) for (k,v) in data[2].items())
        data[3] = dict((k,ctx.convert(v)) for (k,v) in data[3].items())
    ctx._rs_cache[:] = data
    return ctx._rs_cache[2], ctx._rs_cache[3]

#-------------------------------------------------------------------------------#
#                                                                               #
#                          Rzeta_simul(s,k=0)                                   #
#                                                                               #
#-------------------------------------------------------------------------------#
#  This function return a list with the values:
#  Rzeta(sigma+it), conj(Rzeta(1-sigma+it)),Rzeta'(sigma+it), conj(Rzeta'(1-sigma+it)),
#  .... , Rzeta^{(k)}(sigma+it), conj(Rzeta^{(k)}(1-sigma+it))
#
#  Useful to compute  the function zeta(s) and Z(w)  or its derivatives.
#

def aux_M_Fp(ctx, xA, xeps4, a, xB1, xL):
    # COMPUTING M  NUMBER OF DERIVATIVES Fp[m] TO COMPUTE
    #  See II Section 3.11  equations (47) and (48)
    aux1 = 126.0657606*xA/xeps4   # 126.06.. = 316/sqrt(2*pi)
    aux1 = ctx.ln(aux1)
    aux2 = (2*ctx.ln(ctx.pi)+ctx.ln(xB1)+ctx.ln(a))/3 -ctx.ln(2*ctx.pi)/2
    m = 3*xL-3
    aux3= (ctx.loggamma(m+1)-ctx.loggamma(m/3.0+2))/2 -ctx.loggamma((m+1)/2.)
    while((aux1 < m*aux2+ aux3)and (m>1)):
        m = m - 1
        aux3 = (ctx.loggamma(m+1)-ctx.loggamma(m/3.0+2))/2 -ctx.loggamma((m+1)/2.)
    xM = m
    return xM

def aux_J_needed(ctx, xA, xeps4, a, xB1, xM):
    #  DETERMINATION OF  J  THE NUMBER OF TERMS NEEDED
    #            IN THE TAYLOR SERIES OF F.
    #  See II Section 3.11 equation (49))
    #  Only determine one
    h1 = xeps4/(632*xA)
    h2 = xB1*a * 126.31337419529260248  # = pi^2*e^2*sqrt(3)
    h2 = h1 * ctx.power((h2/xM**2),(xM-1)/3) / xM
    h3 = min(h1,h2)
    return h3

def Rzeta_simul(ctx, s, der=0):
    # First we take the value of ctx.prec
    wpinitial = ctx.prec

    # INITIALIZATION
    # Take the real and imaginary part of s
    t = ctx._im(s)
    xsigma = ctx._re(s)
    ysigma = 1 - xsigma

    # Now compute several parameter that appear on the program
    ctx.prec = 15
    a = ctx.sqrt(t/(2*ctx.pi))
    xasigma = a ** xsigma
    yasigma = a ** ysigma

    # We need a simple bound A1 < asigma  (see II Section 3.1 and 3.3)
    xA1=ctx.power(2, ctx.mag(xasigma)-1)
    yA1=ctx.power(2, ctx.mag(yasigma)-1)

    # We compute various epsilon's  (see II end of Section 3.1)
    eps = ctx.power(2, -wpinitial)
    eps1 = eps/6.
    xeps2 = eps * xA1/3.
    yeps2 = eps * yA1/3.

    #  COMPUTING SOME COEFFICIENTS THAT DEPENDS
    #                ON  sigma
    #  constant b and c  (see I  Theorem 2 formula (26) )
    #  coefficients A and B1  (see I Section 6.1 equation (50))
    #
    # here we not need high precision
    ctx.prec = 15
    if xsigma > 0:
        xb = 2.
        xc = math.pow(9,xsigma)/4.44288
        # 4.44288 =(math.sqrt(2)*math.pi)
        xA = math.pow(9,xsigma)
        xB1 = 1
    else:
        xb = 2.25158  #  math.sqrt( (3-2* math.log(2))*math.pi )
        xc = math.pow(2,-xsigma)/4.44288
        xA = math.pow(2,-xsigma)
        xB1 = 1.10789   #  = 2*sqrt(1-log(2))

    if(ysigma > 0):
        yb = 2.
        yc = math.pow(9,ysigma)/4.44288
        # 4.44288 =(math.sqrt(2)*math.pi)
        yA = math.pow(9,ysigma)
        yB1 = 1
    else:
        yb = 2.25158  #  math.sqrt( (3-2* math.log(2))*math.pi )
        yc = math.pow(2,-ysigma)/4.44288
        yA = math.pow(2,-ysigma)
        yB1 = 1.10789   #  = 2*sqrt(1-log(2))

    #  COMPUTING L THE NUMBER OF TERMS NEEDED IN THE RIEMANN-SIEGEL
    #                         CORRECTION
    #  See II Section 3.2
    ctx.prec = 15
    xL = 1
    while 3*xc*ctx.gamma(xL*0.5) * ctx.power(xb*a,-xL) >= xeps2:
        xL = xL+1
    xL = max(2,xL)
    yL = 1
    while 3*yc*ctx.gamma(yL*0.5) * ctx.power(yb*a,-yL) >= yeps2:
        yL = yL+1
    yL = max(2,yL)

    #  The number L has to satify some conditions.
    #  If not RS can not compute Rzeta(s) with the prescribed precision
    #  (see II, Section 3.2 condition (20)  ) and
    #  (II, Section 3.3 condition (22) ). Also we have added
    #  an additional technical  condition in Section 3.17 Proposition 17
    if ((3*xL >= 2*a*a/25.) or (3*xL+2+xsigma<0) or (abs(xsigma) > a/2.) or \
        (3*yL >= 2*a*a/25.) or (3*yL+2+ysigma<0) or (abs(ysigma) > a/2.)):
        ctx.prec = wpinitial
        raise NotImplementedError("Riemann-Siegel can not compute with such precision")

    #  We take the maximum of the two values
    L = max(xL, yL)

    #  INITIALIZATION (CONTINUATION)
    #
    # eps3 is the constant defined on (II, Section 3.5 equation (27) )
    # each term of the RS correction must be computed with error <= eps3
    xeps3 =  xeps2/(4*xL)
    yeps3 =  yeps2/(4*yL)

    # eps4 is defined on (II Section 3.6  equation (30) )
    # each component of the formula (II Section 3.6 equation (29) )
    # must be computed with error <= eps4
    xeps4 = xeps3/(3*xL)
    yeps4 = yeps3/(3*yL)

    # COMPUTING M NUMBER OF DERIVATIVES Fp[m] TO COMPUTE
    xM = aux_M_Fp(ctx, xA, xeps4, a, xB1, xL)
    yM = aux_M_Fp(ctx, yA, yeps4, a, yB1, yL)
    M = max(xM, yM)

    # COMPUTING NUMBER OF TERMS J NEEDED
    h3 = aux_J_needed(ctx, xA, xeps4, a, xB1, xM)
    h4 = aux_J_needed(ctx, yA, yeps4, a, yB1, yM)
    h3 = min(h3,h4)
    J = 12
    jvalue = (2*ctx.pi)**J / ctx.gamma(J+1)
    while jvalue > h3:
        J = J+1
        jvalue = (2*ctx.pi)*jvalue/J

    # COMPUTING eps5[m] for 1 <= m <= 21
    #  See II Section 10 equation (43)
    #  We choose the minimum of the two possibilities
    eps5={}
    xforeps5 = math.pi*math.pi*xB1*a
    yforeps5 = math.pi*math.pi*yB1*a
    for m in range(0,22):
        xaux1 = math.pow(xforeps5, m/3)/(316.*xA)
        yaux1 = math.pow(yforeps5, m/3)/(316.*yA)
        aux1 = min(xaux1, yaux1)
        aux2 = ctx.gamma(m+1)/ctx.gamma(m/3.0+0.5)
        aux2 = math.sqrt(aux2)
        eps5[m] = (aux1*aux2*min(xeps4,yeps4))

    # COMPUTING wpfp
    #  See II Section 3.13 equation (59)
    twenty = min(3*L-3, 21)+1
    aux = 6812*J
    wpfp = ctx.mag(44*J)
    for m in range(0,twenty):
        wpfp = max(wpfp, ctx.mag(aux*ctx.gamma(m+1)/eps5[m]))

    # COMPUTING N AND p
    #  See II Section
    ctx.prec = wpfp + ctx.mag(t)+20
    a = ctx.sqrt(t/(2*ctx.pi))
    N = ctx.floor(a)
    p = 1-2*(a-N)

    # now we get a rounded version of p
    # to the precision wpfp
    # this possibly is not necessary
    num=ctx.floor(p*(ctx.mpf('2')**wpfp))
    difference = p * (ctx.mpf('2')**wpfp)-num
    if (difference < 0.5):
        num = num
    else:
        num = num+1
    p = ctx.convert(num * (ctx.mpf('2')**(-wpfp)))

    # COMPUTING THE COEFFICIENTS c[n] = cc[n]
    # We shall use the notation cc[n], since there is
    # a constant that is called c
    # See II Section 3.14
    # We compute the coefficients and also save then in a
    # cache.  The bulk of the computation is passed to
    # the function  coef()
    #
    #  eps6 is defined in II Section 3.13  equation (58)
    eps6 = ctx.power(ctx.convert(2*ctx.pi), J)/(ctx.gamma(J+1)*3*J)

    #  Now we compute the coefficients
    cc = {}
    cont = {}
    cont, pipowers = coef(ctx, J, eps6)
    cc=cont.copy()   # we need a copy since we have to change his values.
    Fp={}            # this is the adequate locus of this
    for n in range(M, 3*L-2):
        Fp[n] = 0
    Fp={}
    ctx.prec = wpfp
    for m in range(0,M+1):
        sumP = 0
        for k in range(2*J-m-1,-1,-1):
            sumP = (sumP * p)+ cc[k]
        Fp[m] = sumP
        # preparation of the new coefficients
        for k in range(0,2*J-m-1):
            cc[k] = (k+1)* cc[k+1]

    # COMPUTING THE NUMBERS  xd[u,n,k], yd[u,n,k]
    #  See II Section 3.17
    #
    #  First we compute the working precisions xwpd[k]
    #   Se II equation (92)
    xwpd={}
    d1 = max(6,ctx.mag(40*L*L))
    xd2 = 13+ctx.mag((1+abs(xsigma))*xA)-ctx.mag(xeps4)-1
    xconst = ctx.ln(8/(ctx.pi*ctx.pi*a*a*xB1*xB1)) /2
    for n in range(0,L):
        xd3 = ctx.mag(ctx.sqrt(ctx.gamma(n-0.5)))-ctx.floor(n*xconst)+xd2
        xwpd[n]=max(xd3,d1)

    # procedure of II Section 3.17
    ctx.prec = xwpd[1]+10
    xpsigma = 1-(2*xsigma)
    xd = {}
    xd[0,0,-2]=0; xd[0,0,-1]=0; xd[0,0,0]=1; xd[0,0,1]=0
    xd[0,-1,-2]=0; xd[0,-1,-1]=0; xd[0,-1,0]=1; xd[0,-1,1]=0
    for n in range(1,L):
        ctx.prec = xwpd[n]+10
        for k in range(0,3*n//2+1):
            m = 3*n-2*k
            if(m!=0):
                m1 = ctx.one/m
                c1= m1/4
                c2=(xpsigma*m1)/2
                c3=-(m+1)
                xd[0,n,k]=c3*xd[0,n-1,k-2]+c1*xd[0,n-1,k]+c2*xd[0,n-1,k-1]
            else:
                xd[0,n,k]=0
                for r in range(0,k):
                    add=xd[0,n,r]*(ctx.mpf('1.0')*ctx.fac(2*k-2*r)/ctx.fac(k-r))
                    xd[0,n,k] -= ((-1)**(k-r))*add
        xd[0,n,-2]=0; xd[0,n,-1]=0; xd[0,n,3*n//2+1]=0
    for mu in range(-2,der+1):
        for n in range(-2,L):
            for k in range(-3,max(1,3*n//2+2)):
                if( (mu<0)or (n<0) or(k<0)or (k>3*n//2)):
                    xd[mu,n,k] = 0
    for mu in range(1,der+1):
        for n in range(0,L):
            ctx.prec = xwpd[n]+10
            for k in range(0,3*n//2+1):
                aux=(2*mu-2)*xd[mu-2,n-2,k-3]+2*(xsigma+n-2)*xd[mu-1,n-2,k-3]
                xd[mu,n,k] = aux - xd[mu-1,n-1,k-1]

    #  Now we compute the working precisions ywpd[k]
    #   Se II equation (92)
    ywpd={}
    d1 = max(6,ctx.mag(40*L*L))
    yd2 = 13+ctx.mag((1+abs(ysigma))*yA)-ctx.mag(yeps4)-1
    yconst = ctx.ln(8/(ctx.pi*ctx.pi*a*a*yB1*yB1)) /2
    for n in range(0,L):
        yd3 = ctx.mag(ctx.sqrt(ctx.gamma(n-0.5)))-ctx.floor(n*yconst)+yd2
        ywpd[n]=max(yd3,d1)

    # procedure of II Section 3.17
    ctx.prec = ywpd[1]+10
    ypsigma = 1-(2*ysigma)
    yd = {}
    yd[0,0,-2]=0; yd[0,0,-1]=0; yd[0,0,0]=1; yd[0,0,1]=0
    yd[0,-1,-2]=0; yd[0,-1,-1]=0; yd[0,-1,0]=1; yd[0,-1,1]=0
    for n in range(1,L):
        ctx.prec = ywpd[n]+10
        for k in range(0,3*n//2+1):
            m = 3*n-2*k
            if(m!=0):
                m1 = ctx.one/m
                c1= m1/4
                c2=(ypsigma*m1)/2
                c3=-(m+1)
                yd[0,n,k]=c3*yd[0,n-1,k-2]+c1*yd[0,n-1,k]+c2*yd[0,n-1,k-1]
            else:
                yd[0,n,k]=0
                for r in range(0,k):
                    add=yd[0,n,r]*(ctx.mpf('1.0')*ctx.fac(2*k-2*r)/ctx.fac(k-r))
                    yd[0,n,k] -= ((-1)**(k-r))*add
        yd[0,n,-2]=0; yd[0,n,-1]=0; yd[0,n,3*n//2+1]=0

    for mu in range(-2,der+1):
        for n in range(-2,L):
            for k in range(-3,max(1,3*n//2+2)):
                if( (mu<0)or (n<0) or(k<0)or (k>3*n//2)):
                    yd[mu,n,k] = 0
    for mu in range(1,der+1):
        for n in range(0,L):
            ctx.prec = ywpd[n]+10
            for k in range(0,3*n//2+1):
                aux=(2*mu-2)*yd[mu-2,n-2,k-3]+2*(ysigma+n-2)*yd[mu-1,n-2,k-3]
                yd[mu,n,k] = aux - yd[mu-1,n-1,k-1]

    # COMPUTING THE COEFFICIENTS xtcoef[k,l]
    #  See II Section 3.9
    #
    # computing the needed wp
    xwptcoef={}
    xwpterm={}
    ctx.prec = 15
    c1 = ctx.mag(40*(L+2))
    xc2 = ctx.mag(68*(L+2)*xA)
    xc4 = ctx.mag(xB1*a*math.sqrt(ctx.pi))-1
    for k in range(0,L):
        xc3 = xc2 - k*xc4+ctx.mag(ctx.fac(k+0.5))/2.
        xwptcoef[k] = (max(c1,xc3-ctx.mag(xeps4)+1)+1 +20)*1.5
        xwpterm[k] = (max(c1,ctx.mag(L+2)+xc3-ctx.mag(xeps3)+1)+1 +20)
    ywptcoef={}
    ywpterm={}
    ctx.prec = 15
    c1 = ctx.mag(40*(L+2))
    yc2 = ctx.mag(68*(L+2)*yA)
    yc4 = ctx.mag(yB1*a*math.sqrt(ctx.pi))-1
    for k in range(0,L):
        yc3 = yc2 - k*yc4+ctx.mag(ctx.fac(k+0.5))/2.
        ywptcoef[k] = ((max(c1,yc3-ctx.mag(yeps4)+1))+10)*1.5
        ywpterm[k] = (max(c1,ctx.mag(L+2)+yc3-ctx.mag(yeps3)+1)+1)+10

    # check of power of pi
    # computing the fortcoef[mu,k,ell]
    xfortcoef={}
    for mu in range(0,der+1):
        for k in range(0,L):
            for ell in range(-2,3*k//2+1):
                xfortcoef[mu,k,ell]=0
    for mu in range(0,der+1):
        for k in range(0,L):
            ctx.prec = xwptcoef[k]
            for ell in range(0,3*k//2+1):
                xfortcoef[mu,k,ell]=xd[mu,k,ell]*Fp[3*k-2*ell]/pipowers[2*k-ell]
                xfortcoef[mu,k,ell]=xfortcoef[mu,k,ell]/((2*ctx.j)**ell)

    def trunc_a(t):
        wp = ctx.prec
        ctx.prec = wp + 2
        aa = ctx.sqrt(t/(2*ctx.pi))
        ctx.prec = wp
        return aa

    # computing the tcoef[k,ell]
    xtcoef={}
    for mu in range(0,der+1):
        for k in range(0,L):
            for ell in range(-2,3*k//2+1):
                xtcoef[mu,k,ell]=0
    ctx.prec = max(xwptcoef[0],ywptcoef[0])+3
    aa= trunc_a(t)
    la = -ctx.ln(aa)

    for chi in range(0,der+1):
        for k in range(0,L):
            ctx.prec = xwptcoef[k]
            for ell in range(0,3*k//2+1):
                xtcoef[chi,k,ell] =0
                for mu in range(0, chi+1):
                    tcoefter=ctx.binomial(chi,mu)*ctx.power(la,mu)*xfortcoef[chi-mu,k,ell]
                    xtcoef[chi,k,ell] += tcoefter

    # COMPUTING THE COEFFICIENTS ytcoef[k,l]
    #  See II Section 3.9
    #
    # computing the needed wp
    # check of power of pi
    # computing the fortcoef[mu,k,ell]
    yfortcoef={}
    for mu in range(0,der+1):
        for k in range(0,L):
            for ell in range(-2,3*k//2+1):
                yfortcoef[mu,k,ell]=0
    for mu in range(0,der+1):
        for k in range(0,L):
            ctx.prec = ywptcoef[k]
            for ell in range(0,3*k//2+1):
                yfortcoef[mu,k,ell]=yd[mu,k,ell]*Fp[3*k-2*ell]/pipowers[2*k-ell]
                yfortcoef[mu,k,ell]=yfortcoef[mu,k,ell]/((2*ctx.j)**ell)
    # computing the tcoef[k,ell]
    ytcoef={}
    for chi in range(0,der+1):
        for k in range(0,L):
            for ell in range(-2,3*k//2+1):
                ytcoef[chi,k,ell]=0
    for chi in range(0,der+1):
        for k in range(0,L):
            ctx.prec = ywptcoef[k]
            for ell in range(0,3*k//2+1):
                ytcoef[chi,k,ell] =0
                for mu in range(0, chi+1):
                    tcoefter=ctx.binomial(chi,mu)*ctx.power(la,mu)*yfortcoef[chi-mu,k,ell]
                    ytcoef[chi,k,ell] += tcoefter

    # COMPUTING tv[k,ell]
    # See II Section 3.8
    #
    #  a has a good value
    ctx.prec = max(xwptcoef[0], ywptcoef[0])+2
    av = {}
    av[0] = 1
    av[1] = av[0]/a

    ctx.prec = max(xwptcoef[0],ywptcoef[0])
    for k in range(2,L):
        av[k] = av[k-1] * av[1]

    # Computing the quotients
    xtv = {}
    for chi in range(0,der+1):
        for k in range(0,L):
            ctx.prec = xwptcoef[k]
            for ell in range(0,3*k//2+1):
                xtv[chi,k,ell] = xtcoef[chi,k,ell]* av[k]
    # Computing the quotients
    ytv = {}
    for chi in range(0,der+1):
        for k in range(0,L):
            ctx.prec = ywptcoef[k]
            for ell in range(0,3*k//2+1):
                ytv[chi,k,ell] = ytcoef[chi,k,ell]* av[k]

    # COMPUTING THE TERMS xterm[k]
    # See II Section 3.6
    xterm = {}
    for chi in range(0,der+1):
        for n in range(0,L):
            ctx.prec = xwpterm[n]
            te = 0
            for k in range(0, 3*n//2+1):
                te += xtv[chi,n,k]
            xterm[chi,n] = te

    # COMPUTING THE TERMS yterm[k]
    # See II Section 3.6
    yterm = {}
    for chi in range(0,der+1):
        for n in range(0,L):
            ctx.prec = ywpterm[n]
            te = 0
            for k in range(0, 3*n//2+1):
                te += ytv[chi,n,k]
            yterm[chi,n] = te

    # COMPUTING  rssum
    # See II Section 3.5
    xrssum={}
    ctx.prec=15
    xrsbound = math.sqrt(ctx.pi) * xc /(xb*a)
    ctx.prec=15
    xwprssum = ctx.mag(4.4*((L+3)**2)*xrsbound / xeps2)
    xwprssum = max(xwprssum, ctx.mag(10*(L+1)))
    ctx.prec = xwprssum
    for chi in range(0,der+1):
        xrssum[chi] = 0
        for k in range(1,L+1):
            xrssum[chi] += xterm[chi,L-k]
    yrssum={}
    ctx.prec=15
    yrsbound = math.sqrt(ctx.pi) * yc /(yb*a)
    ctx.prec=15
    ywprssum = ctx.mag(4.4*((L+3)**2)*yrsbound / yeps2)
    ywprssum = max(ywprssum, ctx.mag(10*(L+1)))
    ctx.prec = ywprssum
    for chi in range(0,der+1):
        yrssum[chi] = 0
        for k in range(1,L+1):
            yrssum[chi] += yterm[chi,L-k]

    # COMPUTING S3
    # See II Section 3.19
    ctx.prec = 15
    A2 = 2**(max(ctx.mag(abs(xrssum[0])), ctx.mag(abs(yrssum[0]))))
    eps8 = eps/(3*A2)
    T = t *ctx.ln(t/(2*ctx.pi))
    xwps3 = 5 +  ctx.mag((1+(2/eps8)*ctx.power(a,-xsigma))*T)
    ywps3 = 5 +  ctx.mag((1+(2/eps8)*ctx.power(a,-ysigma))*T)

    ctx.prec = max(xwps3, ywps3)

    tpi = t/(2*ctx.pi)
    arg = (t/2)*ctx.ln(tpi)-(t/2)-ctx.pi/8
    U = ctx.expj(-arg)
    a = trunc_a(t)
    xasigma = ctx.power(a, -xsigma)
    yasigma = ctx.power(a, -ysigma)
    xS3 = ((-1)**(N-1)) * xasigma * U
    yS3 = ((-1)**(N-1)) * yasigma * U

    # COMPUTING S1 the zetasum
    # See II Section 3.18
    ctx.prec = 15
    xwpsum =  4+ ctx.mag((N+ctx.power(N,1-xsigma))*ctx.ln(N) /eps1)
    ywpsum =  4+ ctx.mag((N+ctx.power(N,1-ysigma))*ctx.ln(N) /eps1)
    wpsum = max(xwpsum, ywpsum)

    ctx.prec = wpsum +10
    '''
    # This can be improved
    xS1={}
    yS1={}
    for chi in range(0,der+1):
        xS1[chi] = 0
        yS1[chi] = 0
    for n in range(1,int(N)+1):
        ln = ctx.ln(n)
        xexpn = ctx.exp(-ln*(xsigma+ctx.j*t))
        yexpn = ctx.conj(1/(n*xexpn))
        for chi in range(0,der+1):
            pown = ctx.power(-ln, chi)
            xterm = pown*xexpn
            yterm = pown*yexpn
            xS1[chi] += xterm
            yS1[chi] += yterm
    '''
    xS1, yS1 = ctx._zetasum(s, 1, int(N)-1, range(0,der+1), True)

    # END OF COMPUTATION of xrz, yrz
    #  See II Section 3.1
    ctx.prec = 15
    xabsS1 = abs(xS1[der])
    xabsS2 = abs(xrssum[der] * xS3)
    xwpend = max(6, wpinitial+ctx.mag(6*(3*xabsS1+7*xabsS2) ) )

    ctx.prec = xwpend
    xrz={}
    for chi in range(0,der+1):
        xrz[chi] = xS1[chi]+xrssum[chi]*xS3

    ctx.prec = 15
    yabsS1 = abs(yS1[der])
    yabsS2 = abs(yrssum[der] * yS3)
    ywpend = max(6, wpinitial+ctx.mag(6*(3*yabsS1+7*yabsS2) ) )

    ctx.prec = ywpend
    yrz={}
    for chi in range(0,der+1):
        yrz[chi] = yS1[chi]+yrssum[chi]*yS3
        yrz[chi] = ctx.conj(yrz[chi])
    ctx.prec = wpinitial
    return xrz, yrz

def Rzeta_set(ctx, s, derivatives=[0]):
    r"""
    Computes several derivatives of the auxiliary function of Riemann `R(s)`.

    **Definition**

    The function is defined by

    .. math ::

        \begin{equation}
        {\mathop{\mathcal R }\nolimits}(s)=
        \int_{0\swarrow1}\frac{x^{-s} e^{\pi i x^2}}{e^{\pi i x}-
        e^{-\pi i x}}\,dx
        \end{equation}

    To this function we apply the Riemann-Siegel expansion.
    """
    der = max(derivatives)
    # First we take the value of ctx.prec
    # During the computation we will change ctx.prec, and finally we will
    # restaurate the initial value
    wpinitial = ctx.prec
    # Take the real and imaginary part of s
    t = ctx._im(s)
    sigma = ctx._re(s)
    # Now compute several parameter that appear on the program
    ctx.prec = 15
    a = ctx.sqrt(t/(2*ctx.pi))     #  Careful
    asigma = ctx.power(a, sigma)  #  Careful
    # We need a simple bound A1 < asigma  (see II Section 3.1 and 3.3)
    A1 = ctx.power(2, ctx.mag(asigma)-1)
    # We compute various epsilon's  (see II end of Section 3.1)
    eps = ctx.power(2, -wpinitial)
    eps1 = eps/6.
    eps2 = eps * A1/3.
    # COMPUTING SOME COEFFICIENTS THAT DEPENDS
    #               ON  sigma
    # constant b and c  (see I  Theorem 2 formula (26) )
    # coefficients A and B1  (see I Section 6.1 equation (50))
    # here we not need high precision
    ctx.prec = 15
    if sigma > 0:
        b = 2.
        c = math.pow(9,sigma)/4.44288
        # 4.44288 =(math.sqrt(2)*math.pi)
        A = math.pow(9,sigma)
        B1 = 1
    else:
        b = 2.25158  #  math.sqrt( (3-2* math.log(2))*math.pi )
        c = math.pow(2,-sigma)/4.44288
        A = math.pow(2,-sigma)
        B1 = 1.10789   #  = 2*sqrt(1-log(2))
    #  COMPUTING L THE NUMBER OF TERMS NEEDED IN THE RIEMANN-SIEGEL
    #                         CORRECTION
    #  See II Section 3.2
    ctx.prec = 15
    L = 1
    while 3*c*ctx.gamma(L*0.5) * ctx.power(b*a,-L) >= eps2:
        L = L+1
    L = max(2,L)
    #  The number L has to satify some conditions.
    #  If not RS can not compute Rzeta(s) with the prescribed precision
    #  (see II, Section 3.2 condition (20)  ) and
    #  (II, Section 3.3 condition (22) ). Also we have added
    #  an additional technical  condition in Section 3.17 Proposition 17
    if ((3*L >= 2*a*a/25.) or (3*L+2+sigma<0) or (abs(sigma)> a/2.)):
        #print 'Error Riemann-Siegel can not compute with such precision'
        ctx.prec = wpinitial
        raise NotImplementedError("Riemann-Siegel can not compute with such precision")

    #  INITIALIZATION (CONTINUATION)
    #
    # eps3 is the constant defined on (II, Section 3.5 equation (27) )
    # each term of the RS correction must be computed with error <= eps3
    eps3 =  eps2/(4*L)

    # eps4 is defined on (II Section 3.6  equation (30) )
    # each component of the formula (II Section 3.6 equation (29) )
    # must be computed with error <= eps4
    eps4 = eps3/(3*L)

    # COMPUTING M.  NUMBER OF DERIVATIVES Fp[m] TO COMPUTE
    M = aux_M_Fp(ctx, A, eps4, a, B1, L)
    Fp = {}
    for n in range(M, 3*L-2):
        Fp[n] = 0

    #  But I have not seen an instance of  M != 3*L-3
    #
    #  DETERMINATION OF  J  THE NUMBER OF TERMS NEEDED
    #            IN THE TAYLOR SERIES OF F.
    #  See II Section 3.11 equation (49))
    h1 = eps4/(632*A)
    h2 = ctx.pi*ctx.pi*B1*a *ctx.sqrt(3)*math.e*math.e
    h2 = h1 * ctx.power((h2/M**2),(M-1)/3) / M
    h3 = min(h1,h2)
    J=12
    jvalue = (2*ctx.pi)**J / ctx.gamma(J+1)
    while jvalue > h3:
        J = J+1
        jvalue = (2*ctx.pi)*jvalue/J

    # COMPUTING eps5[m] for 1 <= m <= 21
    #  See II Section 10 equation (43)
    eps5={}
    foreps5 = math.pi*math.pi*B1*a
    for m in range(0,22):
        aux1 = math.pow(foreps5, m/3)/(316.*A)
        aux2 = ctx.gamma(m+1)/ctx.gamma(m/3.0+0.5)
        aux2 = math.sqrt(aux2)
        eps5[m] = aux1*aux2*eps4

    # COMPUTING wpfp
    #  See II Section 3.13 equation (59)
    twenty = min(3*L-3, 21)+1
    aux = 6812*J
    wpfp = ctx.mag(44*J)
    for m in range(0, twenty):
        wpfp = max(wpfp, ctx.mag(aux*ctx.gamma(m+1)/eps5[m]))
    # COMPUTING N AND p
    #  See II Section
    ctx.prec = wpfp + ctx.mag(t) + 20
    a = ctx.sqrt(t/(2*ctx.pi))
    N = ctx.floor(a)
    p = 1-2*(a-N)

    # now we get a rounded version of p to the precision wpfp
    # this possibly is not necessary
    num = ctx.floor(p*(ctx.mpf(2)**wpfp))
    difference = p * (ctx.mpf(2)**wpfp)-num
    if difference < 0.5:
        num = num
    else:
        num = num+1
    p = ctx.convert(num * (ctx.mpf(2)**(-wpfp)))

    # COMPUTING THE COEFFICIENTS c[n] = cc[n]
    # We shall use the notation cc[n], since there is
    # a constant that is called c
    # See II Section 3.14
    # We compute the coefficients and also save then in a
    # cache.  The bulk of the computation is passed to
    # the function  coef()
    #
    #  eps6 is defined in II Section 3.13  equation (58)
    eps6 = ctx.power(2*ctx.pi, J)/(ctx.gamma(J+1)*3*J)

    #  Now we compute the coefficients
    cc={}
    cont={}
    cont, pipowers = coef(ctx, J, eps6)
    cc = cont.copy()   # we need a copy since we have
    Fp={}
    for n in range(M, 3*L-2):
        Fp[n] = 0
    ctx.prec = wpfp
    for m in range(0,M+1):
        sumP = 0
        for k in range(2*J-m-1,-1,-1):
            sumP = (sumP * p) + cc[k]
        Fp[m] = sumP
        # preparation of the new coefficients
        for k in range(0, 2*J-m-1):
            cc[k] = (k+1) * cc[k+1]

    # COMPUTING THE NUMBERS  d[n,k]
    #  See II Section 3.17

    #  First we compute the working precisions wpd[k]
    #   Se II equation (92)
    wpd = {}
    d1 = max(6, ctx.mag(40*L*L))
    d2 = 13+ctx.mag((1+abs(sigma))*A)-ctx.mag(eps4)-1
    const = ctx.ln(8/(ctx.pi*ctx.pi*a*a*B1*B1)) /2
    for n in range(0,L):
        d3 = ctx.mag(ctx.sqrt(ctx.gamma(n-0.5)))-ctx.floor(n*const)+d2
        wpd[n] = max(d3,d1)

    # procedure of II Section 3.17
    ctx.prec = wpd[1]+10
    psigma = 1-(2*sigma)
    d = {}
    d[0,0,-2]=0; d[0,0,-1]=0; d[0,0,0]=1; d[0,0,1]=0
    d[0,-1,-2]=0; d[0,-1,-1]=0; d[0,-1,0]=1; d[0,-1,1]=0
    for n in range(1,L):
        ctx.prec = wpd[n]+10
        for k in range(0,3*n//2+1):
            m = 3*n-2*k
            if (m!=0):
                m1 = ctx.one/m
                c1 = m1/4
                c2 = (psigma*m1)/2
                c3 = -(m+1)
                d[0,n,k] = c3*d[0,n-1,k-2]+c1*d[0,n-1,k]+c2*d[0,n-1,k-1]
            else:
                d[0,n,k]=0
                for r in range(0,k):
                    add = d[0,n,r]*(ctx.one*ctx.fac(2*k-2*r)/ctx.fac(k-r))
                    d[0,n,k] -= ((-1)**(k-r))*add
        d[0,n,-2]=0; d[0,n,-1]=0; d[0,n,3*n//2+1]=0

    for mu in range(-2,der+1):
        for n in range(-2,L):
            for k in range(-3,max(1,3*n//2+2)):
                if ((mu<0)or (n<0) or(k<0)or (k>3*n//2)):
                    d[mu,n,k] = 0

    for mu in range(1,der+1):
        for n in range(0,L):
            ctx.prec = wpd[n]+10
            for k in range(0,3*n//2+1):
                aux=(2*mu-2)*d[mu-2,n-2,k-3]+2*(sigma+n-2)*d[mu-1,n-2,k-3]
                d[mu,n,k] = aux - d[mu-1,n-1,k-1]

    # COMPUTING THE COEFFICIENTS t[k,l]
    #  See II Section 3.9
    #
    # computing the needed wp
    wptcoef = {}
    wpterm = {}
    ctx.prec = 15
    c1 = ctx.mag(40*(L+2))
    c2 = ctx.mag(68*(L+2)*A)
    c4 = ctx.mag(B1*a*math.sqrt(ctx.pi))-1
    for k in range(0,L):
        c3 = c2 - k*c4+ctx.mag(ctx.fac(k+0.5))/2.
        wptcoef[k] = max(c1,c3-ctx.mag(eps4)+1)+1 +10
        wpterm[k] = max(c1,ctx.mag(L+2)+c3-ctx.mag(eps3)+1)+1 +10

    # check of power of pi

    # computing the fortcoef[mu,k,ell]
    fortcoef={}
    for mu in derivatives:
        for k in range(0,L):
            for ell in range(-2,3*k//2+1):
                fortcoef[mu,k,ell]=0

    for mu in derivatives:
        for k in range(0,L):
            ctx.prec = wptcoef[k]
            for ell in range(0,3*k//2+1):
                fortcoef[mu,k,ell]=d[mu,k,ell]*Fp[3*k-2*ell]/pipowers[2*k-ell]
                fortcoef[mu,k,ell]=fortcoef[mu,k,ell]/((2*ctx.j)**ell)

    def trunc_a(t):
        wp = ctx.prec
        ctx.prec = wp + 2
        aa = ctx.sqrt(t/(2*ctx.pi))
        ctx.prec = wp
        return aa

    # computing the tcoef[chi,k,ell]
    tcoef={}
    for chi in derivatives:
        for k in range(0,L):
            for ell in range(-2,3*k//2+1):
                tcoef[chi,k,ell]=0
    ctx.prec = wptcoef[0]+3
    aa = trunc_a(t)
    la = -ctx.ln(aa)

    for chi in derivatives:
        for k in range(0,L):
            ctx.prec = wptcoef[k]
            for ell in range(0,3*k//2+1):
                tcoef[chi,k,ell] = 0
                for mu in range(0, chi+1):
                    tcoefter = ctx.binomial(chi,mu) * la**mu * \
                        fortcoef[chi-mu,k,ell]
                    tcoef[chi,k,ell] += tcoefter

    # COMPUTING tv[k,ell]
    # See II Section 3.8

    # Computing the powers av[k] = a**(-k)
    ctx.prec = wptcoef[0] + 2

    # a has a good value of a.
    # See II Section 3.6
    av = {}
    av[0] = 1
    av[1] = av[0]/a

    ctx.prec = wptcoef[0]
    for k in range(2,L):
        av[k] = av[k-1] * av[1]

    # Computing the quotients
    tv = {}
    for chi in derivatives:
        for k in range(0,L):
            ctx.prec = wptcoef[k]
            for ell in range(0,3*k//2+1):
                tv[chi,k,ell] = tcoef[chi,k,ell]* av[k]

    # COMPUTING THE TERMS term[k]
    # See II Section 3.6
    term = {}
    for chi in derivatives:
        for n in range(0,L):
            ctx.prec = wpterm[n]
            te = 0
            for k in range(0, 3*n//2+1):
                te += tv[chi,n,k]
            term[chi,n] = te

    # COMPUTING  rssum
    # See II Section 3.5
    rssum={}
    ctx.prec=15
    rsbound = math.sqrt(ctx.pi) * c /(b*a)
    ctx.prec=15
    wprssum = ctx.mag(4.4*((L+3)**2)*rsbound / eps2)
    wprssum = max(wprssum, ctx.mag(10*(L+1)))
    ctx.prec = wprssum
    for chi in derivatives:
        rssum[chi] = 0
        for k in range(1,L+1):
            rssum[chi] += term[chi,L-k]

    # COMPUTING S3
    # See II Section 3.19
    ctx.prec = 15
    A2 = 2**(ctx.mag(rssum[0]))
    eps8 = eps/(3* A2)
    T = t * ctx.ln(t/(2*ctx.pi))
    wps3 = 5 + ctx.mag((1+(2/eps8)*ctx.power(a,-sigma))*T)

    ctx.prec = wps3
    tpi = t/(2*ctx.pi)
    arg = (t/2)*ctx.ln(tpi)-(t/2)-ctx.pi/8
    U = ctx.expj(-arg)
    a = trunc_a(t)
    asigma = ctx.power(a, -sigma)
    S3 = ((-1)**(N-1)) * asigma * U

    # COMPUTING S1 the zetasum
    # See II Section 3.18
    ctx.prec = 15
    wpsum = 4 + ctx.mag((N+ctx.power(N,1-sigma))*ctx.ln(N)/eps1)

    ctx.prec = wpsum + 10
    '''
    # This can be improved
    S1 = {}
    for chi in derivatives:
        S1[chi] = 0
    for n in range(1,int(N)+1):
        ln = ctx.ln(n)
        expn = ctx.exp(-ln*(sigma+ctx.j*t))
        for chi in derivatives:
            term = ctx.power(-ln, chi)*expn
            S1[chi] += term
    '''
    S1 = ctx._zetasum(s, 1, int(N)-1, derivatives)[0]

    # END OF COMPUTATION
    #  See II Section 3.1
    ctx.prec = 15
    absS1 = abs(S1[der])
    absS2 = abs(rssum[der] * S3)
    wpend = max(6, wpinitial + ctx.mag(6*(3*absS1+7*absS2)))
    ctx.prec = wpend
    rz = {}
    for chi in derivatives:
        rz[chi] = S1[chi]+rssum[chi]*S3
    ctx.prec = wpinitial
    return rz


def z_half(ctx,t,der=0):
    r"""
    z_half(t,der=0) Computes Z^(der)(t)
    """
    s=ctx.mpf('0.5')+ctx.j*t
    wpinitial = ctx.prec
    ctx.prec = 15
    tt = t/(2*ctx.pi)
    wptheta = wpinitial +1 + ctx.mag(3*(tt**1.5)*ctx.ln(tt))
    wpz = wpinitial + 1 + ctx.mag(12*tt*ctx.ln(tt))
    ctx.prec = wptheta
    theta = ctx.siegeltheta(t)
    ctx.prec = wpz
    rz = Rzeta_set(ctx,s, range(der+1))
    if der > 0: ps1 = ctx._re(ctx.psi(0,s/2)/2 - ctx.ln(ctx.pi)/2)
    if der > 1: ps2 = ctx._re(ctx.j*ctx.psi(1,s/2)/4)
    if der > 2: ps3 = ctx._re(-ctx.psi(2,s/2)/8)
    if der > 3: ps4 = ctx._re(-ctx.j*ctx.psi(3,s/2)/16)
    exptheta = ctx.expj(theta)
    if der == 0:
        z = 2*exptheta*rz[0]
    if der == 1:
        zf = 2j*exptheta
        z = zf*(ps1*rz[0]+rz[1])
    if der == 2:
        zf = 2 * exptheta
        z = -zf*(2*rz[1]*ps1+rz[0]*ps1**2+rz[2]-ctx.j*rz[0]*ps2)
    if der == 3:
        zf = -2j*exptheta
        z = 3*rz[1]*ps1**2+rz[0]*ps1**3+3*ps1*rz[2]
        z = zf*(z-3j*rz[1]*ps2-3j*rz[0]*ps1*ps2+rz[3]-rz[0]*ps3)
    if der == 4:
        zf = 2*exptheta
        z = 4*rz[1]*ps1**3+rz[0]*ps1**4+6*ps1**2*rz[2]
        z = z-12j*rz[1]*ps1*ps2-6j*rz[0]*ps1**2*ps2-6j*rz[2]*ps2-3*rz[0]*ps2*ps2
        z = z + 4*ps1*rz[3]-4*rz[1]*ps3-4*rz[0]*ps1*ps3+rz[4]+ctx.j*rz[0]*ps4
        z = zf*z
    ctx.prec = wpinitial
    return ctx._re(z)

def zeta_half(ctx, s, k=0):
    """
    zeta_half(s,k=0) Computes zeta^(k)(s) when Re s = 0.5
    """
    wpinitial = ctx.prec
    sigma = ctx._re(s)
    t = ctx._im(s)
    #--- compute wptheta, wpR, wpbasic ---
    ctx.prec = 53
    #  X see II Section 3.21 (109) and (110)
    if sigma > 0:
        X = ctx.sqrt(abs(s))
    else:
        X = (2*ctx.pi)**(sigma-1) * abs(1-s)**(0.5-sigma)
    # M1  see II Section 3.21 (111) and (112)
    if sigma > 0:
        M1 = 2*ctx.sqrt(t/(2*ctx.pi))
    else:
        M1 = 4 * t * X
    # T  see II Section 3.21 (113)
    abst = abs(0.5-s)
    T = 2* abst*math.log(abst)
    # computing wpbasic, wptheta, wpR  see II Section 3.21
    wpbasic = max(6,3+ctx.mag(t))
    wpbasic2 = 2+ctx.mag(2.12*M1+21.2*M1*X+1.3*M1*X*T)+wpinitial+1
    wpbasic = max(wpbasic, wpbasic2)
    wptheta = max(4, 3+ctx.mag(2.7*M1*X)+wpinitial+1)
    wpR = 3+ctx.mag(1.1+2*X)+wpinitial+1
    ctx.prec = wptheta
    theta = ctx.siegeltheta(t-ctx.j*(sigma-ctx.mpf('0.5')))
    if k > 0: ps1 = (ctx._re(ctx.psi(0,s/2)))/2 - ctx.ln(ctx.pi)/2
    if k > 1: ps2 = -(ctx._im(ctx.psi(1,s/2)))/4
    if k > 2: ps3 = -(ctx._re(ctx.psi(2,s/2)))/8
    if k > 3: ps4 = (ctx._im(ctx.psi(3,s/2)))/16
    ctx.prec = wpR
    xrz = Rzeta_set(ctx,s,range(k+1))
    yrz={}
    for chi in range(0,k+1):
        yrz[chi] = ctx.conj(xrz[chi])
    ctx.prec = wpbasic
    exptheta = ctx.expj(-2*theta)
    if k==0:
        zv = xrz[0]+exptheta*yrz[0]
    if k==1:
        zv1 = -yrz[1] - 2*yrz[0]*ps1
        zv = xrz[1] + exptheta*zv1
    if k==2:
        zv1 = 4*yrz[1]*ps1+4*yrz[0]*(ps1**2)+yrz[2]+2j*yrz[0]*ps2
        zv = xrz[2]+exptheta*zv1
    if k==3:
        zv1 = -12*yrz[1]*ps1**2-8*yrz[0]*ps1**3-6*yrz[2]*ps1-6j*yrz[1]*ps2
        zv1 = zv1 - 12j*yrz[0]*ps1*ps2-yrz[3]+2*yrz[0]*ps3
        zv = xrz[3]+exptheta*zv1
    if k == 4:
        zv1 = 32*yrz[1]*ps1**3 +16*yrz[0]*ps1**4+24*yrz[2]*ps1**2
        zv1 = zv1 +48j*yrz[1]*ps1*ps2+48j*yrz[0]*(ps1**2)*ps2
        zv1 = zv1+12j*yrz[2]*ps2-12*yrz[0]*ps2**2+8*yrz[3]*ps1-8*yrz[1]*ps3
        zv1 = zv1-16*yrz[0]*ps1*ps3+yrz[4]-2j*yrz[0]*ps4
        zv = xrz[4]+exptheta*zv1
    ctx.prec = wpinitial
    return zv

def zeta_offline(ctx, s, k=0):
    """
    Computes zeta^(k)(s) off the line
    """
    wpinitial = ctx.prec
    sigma = ctx._re(s)
    t = ctx._im(s)
    #--- compute wptheta, wpR, wpbasic ---
    ctx.prec = 53
    #  X see II Section 3.21 (109) and (110)
    if sigma > 0:
        X = ctx.power(abs(s), 0.5)
    else:
        X = ctx.power(2*ctx.pi, sigma-1)*ctx.power(abs(1-s),0.5-sigma)
    # M1  see II Section 3.21 (111) and (112)
    if (sigma > 0):
        M1 = 2*ctx.sqrt(t/(2*ctx.pi))
    else:
        M1 = 4 * t * X
    # M2  see II Section 3.21 (111) and (112)
    if (1-sigma > 0):
        M2 = 2*ctx.sqrt(t/(2*ctx.pi))
    else:
        M2 = 4*t*ctx.power(2*ctx.pi, -sigma)*ctx.power(abs(s),sigma-0.5)
    # T  see II Section 3.21 (113)
    abst = abs(0.5-s)
    T = 2* abst*math.log(abst)
    # computing wpbasic, wptheta, wpR  see II Section 3.21
    wpbasic = max(6,3+ctx.mag(t))
    wpbasic2 = 2+ctx.mag(2.12*M1+21.2*M2*X+1.3*M2*X*T)+wpinitial+1
    wpbasic = max(wpbasic, wpbasic2)
    wptheta = max(4, 3+ctx.mag(2.7*M2*X)+wpinitial+1)
    wpR = 3+ctx.mag(1.1+2*X)+wpinitial+1
    ctx.prec = wptheta
    theta = ctx.siegeltheta(t-ctx.j*(sigma-ctx.mpf('0.5')))
    s1 = s
    s2 = ctx.conj(1-s1)
    ctx.prec = wpR
    xrz, yrz = Rzeta_simul(ctx, s, k)
    if k > 0: ps1 = (ctx.psi(0,s1/2)+ctx.psi(0,(1-s1)/2))/4 - ctx.ln(ctx.pi)/2
    if k > 1: ps2 = ctx.j*(ctx.psi(1,s1/2)-ctx.psi(1,(1-s1)/2))/8
    if k > 2: ps3 = -(ctx.psi(2,s1/2)+ctx.psi(2,(1-s1)/2))/16
    if k > 3: ps4 = -ctx.j*(ctx.psi(3,s1/2)-ctx.psi(3,(1-s1)/2))/32
    ctx.prec = wpbasic
    exptheta = ctx.expj(-2*theta)
    if k == 0:
        zv = xrz[0]+exptheta*yrz[0]
    if k == 1:
        zv1 = -yrz[1]-2*yrz[0]*ps1
        zv = xrz[1]+exptheta*zv1
    if k == 2:
        zv1 = 4*yrz[1]*ps1+4*yrz[0]*(ps1**2) +yrz[2]+2j*yrz[0]*ps2
        zv = xrz[2]+exptheta*zv1
    if k == 3:
        zv1 = -12*yrz[1]*ps1**2 -8*yrz[0]*ps1**3-6*yrz[2]*ps1-6j*yrz[1]*ps2
        zv1 = zv1 - 12j*yrz[0]*ps1*ps2-yrz[3]+2*yrz[0]*ps3
        zv = xrz[3]+exptheta*zv1
    if k == 4:
        zv1 = 32*yrz[1]*ps1**3 +16*yrz[0]*ps1**4+24*yrz[2]*ps1**2
        zv1 = zv1 +48j*yrz[1]*ps1*ps2+48j*yrz[0]*(ps1**2)*ps2
        zv1 = zv1+12j*yrz[2]*ps2-12*yrz[0]*ps2**2+8*yrz[3]*ps1-8*yrz[1]*ps3
        zv1 = zv1-16*yrz[0]*ps1*ps3+yrz[4]-2j*yrz[0]*ps4
        zv = xrz[4]+exptheta*zv1
    ctx.prec = wpinitial
    return zv

def z_offline(ctx, w, k=0):
    r"""
    Computes Z(w) and its derivatives off the line
    """
    s = ctx.mpf('0.5')+ctx.j*w
    s1 = s
    s2 = ctx.conj(1-s1)
    wpinitial = ctx.prec
    ctx.prec = 35
    #  X see II Section 3.21 (109) and (110)
    # M1  see II Section 3.21 (111) and (112)
    if (ctx._re(s1) >= 0):
        M1 = 2*ctx.sqrt(ctx._im(s1)/(2 * ctx.pi))
        X = ctx.sqrt(abs(s1))
    else:
        X = (2*ctx.pi)**(ctx._re(s1)-1) * abs(1-s1)**(0.5-ctx._re(s1))
        M1 = 4 * ctx._im(s1)*X
    # M2  see II Section 3.21 (111) and (112)
    if (ctx._re(s2) >= 0):
        M2 = 2*ctx.sqrt(ctx._im(s2)/(2 * ctx.pi))
    else:
        M2 = 4 * ctx._im(s2)*(2*ctx.pi)**(ctx._re(s2)-1)*abs(1-s2)**(0.5-ctx._re(s2))
    # T  see II Section 3.21  Prop. 27
    T = 2*abs(ctx.siegeltheta(w))
    # defining some precisions
    # see II Section 3.22 (115), (116), (117)
    aux1 = ctx.sqrt(X)
    aux2 = aux1*(M1+M2)
    aux3 = 3 +wpinitial
    wpbasic = max(6, 3+ctx.mag(T), ctx.mag(aux2*(26+2*T))+aux3)
    wptheta = max(4,ctx.mag(2.04*aux2)+aux3)
    wpR = ctx.mag(4*aux1)+aux3
    # now the computations
    ctx.prec = wptheta
    theta = ctx.siegeltheta(w)
    ctx.prec = wpR
    xrz, yrz = Rzeta_simul(ctx,s,k)
    pta = 0.25 + 0.5j*w
    ptb = 0.25 - 0.5j*w
    if k > 0: ps1 = 0.25*(ctx.psi(0,pta)+ctx.psi(0,ptb)) - ctx.ln(ctx.pi)/2
    if k > 1: ps2 = (1j/8)*(ctx.psi(1,pta)-ctx.psi(1,ptb))
    if k > 2: ps3 = (-1./16)*(ctx.psi(2,pta)+ctx.psi(2,ptb))
    if k > 3: ps4 = (-1j/32)*(ctx.psi(3,pta)-ctx.psi(3,ptb))
    ctx.prec = wpbasic
    exptheta = ctx.expj(theta)
    if k == 0:
        zv = exptheta*xrz[0]+yrz[0]/exptheta
    j = ctx.j
    if k == 1:
        zv = j*exptheta*(xrz[1]+xrz[0]*ps1)-j*(yrz[1]+yrz[0]*ps1)/exptheta
    if k == 2:
        zv = exptheta*(-2*xrz[1]*ps1-xrz[0]*ps1**2-xrz[2]+j*xrz[0]*ps2)
        zv =zv + (-2*yrz[1]*ps1-yrz[0]*ps1**2-yrz[2]-j*yrz[0]*ps2)/exptheta
    if k == 3:
        zv1 = -3*xrz[1]*ps1**2-xrz[0]*ps1**3-3*xrz[2]*ps1+j*3*xrz[1]*ps2
        zv1 = (zv1+ 3j*xrz[0]*ps1*ps2-xrz[3]+xrz[0]*ps3)*j*exptheta
        zv2 = 3*yrz[1]*ps1**2+yrz[0]*ps1**3+3*yrz[2]*ps1+j*3*yrz[1]*ps2
        zv2 = j*(zv2 + 3j*yrz[0]*ps1*ps2+ yrz[3]-yrz[0]*ps3)/exptheta
        zv = zv1+zv2
    if k == 4:
        zv1 = 4*xrz[1]*ps1**3+xrz[0]*ps1**4 + 6*xrz[2]*ps1**2
        zv1 = zv1-12j*xrz[1]*ps1*ps2-6j*xrz[0]*ps1**2*ps2-6j*xrz[2]*ps2
        zv1 = zv1-3*xrz[0]*ps2*ps2+4*xrz[3]*ps1-4*xrz[1]*ps3-4*xrz[0]*ps1*ps3
        zv1 = zv1+xrz[4]+j*xrz[0]*ps4
        zv2 = 4*yrz[1]*ps1**3+yrz[0]*ps1**4 + 6*yrz[2]*ps1**2
        zv2 = zv2+12j*yrz[1]*ps1*ps2+6j*yrz[0]*ps1**2*ps2+6j*yrz[2]*ps2
        zv2 = zv2-3*yrz[0]*ps2*ps2+4*yrz[3]*ps1-4*yrz[1]*ps3-4*yrz[0]*ps1*ps3
        zv2 = zv2+yrz[4]-j*yrz[0]*ps4
        zv = exptheta*zv1+zv2/exptheta
    ctx.prec = wpinitial
    return zv

@defun
def rs_zeta(ctx, s, derivative=0, **kwargs):
    if derivative > 4:
        raise NotImplementedError
    s = ctx.convert(s)
    re = ctx._re(s); im = ctx._im(s)
    if im < 0:
        z = ctx.conj(ctx.rs_zeta(ctx.conj(s), derivative))
        return z
    critical_line = (re == 0.5)
    if critical_line:
        return zeta_half(ctx, s, derivative)
    else:
        return zeta_offline(ctx, s, derivative)

@defun
def rs_z(ctx, w, derivative=0):
    w = ctx.convert(w)
    re = ctx._re(w); im = ctx._im(w)
    if re < 0:
        return rs_z(ctx, -w, derivative)
    critical_line = (im == 0)
    if critical_line :
        return z_half(ctx, w, derivative)
    else:
        return z_offline(ctx, w, derivative)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\distro\distro.py ===
#!/usr/bin/env python
# Copyright 2015-2021 Nir Cohen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
The ``distro`` package (``distro`` stands for Linux Distribution) provides
information about the Linux distribution it runs on, such as a reliable
machine-readable distro ID, or version information.

It is the recommended replacement for Python's original
:py:func:`platform.linux_distribution` function, but it provides much more
functionality. An alternative implementation became necessary because Python
3.5 deprecated this function, and Python 3.8 removed it altogether. Its
predecessor function :py:func:`platform.dist` was already deprecated since
Python 2.6 and removed in Python 3.8. Still, there are many cases in which
access to OS distribution information is needed. See `Python issue 1322
<https://bugs.python.org/issue1322>`_ for more information.
"""

import argparse
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Optional,
    Sequence,
    TextIO,
    Tuple,
    Type,
)

try:
    from typing import TypedDict
except ImportError:
    # Python 3.7
    TypedDict = dict

__version__ = "1.9.0"


class VersionDict(TypedDict):
    major: str
    minor: str
    build_number: str


class InfoDict(TypedDict):
    id: str
    version: str
    version_parts: VersionDict
    like: str
    codename: str


_UNIXCONFDIR = os.environ.get("UNIXCONFDIR", "/etc")
_UNIXUSRLIBDIR = os.environ.get("UNIXUSRLIBDIR", "/usr/lib")
_OS_RELEASE_BASENAME = "os-release"

#: Translation table for normalizing the "ID" attribute defined in os-release
#: files, for use by the :func:`distro.id` method.
#:
#: * Key: Value as defined in the os-release file, translated to lower case,
#:   with blanks translated to underscores.
#:
#: * Value: Normalized value.
NORMALIZED_OS_ID = {
    "ol": "oracle",  # Oracle Linux
    "opensuse-leap": "opensuse",  # Newer versions of OpenSuSE report as opensuse-leap
}

#: Translation table for normalizing the "Distributor ID" attribute returned by
#: the lsb_release command, for use by the :func:`distro.id` method.
#:
#: * Key: Value as returned by the lsb_release command, translated to lower
#:   case, with blanks translated to underscores.
#:
#: * Value: Normalized value.
NORMALIZED_LSB_ID = {
    "enterpriseenterpriseas": "oracle",  # Oracle Enterprise Linux 4
    "enterpriseenterpriseserver": "oracle",  # Oracle Linux 5
    "redhatenterpriseworkstation": "rhel",  # RHEL 6, 7 Workstation
    "redhatenterpriseserver": "rhel",  # RHEL 6, 7 Server
    "redhatenterprisecomputenode": "rhel",  # RHEL 6 ComputeNode
}

#: Translation table for normalizing the distro ID derived from the file name
#: of distro release files, for use by the :func:`distro.id` method.
#:
#: * Key: Value as derived from the file name of a distro release file,
#:   translated to lower case, with blanks translated to underscores.
#:
#: * Value: Normalized value.
NORMALIZED_DISTRO_ID = {
    "redhat": "rhel",  # RHEL 6.x, 7.x
}

# Pattern for content of distro release file (reversed)
_DISTRO_RELEASE_CONTENT_REVERSED_PATTERN = re.compile(
    r"(?:[^)]*\)(.*)\()? *(?:STL )?([\d.+\-a-z]*\d) *(?:esaeler *)?(.+)"
)

# Pattern for base file name of distro release file
_DISTRO_RELEASE_BASENAME_PATTERN = re.compile(r"(\w+)[-_](release|version)$")

# Base file names to be looked up for if _UNIXCONFDIR is not readable.
_DISTRO_RELEASE_BASENAMES = [
    "SuSE-release",
    "altlinux-release",
    "arch-release",
    "base-release",
    "centos-release",
    "fedora-release",
    "gentoo-release",
    "mageia-release",
    "mandrake-release",
    "mandriva-release",
    "mandrivalinux-release",
    "manjaro-release",
    "oracle-release",
    "redhat-release",
    "rocky-release",
    "sl-release",
    "slackware-version",
]

# Base file names to be ignored when searching for distro release file
_DISTRO_RELEASE_IGNORE_BASENAMES = (
    "debian_version",
    "lsb-release",
    "oem-release",
    _OS_RELEASE_BASENAME,
    "system-release",
    "plesk-release",
    "iredmail-release",
    "board-release",
    "ec2_version",
)


def linux_distribution(full_distribution_name: bool = True) -> Tuple[str, str, str]:
    """
    .. deprecated:: 1.6.0

        :func:`distro.linux_distribution()` is deprecated. It should only be
        used as a compatibility shim with Python's
        :py:func:`platform.linux_distribution()`. Please use :func:`distro.id`,
        :func:`distro.version` and :func:`distro.name` instead.

    Return information about the current OS distribution as a tuple
    ``(id_name, version, codename)`` with items as follows:

    * ``id_name``:  If *full_distribution_name* is false, the result of
      :func:`distro.id`. Otherwise, the result of :func:`distro.name`.

    * ``version``:  The result of :func:`distro.version`.

    * ``codename``:  The extra item (usually in parentheses) after the
      os-release version number, or the result of :func:`distro.codename`.

    The interface of this function is compatible with the original
    :py:func:`platform.linux_distribution` function, supporting a subset of
    its parameters.

    The data it returns may not exactly be the same, because it uses more data
    sources than the original function, and that may lead to different data if
    the OS distribution is not consistent across multiple data sources it
    provides (there are indeed such distributions ...).

    Another reason for differences is the fact that the :func:`distro.id`
    method normalizes the distro ID string to a reliable machine-readable value
    for a number of popular OS distributions.
    """
    warnings.warn(
        "distro.linux_distribution() is deprecated. It should only be used as a "
        "compatibility shim with Python's platform.linux_distribution(). Please use "
        "distro.id(), distro.version() and distro.name() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _distro.linux_distribution(full_distribution_name)


def id() -> str:
    """
    Return the distro ID of the current distribution, as a
    machine-readable string.

    For a number of OS distributions, the returned distro ID value is
    *reliable*, in the sense that it is documented and that it does not change
    across releases of the distribution.

    This package maintains the following reliable distro ID values:

    ==============  =========================================
    Distro ID       Distribution
    ==============  =========================================
    "ubuntu"        Ubuntu
    "debian"        Debian
    "rhel"          RedHat Enterprise Linux
    "centos"        CentOS
    "fedora"        Fedora
    "sles"          SUSE Linux Enterprise Server
    "opensuse"      openSUSE
    "amzn"          Amazon Linux
    "arch"          Arch Linux
    "buildroot"     Buildroot
    "cloudlinux"    CloudLinux OS
    "exherbo"       Exherbo Linux
    "gentoo"        GenToo Linux
    "ibm_powerkvm"  IBM PowerKVM
    "kvmibm"        KVM for IBM z Systems
    "linuxmint"     Linux Mint
    "mageia"        Mageia
    "mandriva"      Mandriva Linux
    "parallels"     Parallels
    "pidora"        Pidora
    "raspbian"      Raspbian
    "oracle"        Oracle Linux (and Oracle Enterprise Linux)
    "scientific"    Scientific Linux
    "slackware"     Slackware
    "xenserver"     XenServer
    "openbsd"       OpenBSD
    "netbsd"        NetBSD
    "freebsd"       FreeBSD
    "midnightbsd"   MidnightBSD
    "rocky"         Rocky Linux
    "aix"           AIX
    "guix"          Guix System
    "altlinux"      ALT Linux
    ==============  =========================================

    If you have a need to get distros for reliable IDs added into this set,
    or if you find that the :func:`distro.id` function returns a different
    distro ID for one of the listed distros, please create an issue in the
    `distro issue tracker`_.

    **Lookup hierarchy and transformations:**

    First, the ID is obtained from the following sources, in the specified
    order. The first available and non-empty value is used:

    * the value of the "ID" attribute of the os-release file,

    * the value of the "Distributor ID" attribute returned by the lsb_release
      command,

    * the first part of the file name of the distro release file,

    The so determined ID value then passes the following transformations,
    before it is returned by this method:

    * it is translated to lower case,

    * blanks (which should not be there anyway) are translated to underscores,

    * a normalization of the ID is performed, based upon
      `normalization tables`_. The purpose of this normalization is to ensure
      that the ID is as reliable as possible, even across incompatible changes
      in the OS distributions. A common reason for an incompatible change is
      the addition of an os-release file, or the addition of the lsb_release
      command, with ID values that differ from what was previously determined
      from the distro release file name.
    """
    return _distro.id()


def name(pretty: bool = False) -> str:
    """
    Return the name of the current OS distribution, as a human-readable
    string.

    If *pretty* is false, the name is returned without version or codename.
    (e.g. "CentOS Linux")

    If *pretty* is true, the version and codename are appended.
    (e.g. "CentOS Linux 7.1.1503 (Core)")

    **Lookup hierarchy:**

    The name is obtained from the following sources, in the specified order.
    The first available and non-empty value is used:

    * If *pretty* is false:

      - the value of the "NAME" attribute of the os-release file,

      - the value of the "Distributor ID" attribute returned by the lsb_release
        command,

      - the value of the "<name>" field of the distro release file.

    * If *pretty* is true:

      - the value of the "PRETTY_NAME" attribute of the os-release file,

      - the value of the "Description" attribute returned by the lsb_release
        command,

      - the value of the "<name>" field of the distro release file, appended
        with the value of the pretty version ("<version_id>" and "<codename>"
        fields) of the distro release file, if available.
    """
    return _distro.name(pretty)


def version(pretty: bool = False, best: bool = False) -> str:
    """
    Return the version of the current OS distribution, as a human-readable
    string.

    If *pretty* is false, the version is returned without codename (e.g.
    "7.0").

    If *pretty* is true, the codename in parenthesis is appended, if the
    codename is non-empty (e.g. "7.0 (Maipo)").

    Some distributions provide version numbers with different precisions in
    the different sources of distribution information. Examining the different
    sources in a fixed priority order does not always yield the most precise
    version (e.g. for Debian 8.2, or CentOS 7.1).

    Some other distributions may not provide this kind of information. In these
    cases, an empty string would be returned. This behavior can be observed
    with rolling releases distributions (e.g. Arch Linux).

    The *best* parameter can be used to control the approach for the returned
    version:

    If *best* is false, the first non-empty version number in priority order of
    the examined sources is returned.

    If *best* is true, the most precise version number out of all examined
    sources is returned.

    **Lookup hierarchy:**

    In all cases, the version number is obtained from the following sources.
    If *best* is false, this order represents the priority order:

    * the value of the "VERSION_ID" attribute of the os-release file,
    * the value of the "Release" attribute returned by the lsb_release
      command,
    * the version number parsed from the "<version_id>" field of the first line
      of the distro release file,
    * the version number parsed from the "PRETTY_NAME" attribute of the
      os-release file, if it follows the format of the distro release files.
    * the version number parsed from the "Description" attribute returned by
      the lsb_release command, if it follows the format of the distro release
      files.
    """
    return _distro.version(pretty, best)


def version_parts(best: bool = False) -> Tuple[str, str, str]:
    """
    Return the version of the current OS distribution as a tuple
    ``(major, minor, build_number)`` with items as follows:

    * ``major``:  The result of :func:`distro.major_version`.

    * ``minor``:  The result of :func:`distro.minor_version`.

    * ``build_number``:  The result of :func:`distro.build_number`.

    For a description of the *best* parameter, see the :func:`distro.version`
    method.
    """
    return _distro.version_parts(best)


def major_version(best: bool = False) -> str:
    """
    Return the major version of the current OS distribution, as a string,
    if provided.
    Otherwise, the empty string is returned. The major version is the first
    part of the dot-separated version string.

    For a description of the *best* parameter, see the :func:`distro.version`
    method.
    """
    return _distro.major_version(best)


def minor_version(best: bool = False) -> str:
    """
    Return the minor version of the current OS distribution, as a string,
    if provided.
    Otherwise, the empty string is returned. The minor version is the second
    part of the dot-separated version string.

    For a description of the *best* parameter, see the :func:`distro.version`
    method.
    """
    return _distro.minor_version(best)


def build_number(best: bool = False) -> str:
    """
    Return the build number of the current OS distribution, as a string,
    if provided.
    Otherwise, the empty string is returned. The build number is the third part
    of the dot-separated version string.

    For a description of the *best* parameter, see the :func:`distro.version`
    method.
    """
    return _distro.build_number(best)


def like() -> str:
    """
    Return a space-separated list of distro IDs of distributions that are
    closely related to the current OS distribution in regards to packaging
    and programming interfaces, for example distributions the current
    distribution is a derivative from.

    **Lookup hierarchy:**

    This information item is only provided by the os-release file.
    For details, see the description of the "ID_LIKE" attribute in the
    `os-release man page
    <http://www.freedesktop.org/software/systemd/man/os-release.html>`_.
    """
    return _distro.like()


def codename() -> str:
    """
    Return the codename for the release of the current OS distribution,
    as a string.

    If the distribution does not have a codename, an empty string is returned.

    Note that the returned codename is not always really a codename. For
    example, openSUSE returns "x86_64". This function does not handle such
    cases in any special way and just returns the string it finds, if any.

    **Lookup hierarchy:**

    * the codename within the "VERSION" attribute of the os-release file, if
      provided,

    * the value of the "Codename" attribute returned by the lsb_release
      command,

    * the value of the "<codename>" field of the distro release file.
    """
    return _distro.codename()


def info(pretty: bool = False, best: bool = False) -> InfoDict:
    """
    Return certain machine-readable information items about the current OS
    distribution in a dictionary, as shown in the following example:

    .. sourcecode:: python

        {
            'id': 'rhel',
            'version': '7.0',
            'version_parts': {
                'major': '7',
                'minor': '0',
                'build_number': ''
            },
            'like': 'fedora',
            'codename': 'Maipo'
        }

    The dictionary structure and keys are always the same, regardless of which
    information items are available in the underlying data sources. The values
    for the various keys are as follows:

    * ``id``:  The result of :func:`distro.id`.

    * ``version``:  The result of :func:`distro.version`.

    * ``version_parts -> major``:  The result of :func:`distro.major_version`.

    * ``version_parts -> minor``:  The result of :func:`distro.minor_version`.

    * ``version_parts -> build_number``:  The result of
      :func:`distro.build_number`.

    * ``like``:  The result of :func:`distro.like`.

    * ``codename``:  The result of :func:`distro.codename`.

    For a description of the *pretty* and *best* parameters, see the
    :func:`distro.version` method.
    """
    return _distro.info(pretty, best)


def os_release_info() -> Dict[str, str]:
    """
    Return a dictionary containing key-value pairs for the information items
    from the os-release file data source of the current OS distribution.

    See `os-release file`_ for details about these information items.
    """
    return _distro.os_release_info()


def lsb_release_info() -> Dict[str, str]:
    """
    Return a dictionary containing key-value pairs for the information items
    from the lsb_release command data source of the current OS distribution.

    See `lsb_release command output`_ for details about these information
    items.
    """
    return _distro.lsb_release_info()


def distro_release_info() -> Dict[str, str]:
    """
    Return a dictionary containing key-value pairs for the information items
    from the distro release file data source of the current OS distribution.

    See `distro release file`_ for details about these information items.
    """
    return _distro.distro_release_info()


def uname_info() -> Dict[str, str]:
    """
    Return a dictionary containing key-value pairs for the information items
    from the distro release file data source of the current OS distribution.
    """
    return _distro.uname_info()


def os_release_attr(attribute: str) -> str:
    """
    Return a single named information item from the os-release file data source
    of the current OS distribution.

    Parameters:

    * ``attribute`` (string): Key of the information item.

    Returns:

    * (string): Value of the information item, if the item exists.
      The empty string, if the item does not exist.

    See `os-release file`_ for details about these information items.
    """
    return _distro.os_release_attr(attribute)


def lsb_release_attr(attribute: str) -> str:
    """
    Return a single named information item from the lsb_release command output
    data source of the current OS distribution.

    Parameters:

    * ``attribute`` (string): Key of the information item.

    Returns:

    * (string): Value of the information item, if the item exists.
      The empty string, if the item does not exist.

    See `lsb_release command output`_ for details about these information
    items.
    """
    return _distro.lsb_release_attr(attribute)


def distro_release_attr(attribute: str) -> str:
    """
    Return a single named information item from the distro release file
    data source of the current OS distribution.

    Parameters:

    * ``attribute`` (string): Key of the information item.

    Returns:

    * (string): Value of the information item, if the item exists.
      The empty string, if the item does not exist.

    See `distro release file`_ for details about these information items.
    """
    return _distro.distro_release_attr(attribute)


def uname_attr(attribute: str) -> str:
    """
    Return a single named information item from the distro release file
    data source of the current OS distribution.

    Parameters:

    * ``attribute`` (string): Key of the information item.

    Returns:

    * (string): Value of the information item, if the item exists.
                The empty string, if the item does not exist.
    """
    return _distro.uname_attr(attribute)


try:
    from functools import cached_property
except ImportError:
    # Python < 3.8
    class cached_property:  # type: ignore
        """A version of @property which caches the value.  On access, it calls the
        underlying function and sets the value in `__dict__` so future accesses
        will not re-call the property.
        """

        def __init__(self, f: Callable[[Any], Any]) -> None:
            self._fname = f.__name__
            self._f = f

        def __get__(self, obj: Any, owner: Type[Any]) -> Any:
            assert obj is not None, f"call {self._fname} on an instance"
            ret = obj.__dict__[self._fname] = self._f(obj)
            return ret


class LinuxDistribution:
    """
    Provides information about a OS distribution.

    This package creates a private module-global instance of this class with
    default initialization arguments, that is used by the
    `consolidated accessor functions`_ and `single source accessor functions`_.
    By using default initialization arguments, that module-global instance
    returns data about the current OS distribution (i.e. the distro this
    package runs on).

    Normally, it is not necessary to create additional instances of this class.
    However, in situations where control is needed over the exact data sources
    that are used, instances of this class can be created with a specific
    distro release file, or a specific os-release file, or without invoking the
    lsb_release command.
    """

    def __init__(
        self,
        include_lsb: Optional[bool] = None,
        os_release_file: str = "",
        distro_release_file: str = "",
        include_uname: Optional[bool] = None,
        root_dir: Optional[str] = None,
        include_oslevel: Optional[bool] = None,
    ) -> None:
        """
        The initialization method of this class gathers information from the
        available data sources, and stores that in private instance attributes.
        Subsequent access to the information items uses these private instance
        attributes, so that the data sources are read only once.

        Parameters:

        * ``include_lsb`` (bool): Controls whether the
          `lsb_release command output`_ is included as a data source.

          If the lsb_release command is not available in the program execution
          path, the data source for the lsb_release command will be empty.

        * ``os_release_file`` (string): The path name of the
          `os-release file`_ that is to be used as a data source.

          An empty string (the default) will cause the default path name to
          be used (see `os-release file`_ for details).

          If the specified or defaulted os-release file does not exist, the
          data source for the os-release file will be empty.

        * ``distro_release_file`` (string): The path name of the
          `distro release file`_ that is to be used as a data source.

          An empty string (the default) will cause a default search algorithm
          to be used (see `distro release file`_ for details).

          If the specified distro release file does not exist, or if no default
          distro release file can be found, the data source for the distro
          release file will be empty.

        * ``include_uname`` (bool): Controls whether uname command output is
          included as a data source. If the uname command is not available in
          the program execution path the data source for the uname command will
          be empty.

        * ``root_dir`` (string): The absolute path to the root directory to use
          to find distro-related information files. Note that ``include_*``
          parameters must not be enabled in combination with ``root_dir``.

        * ``include_oslevel`` (bool): Controls whether (AIX) oslevel command
          output is included as a data source. If the oslevel command is not
          available in the program execution path the data source will be
          empty.

        Public instance attributes:

        * ``os_release_file`` (string): The path name of the
          `os-release file`_ that is actually used as a data source. The
          empty string if no distro release file is used as a data source.

        * ``distro_release_file`` (string): The path name of the
          `distro release file`_ that is actually used as a data source. The
          empty string if no distro release file is used as a data source.

        * ``include_lsb`` (bool): The result of the ``include_lsb`` parameter.
          This controls whether the lsb information will be loaded.

        * ``include_uname`` (bool): The result of the ``include_uname``
          parameter. This controls whether the uname information will
          be loaded.

        * ``include_oslevel`` (bool): The result of the ``include_oslevel``
          parameter. This controls whether (AIX) oslevel information will be
          loaded.

        * ``root_dir`` (string): The result of the ``root_dir`` parameter.
          The absolute path to the root directory to use to find distro-related
          information files.

        Raises:

        * :py:exc:`ValueError`: Initialization parameters combination is not
           supported.

        * :py:exc:`OSError`: Some I/O issue with an os-release file or distro
          release file.

        * :py:exc:`UnicodeError`: A data source has unexpected characters or
          uses an unexpected encoding.
        """
        self.root_dir = root_dir
        self.etc_dir = os.path.join(root_dir, "etc") if root_dir else _UNIXCONFDIR
        self.usr_lib_dir = (
            os.path.join(root_dir, "usr/lib") if root_dir else _UNIXUSRLIBDIR
        )

        if os_release_file:
            self.os_release_file = os_release_file
        else:
            etc_dir_os_release_file = os.path.join(self.etc_dir, _OS_RELEASE_BASENAME)
            usr_lib_os_release_file = os.path.join(
                self.usr_lib_dir, _OS_RELEASE_BASENAME
            )

            # NOTE: The idea is to respect order **and** have it set
            #       at all times for API backwards compatibility.
            if os.path.isfile(etc_dir_os_release_file) or not os.path.isfile(
                usr_lib_os_release_file
            ):
                self.os_release_file = etc_dir_os_release_file
            else:
                self.os_release_file = usr_lib_os_release_file

        self.distro_release_file = distro_release_file or ""  # updated later

        is_root_dir_defined = root_dir is not None
        if is_root_dir_defined and (include_lsb or include_uname or include_oslevel):
            raise ValueError(
                "Including subprocess data sources from specific root_dir is disallowed"
                " to prevent false information"
            )
        self.include_lsb = (
            include_lsb if include_lsb is not None else not is_root_dir_defined
        )
        self.include_uname = (
            include_uname if include_uname is not None else not is_root_dir_defined
        )
        self.include_oslevel = (
            include_oslevel if include_oslevel is not None else not is_root_dir_defined
        )

    def __repr__(self) -> str:
        """Return repr of all info"""
        return (
            "LinuxDistribution("
            "os_release_file={self.os_release_file!r}, "
            "distro_release_file={self.distro_release_file!r}, "
            "include_lsb={self.include_lsb!r}, "
            "include_uname={self.include_uname!r}, "
            "include_oslevel={self.include_oslevel!r}, "
            "root_dir={self.root_dir!r}, "
            "_os_release_info={self._os_release_info!r}, "
            "_lsb_release_info={self._lsb_release_info!r}, "
            "_distro_release_info={self._distro_release_info!r}, "
            "_uname_info={self._uname_info!r}, "
            "_oslevel_info={self._oslevel_info!r})".format(self=self)
        )

    def linux_distribution(
        self, full_distribution_name: bool = True
    ) -> Tuple[str, str, str]:
        """
        Return information about the OS distribution that is compatible
        with Python's :func:`platform.linux_distribution`, supporting a subset
        of its parameters.

        For details, see :func:`distro.linux_distribution`.
        """
        return (
            self.name() if full_distribution_name else self.id(),
            self.version(),
            self._os_release_info.get("release_codename") or self.codename(),
        )

    def id(self) -> str:
        """Return the distro ID of the OS distribution, as a string.

        For details, see :func:`distro.id`.
        """

        def normalize(distro_id: str, table: Dict[str, str]) -> str:
            distro_id = distro_id.lower().replace(" ", "_")
            return table.get(distro_id, distro_id)

        distro_id = self.os_release_attr("id")
        if distro_id:
            return normalize(distro_id, NORMALIZED_OS_ID)

        distro_id = self.lsb_release_attr("distributor_id")
        if distro_id:
            return normalize(distro_id, NORMALIZED_LSB_ID)

        distro_id = self.distro_release_attr("id")
        if distro_id:
            return normalize(distro_id, NORMALIZED_DISTRO_ID)

        distro_id = self.uname_attr("id")
        if distro_id:
            return normalize(distro_id, NORMALIZED_DISTRO_ID)

        return ""

    def name(self, pretty: bool = False) -> str:
        """
        Return the name of the OS distribution, as a string.

        For details, see :func:`distro.name`.
        """
        name = (
            self.os_release_attr("name")
            or self.lsb_release_attr("distributor_id")
            or self.distro_release_attr("name")
            or self.uname_attr("name")
        )
        if pretty:
            name = self.os_release_attr("pretty_name") or self.lsb_release_attr(
                "description"
            )
            if not name:
                name = self.distro_release_attr("name") or self.uname_attr("name")
                version = self.version(pretty=True)
                if version:
                    name = f"{name} {version}"
        return name or ""

    def version(self, pretty: bool = False, best: bool = False) -> str:
        """
        Return the version of the OS distribution, as a string.

        For details, see :func:`distro.version`.
        """
        versions = [
            self.os_release_attr("version_id"),
            self.lsb_release_attr("release"),
            self.distro_release_attr("version_id"),
            self._parse_distro_release_content(self.os_release_attr("pretty_name")).get(
                "version_id", ""
            ),
            self._parse_distro_release_content(
                self.lsb_release_attr("description")
            ).get("version_id", ""),
            self.uname_attr("release"),
        ]
        if self.uname_attr("id").startswith("aix"):
            # On AIX platforms, prefer oslevel command output.
            versions.insert(0, self.oslevel_info())
        elif self.id() == "debian" or "debian" in self.like().split():
            # On Debian-like, add debian_version file content to candidates list.
            versions.append(self._debian_version)
        version = ""
        if best:
            # This algorithm uses the last version in priority order that has
            # the best precision. If the versions are not in conflict, that
            # does not matter; otherwise, using the last one instead of the
            # first one might be considered a surprise.
            for v in versions:
                if v.count(".") > version.count(".") or version == "":
                    version = v
        else:
            for v in versions:
                if v != "":
                    version = v
                    break
        if pretty and version and self.codename():
            version = f"{version} ({self.codename()})"
        return version

    def version_parts(self, best: bool = False) -> Tuple[str, str, str]:
        """
        Return the version of the OS distribution, as a tuple of version
        numbers.

        For details, see :func:`distro.version_parts`.
        """
        version_str = self.version(best=best)
        if version_str:
            version_regex = re.compile(r"(\d+)\.?(\d+)?\.?(\d+)?")
            matches = version_regex.match(version_str)
            if matches:
                major, minor, build_number = matches.groups()
                return major, minor or "", build_number or ""
        return "", "", ""

    def major_version(self, best: bool = False) -> str:
        """
        Return the major version number of the current distribution.

        For details, see :func:`distro.major_version`.
        """
        return self.version_parts(best)[0]

    def minor_version(self, best: bool = False) -> str:
        """
        Return the minor version number of the current distribution.

        For details, see :func:`distro.minor_version`.
        """
        return self.version_parts(best)[1]

    def build_number(self, best: bool = False) -> str:
        """
        Return the build number of the current distribution.

        For details, see :func:`distro.build_number`.
        """
        return self.version_parts(best)[2]

    def like(self) -> str:
        """
        Return the IDs of distributions that are like the OS distribution.

        For details, see :func:`distro.like`.
        """
        return self.os_release_attr("id_like") or ""

    def codename(self) -> str:
        """
        Return the codename of the OS distribution.

        For details, see :func:`distro.codename`.
        """
        try:
            # Handle os_release specially since distros might purposefully set
            # this to empty string to have no codename
            return self._os_release_info["codename"]
        except KeyError:
            return (
                self.lsb_release_attr("codename")
                or self.distro_release_attr("codename")
                or ""
            )

    def info(self, pretty: bool = False, best: bool = False) -> InfoDict:
        """
        Return certain machine-readable information about the OS
        distribution.

        For details, see :func:`distro.info`.
        """
        return InfoDict(
            id=self.id(),
            version=self.version(pretty, best),
            version_parts=VersionDict(
                major=self.major_version(best),
                minor=self.minor_version(best),
                build_number=self.build_number(best),
            ),
            like=self.like(),
            codename=self.codename(),
        )

    def os_release_info(self) -> Dict[str, str]:
        """
        Return a dictionary containing key-value pairs for the information
        items from the os-release file data source of the OS distribution.

        For details, see :func:`distro.os_release_info`.
        """
        return self._os_release_info

    def lsb_release_info(self) -> Dict[str, str]:
        """
        Return a dictionary containing key-value pairs for the information
        items from the lsb_release command data source of the OS
        distribution.

        For details, see :func:`distro.lsb_release_info`.
        """
        return self._lsb_release_info

    def distro_release_info(self) -> Dict[str, str]:
        """
        Return a dictionary containing key-value pairs for the information
        items from the distro release file data source of the OS
        distribution.

        For details, see :func:`distro.distro_release_info`.
        """
        return self._distro_release_info

    def uname_info(self) -> Dict[str, str]:
        """
        Return a dictionary containing key-value pairs for the information
        items from the uname command data source of the OS distribution.

        For details, see :func:`distro.uname_info`.
        """
        return self._uname_info

    def oslevel_info(self) -> str:
        """
        Return AIX' oslevel command output.
        """
        return self._oslevel_info

    def os_release_attr(self, attribute: str) -> str:
        """
        Return a single named information item from the os-release file data
        source of the OS distribution.

        For details, see :func:`distro.os_release_attr`.
        """
        return self._os_release_info.get(attribute, "")

    def lsb_release_attr(self, attribute: str) -> str:
        """
        Return a single named information item from the lsb_release command
        output data source of the OS distribution.

        For details, see :func:`distro.lsb_release_attr`.
        """
        return self._lsb_release_info.get(attribute, "")

    def distro_release_attr(self, attribute: str) -> str:
        """
        Return a single named information item from the distro release file
        data source of the OS distribution.

        For details, see :func:`distro.distro_release_attr`.
        """
        return self._distro_release_info.get(attribute, "")

    def uname_attr(self, attribute: str) -> str:
        """
        Return a single named information item from the uname command
        output data source of the OS distribution.

        For details, see :func:`distro.uname_attr`.
        """
        return self._uname_info.get(attribute, "")

    @cached_property
    def _os_release_info(self) -> Dict[str, str]:
        """
        Get the information items from the specified os-release file.

        Returns:
            A dictionary containing all information items.
        """
        if os.path.isfile(self.os_release_file):
            with open(self.os_release_file, encoding="utf-8") as release_file:
                return self._parse_os_release_content(release_file)
        return {}

    @staticmethod
    def _parse_os_release_content(lines: TextIO) -> Dict[str, str]:
        """
        Parse the lines of an os-release file.

        Parameters:

        * lines: Iterable through the lines in the os-release file.
                 Each line must be a unicode string or a UTF-8 encoded byte
                 string.

        Returns:
            A dictionary containing all information items.
        """
        props = {}
        lexer = shlex.shlex(lines, posix=True)
        lexer.whitespace_split = True

        tokens = list(lexer)
        for token in tokens:
            # At this point, all shell-like parsing has been done (i.e.
            # comments processed, quotes and backslash escape sequences
            # processed, multi-line values assembled, trailing newlines
            # stripped, etc.), so the tokens are now either:
            # * variable assignments: var=value
            # * commands or their arguments (not allowed in os-release)
            # Ignore any tokens that are not variable assignments
            if "=" in token:
                k, v = token.split("=", 1)
                props[k.lower()] = v

        if "version" in props:
            # extract release codename (if any) from version attribute
            match = re.search(r"\((\D+)\)|,\s*(\D+)", props["version"])
            if match:
                release_codename = match.group(1) or match.group(2)
                props["codename"] = props["release_codename"] = release_codename

        if "version_codename" in props:
            # os-release added a version_codename field.  Use that in
            # preference to anything else Note that some distros purposefully
            # do not have code names.  They should be setting
            # version_codename=""
            props["codename"] = props["version_codename"]
        elif "ubuntu_codename" in props:
            # Same as above but a non-standard field name used on older Ubuntus
            props["codename"] = props["ubuntu_codename"]

        return props

    @cached_property
    def _lsb_release_info(self) -> Dict[str, str]:
        """
        Get the information items from the lsb_release command output.

        Returns:
            A dictionary containing all information items.
        """
        if not self.include_lsb:
            return {}
        try:
            cmd = ("lsb_release", "-a")
            stdout = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        # Command not found or lsb_release returned error
        except (OSError, subprocess.CalledProcessError):
            return {}
        content = self._to_str(stdout).splitlines()
        return self._parse_lsb_release_content(content)

    @staticmethod
    def _parse_lsb_release_content(lines: Iterable[str]) -> Dict[str, str]:
        """
        Parse the output of the lsb_release command.

        Parameters:

        * lines: Iterable through the lines of the lsb_release output.
                 Each line must be a unicode string or a UTF-8 encoded byte
                 string.

        Returns:
            A dictionary containing all information items.
        """
        props = {}
        for line in lines:
            kv = line.strip("\n").split(":", 1)
            if len(kv) != 2:
                # Ignore lines without colon.
                continue
            k, v = kv
            props.update({k.replace(" ", "_").lower(): v.strip()})
        return props

    @cached_property
    def _uname_info(self) -> Dict[str, str]:
        if not self.include_uname:
            return {}
        try:
            cmd = ("uname", "-rs")
            stdout = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        except OSError:
            return {}
        content = self._to_str(stdout).splitlines()
        return self._parse_uname_content(content)

    @cached_property
    def _oslevel_info(self) -> str:
        if not self.include_oslevel:
            return ""
        try:
            stdout = subprocess.check_output("oslevel", stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError):
            return ""
        return self._to_str(stdout).strip()

    @cached_property
    def _debian_version(self) -> str:
        try:
            with open(
                os.path.join(self.etc_dir, "debian_version"), encoding="ascii"
            ) as fp:
                return fp.readline().rstrip()
        except FileNotFoundError:
            return ""

    @staticmethod
    def _parse_uname_content(lines: Sequence[str]) -> Dict[str, str]:
        if not lines:
            return {}
        props = {}
        match = re.search(r"^([^\s]+)\s+([\d\.]+)", lines[0].strip())
        if match:
            name, version = match.groups()

            # This is to prevent the Linux kernel version from
            # appearing as the 'best' version on otherwise
            # identifiable distributions.
            if name == "Linux":
                return {}
            props["id"] = name.lower()
            props["name"] = name
            props["release"] = version
        return props

    @staticmethod
    def _to_str(bytestring: bytes) -> str:
        encoding = sys.getfilesystemencoding()
        return bytestring.decode(encoding)

    @cached_property
    def _distro_release_info(self) -> Dict[str, str]:
        """
        Get the information items from the specified distro release file.

        Returns:
            A dictionary containing all information items.
        """
        if self.distro_release_file:
            # If it was specified, we use it and parse what we can, even if
            # its file name or content does not match the expected pattern.
            distro_info = self._parse_distro_release_file(self.distro_release_file)
            basename = os.path.basename(self.distro_release_file)
            # The file name pattern for user-specified distro release files
            # is somewhat more tolerant (compared to when searching for the
            # file), because we want to use what was specified as best as
            # possible.
            match = _DISTRO_RELEASE_BASENAME_PATTERN.match(basename)
        else:
            try:
                basenames = [
                    basename
                    for basename in os.listdir(self.etc_dir)
                    if basename not in _DISTRO_RELEASE_IGNORE_BASENAMES
                    and os.path.isfile(os.path.join(self.etc_dir, basename))
                ]
                # We sort for repeatability in cases where there are multiple
                # distro specific files; e.g. CentOS, Oracle, Enterprise all
                # containing `redhat-release` on top of their own.
                basenames.sort()
            except OSError:
                # This may occur when /etc is not readable but we can't be
                # sure about the *-release files. Check common entries of
                # /etc for information. If they turn out to not be there the
                # error is handled in `_parse_distro_release_file()`.
                basenames = _DISTRO_RELEASE_BASENAMES
            for basename in basenames:
                match = _DISTRO_RELEASE_BASENAME_PATTERN.match(basename)
                if match is None:
                    continue
                filepath = os.path.join(self.etc_dir, basename)
                distro_info = self._parse_distro_release_file(filepath)
                # The name is always present if the pattern matches.
                if "name" not in distro_info:
                    continue
                self.distro_release_file = filepath
                break
            else:  # the loop didn't "break": no candidate.
                return {}

        if match is not None:
            distro_info["id"] = match.group(1)

        # CloudLinux < 7: manually enrich info with proper id.
        if "cloudlinux" in distro_info.get("name", "").lower():
            distro_info["id"] = "cloudlinux"

        return distro_info

    def _parse_distro_release_file(self, filepath: str) -> Dict[str, str]:
        """
        Parse a distro release file.

        Parameters:

        * filepath: Path name of the distro release file.

        Returns:
            A dictionary containing all information items.
        """
        try:
            with open(filepath, encoding="utf-8") as fp:
                # Only parse the first line. For instance, on SLES there
                # are multiple lines. We don't want them...
                return self._parse_distro_release_content(fp.readline())
        except OSError:
            # Ignore not being able to read a specific, seemingly version
            # related file.
            # See https://github.com/python-distro/distro/issues/162
            return {}

    @staticmethod
    def _parse_distro_release_content(line: str) -> Dict[str, str]:
        """
        Parse a line from a distro release file.

        Parameters:
        * line: Line from the distro release file. Must be a unicode string
                or a UTF-8 encoded byte string.

        Returns:
            A dictionary containing all information items.
        """
        matches = _DISTRO_RELEASE_CONTENT_REVERSED_PATTERN.match(line.strip()[::-1])
        distro_info = {}
        if matches:
            # regexp ensures non-None
            distro_info["name"] = matches.group(3)[::-1]
            if matches.group(2):
                distro_info["version_id"] = matches.group(2)[::-1]
            if matches.group(1):
                distro_info["codename"] = matches.group(1)[::-1]
        elif line:
            distro_info["name"] = line.strip()
        return distro_info


_distro = LinuxDistribution()


def main() -> None:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    parser = argparse.ArgumentParser(description="OS distro info tool")
    parser.add_argument(
        "--json", "-j", help="Output in machine readable format", action="store_true"
    )

    parser.add_argument(
        "--root-dir",
        "-r",
        type=str,
        dest="root_dir",
        help="Path to the root filesystem directory (defaults to /)",
    )

    args = parser.parse_args()

    if args.root_dir:
        dist = LinuxDistribution(
            include_lsb=False,
            include_uname=False,
            include_oslevel=False,
            root_dir=args.root_dir,
        )
    else:
        dist = _distro

    if args.json:
        logger.info(json.dumps(dist.info(), indent=4, sort_keys=True))
    else:
        logger.info("Name: %s", dist.name(pretty=True))
        distribution_version = dist.version(pretty=True)
        logger.info("Version: %s", distribution_version)
        distribution_codename = dist.codename()
        logger.info("Codename: %s", distribution_codename)


if __name__ == "__main__":
    main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\sql\coercions.py ===
# sql/coercions.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: allow-untyped-defs, allow-untyped-calls

from __future__ import annotations

import collections.abc as collections_abc
import numbers
import re
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

from . import roles
from . import visitors
from ._typing import is_from_clause
from .base import ExecutableOption
from .base import Options
from .cache_key import HasCacheKey
from .visitors import Visitable
from .. import exc
from .. import inspection
from .. import util
from ..util.typing import Literal

if typing.TYPE_CHECKING:
    # elements lambdas schema selectable are set by __init__
    from . import elements
    from . import lambdas
    from . import schema
    from . import selectable
    from ._typing import _ColumnExpressionArgument
    from ._typing import _ColumnsClauseArgument
    from ._typing import _DDLColumnArgument
    from ._typing import _DMLTableArgument
    from ._typing import _FromClauseArgument
    from .dml import _DMLTableElement
    from .elements import BindParameter
    from .elements import ClauseElement
    from .elements import ColumnClause
    from .elements import ColumnElement
    from .elements import NamedColumn
    from .elements import SQLCoreOperations
    from .elements import TextClause
    from .schema import Column
    from .selectable import _ColumnsClauseElement
    from .selectable import _JoinTargetProtocol
    from .selectable import FromClause
    from .selectable import HasCTE
    from .selectable import SelectBase
    from .selectable import Subquery
    from .visitors import _TraverseCallableType

_SR = TypeVar("_SR", bound=roles.SQLRole)
_F = TypeVar("_F", bound=Callable[..., Any])
_StringOnlyR = TypeVar("_StringOnlyR", bound=roles.StringRole)
_T = TypeVar("_T", bound=Any)


def _is_literal(element: Any) -> bool:
    """Return whether or not the element is a "literal" in the context
    of a SQL expression construct.

    """

    return not isinstance(
        element,
        (Visitable, schema.SchemaEventTarget),
    ) and not hasattr(element, "__clause_element__")


def _deep_is_literal(element):
    """Return whether or not the element is a "literal" in the context
    of a SQL expression construct.

    does a deeper more esoteric check than _is_literal.   is used
    for lambda elements that have to distinguish values that would
    be bound vs. not without any context.

    """

    if isinstance(element, collections_abc.Sequence) and not isinstance(
        element, str
    ):
        for elem in element:
            if not _deep_is_literal(elem):
                return False
        else:
            return True

    return (
        not isinstance(
            element,
            (
                Visitable,
                schema.SchemaEventTarget,
                HasCacheKey,
                Options,
                util.langhelpers.symbol,
            ),
        )
        and not hasattr(element, "__clause_element__")
        and (
            not isinstance(element, type)
            or not issubclass(element, HasCacheKey)
        )
    )


def _document_text_coercion(
    paramname: str, meth_rst: str, param_rst: str
) -> Callable[[_F], _F]:
    return util.add_parameter_text(
        paramname,
        (
            ".. warning:: "
            "The %s argument to %s can be passed as a Python string argument, "
            "which will be treated "
            "as **trusted SQL text** and rendered as given.  **DO NOT PASS "
            "UNTRUSTED INPUT TO THIS PARAMETER**."
        )
        % (param_rst, meth_rst),
    )


def _expression_collection_was_a_list(
    attrname: str,
    fnname: str,
    args: Union[Sequence[_T], Sequence[Sequence[_T]]],
) -> Sequence[_T]:
    if args and isinstance(args[0], (list, set, dict)) and len(args) == 1:
        if isinstance(args[0], list):
            raise exc.ArgumentError(
                f'The "{attrname}" argument to {fnname}(), when '
                "referring to a sequence "
                "of items, is now passed as a series of positional "
                "elements, rather than as a list. "
            )
        return cast("Sequence[_T]", args[0])

    return cast("Sequence[_T]", args)


@overload
def expect(
    role: Type[roles.TruncatedLabelRole],
    element: Any,
    **kw: Any,
) -> str: ...


@overload
def expect(
    role: Type[roles.DMLColumnRole],
    element: Any,
    *,
    as_key: Literal[True] = ...,
    **kw: Any,
) -> str: ...


@overload
def expect(
    role: Type[roles.LiteralValueRole],
    element: Any,
    **kw: Any,
) -> BindParameter[Any]: ...


@overload
def expect(
    role: Type[roles.DDLReferredColumnRole],
    element: Any,
    **kw: Any,
) -> Union[Column[Any], str]: ...


@overload
def expect(
    role: Type[roles.DDLConstraintColumnRole],
    element: Any,
    **kw: Any,
) -> Union[Column[Any], str]: ...


@overload
def expect(
    role: Type[roles.StatementOptionRole],
    element: Any,
    **kw: Any,
) -> Union[ColumnElement[Any], TextClause]: ...


@overload
def expect(
    role: Type[roles.LabeledColumnExprRole[Any]],
    element: _ColumnExpressionArgument[_T],
    **kw: Any,
) -> NamedColumn[_T]: ...


@overload
def expect(
    role: Union[
        Type[roles.ExpressionElementRole[Any]],
        Type[roles.LimitOffsetRole],
        Type[roles.WhereHavingRole],
    ],
    element: _ColumnExpressionArgument[_T],
    **kw: Any,
) -> ColumnElement[_T]: ...


@overload
def expect(
    role: Union[
        Type[roles.ExpressionElementRole[Any]],
        Type[roles.LimitOffsetRole],
        Type[roles.WhereHavingRole],
        Type[roles.OnClauseRole],
        Type[roles.ColumnArgumentRole],
    ],
    element: Any,
    **kw: Any,
) -> ColumnElement[Any]: ...


@overload
def expect(
    role: Type[roles.DMLTableRole],
    element: _DMLTableArgument,
    **kw: Any,
) -> _DMLTableElement: ...


@overload
def expect(
    role: Type[roles.HasCTERole],
    element: HasCTE,
    **kw: Any,
) -> HasCTE: ...


@overload
def expect(
    role: Type[roles.SelectStatementRole],
    element: SelectBase,
    **kw: Any,
) -> SelectBase: ...


@overload
def expect(
    role: Type[roles.FromClauseRole],
    element: _FromClauseArgument,
    **kw: Any,
) -> FromClause: ...


@overload
def expect(
    role: Type[roles.FromClauseRole],
    element: SelectBase,
    *,
    explicit_subquery: Literal[True] = ...,
    **kw: Any,
) -> Subquery: ...


@overload
def expect(
    role: Type[roles.ColumnsClauseRole],
    element: _ColumnsClauseArgument[Any],
    **kw: Any,
) -> _ColumnsClauseElement: ...


@overload
def expect(
    role: Type[roles.JoinTargetRole],
    element: _JoinTargetProtocol,
    **kw: Any,
) -> _JoinTargetProtocol: ...


# catchall for not-yet-implemented overloads
@overload
def expect(
    role: Type[_SR],
    element: Any,
    **kw: Any,
) -> Any: ...


def expect(
    role: Type[_SR],
    element: Any,
    *,
    apply_propagate_attrs: Optional[ClauseElement] = None,
    argname: Optional[str] = None,
    post_inspect: bool = False,
    disable_inspection: bool = False,
    **kw: Any,
) -> Any:
    if (
        role.allows_lambda
        # note callable() will not invoke a __getattr__() method, whereas
        # hasattr(obj, "__call__") will. by keeping the callable() check here
        # we prevent most needless calls to hasattr()  and therefore
        # __getattr__(), which is present on ColumnElement.
        and callable(element)
        and hasattr(element, "__code__")
    ):
        return lambdas.LambdaElement(
            element,
            role,
            lambdas.LambdaOptions(**kw),
            apply_propagate_attrs=apply_propagate_attrs,
        )

    # major case is that we are given a ClauseElement already, skip more
    # elaborate logic up front if possible
    impl = _impl_lookup[role]

    original_element = element

    if not isinstance(
        element,
        (
            elements.CompilerElement,
            schema.SchemaItem,
            schema.FetchedValue,
            lambdas.PyWrapper,
        ),
    ):
        resolved = None

        if impl._resolve_literal_only:
            resolved = impl._literal_coercion(element, **kw)
        else:
            original_element = element

            is_clause_element = False

            # this is a special performance optimization for ORM
            # joins used by JoinTargetImpl that we don't go through the
            # work of creating __clause_element__() when we only need the
            # original QueryableAttribute, as the former will do clause
            # adaption and all that which is just thrown away here.
            if (
                impl._skip_clauseelement_for_target_match
                and isinstance(element, role)
                and hasattr(element, "__clause_element__")
            ):
                is_clause_element = True
            else:
                while hasattr(element, "__clause_element__"):
                    is_clause_element = True

                    if not getattr(element, "is_clause_element", False):
                        element = element.__clause_element__()
                    else:
                        break

            if not is_clause_element:
                if impl._use_inspection and not disable_inspection:
                    insp = inspection.inspect(element, raiseerr=False)
                    if insp is not None:
                        if post_inspect:
                            insp._post_inspect
                        try:
                            resolved = insp.__clause_element__()
                        except AttributeError:
                            impl._raise_for_expected(original_element, argname)

                if resolved is None:
                    resolved = impl._literal_coercion(
                        element, argname=argname, **kw
                    )
            else:
                resolved = element
    elif isinstance(element, lambdas.PyWrapper):
        resolved = element._sa__py_wrapper_literal(**kw)
    else:
        resolved = element

    if apply_propagate_attrs is not None:
        if typing.TYPE_CHECKING:
            assert isinstance(resolved, (SQLCoreOperations, ClauseElement))

        if not apply_propagate_attrs._propagate_attrs and getattr(
            resolved, "_propagate_attrs", None
        ):
            apply_propagate_attrs._propagate_attrs = resolved._propagate_attrs

    if impl._role_class in resolved.__class__.__mro__:
        if impl._post_coercion:
            resolved = impl._post_coercion(
                resolved,
                argname=argname,
                original_element=original_element,
                **kw,
            )
        return resolved
    else:
        return impl._implicit_coercions(
            original_element, resolved, argname=argname, **kw
        )


def expect_as_key(
    role: Type[roles.DMLColumnRole], element: Any, **kw: Any
) -> str:
    kw.pop("as_key", None)
    return expect(role, element, as_key=True, **kw)


def expect_col_expression_collection(
    role: Type[roles.DDLConstraintColumnRole],
    expressions: Iterable[_DDLColumnArgument],
) -> Iterator[
    Tuple[
        Union[str, Column[Any]],
        Optional[ColumnClause[Any]],
        Optional[str],
        Optional[Union[Column[Any], str]],
    ]
]:
    for expr in expressions:
        strname = None
        column = None

        resolved: Union[Column[Any], str] = expect(role, expr)
        if isinstance(resolved, str):
            assert isinstance(expr, str)
            strname = resolved = expr
        else:
            cols: List[Column[Any]] = []
            col_append: _TraverseCallableType[Column[Any]] = cols.append
            visitors.traverse(resolved, {}, {"column": col_append})
            if cols:
                column = cols[0]
        add_element = column if column is not None else strname

        yield resolved, column, strname, add_element


class RoleImpl:
    __slots__ = ("_role_class", "name", "_use_inspection")

    def _literal_coercion(self, element, **kw):
        raise NotImplementedError()

    _post_coercion: Any = None
    _resolve_literal_only = False
    _skip_clauseelement_for_target_match = False

    def __init__(self, role_class):
        self._role_class = role_class
        self.name = role_class._role_name
        self._use_inspection = issubclass(role_class, roles.UsesInspection)

    def _implicit_coercions(
        self,
        element: Any,
        resolved: Any,
        argname: Optional[str] = None,
        **kw: Any,
    ) -> Any:
        self._raise_for_expected(element, argname, resolved)

    def _raise_for_expected(
        self,
        element: Any,
        argname: Optional[str] = None,
        resolved: Optional[Any] = None,
        *,
        advice: Optional[str] = None,
        code: Optional[str] = None,
        err: Optional[Exception] = None,
        **kw: Any,
    ) -> NoReturn:
        if resolved is not None and resolved is not element:
            got = "%r object resolved from %r object" % (resolved, element)
        else:
            got = repr(element)

        if argname:
            msg = "%s expected for argument %r; got %s." % (
                self.name,
                argname,
                got,
            )
        else:
            msg = "%s expected, got %s." % (self.name, got)

        if advice:
            msg += " " + advice

        raise exc.ArgumentError(msg, code=code) from err


class _Deannotate:
    __slots__ = ()

    def _post_coercion(self, resolved, **kw):
        from .util import _deep_deannotate

        return _deep_deannotate(resolved)


class _StringOnly:
    __slots__ = ()

    _resolve_literal_only = True


class _ReturnsStringKey(RoleImpl):
    __slots__ = ()

    def _implicit_coercions(self, element, resolved, argname=None, **kw):
        if isinstance(element, str):
            return element
        else:
            self._raise_for_expected(element, argname, resolved)

    def _literal_coercion(self, element, **kw):
        return element


class _ColumnCoercions(RoleImpl):
    __slots__ = ()

    def _warn_for_scalar_subquery_coercion(self):
        util.warn(
            "implicitly coercing SELECT object to scalar subquery; "
            "please use the .scalar_subquery() method to produce a scalar "
            "subquery.",
        )

    def _implicit_coercions(self, element, resolved, argname=None, **kw):
        original_element = element
        if not getattr(resolved, "is_clause_element", False):
            self._raise_for_expected(original_element, argname, resolved)
        elif resolved._is_select_base:
            self._warn_for_scalar_subquery_coercion()
            return resolved.scalar_subquery()
        elif resolved._is_from_clause and isinstance(
            resolved, selectable.Subquery
        ):
            self._warn_for_scalar_subquery_coercion()
            return resolved.element.scalar_subquery()
        elif self._role_class.allows_lambda and resolved._is_lambda_element:
            return resolved
        else:
            self._raise_for_expected(original_element, argname, resolved)


def _no_text_coercion(
    element: Any,
    argname: Optional[str] = None,
    exc_cls: Type[exc.SQLAlchemyError] = exc.ArgumentError,
    extra: Optional[str] = None,
    err: Optional[Exception] = None,
) -> NoReturn:
    raise exc_cls(
        "%(extra)sTextual SQL expression %(expr)r %(argname)sshould be "
        "explicitly declared as text(%(expr)r)"
        % {
            "expr": util.ellipses_string(element),
            "argname": "for argument %s" % (argname,) if argname else "",
            "extra": "%s " % extra if extra else "",
        }
    ) from err


class _NoTextCoercion(RoleImpl):
    __slots__ = ()

    def _literal_coercion(self, element, *, argname=None, **kw):
        if isinstance(element, str) and issubclass(
            elements.TextClause, self._role_class
        ):
            _no_text_coercion(element, argname)
        else:
            self._raise_for_expected(element, argname)


class _CoerceLiterals(RoleImpl):
    __slots__ = ()
    _coerce_consts = False
    _coerce_star = False
    _coerce_numerics = False

    def _text_coercion(self, element, argname=None):
        return _no_text_coercion(element, argname)

    def _literal_coercion(self, element, *, argname=None, **kw):
        if isinstance(element, str):
            if self._coerce_star and element == "*":
                return elements.ColumnClause("*", is_literal=True)
            else:
                return self._text_coercion(element, argname, **kw)

        if self._coerce_consts:
            if element is None:
                return elements.Null()
            elif element is False:
                return elements.False_()
            elif element is True:
                return elements.True_()

        if self._coerce_numerics and isinstance(element, (numbers.Number)):
            return elements.ColumnClause(str(element), is_literal=True)

        self._raise_for_expected(element, argname)


class LiteralValueImpl(RoleImpl):
    _resolve_literal_only = True

    def _implicit_coercions(
        self,
        element,
        resolved,
        argname=None,
        *,
        type_=None,
        literal_execute=False,
        **kw,
    ):
        if not _is_literal(resolved):
            self._raise_for_expected(
                element, resolved=resolved, argname=argname, **kw
            )

        return elements.BindParameter(
            None,
            element,
            type_=type_,
            unique=True,
            literal_execute=literal_execute,
        )

    def _literal_coercion(self, element, **kw):
        return element


class _SelectIsNotFrom(RoleImpl):
    __slots__ = ()

    def _raise_for_expected(
        self,
        element: Any,
        argname: Optional[str] = None,
        resolved: Optional[Any] = None,
        *,
        advice: Optional[str] = None,
        code: Optional[str] = None,
        err: Optional[Exception] = None,
        **kw: Any,
    ) -> NoReturn:
        if (
            not advice
            and isinstance(element, roles.SelectStatementRole)
            or isinstance(resolved, roles.SelectStatementRole)
        ):
            advice = (
                "To create a "
                "FROM clause from a %s object, use the .subquery() method."
                % (resolved.__class__ if resolved is not None else element,)
            )
            code = "89ve"
        else:
            code = None

        super()._raise_for_expected(
            element,
            argname=argname,
            resolved=resolved,
            advice=advice,
            code=code,
            err=err,
            **kw,
        )
        # never reached
        assert False


class HasCacheKeyImpl(RoleImpl):
    __slots__ = ()

    def _implicit_coercions(
        self,
        element: Any,
        resolved: Any,
        argname: Optional[str] = None,
        **kw: Any,
    ) -> Any:
        if isinstance(element, HasCacheKey):
            return element
        else:
            self._raise_for_expected(element, argname, resolved)

    def _literal_coercion(self, element, **kw):
        return element


class ExecutableOptionImpl(RoleImpl):
    __slots__ = ()

    def _implicit_coercions(
        self,
        element: Any,
        resolved: Any,
        argname: Optional[str] = None,
        **kw: Any,
    ) -> Any:
        if isinstance(element, ExecutableOption):
            return element
        else:
            self._raise_for_expected(element, argname, resolved)

    def _literal_coercion(self, element, **kw):
        return element


class ExpressionElementImpl(_ColumnCoercions, RoleImpl):
    __slots__ = ()

    def _literal_coercion(
        self, element, *, name=None, type_=None, is_crud=False, **kw
    ):
        if (
            element is None
            and not is_crud
            and (type_ is None or not type_.should_evaluate_none)
        ):
            # TODO: there's no test coverage now for the
            # "should_evaluate_none" part of this, as outside of "crud" this
            # codepath is not normally used except in some special cases
            return elements.Null()
        else:
            try:
                return elements.BindParameter(
                    name, element, type_, unique=True, _is_crud=is_crud
                )
            except exc.ArgumentError as err:
                self._raise_for_expected(element, err=err)

    def _raise_for_expected(self, element, argname=None, resolved=None, **kw):
        # select uses implicit coercion with warning instead of raising
        if isinstance(element, selectable.Values):
            advice = (
                "To create a column expression from a VALUES clause, "
                "use the .scalar_values() method."
            )
        elif isinstance(element, roles.AnonymizedFromClauseRole):
            advice = (
                "To create a column expression from a FROM clause row "
                "as a whole, use the .table_valued() method."
            )
        else:
            advice = None

        return super()._raise_for_expected(
            element, argname=argname, resolved=resolved, advice=advice, **kw
        )


class BinaryElementImpl(ExpressionElementImpl, RoleImpl):
    __slots__ = ()

    def _literal_coercion(  # type: ignore[override]
        self,
        element,
        *,
        expr,
        operator,
        bindparam_type=None,
        argname=None,
        **kw,
    ):
        try:
            return expr._bind_param(operator, element, type_=bindparam_type)
        except exc.ArgumentError as err:
            self._raise_for_expected(element, err=err)

    def _post_coercion(self, resolved, *, expr, bindparam_type=None, **kw):
        if resolved.type._isnull and not expr.type._isnull:
            resolved = resolved._with_binary_element_type(
                bindparam_type if bindparam_type is not None else expr.type
            )
        return resolved


class InElementImpl(RoleImpl):
    __slots__ = ()

    def _implicit_coercions(
        self,
        element: Any,
        resolved: Any,
        argname: Optional[str] = None,
        **kw: Any,
    ) -> Any:
        if resolved._is_from_clause:
            if (
                isinstance(resolved, selectable.Alias)
                and resolved.element._is_select_base
            ):
                self._warn_for_implicit_coercion(resolved)
                return self._post_coercion(resolved.element, **kw)
            else:
                self._warn_for_implicit_coercion(resolved)
                return self._post_coercion(resolved.select(), **kw)
        else:
            self._raise_for_expected(element, argname, resolved)

    def _warn_for_implicit_coercion(self, elem):
        util.warn(
            "Coercing %s object into a select() for use in IN(); "
            "please pass a select() construct explicitly"
            % (elem.__class__.__name__)
        )

    @util.preload_module("sqlalchemy.sql.elements")
    def _literal_coercion(self, element, *, expr, operator, **kw):
        if util.is_non_string_iterable(element):
            non_literal_expressions: Dict[
                Optional[_ColumnExpressionArgument[Any]],
                _ColumnExpressionArgument[Any],
            ] = {}
            element = list(element)
            for o in element:
                if not _is_literal(o):
                    if not isinstance(
                        o, util.preloaded.sql_elements.ColumnElement
                    ) and not hasattr(o, "__clause_element__"):
                        self._raise_for_expected(element, **kw)

                    else:
                        non_literal_expressions[o] = o

            if non_literal_expressions:
                return elements.ClauseList(
                    *[
                        (
                            non_literal_expressions[o]
                            if o in non_literal_expressions
                            else expr._bind_param(operator, o)
                        )
                        for o in element
                    ]
                )
            else:
                return expr._bind_param(operator, element, expanding=True)

        else:
            self._raise_for_expected(element, **kw)

    def _post_coercion(self, element, *, expr, operator, **kw):
        if element._is_select_base:
            # for IN, we are doing scalar_subquery() coercion without
            # a warning
            return element.scalar_subquery()
        elif isinstance(element, elements.ClauseList):
            assert not len(element.clauses) == 0
            return element.self_group(against=operator)

        elif isinstance(element, elements.BindParameter):
            element = element._clone(maintain_key=True)
            element.expanding = True
            element.expand_op = operator

            return element
        elif isinstance(element, selectable.Values):
            return element.scalar_values()
        else:
            return element


class OnClauseImpl(_ColumnCoercions, RoleImpl):
    __slots__ = ()

    _coerce_consts = True

    def _literal_coercion(self, element, **kw):
        self._raise_for_expected(element)

    def _post_coercion(self, resolved, *, original_element=None, **kw):
        # this is a hack right now as we want to use coercion on an
        # ORM InstrumentedAttribute, but we want to return the object
        # itself if it is one, not its clause element.
        # ORM context _join and _legacy_join() would need to be improved
        # to look for annotations in a clause element form.
        if isinstance(original_element, roles.JoinTargetRole):
            return original_element
        return resolved


class WhereHavingImpl(_CoerceLiterals, _ColumnCoercions, RoleImpl):
    __slots__ = ()

    _coerce_consts = True

    def _text_coercion(self, element, argname=None):
        return _no_text_coercion(element, argname)


class StatementOptionImpl(_CoerceLiterals, RoleImpl):
    __slots__ = ()

    _coerce_consts = True

    def _text_coercion(self, element, argname=None):
        return elements.TextClause(element)


class ColumnArgumentImpl(_NoTextCoercion, RoleImpl):
    __slots__ = ()


class ColumnArgumentOrKeyImpl(_ReturnsStringKey, RoleImpl):
    __slots__ = ()


class StrAsPlainColumnImpl(_CoerceLiterals, RoleImpl):
    __slots__ = ()

    def _text_coercion(self, element, argname=None):
        return elements.ColumnClause(element)


class ByOfImpl(_CoerceLiterals, _ColumnCoercions, RoleImpl, roles.ByOfRole):
    __slots__ = ()

    _coerce_consts = True

    def _text_coercion(self, element, argname=None):
        return elements._textual_label_reference(element)


class OrderByImpl(ByOfImpl, RoleImpl):
    __slots__ = ()

    def _post_coercion(self, resolved, **kw):
        if (
            isinstance(resolved, self._role_class)
            and resolved._order_by_label_element is not None
        ):
            return elements._label_reference(resolved)
        else:
            return resolved


class GroupByImpl(ByOfImpl, RoleImpl):
    __slots__ = ()

    def _implicit_coercions(
        self,
        element: Any,
        resolved: Any,
        argname: Optional[str] = None,
        **kw: Any,
    ) -> Any:
        if is_from_clause(resolved):
            return elements.ClauseList(*resolved.c)
        else:
            return resolved


class DMLColumnImpl(_ReturnsStringKey, RoleImpl):
    __slots__ = ()

    def _post_coercion(self, element, *, as_key=False, **kw):
        if as_key:
            return element.key
        else:
            return element


class ConstExprImpl(RoleImpl):
    __slots__ = ()

    def _literal_coercion(self, element, *, argname=None, **kw):
        if element is None:
            return elements.Null()
        elif element is False:
            return elements.False_()
        elif element is True:
            return elements.True_()
        else:
            self._raise_for_expected(element, argname)


class TruncatedLabelImpl(_StringOnly, RoleImpl):
    __slots__ = ()

    def _implicit_coercions(
        self,
        element: Any,
        resolved: Any,
        argname: Optional[str] = None,
        **kw: Any,
    ) -> Any:
        if isinstance(element, str):
            return resolved
        else:
            self._raise_for_expected(element, argname, resolved)

    def _literal_coercion(self, element, **kw):
        """coerce the given value to :class:`._truncated_label`.

        Existing :class:`._truncated_label` and
        :class:`._anonymous_label` objects are passed
        unchanged.
        """

        if isinstance(element, elements._truncated_label):
            return element
        else:
            return elements._truncated_label(element)


class DDLExpressionImpl(_Deannotate, _CoerceLiterals, RoleImpl):
    __slots__ = ()

    _coerce_consts = True

    def _text_coercion(self, element, argname=None):
        # see #5754 for why we can't easily deprecate this coercion.
        # essentially expressions like postgresql_where would have to be
        # text() as they come back from reflection and we don't want to
        # have text() elements wired into the inspection dictionaries.
        return elements.TextClause(element)


class DDLConstraintColumnImpl(_Deannotate, _ReturnsStringKey, RoleImpl):
    __slots__ = ()


class DDLReferredColumnImpl(DDLConstraintColumnImpl):
    __slots__ = ()


class LimitOffsetImpl(RoleImpl):
    __slots__ = ()

    def _implicit_coercions(
        self,
        element: Any,
        resolved: Any,
        argname: Optional[str] = None,
        **kw: Any,
    ) -> Any:
        if resolved is None:
            return None
        else:
            self._raise_for_expected(element, argname, resolved)

    def _literal_coercion(  # type: ignore[override]
        self, element, *, name, type_, **kw
    ):
        if element is None:
            return None
        else:
            value = util.asint(element)
            return selectable._OffsetLimitParam(
                name, value, type_=type_, unique=True
            )


class LabeledColumnExprImpl(ExpressionElementImpl):
    __slots__ = ()

    def _implicit_coercions(
        self,
        element: Any,
        resolved: Any,
        argname: Optional[str] = None,
        **kw: Any,
    ) -> Any:
        if isinstance(resolved, roles.ExpressionElementRole):
            return resolved.label(None)
        else:
            new = super()._implicit_coercions(
                element, resolved, argname=argname, **kw
            )
            if isinstance(new, roles.ExpressionElementRole):
                return new.label(None)
            else:
                self._raise_for_expected(element, argname, resolved)


class ColumnsClauseImpl(_SelectIsNotFrom, _CoerceLiterals, RoleImpl):
    __slots__ = ()

    _coerce_consts = True
    _coerce_numerics = True
    _coerce_star = True

    _guess_straight_column = re.compile(r"^\w\S*$", re.I)

    def _raise_for_expected(
        self, element, argname=None, resolved=None, *, advice=None, **kw
    ):
        if not advice and isinstance(element, list):
            advice = (
                f"Did you mean to say select("
                f"{', '.join(repr(e) for e in element)})?"
            )

        return super()._raise_for_expected(
            element, argname=argname, resolved=resolved, advice=advice, **kw
        )

    def _text_coercion(self, element, argname=None):
        element = str(element)

        guess_is_literal = not self._guess_straight_column.match(element)
        raise exc.ArgumentError(
            "Textual column expression %(column)r %(argname)sshould be "
            "explicitly declared with text(%(column)r), "
            "or use %(literal_column)s(%(column)r) "
            "for more specificity"
            % {
                "column": util.ellipses_string(element),
                "argname": "for argument %s" % (argname,) if argname else "",
                "literal_column": (
                    "literal_column" if guess_is_literal else "column"
                ),
            }
        )


class ReturnsRowsImpl(RoleImpl):
    __slots__ = ()


class StatementImpl(_CoerceLiterals, RoleImpl):
    __slots__ = ()

    def _post_coercion(
        self, resolved, *, original_element, argname=None, **kw
    ):
        if resolved is not original_element and not isinstance(
            original_element, str
        ):
            # use same method as Connection uses; this will later raise
            # ObjectNotExecutableError
            try:
                original_element._execute_on_connection
            except AttributeError:
                util.warn_deprecated(
                    "Object %r should not be used directly in a SQL statement "
                    "context, such as passing to methods such as "
                    "session.execute().  This usage will be disallowed in a "
                    "future release.  "
                    "Please use Core select() / update() / delete() etc. "
                    "with Session.execute() and other statement execution "
                    "methods." % original_element,
                    "1.4",
                )

        return resolved

    def _implicit_coercions(
        self,
        element: Any,
        resolved: Any,
        argname: Optional[str] = None,
        **kw: Any,
    ) -> Any:
        if resolved._is_lambda_element:
            return resolved
        else:
            return super()._implicit_coercions(
                element, resolved, argname=argname, **kw
            )


class SelectStatementImpl(_NoTextCoercion, RoleImpl):
    __slots__ = ()

    def _implicit_coercions(
        self,
        element: Any,
        resolved: Any,
        argname: Optional[str] = None,
        **kw: Any,
    ) -> Any:
        if resolved._is_text_clause:
            return resolved.columns()
        else:
            self._raise_for_expected(element, argname, resolved)


class HasCTEImpl(ReturnsRowsImpl):
    __slots__ = ()


class IsCTEImpl(RoleImpl):
    __slots__ = ()


class JoinTargetImpl(RoleImpl):
    __slots__ = ()

    _skip_clauseelement_for_target_match = True

    def _literal_coercion(self, element, *, argname=None, **kw):
        self._raise_for_expected(element, argname)

    def _implicit_coercions(
        self,
        element: Any,
        resolved: Any,
        argname: Optional[str] = None,
        *,
        legacy: bool = False,
        **kw: Any,
    ) -> Any:
        if isinstance(element, roles.JoinTargetRole):
            # note that this codepath no longer occurs as of
            # #6550, unless JoinTargetImpl._skip_clauseelement_for_target_match
            # were set to False.
            return element
        elif legacy and resolved._is_select_base:
            util.warn_deprecated(
                "Implicit coercion of SELECT and textual SELECT "
                "constructs into FROM clauses is deprecated; please call "
                ".subquery() on any Core select or ORM Query object in "
                "order to produce a subquery object.",
                version="1.4",
            )
            # TODO: doing _implicit_subquery here causes tests to fail,
            # how was this working before?  probably that ORM
            # join logic treated it as a select and subquery would happen
            # in _ORMJoin->Join
            return resolved
        else:
            self._raise_for_expected(element, argname, resolved)


class FromClauseImpl(_SelectIsNotFrom, _NoTextCoercion, RoleImpl):
    __slots__ = ()

    def _implicit_coercions(
        self,
        element: Any,
        resolved: Any,
        argname: Optional[str] = None,
        *,
        explicit_subquery: bool = False,
        allow_select: bool = True,
        **kw: Any,
    ) -> Any:
        if resolved._is_select_base:
            if explicit_subquery:
                return resolved.subquery()
            elif allow_select:
                util.warn_deprecated(
                    "Implicit coercion of SELECT and textual SELECT "
                    "constructs into FROM clauses is deprecated; please call "
                    ".subquery() on any Core select or ORM Query object in "
                    "order to produce a subquery object.",
                    version="1.4",
                )
                return resolved._implicit_subquery
        elif resolved._is_text_clause:
            return resolved
        else:
            self._raise_for_expected(element, argname, resolved)

    def _post_coercion(self, element, *, deannotate=False, **kw):
        if deannotate:
            return element._deannotate()
        else:
            return element


class StrictFromClauseImpl(FromClauseImpl):
    __slots__ = ()

    def _implicit_coercions(
        self,
        element: Any,
        resolved: Any,
        argname: Optional[str] = None,
        *,
        allow_select: bool = False,
        **kw: Any,
    ) -> Any:
        if resolved._is_select_base and allow_select:
            util.warn_deprecated(
                "Implicit coercion of SELECT and textual SELECT constructs "
                "into FROM clauses is deprecated; please call .subquery() "
                "on any Core select or ORM Query object in order to produce a "
                "subquery object.",
                version="1.4",
            )
            return resolved._implicit_subquery
        else:
            self._raise_for_expected(element, argname, resolved)


class AnonymizedFromClauseImpl(StrictFromClauseImpl):
    __slots__ = ()

    def _post_coercion(self, element, *, flat=False, name=None, **kw):
        assert name is None

        return element._anonymous_fromclause(flat=flat)


class DMLTableImpl(_SelectIsNotFrom, _NoTextCoercion, RoleImpl):
    __slots__ = ()

    def _post_coercion(self, element, **kw):
        if "dml_table" in element._annotations:
            return element._annotations["dml_table"]
        else:
            return element


class DMLSelectImpl(_NoTextCoercion, RoleImpl):
    __slots__ = ()

    def _implicit_coercions(
        self,
        element: Any,
        resolved: Any,
        argname: Optional[str] = None,
        **kw: Any,
    ) -> Any:
        if resolved._is_from_clause:
            if (
                isinstance(resolved, selectable.Alias)
                and resolved.element._is_select_base
            ):
                return resolved.element
            else:
                return resolved.select()
        else:
            self._raise_for_expected(element, argname, resolved)


class CompoundElementImpl(_NoTextCoercion, RoleImpl):
    __slots__ = ()

    def _raise_for_expected(self, element, argname=None, resolved=None, **kw):
        if isinstance(element, roles.FromClauseRole):
            if element._is_subquery:
                advice = (
                    "Use the plain select() object without "
                    "calling .subquery() or .alias()."
                )
            else:
                advice = (
                    "To SELECT from any FROM clause, use the .select() method."
                )
        else:
            advice = None
        return super()._raise_for_expected(
            element, argname=argname, resolved=resolved, advice=advice, **kw
        )


_impl_lookup = {}


for name in dir(roles):
    cls = getattr(roles, name)
    if name.endswith("Role"):
        name = name.replace("Role", "Impl")
        if name in globals():
            impl = globals()[name](cls)
            _impl_lookup[cls] = impl

if not TYPE_CHECKING:
    ee_impl = _impl_lookup[roles.ExpressionElementRole]

    for py_type in (int, bool, str, float):
        _impl_lookup[roles.ExpressionElementRole[py_type]] = ee_impl

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\base.py ===
"""
Base and utility classes for pandas objects.
"""

from __future__ import annotations

import textwrap
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    cast,
    final,
    overload,
)
import warnings

import numpy as np

from pandas._config import using_copy_on_write

from pandas._libs import lib
from pandas._typing import (
    AxisInt,
    DtypeObj,
    IndexLabel,
    NDFrameT,
    Self,
    Shape,
    npt,
)
from pandas.compat import PYPY
from pandas.compat.numpy import function as nv
from pandas.errors import AbstractMethodError
from pandas.util._decorators import (
    cache_readonly,
    doc,
)
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.cast import can_hold_element
from pandas.core.dtypes.common import (
    is_object_dtype,
    is_scalar,
)
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCIndex,
    ABCMultiIndex,
    ABCSeries,
)
from pandas.core.dtypes.missing import (
    isna,
    remove_na_arraylike,
)

from pandas.core import (
    algorithms,
    nanops,
    ops,
)
from pandas.core.accessor import DirNamesMixin
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays import ExtensionArray
from pandas.core.construction import (
    ensure_wrapped_if_datetimelike,
    extract_array,
)

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Iterator,
    )

    from pandas._typing import (
        DropKeep,
        NumpySorter,
        NumpyValueArrayLike,
        ScalarLike_co,
    )

    from pandas import (
        DataFrame,
        Index,
        Series,
    )


_shared_docs: dict[str, str] = {}
_indexops_doc_kwargs = {
    "klass": "IndexOpsMixin",
    "inplace": "",
    "unique": "IndexOpsMixin",
    "duplicated": "IndexOpsMixin",
}


class PandasObject(DirNamesMixin):
    """
    Baseclass for various pandas objects.
    """

    # results from calls to methods decorated with cache_readonly get added to _cache
    _cache: dict[str, Any]

    @property
    def _constructor(self):
        """
        Class constructor (for this class it's just `__class__`).
        """
        return type(self)

    def __repr__(self) -> str:
        """
        Return a string representation for a particular object.
        """
        # Should be overwritten by base classes
        return object.__repr__(self)

    def _reset_cache(self, key: str | None = None) -> None:
        """
        Reset cached properties. If ``key`` is passed, only clears that key.
        """
        if not hasattr(self, "_cache"):
            return
        if key is None:
            self._cache.clear()
        else:
            self._cache.pop(key, None)

    def __sizeof__(self) -> int:
        """
        Generates the total memory usage for an object that returns
        either a value or Series of values
        """
        memory_usage = getattr(self, "memory_usage", None)
        if memory_usage:
            mem = memory_usage(deep=True)  # pylint: disable=not-callable
            return int(mem if is_scalar(mem) else mem.sum())

        # no memory_usage attribute, so fall back to object's 'sizeof'
        return super().__sizeof__()


class NoNewAttributesMixin:
    """
    Mixin which prevents adding new attributes.

    Prevents additional attributes via xxx.attribute = "something" after a
    call to `self.__freeze()`. Mainly used to prevent the user from using
    wrong attributes on an accessor (`Series.cat/.str/.dt`).

    If you really want to add a new attribute at a later time, you need to use
    `object.__setattr__(self, key, value)`.
    """

    def _freeze(self) -> None:
        """
        Prevents setting additional attributes.
        """
        object.__setattr__(self, "__frozen", True)

    # prevent adding any attribute via s.xxx.new_attribute = ...
    def __setattr__(self, key: str, value) -> None:
        # _cache is used by a decorator
        # We need to check both 1.) cls.__dict__ and 2.) getattr(self, key)
        # because
        # 1.) getattr is false for attributes that raise errors
        # 2.) cls.__dict__ doesn't traverse into base classes
        if getattr(self, "__frozen", False) and not (
            key == "_cache"
            or key in type(self).__dict__
            or getattr(self, key, None) is not None
        ):
            raise AttributeError(f"You cannot add any new attribute '{key}'")
        object.__setattr__(self, key, value)


class SelectionMixin(Generic[NDFrameT]):
    """
    mixin implementing the selection & aggregation interface on a group-like
    object sub-classes need to define: obj, exclusions
    """

    obj: NDFrameT
    _selection: IndexLabel | None = None
    exclusions: frozenset[Hashable]
    _internal_names = ["_cache", "__setstate__"]
    _internal_names_set = set(_internal_names)

    @final
    @property
    def _selection_list(self):
        if not isinstance(
            self._selection, (list, tuple, ABCSeries, ABCIndex, np.ndarray)
        ):
            return [self._selection]
        return self._selection

    @cache_readonly
    def _selected_obj(self):
        if self._selection is None or isinstance(self.obj, ABCSeries):
            return self.obj
        else:
            return self.obj[self._selection]

    @final
    @cache_readonly
    def ndim(self) -> int:
        return self._selected_obj.ndim

    @final
    @cache_readonly
    def _obj_with_exclusions(self):
        if isinstance(self.obj, ABCSeries):
            return self.obj

        if self._selection is not None:
            return self.obj._getitem_nocopy(self._selection_list)

        if len(self.exclusions) > 0:
            # equivalent to `self.obj.drop(self.exclusions, axis=1)
            #  but this avoids consolidating and making a copy
            # TODO: following GH#45287 can we now use .drop directly without
            #  making a copy?
            return self.obj._drop_axis(self.exclusions, axis=1, only_slice=True)
        else:
            return self.obj

    def __getitem__(self, key):
        if self._selection is not None:
            raise IndexError(f"Column(s) {self._selection} already selected")

        if isinstance(key, (list, tuple, ABCSeries, ABCIndex, np.ndarray)):
            if len(self.obj.columns.intersection(key)) != len(set(key)):
                bad_keys = list(set(key).difference(self.obj.columns))
                raise KeyError(f"Columns not found: {str(bad_keys)[1:-1]}")
            return self._gotitem(list(key), ndim=2)

        else:
            if key not in self.obj:
                raise KeyError(f"Column not found: {key}")
            ndim = self.obj[key].ndim
            return self._gotitem(key, ndim=ndim)

    def _gotitem(self, key, ndim: int, subset=None):
        """
        sub-classes to define
        return a sliced object

        Parameters
        ----------
        key : str / list of selections
        ndim : {1, 2}
            requested ndim of result
        subset : object, default None
            subset to act on
        """
        raise AbstractMethodError(self)

    @final
    def _infer_selection(self, key, subset: Series | DataFrame):
        """
        Infer the `selection` to pass to our constructor in _gotitem.
        """
        # Shared by Rolling and Resample
        selection = None
        if subset.ndim == 2 and (
            (lib.is_scalar(key) and key in subset) or lib.is_list_like(key)
        ):
            selection = key
        elif subset.ndim == 1 and lib.is_scalar(key) and key == subset.name:
            selection = key
        return selection

    def aggregate(self, func, *args, **kwargs):
        raise AbstractMethodError(self)

    agg = aggregate


class IndexOpsMixin(OpsMixin):
    """
    Common ops mixin to support a unified interface / docs for Series / Index
    """

    # ndarray compatibility
    __array_priority__ = 1000
    _hidden_attrs: frozenset[str] = frozenset(
        ["tolist"]  # tolist is not deprecated, just suppressed in the __dir__
    )

    @property
    def dtype(self) -> DtypeObj:
        # must be defined here as a property for mypy
        raise AbstractMethodError(self)

    @property
    def _values(self) -> ExtensionArray | np.ndarray:
        # must be defined here as a property for mypy
        raise AbstractMethodError(self)

    @final
    def transpose(self, *args, **kwargs) -> Self:
        """
        Return the transpose, which is by definition self.

        Returns
        -------
        %(klass)s
        """
        nv.validate_transpose(args, kwargs)
        return self

    T = property(
        transpose,
        doc="""
        Return the transpose, which is by definition self.

        Examples
        --------
        For Series:

        >>> s = pd.Series(['Ant', 'Bear', 'Cow'])
        >>> s
        0     Ant
        1    Bear
        2     Cow
        dtype: object
        >>> s.T
        0     Ant
        1    Bear
        2     Cow
        dtype: object

        For Index:

        >>> idx = pd.Index([1, 2, 3])
        >>> idx.T
        Index([1, 2, 3], dtype='int64')
        """,
    )

    @property
    def shape(self) -> Shape:
        """
        Return a tuple of the shape of the underlying data.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> s.shape
        (3,)
        """
        return self._values.shape

    def __len__(self) -> int:
        # We need this defined here for mypy
        raise AbstractMethodError(self)

    # Temporarily avoid using `-> Literal[1]:` because of an IPython (jedi) bug
    # https://github.com/ipython/ipython/issues/14412
    # https://github.com/davidhalter/jedi/issues/1990
    @property
    def ndim(self) -> int:
        """
        Number of dimensions of the underlying data, by definition 1.

        Examples
        --------
        >>> s = pd.Series(['Ant', 'Bear', 'Cow'])
        >>> s
        0     Ant
        1    Bear
        2     Cow
        dtype: object
        >>> s.ndim
        1

        For Index:

        >>> idx = pd.Index([1, 2, 3])
        >>> idx
        Index([1, 2, 3], dtype='int64')
        >>> idx.ndim
        1
        """
        return 1

    @final
    def item(self):
        """
        Return the first element of the underlying data as a Python scalar.

        Returns
        -------
        scalar
            The first element of Series or Index.

        Raises
        ------
        ValueError
            If the data is not length = 1.

        Examples
        --------
        >>> s = pd.Series([1])
        >>> s.item()
        1

        For an index:

        >>> s = pd.Series([1], index=['a'])
        >>> s.index.item()
        'a'
        """
        if len(self) == 1:
            return next(iter(self))
        raise ValueError("can only convert an array of size 1 to a Python scalar")

    @property
    def nbytes(self) -> int:
        """
        Return the number of bytes in the underlying data.

        Examples
        --------
        For Series:

        >>> s = pd.Series(['Ant', 'Bear', 'Cow'])
        >>> s
        0     Ant
        1    Bear
        2     Cow
        dtype: object
        >>> s.nbytes
        24

        For Index:

        >>> idx = pd.Index([1, 2, 3])
        >>> idx
        Index([1, 2, 3], dtype='int64')
        >>> idx.nbytes
        24
        """
        return self._values.nbytes

    @property
    def size(self) -> int:
        """
        Return the number of elements in the underlying data.

        Examples
        --------
        For Series:

        >>> s = pd.Series(['Ant', 'Bear', 'Cow'])
        >>> s
        0     Ant
        1    Bear
        2     Cow
        dtype: object
        >>> s.size
        3

        For Index:

        >>> idx = pd.Index([1, 2, 3])
        >>> idx
        Index([1, 2, 3], dtype='int64')
        >>> idx.size
        3
        """
        return len(self._values)

    @property
    def array(self) -> ExtensionArray:
        """
        The ExtensionArray of the data backing this Series or Index.

        Returns
        -------
        ExtensionArray
            An ExtensionArray of the values stored within. For extension
            types, this is the actual array. For NumPy native types, this
            is a thin (no copy) wrapper around :class:`numpy.ndarray`.

            ``.array`` differs from ``.values``, which may require converting
            the data to a different form.

        See Also
        --------
        Index.to_numpy : Similar method that always returns a NumPy array.
        Series.to_numpy : Similar method that always returns a NumPy array.

        Notes
        -----
        This table lays out the different array types for each extension
        dtype within pandas.

        ================== =============================
        dtype              array type
        ================== =============================
        category           Categorical
        period             PeriodArray
        interval           IntervalArray
        IntegerNA          IntegerArray
        string             StringArray
        boolean            BooleanArray
        datetime64[ns, tz] DatetimeArray
        ================== =============================

        For any 3rd-party extension types, the array type will be an
        ExtensionArray.

        For all remaining dtypes ``.array`` will be a
        :class:`arrays.NumpyExtensionArray` wrapping the actual ndarray
        stored within. If you absolutely need a NumPy array (possibly with
        copying / coercing data), then use :meth:`Series.to_numpy` instead.

        Examples
        --------
        For regular NumPy types like int, and float, a NumpyExtensionArray
        is returned.

        >>> pd.Series([1, 2, 3]).array
        <NumpyExtensionArray>
        [1, 2, 3]
        Length: 3, dtype: int64

        For extension types, like Categorical, the actual ExtensionArray
        is returned

        >>> ser = pd.Series(pd.Categorical(['a', 'b', 'a']))
        >>> ser.array
        ['a', 'b', 'a']
        Categories (2, object): ['a', 'b']
        """
        raise AbstractMethodError(self)

    @final
    def to_numpy(
        self,
        dtype: npt.DTypeLike | None = None,
        copy: bool = False,
        na_value: object = lib.no_default,
        **kwargs,
    ) -> np.ndarray:
        """
        A NumPy ndarray representing the values in this Series or Index.

        Parameters
        ----------
        dtype : str or numpy.dtype, optional
            The dtype to pass to :meth:`numpy.asarray`.
        copy : bool, default False
            Whether to ensure that the returned value is not a view on
            another array. Note that ``copy=False`` does not *ensure* that
            ``to_numpy()`` is no-copy. Rather, ``copy=True`` ensure that
            a copy is made, even if not strictly necessary.
        na_value : Any, optional
            The value to use for missing values. The default value depends
            on `dtype` and the type of the array.
        **kwargs
            Additional keywords passed through to the ``to_numpy`` method
            of the underlying array (for extension arrays).

        Returns
        -------
        numpy.ndarray

        See Also
        --------
        Series.array : Get the actual data stored within.
        Index.array : Get the actual data stored within.
        DataFrame.to_numpy : Similar method for DataFrame.

        Notes
        -----
        The returned array will be the same up to equality (values equal
        in `self` will be equal in the returned array; likewise for values
        that are not equal). When `self` contains an ExtensionArray, the
        dtype may be different. For example, for a category-dtype Series,
        ``to_numpy()`` will return a NumPy array and the categorical dtype
        will be lost.

        For NumPy dtypes, this will be a reference to the actual data stored
        in this Series or Index (assuming ``copy=False``). Modifying the result
        in place will modify the data stored in the Series or Index (not that
        we recommend doing that).

        For extension types, ``to_numpy()`` *may* require copying data and
        coercing the result to a NumPy type (possibly object), which may be
        expensive. When you need a no-copy reference to the underlying data,
        :attr:`Series.array` should be used instead.

        This table lays out the different dtypes and default return types of
        ``to_numpy()`` for various dtypes within pandas.

        ================== ================================
        dtype              array type
        ================== ================================
        category[T]        ndarray[T] (same dtype as input)
        period             ndarray[object] (Periods)
        interval           ndarray[object] (Intervals)
        IntegerNA          ndarray[object]
        datetime64[ns]     datetime64[ns]
        datetime64[ns, tz] ndarray[object] (Timestamps)
        ================== ================================

        Examples
        --------
        >>> ser = pd.Series(pd.Categorical(['a', 'b', 'a']))
        >>> ser.to_numpy()
        array(['a', 'b', 'a'], dtype=object)

        Specify the `dtype` to control how datetime-aware data is represented.
        Use ``dtype=object`` to return an ndarray of pandas :class:`Timestamp`
        objects, each with the correct ``tz``.

        >>> ser = pd.Series(pd.date_range('2000', periods=2, tz="CET"))
        >>> ser.to_numpy(dtype=object)
        array([Timestamp('2000-01-01 00:00:00+0100', tz='CET'),
               Timestamp('2000-01-02 00:00:00+0100', tz='CET')],
              dtype=object)

        Or ``dtype='datetime64[ns]'`` to return an ndarray of native
        datetime64 values. The values are converted to UTC and the timezone
        info is dropped.

        >>> ser.to_numpy(dtype="datetime64[ns]")
        ... # doctest: +ELLIPSIS
        array(['1999-12-31T23:00:00.000000000', '2000-01-01T23:00:00...'],
              dtype='datetime64[ns]')
        """
        if isinstance(self.dtype, ExtensionDtype):
            return self.array.to_numpy(dtype, copy=copy, na_value=na_value, **kwargs)
        elif kwargs:
            bad_keys = next(iter(kwargs.keys()))
            raise TypeError(
                f"to_numpy() got an unexpected keyword argument '{bad_keys}'"
            )

        fillna = (
            na_value is not lib.no_default
            # no need to fillna with np.nan if we already have a float dtype
            and not (na_value is np.nan and np.issubdtype(self.dtype, np.floating))
        )

        values = self._values
        if fillna:
            if not can_hold_element(values, na_value):
                # if we can't hold the na_value asarray either makes a copy or we
                # error before modifying values. The asarray later on thus won't make
                # another copy
                values = np.asarray(values, dtype=dtype)
            else:
                values = values.copy()

            values[np.asanyarray(isna(self))] = na_value

        result = np.asarray(values, dtype=dtype)

        if (copy and not fillna) or (not copy and using_copy_on_write()):
            if np.shares_memory(self._values[:2], result[:2]):
                # Take slices to improve performance of check
                if using_copy_on_write() and not copy:
                    result = result.view()
                    result.flags.writeable = False
                else:
                    result = result.copy()

        return result

    @final
    @property
    def empty(self) -> bool:
        return not self.size

    @doc(op="max", oppose="min", value="largest")
    def argmax(
        self, axis: AxisInt | None = None, skipna: bool = True, *args, **kwargs
    ) -> int:
        """
        Return int position of the {value} value in the Series.

        If the {op}imum is achieved in multiple locations,
        the first row position is returned.

        Parameters
        ----------
        axis : {{None}}
            Unused. Parameter needed for compatibility with DataFrame.
        skipna : bool, default True
            Exclude NA/null values when showing the result.
        *args, **kwargs
            Additional arguments and keywords for compatibility with NumPy.

        Returns
        -------
        int
            Row position of the {op}imum value.

        See Also
        --------
        Series.arg{op} : Return position of the {op}imum value.
        Series.arg{oppose} : Return position of the {oppose}imum value.
        numpy.ndarray.arg{op} : Equivalent method for numpy arrays.
        Series.idxmax : Return index label of the maximum values.
        Series.idxmin : Return index label of the minimum values.

        Examples
        --------
        Consider dataset containing cereal calories

        >>> s = pd.Series({{'Corn Flakes': 100.0, 'Almond Delight': 110.0,
        ...                'Cinnamon Toast Crunch': 120.0, 'Cocoa Puff': 110.0}})
        >>> s
        Corn Flakes              100.0
        Almond Delight           110.0
        Cinnamon Toast Crunch    120.0
        Cocoa Puff               110.0
        dtype: float64

        >>> s.argmax()
        2
        >>> s.argmin()
        0

        The maximum cereal calories is the third element and
        the minimum cereal calories is the first element,
        since series is zero-indexed.
        """
        delegate = self._values
        nv.validate_minmax_axis(axis)
        skipna = nv.validate_argmax_with_skipna(skipna, args, kwargs)

        if isinstance(delegate, ExtensionArray):
            if not skipna and delegate.isna().any():
                warnings.warn(
                    f"The behavior of {type(self).__name__}.argmax/argmin "
                    "with skipna=False and NAs, or with all-NAs is deprecated. "
                    "In a future version this will raise ValueError.",
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )
                return -1
            else:
                return delegate.argmax()
        else:
            result = nanops.nanargmax(delegate, skipna=skipna)
            if result == -1:
                warnings.warn(
                    f"The behavior of {type(self).__name__}.argmax/argmin "
                    "with skipna=False and NAs, or with all-NAs is deprecated. "
                    "In a future version this will raise ValueError.",
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )
            # error: Incompatible return value type (got "Union[int, ndarray]", expected
            # "int")
            return result  # type: ignore[return-value]

    @doc(argmax, op="min", oppose="max", value="smallest")
    def argmin(
        self, axis: AxisInt | None = None, skipna: bool = True, *args, **kwargs
    ) -> int:
        delegate = self._values
        nv.validate_minmax_axis(axis)
        skipna = nv.validate_argmin_with_skipna(skipna, args, kwargs)

        if isinstance(delegate, ExtensionArray):
            if not skipna and delegate.isna().any():
                warnings.warn(
                    f"The behavior of {type(self).__name__}.argmax/argmin "
                    "with skipna=False and NAs, or with all-NAs is deprecated. "
                    "In a future version this will raise ValueError.",
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )
                return -1
            else:
                return delegate.argmin()
        else:
            result = nanops.nanargmin(delegate, skipna=skipna)
            if result == -1:
                warnings.warn(
                    f"The behavior of {type(self).__name__}.argmax/argmin "
                    "with skipna=False and NAs, or with all-NAs is deprecated. "
                    "In a future version this will raise ValueError.",
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )
            # error: Incompatible return value type (got "Union[int, ndarray]", expected
            # "int")
            return result  # type: ignore[return-value]

    def tolist(self):
        """
        Return a list of the values.

        These are each a scalar type, which is a Python scalar
        (for str, int, float) or a pandas scalar
        (for Timestamp/Timedelta/Interval/Period)

        Returns
        -------
        list

        See Also
        --------
        numpy.ndarray.tolist : Return the array as an a.ndim-levels deep
            nested list of Python scalars.

        Examples
        --------
        For Series

        >>> s = pd.Series([1, 2, 3])
        >>> s.to_list()
        [1, 2, 3]

        For Index:

        >>> idx = pd.Index([1, 2, 3])
        >>> idx
        Index([1, 2, 3], dtype='int64')

        >>> idx.to_list()
        [1, 2, 3]
        """
        return self._values.tolist()

    to_list = tolist

    def __iter__(self) -> Iterator:
        """
        Return an iterator of the values.

        These are each a scalar type, which is a Python scalar
        (for str, int, float) or a pandas scalar
        (for Timestamp/Timedelta/Interval/Period)

        Returns
        -------
        iterator

        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> for x in s:
        ...     print(x)
        1
        2
        3
        """
        # We are explicitly making element iterators.
        if not isinstance(self._values, np.ndarray):
            # Check type instead of dtype to catch DTA/TDA
            return iter(self._values)
        else:
            return map(self._values.item, range(self._values.size))

    @cache_readonly
    def hasnans(self) -> bool:
        """
        Return True if there are any NaNs.

        Enables various performance speedups.

        Returns
        -------
        bool

        Examples
        --------
        >>> s = pd.Series([1, 2, 3, None])
        >>> s
        0    1.0
        1    2.0
        2    3.0
        3    NaN
        dtype: float64
        >>> s.hasnans
        True
        """
        # error: Item "bool" of "Union[bool, ndarray[Any, dtype[bool_]], NDFrame]"
        # has no attribute "any"
        return bool(isna(self).any())  # type: ignore[union-attr]

    @final
    def _map_values(self, mapper, na_action=None, convert: bool = True):
        """
        An internal function that maps values using the input
        correspondence (which can be a dict, Series, or function).

        Parameters
        ----------
        mapper : function, dict, or Series
            The input correspondence object
        na_action : {None, 'ignore'}
            If 'ignore', propagate NA values, without passing them to the
            mapping function
        convert : bool, default True
            Try to find better dtype for elementwise function results. If
            False, leave as dtype=object. Note that the dtype is always
            preserved for some extension array dtypes, such as Categorical.

        Returns
        -------
        Union[Index, MultiIndex], inferred
            The output of the mapping function applied to the index.
            If the function returns a tuple with more than one element
            a MultiIndex will be returned.
        """
        arr = self._values

        if isinstance(arr, ExtensionArray):
            return arr.map(mapper, na_action=na_action)

        return algorithms.map_array(arr, mapper, na_action=na_action, convert=convert)

    @final
    def value_counts(
        self,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        bins=None,
        dropna: bool = True,
    ) -> Series:
        """
        Return a Series containing counts of unique values.

        The resulting object will be in descending order so that the
        first element is the most frequently-occurring element.
        Excludes NA values by default.

        Parameters
        ----------
        normalize : bool, default False
            If True then the object returned will contain the relative
            frequencies of the unique values.
        sort : bool, default True
            Sort by frequencies when True. Preserve the order of the data when False.
        ascending : bool, default False
            Sort in ascending order.
        bins : int, optional
            Rather than count values, group them into half-open bins,
            a convenience for ``pd.cut``, only works with numeric data.
        dropna : bool, default True
            Don't include counts of NaN.

        Returns
        -------
        Series

        See Also
        --------
        Series.count: Number of non-NA elements in a Series.
        DataFrame.count: Number of non-NA elements in a DataFrame.
        DataFrame.value_counts: Equivalent method on DataFrames.

        Examples
        --------
        >>> index = pd.Index([3, 1, 2, 3, 4, np.nan])
        >>> index.value_counts()
        3.0    2
        1.0    1
        2.0    1
        4.0    1
        Name: count, dtype: int64

        With `normalize` set to `True`, returns the relative frequency by
        dividing all values by the sum of values.

        >>> s = pd.Series([3, 1, 2, 3, 4, np.nan])
        >>> s.value_counts(normalize=True)
        3.0    0.4
        1.0    0.2
        2.0    0.2
        4.0    0.2
        Name: proportion, dtype: float64

        **bins**

        Bins can be useful for going from a continuous variable to a
        categorical variable; instead of counting unique
        apparitions of values, divide the index in the specified
        number of half-open bins.

        >>> s.value_counts(bins=3)
        (0.996, 2.0]    2
        (2.0, 3.0]      2
        (3.0, 4.0]      1
        Name: count, dtype: int64

        **dropna**

        With `dropna` set to `False` we can also see NaN index values.

        >>> s.value_counts(dropna=False)
        3.0    2
        1.0    1
        2.0    1
        4.0    1
        NaN    1
        Name: count, dtype: int64
        """
        return algorithms.value_counts_internal(
            self,
            sort=sort,
            ascending=ascending,
            normalize=normalize,
            bins=bins,
            dropna=dropna,
        )

    def unique(self):
        values = self._values
        if not isinstance(values, np.ndarray):
            # i.e. ExtensionArray
            result = values.unique()
        else:
            result = algorithms.unique1d(values)
        return result

    @final
    def nunique(self, dropna: bool = True) -> int:
        """
        Return number of unique elements in the object.

        Excludes NA values by default.

        Parameters
        ----------
        dropna : bool, default True
            Don't include NaN in the count.

        Returns
        -------
        int

        See Also
        --------
        DataFrame.nunique: Method nunique for DataFrame.
        Series.count: Count non-NA/null observations in the Series.

        Examples
        --------
        >>> s = pd.Series([1, 3, 5, 7, 7])
        >>> s
        0    1
        1    3
        2    5
        3    7
        4    7
        dtype: int64

        >>> s.nunique()
        4
        """
        uniqs = self.unique()
        if dropna:
            uniqs = remove_na_arraylike(uniqs)
        return len(uniqs)

    @property
    def is_unique(self) -> bool:
        """
        Return boolean if values in the object are unique.

        Returns
        -------
        bool

        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> s.is_unique
        True

        >>> s = pd.Series([1, 2, 3, 1])
        >>> s.is_unique
        False
        """
        return self.nunique(dropna=False) == len(self)

    @property
    def is_monotonic_increasing(self) -> bool:
        """
        Return boolean if values in the object are monotonically increasing.

        Returns
        -------
        bool

        Examples
        --------
        >>> s = pd.Series([1, 2, 2])
        >>> s.is_monotonic_increasing
        True

        >>> s = pd.Series([3, 2, 1])
        >>> s.is_monotonic_increasing
        False
        """
        from pandas import Index

        return Index(self).is_monotonic_increasing

    @property
    def is_monotonic_decreasing(self) -> bool:
        """
        Return boolean if values in the object are monotonically decreasing.

        Returns
        -------
        bool

        Examples
        --------
        >>> s = pd.Series([3, 2, 2, 1])
        >>> s.is_monotonic_decreasing
        True

        >>> s = pd.Series([1, 2, 3])
        >>> s.is_monotonic_decreasing
        False
        """
        from pandas import Index

        return Index(self).is_monotonic_decreasing

    @final
    def _memory_usage(self, deep: bool = False) -> int:
        """
        Memory usage of the values.

        Parameters
        ----------
        deep : bool, default False
            Introspect the data deeply, interrogate
            `object` dtypes for system-level memory consumption.

        Returns
        -------
        bytes used

        See Also
        --------
        numpy.ndarray.nbytes : Total bytes consumed by the elements of the
            array.

        Notes
        -----
        Memory usage does not include memory consumed by elements that
        are not components of the array if deep=False or if used on PyPy

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3])
        >>> idx.memory_usage()
        24
        """
        if hasattr(self.array, "memory_usage"):
            return self.array.memory_usage(  # pyright: ignore[reportGeneralTypeIssues]
                deep=deep,
            )

        v = self.array.nbytes
        if deep and is_object_dtype(self.dtype) and not PYPY:
            values = cast(np.ndarray, self._values)
            v += lib.memory_usage_of_objects(values)
        return v

    @doc(
        algorithms.factorize,
        values="",
        order="",
        size_hint="",
        sort=textwrap.dedent(
            """\
            sort : bool, default False
                Sort `uniques` and shuffle `codes` to maintain the
                relationship.
            """
        ),
    )
    def factorize(
        self,
        sort: bool = False,
        use_na_sentinel: bool = True,
    ) -> tuple[npt.NDArray[np.intp], Index]:
        codes, uniques = algorithms.factorize(
            self._values, sort=sort, use_na_sentinel=use_na_sentinel
        )
        if uniques.dtype == np.float16:
            uniques = uniques.astype(np.float32)

        if isinstance(self, ABCMultiIndex):
            # preserve MultiIndex
            uniques = self._constructor(uniques)
        else:
            from pandas import Index

            try:
                uniques = Index(uniques, dtype=self.dtype)
            except NotImplementedError:
                # not all dtypes are supported in Index that are allowed for Series
                # e.g. float16 or bytes
                uniques = Index(uniques)
        return codes, uniques

    _shared_docs[
        "searchsorted"
    ] = """
        Find indices where elements should be inserted to maintain order.

        Find the indices into a sorted {klass} `self` such that, if the
        corresponding elements in `value` were inserted before the indices,
        the order of `self` would be preserved.

        .. note::

            The {klass} *must* be monotonically sorted, otherwise
            wrong locations will likely be returned. Pandas does *not*
            check this for you.

        Parameters
        ----------
        value : array-like or scalar
            Values to insert into `self`.
        side : {{'left', 'right'}}, optional
            If 'left', the index of the first suitable location found is given.
            If 'right', return the last such index.  If there is no suitable
            index, return either 0 or N (where N is the length of `self`).
        sorter : 1-D array-like, optional
            Optional array of integer indices that sort `self` into ascending
            order. They are typically the result of ``np.argsort``.

        Returns
        -------
        int or array of int
            A scalar or array of insertion points with the
            same shape as `value`.

        See Also
        --------
        sort_values : Sort by the values along either axis.
        numpy.searchsorted : Similar method from NumPy.

        Notes
        -----
        Binary search is used to find the required insertion points.

        Examples
        --------
        >>> ser = pd.Series([1, 2, 3])
        >>> ser
        0    1
        1    2
        2    3
        dtype: int64

        >>> ser.searchsorted(4)
        3

        >>> ser.searchsorted([0, 4])
        array([0, 3])

        >>> ser.searchsorted([1, 3], side='left')
        array([0, 2])

        >>> ser.searchsorted([1, 3], side='right')
        array([1, 3])

        >>> ser = pd.Series(pd.to_datetime(['3/11/2000', '3/12/2000', '3/13/2000']))
        >>> ser
        0   2000-03-11
        1   2000-03-12
        2   2000-03-13
        dtype: datetime64[ns]

        >>> ser.searchsorted('3/14/2000')
        3

        >>> ser = pd.Categorical(
        ...     ['apple', 'bread', 'bread', 'cheese', 'milk'], ordered=True
        ... )
        >>> ser
        ['apple', 'bread', 'bread', 'cheese', 'milk']
        Categories (4, object): ['apple' < 'bread' < 'cheese' < 'milk']

        >>> ser.searchsorted('bread')
        1

        >>> ser.searchsorted(['bread'], side='right')
        array([3])

        If the values are not monotonically sorted, wrong locations
        may be returned:

        >>> ser = pd.Series([2, 1, 3])
        >>> ser
        0    2
        1    1
        2    3
        dtype: int64

        >>> ser.searchsorted(1)  # doctest: +SKIP
        0  # wrong result, correct would be 1
        """

    # This overload is needed so that the call to searchsorted in
    # pandas.core.resample.TimeGrouper._get_period_bins picks the correct result

    # error: Overloaded function signatures 1 and 2 overlap with incompatible
    # return types
    @overload
    def searchsorted(  # type: ignore[overload-overlap]
        self,
        value: ScalarLike_co,
        side: Literal["left", "right"] = ...,
        sorter: NumpySorter = ...,
    ) -> np.intp:
        ...

    @overload
    def searchsorted(
        self,
        value: npt.ArrayLike | ExtensionArray,
        side: Literal["left", "right"] = ...,
        sorter: NumpySorter = ...,
    ) -> npt.NDArray[np.intp]:
        ...

    @doc(_shared_docs["searchsorted"], klass="Index")
    def searchsorted(
        self,
        value: NumpyValueArrayLike | ExtensionArray,
        side: Literal["left", "right"] = "left",
        sorter: NumpySorter | None = None,
    ) -> npt.NDArray[np.intp] | np.intp:
        if isinstance(value, ABCDataFrame):
            msg = (
                "Value must be 1-D array-like or scalar, "
                f"{type(value).__name__} is not supported"
            )
            raise ValueError(msg)

        values = self._values
        if not isinstance(values, np.ndarray):
            # Going through EA.searchsorted directly improves performance GH#38083
            return values.searchsorted(value, side=side, sorter=sorter)

        return algorithms.searchsorted(
            values,
            value,
            side=side,
            sorter=sorter,
        )

    def drop_duplicates(self, *, keep: DropKeep = "first"):
        duplicated = self._duplicated(keep=keep)
        # error: Value of type "IndexOpsMixin" is not indexable
        return self[~duplicated]  # type: ignore[index]

    @final
    def _duplicated(self, keep: DropKeep = "first") -> npt.NDArray[np.bool_]:
        arr = self._values
        if isinstance(arr, ExtensionArray):
            return arr.duplicated(keep=keep)
        return algorithms.duplicated(arr, keep=keep)

    def _arith_method(self, other, op):
        res_name = ops.get_op_result_name(self, other)

        lvalues = self._values
        rvalues = extract_array(other, extract_numpy=True, extract_range=True)
        rvalues = ops.maybe_prepare_scalar_for_op(rvalues, lvalues.shape)
        rvalues = ensure_wrapped_if_datetimelike(rvalues)
        if isinstance(rvalues, range):
            rvalues = np.arange(rvalues.start, rvalues.stop, rvalues.step)

        with np.errstate(all="ignore"):
            result = ops.arithmetic_op(lvalues, rvalues, op)

        return self._construct_result(result, name=res_name)

    def _construct_result(self, result, name):
        """
        Construct an appropriately-wrapped result from the ArrayLike result
        of an arithmetic-like operation.
        """
        raise AbstractMethodError(self)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\styles\builtins.py ===
# Copyright (c) 2010-2024 openpyxl

# Builtins styles as defined in Part 4 Annex G.2

from .named_styles import NamedStyle
from openpyxl.xml.functions import fromstring


normal = """
  <namedStyle builtinId="0" name="Normal">
    <alignment/>
    <border>
      <left/>
      <right/>
      <top/>
      <bottom/>
      <diagonal/>
    </border>
    <fill>
      <patternFill/>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <color theme="1"/>
      <sz val="12"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>
"""

comma = """
  <namedStyle builtinId="3" name="Comma">
    <alignment/>
    <number_format>_-* #,##0.00\\ _$_-;\\-* #,##0.00\\ _$_-;_-* "-"??\\ _$_-;_-@_-</number_format>
    <border>
      <left/>
      <right/>
      <top/>
      <bottom/>
      <diagonal/>
    </border>
    <fill>
      <patternFill/>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <color theme="1"/>
      <sz val="12"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>
"""

comma_0 = """
  <namedStyle builtinId="6" name="Comma [0]">
    <alignment/>
    <number_format>_-* #,##0\\ _$_-;\\-* #,##0\\ _$_-;_-* "-"\\ _$_-;_-@_-</number_format>
    <border>
      <left/>
      <right/>
      <top/>
      <bottom/>
      <diagonal/>
    </border>
    <fill>
      <patternFill/>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <color theme="1"/>
      <sz val="12"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>
"""

currency = """
  <namedStyle builtinId="4" name="Currency">
    <alignment/>
    <number_format>_-* #,##0.00\\ "$"_-;\\-* #,##0.00\\ "$"_-;_-* "-"??\\ "$"_-;_-@_-</number_format>
    <border>
      <left/>
      <right/>
      <top/>
      <bottom/>
      <diagonal/>
    </border>
    <fill>
      <patternFill/>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <color theme="1"/>
      <sz val="12"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>
"""

currency_0 = """
  <namedStyle builtinId="7" name="Currency [0]">
    <alignment/>
    <number_format>_-* #,##0\\ "$"_-;\\-* #,##0\\ "$"_-;_-* "-"\\ "$"_-;_-@_-</number_format>
    <border>
      <left/>
      <right/>
      <top/>
      <bottom/>
      <diagonal/>
    </border>
    <fill>
      <patternFill/>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <color theme="1"/>
      <sz val="12"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>
"""

percent = """
  <namedStyle builtinId="5" name="Percent">
    <alignment/>
    <number_format>0%</number_format>
    <border>
      <left/>
      <right/>
      <top/>
      <bottom/>
      <diagonal/>
    </border>
    <fill>
      <patternFill/>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <color theme="1"/>
      <sz val="12"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>
"""

hyperlink = """
  <namedStyle builtinId="8" name="Hyperlink" >
    <alignment/>
    <border>
      <left/>
      <right/>
      <top/>
      <bottom/>
      <diagonal/>
    </border>
    <fill>
      <patternFill/>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <color theme="10"/>
      <sz val="12"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>"""

followed_hyperlink = """
  <namedStyle builtinId="9" name="Followed Hyperlink" >
    <alignment/>
    <border>
      <left/>
      <right/>
      <top/>
      <bottom/>
      <diagonal/>
    </border>
    <fill>
      <patternFill/>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <color theme="11"/>
      <sz val="12"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>"""

title = """
  <namedStyle builtinId="15" name="Title">
    <alignment/>
    <border>
      <left/>
      <right/>
      <top/>
      <bottom/>
      <diagonal/>
    </border>
    <fill>
      <patternFill/>
    </fill>
    <font>
      <name val="Cambria"/>
      <family val="2"/>
      <b val="1"/>
      <color theme="3"/>
      <sz val="18"/>
      <scheme val="major"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>
"""

headline_1 = """
  <namedStyle builtinId="16" name="Headline 1" >
    <alignment/>
    <border>
      <left/>
      <right/>
      <top/>
      <bottom style="thick">
        <color theme="4"/>
      </bottom>
      <diagonal/>
    </border>
    <fill>
      <patternFill/>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <b val="1"/>
      <color theme="3"/>
      <sz val="15"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>
"""

headline_2 = """
  <namedStyle builtinId="17" name="Headline 2" >
    <alignment/>
    <border>
      <left/>
      <right/>
      <top/>
      <bottom style="thick">
        <color theme="4" tint="0.5"/>
      </bottom>
      <diagonal/>
    </border>
    <fill>
      <patternFill/>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <b val="1"/>
      <color theme="3"/>
      <sz val="13"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>
"""

headline_3 = """
   <namedStyle builtinId="18" name="Headline 3" >
    <alignment/>
    <border>
      <left/>
      <right/>
      <top/>
      <bottom style="medium">
        <color theme="4" tint="0.4"/>
      </bottom>
      <diagonal/>
    </border>
    <fill>
      <patternFill/>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <b val="1"/>
      <color theme="3"/>
      <sz val="11"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>

"""

headline_4 = """
  <namedStyle builtinId="19" name="Headline 4">
    <alignment/>
    <border>
      <left/>
      <right/>
      <top/>
      <bottom/>
      <diagonal/>
    </border>
    <fill>
      <patternFill/>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <b val="1"/>
      <color theme="3"/>
      <sz val="11"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>
"""

good = """
  <namedStyle builtinId="26" name="Good" >
    <alignment/>
    <border>
      <left/>
      <right/>
      <top/>
      <bottom/>
      <diagonal/>
    </border>
    <fill>
      <patternFill patternType="solid">
        <fgColor rgb="FFC6EFCE"/>
      </patternFill>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <color rgb="FF006100"/>
      <sz val="12"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>
"""

bad = """
  <namedStyle builtinId="27" name="Bad" >
    <alignment/>
    <border>
      <left/>
      <right/>
      <top/>
      <bottom/>
      <diagonal/>
    </border>
    <fill>
      <patternFill patternType="solid">
        <fgColor rgb="FFFFC7CE"/>
      </patternFill>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <color rgb="FF9C0006"/>
      <sz val="12"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>
"""

neutral = """
  <namedStyle builtinId="28" name="Neutral" >
    <alignment/>
    <border>
      <left/>
      <right/>
      <top/>
      <bottom/>
      <diagonal/>
    </border>
    <fill>
      <patternFill patternType="solid">
        <fgColor rgb="FFFFEB9C"/>
      </patternFill>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <color rgb="FF9C6500"/>
      <sz val="12"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>
"""

input = """
  <namedStyle builtinId="20" name="Input" >
    <alignment/>
    <border>
      <left style="thin">
        <color rgb="FF7F7F7F"/>
      </left>
      <right style="thin">
        <color rgb="FF7F7F7F"/>
      </right>
      <top style="thin">
        <color rgb="FF7F7F7F"/>
      </top>
      <bottom style="thin">
        <color rgb="FF7F7F7F"/>
      </bottom>
      <diagonal/>
    </border>
    <fill>
      <patternFill patternType="solid">
        <fgColor rgb="FFFFCC99"/>
      </patternFill>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <color rgb="FF3F3F76"/>
      <sz val="12"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>
"""

output = """
  <namedStyle builtinId="21" name="Output" >
    <alignment/>
    <border>
      <left style="thin">
        <color rgb="FF3F3F3F"/>
      </left>
      <right style="thin">
        <color rgb="FF3F3F3F"/>
      </right>
      <top style="thin">
        <color rgb="FF3F3F3F"/>
      </top>
      <bottom style="thin">
        <color rgb="FF3F3F3F"/>
      </bottom>
      <diagonal/>
    </border>
    <fill>
      <patternFill patternType="solid">
        <fgColor rgb="FFF2F2F2"/>
      </patternFill>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <b val="1"/>
      <color rgb="FF3F3F3F"/>
      <sz val="12"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>
"""

calculation = """
  <namedStyle builtinId="22" name="Calculation" >
    <alignment/>
    <border>
      <left style="thin">
        <color rgb="FF7F7F7F"/>
      </left>
      <right style="thin">
        <color rgb="FF7F7F7F"/>
      </right>
      <top style="thin">
        <color rgb="FF7F7F7F"/>
      </top>
      <bottom style="thin">
        <color rgb="FF7F7F7F"/>
      </bottom>
      <diagonal/>
    </border>
    <fill>
      <patternFill patternType="solid">
        <fgColor rgb="FFF2F2F2"/>
      </patternFill>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <b val="1"/>
      <color rgb="FFFA7D00"/>
      <sz val="12"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>
"""

linked_cell = """
  <namedStyle builtinId="24" name="Linked Cell" >
    <alignment/>
    <border>
      <left/>
      <right/>
      <top/>
      <bottom style="double">
        <color rgb="FFFF8001"/>
      </bottom>
      <diagonal/>
    </border>
    <fill>
      <patternFill/>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <color rgb="FFFA7D00"/>
      <sz val="12"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>
"""

check_cell = """
  <namedStyle builtinId="23" name="Check Cell" >
    <alignment/>
    <border>
      <left style="double">
        <color rgb="FF3F3F3F"/>
      </left>
      <right style="double">
        <color rgb="FF3F3F3F"/>
      </right>
      <top style="double">
        <color rgb="FF3F3F3F"/>
      </top>
      <bottom style="double">
        <color rgb="FF3F3F3F"/>
      </bottom>
      <diagonal/>
    </border>
    <fill>
      <patternFill patternType="solid">
        <fgColor rgb="FFA5A5A5"/>
      </patternFill>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <b val="1"/>
      <color theme="0"/>
      <sz val="12"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>
"""

warning = """
  <namedStyle builtinId="11" name="Warning Text" >
    <alignment/>
    <border>
      <left/>
      <right/>
      <top/>
      <bottom/>
      <diagonal/>
    </border>
    <fill>
      <patternFill/>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <color rgb="FFFF0000"/>
      <sz val="12"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>
"""

note = """
  <namedStyle builtinId="10" name="Note" >
    <alignment/>
    <border>
      <left style="thin">
        <color rgb="FFB2B2B2"/>
      </left>
      <right style="thin">
        <color rgb="FFB2B2B2"/>
      </right>
      <top style="thin">
        <color rgb="FFB2B2B2"/>
      </top>
      <bottom style="thin">
        <color rgb="FFB2B2B2"/>
      </bottom>
      <diagonal/>
    </border>
    <fill>
      <patternFill patternType="solid">
        <fgColor rgb="FFFFFFCC"/>
      </patternFill>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <color theme="1"/>
      <sz val="12"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>
"""

explanatory = """
  <namedStyle builtinId="53" name="Explanatory Text" >
    <alignment/>
    <border>
      <left/>
      <right/>
      <top/>
      <bottom/>
      <diagonal/>
    </border>
    <fill>
      <patternFill/>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <i val="1"/>
      <color rgb="FF7F7F7F"/>
      <sz val="12"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>
"""

total = """
  <namedStyle builtinId="25" name="Total" >
    <alignment/>
    <border>
      <left/>
      <right/>
      <top style="thin">
        <color theme="4"/>
      </top>
      <bottom style="double">
        <color theme="4"/>
      </bottom>
      <diagonal/>
    </border>
    <fill>
      <patternFill/>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <b val="1"/>
      <color theme="1"/>
      <sz val="12"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>
"""

accent_1 = """
  <namedStyle builtinId="29" name="Accent1" >
    <alignment/>
    <border>
      <left/>
      <right/>
      <top/>
      <bottom/>
      <diagonal/>
    </border>
    <fill>
      <patternFill patternType="solid">
        <fgColor theme="4"/>
      </patternFill>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <color theme="0"/>
      <sz val="12"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>
"""

accent_1_20 = """
  <namedStyle builtinId="30" name="20 % - Accent1" >
    <alignment/>
    <border>
      <left/>
      <right/>
      <top/>
      <bottom/>
      <diagonal/>
    </border>
    <fill>
      <patternFill patternType="solid">
        <fgColor theme="4" tint="0.7999816888943144"/>
        <bgColor indexed="65"/>
      </patternFill>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <color theme="1"/>
      <sz val="12"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>
"""

accent_1_40 = """
  <namedStyle builtinId="31" name="40 % - Accent1" >
    <alignment/>
    <border>
      <left/>
      <right/>
      <top/>
      <bottom/>
      <diagonal/>
    </border>
    <fill>
      <patternFill patternType="solid">
        <fgColor theme="4" tint="0.5999938962981048"/>
        <bgColor indexed="65"/>
      </patternFill>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <color theme="1"/>
      <sz val="12"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>
"""

accent_1_60 = """
  <namedStyle builtinId="32" name="60 % - Accent1" >
    <alignment/>
    <border>
      <left/>
      <right/>
      <top/>
      <bottom/>
      <diagonal/>
    </border>
    <fill>
      <patternFill patternType="solid">
        <fgColor theme="4" tint="0.3999755851924192"/>
        <bgColor indexed="65"/>
      </patternFill>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <color theme="0"/>
      <sz val="12"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>
"""

accent_2 = """<namedStyle builtinId="33" name="Accent2" >
    <alignment/>
    <border>
      <left/>
      <right/>
      <top/>
      <bottom/>
      <diagonal/>
    </border>
    <fill>
      <patternFill patternType="solid">
        <fgColor theme="5"/>
      </patternFill>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <color theme="0"/>
      <sz val="12"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>"""

accent_2_20 = """
  <namedStyle builtinId="34" name="20 % - Accent2" >
    <alignment/>
    <border>
      <left/>
      <right/>
      <top/>
      <bottom/>
      <diagonal/>
    </border>
    <fill>
      <patternFill patternType="solid">
        <fgColor theme="5" tint="0.7999816888943144"/>
        <bgColor indexed="65"/>
      </patternFill>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <color theme="1"/>
      <sz val="12"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>"""

accent_2_40 = """
<namedStyle builtinId="35" name="40 % - Accent2" >
    <alignment/>
    <border>
      <left/>
      <right/>
      <top/>
      <bottom/>
      <diagonal/>
    </border>
    <fill>
      <patternFill patternType="solid">
        <fgColor theme="5" tint="0.5999938962981048"/>
        <bgColor indexed="65"/>
      </patternFill>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <color theme="1"/>
      <sz val="12"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>"""

accent_2_60 = """
<namedStyle builtinId="36" name="60 % - Accent2" >
    <alignment/>
    <border>
      <left/>
      <right/>
      <top/>
      <bottom/>
      <diagonal/>
    </border>
    <fill>
      <patternFill patternType="solid">
        <fgColor theme="5" tint="0.3999755851924192"/>
        <bgColor indexed="65"/>
      </patternFill>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <color theme="0"/>
      <sz val="12"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>"""

accent_3 = """
<namedStyle builtinId="37" name="Accent3" >
    <alignment/>
    <border>
      <left/>
      <right/>
      <top/>
      <bottom/>
      <diagonal/>
    </border>
    <fill>
      <patternFill patternType="solid">
        <fgColor theme="6"/>
      </patternFill>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <color theme="0"/>
      <sz val="12"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>"""

accent_3_20 = """
  <namedStyle builtinId="38" name="20 % - Accent3" >
    <alignment/>
    <border>
      <left/>
      <right/>
      <top/>
      <bottom/>
      <diagonal/>
    </border>
    <fill>
      <patternFill patternType="solid">
        <fgColor theme="6" tint="0.7999816888943144"/>
        <bgColor indexed="65"/>
      </patternFill>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <color theme="1"/>
      <sz val="12"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>"""

accent_3_40 = """
  <namedStyle builtinId="39" name="40 % - Accent3" >
    <alignment/>
    <border>
      <left/>
      <right/>
      <top/>
      <bottom/>
      <diagonal/>
    </border>
    <fill>
      <patternFill patternType="solid">
        <fgColor theme="6" tint="0.5999938962981048"/>
        <bgColor indexed="65"/>
      </patternFill>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <color theme="1"/>
      <sz val="12"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>
"""
accent_3_60 = """
  <namedStyle builtinId="40" name="60 % - Accent3" >
    <alignment/>
    <border>
      <left/>
      <right/>
      <top/>
      <bottom/>
      <diagonal/>
    </border>
    <fill>
      <patternFill patternType="solid">
        <fgColor theme="6" tint="0.3999755851924192"/>
        <bgColor indexed="65"/>
      </patternFill>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <color theme="0"/>
      <sz val="12"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>
"""
accent_4 = """
  <namedStyle builtinId="41" name="Accent4" >
    <alignment/>
    <border>
      <left/>
      <right/>
      <top/>
      <bottom/>
      <diagonal/>
    </border>
    <fill>
      <patternFill patternType="solid">
        <fgColor theme="7"/>
      </patternFill>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <color theme="0"/>
      <sz val="12"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>
"""

accent_4_20 = """
  <namedStyle builtinId="42" name="20 % - Accent4" >
    <alignment/>
    <border>
      <left/>
      <right/>
      <top/>
      <bottom/>
      <diagonal/>
    </border>
    <fill>
      <patternFill patternType="solid">
        <fgColor theme="7" tint="0.7999816888943144"/>
        <bgColor indexed="65"/>
      </patternFill>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <color theme="1"/>
      <sz val="12"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>
"""

accent_4_40 = """
  <namedStyle builtinId="43" name="40 % - Accent4" >
    <alignment/>
    <border>
      <left/>
      <right/>
      <top/>
      <bottom/>
      <diagonal/>
    </border>
    <fill>
      <patternFill patternType="solid">
        <fgColor theme="7" tint="0.5999938962981048"/>
        <bgColor indexed="65"/>
      </patternFill>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <color theme="1"/>
      <sz val="12"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>
"""

accent_4_60 = """
<namedStyle builtinId="44" name="60 % - Accent4" >
    <alignment/>
    <border>
      <left/>
      <right/>
      <top/>
      <bottom/>
      <diagonal/>
    </border>
    <fill>
      <patternFill patternType="solid">
        <fgColor theme="7" tint="0.3999755851924192"/>
        <bgColor indexed="65"/>
      </patternFill>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <color theme="0"/>
      <sz val="12"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>
"""

accent_5 = """
  <namedStyle builtinId="45" name="Accent5" >
    <alignment/>
    <border>
      <left/>
      <right/>
      <top/>
      <bottom/>
      <diagonal/>
    </border>
    <fill>
      <patternFill patternType="solid">
        <fgColor theme="8"/>
      </patternFill>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <color theme="0"/>
      <sz val="12"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>
"""

accent_5_20 = """
  <namedStyle builtinId="46" name="20 % - Accent5" >
    <alignment/>
    <border>
      <left/>
      <right/>
      <top/>
      <bottom/>
      <diagonal/>
    </border>
    <fill>
      <patternFill patternType="solid">
        <fgColor theme="8" tint="0.7999816888943144"/>
        <bgColor indexed="65"/>
      </patternFill>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <color theme="1"/>
      <sz val="12"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>
"""

accent_5_40 = """
  <namedStyle builtinId="47" name="40 % - Accent5" >
    <alignment/>
    <border>
      <left/>
      <right/>
      <top/>
      <bottom/>
      <diagonal/>
    </border>
    <fill>
      <patternFill patternType="solid">
        <fgColor theme="8" tint="0.5999938962981048"/>
        <bgColor indexed="65"/>
      </patternFill>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <color theme="1"/>
      <sz val="12"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>
"""

accent_5_60 = """
  <namedStyle builtinId="48" name="60 % - Accent5" >
    <alignment/>
    <border>
      <left/>
      <right/>
      <top/>
      <bottom/>
      <diagonal/>
    </border>
    <fill>
      <patternFill patternType="solid">
        <fgColor theme="8" tint="0.3999755851924192"/>
        <bgColor indexed="65"/>
      </patternFill>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <color theme="0"/>
      <sz val="12"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>
"""

accent_6 = """
  <namedStyle builtinId="49" name="Accent6" >
    <alignment/>
    <border>
      <left/>
      <right/>
      <top/>
      <bottom/>
      <diagonal/>
    </border>
    <fill>
      <patternFill patternType="solid">
        <fgColor theme="9"/>
      </patternFill>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <color theme="0"/>
      <sz val="12"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>
"""

accent_6_20 = """
  <namedStyle builtinId="50" name="20 % - Accent6" >
    <alignment/>
    <border>
      <left/>
      <right/>
      <top/>
      <bottom/>
      <diagonal/>
    </border>
    <fill>
      <patternFill patternType="solid">
        <fgColor theme="9" tint="0.7999816888943144"/>
        <bgColor indexed="65"/>
      </patternFill>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <color theme="1"/>
      <sz val="12"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>
"""

accent_6_40 = """
  <namedStyle builtinId="51" name="40 % - Accent6" >
    <alignment/>
    <border>
      <left/>
      <right/>
      <top/>
      <bottom/>
      <diagonal/>
    </border>
    <fill>
      <patternFill patternType="solid">
        <fgColor theme="9" tint="0.5999938962981048"/>
        <bgColor indexed="65"/>
      </patternFill>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <color theme="1"/>
      <sz val="12"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>
"""

accent_6_60 = """
  <namedStyle builtinId="52" name="60 % - Accent6" >
    <alignment/>
    <border>
      <left/>
      <right/>
      <top/>
      <bottom/>
      <diagonal/>
    </border>
    <fill>
      <patternFill patternType="solid">
        <fgColor theme="9" tint="0.3999755851924192"/>
        <bgColor indexed="65"/>
      </patternFill>
    </fill>
    <font>
      <name val="Calibri"/>
      <family val="2"/>
      <color theme="0"/>
      <sz val="12"/>
      <scheme val="minor"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>
"""

pandas_highlight = """
  <namedStyle hidden="0" name="Pandas">
    <alignment horizontal="center"/>
    <border>
      <left style="thin"><color rgb="00000000"/></left>
      <right style="thin"><color rgb="00000000"/></right>
      <top style="thin"><color rgb="00000000"/></top>
      <bottom style="thin"><color rgb="00000000"/></bottom>
      <diagonal/>
    </border>
    <fill>
      <patternFill/>
    </fill>
    <font>
      <b val="1"/>
    </font>
    <protection hidden="0" locked="1"/>
  </namedStyle>
"""

styles = dict(
    [
        ('Normal', NamedStyle.from_tree(fromstring(normal))),
        ('Comma', NamedStyle.from_tree(fromstring(comma))),
        ('Currency', NamedStyle.from_tree(fromstring(currency))),
        ('Percent', NamedStyle.from_tree(fromstring(percent))),
        ('Comma [0]', NamedStyle.from_tree(fromstring(comma_0))),
        ('Currency [0]', NamedStyle.from_tree(fromstring(currency_0))),
        ('Hyperlink', NamedStyle.from_tree(fromstring(hyperlink))),
        ('Followed Hyperlink', NamedStyle.from_tree(fromstring(followed_hyperlink))),
        ('Note', NamedStyle.from_tree(fromstring(note))),
        ('Warning Text', NamedStyle.from_tree(fromstring(warning))),
        ('Title', NamedStyle.from_tree(fromstring(title))),
        ('Headline 1', NamedStyle.from_tree(fromstring(headline_1))),
        ('Headline 2', NamedStyle.from_tree(fromstring(headline_2))),
        ('Headline 3', NamedStyle.from_tree(fromstring(headline_3))),
        ('Headline 4', NamedStyle.from_tree(fromstring(headline_4))),
        ('Input', NamedStyle.from_tree(fromstring(input))),
        ('Output', NamedStyle.from_tree(fromstring(output))),
        ('Calculation',NamedStyle.from_tree(fromstring(calculation))),
        ('Check Cell', NamedStyle.from_tree(fromstring(check_cell))),
        ('Linked Cell', NamedStyle.from_tree(fromstring(linked_cell))),
        ('Total', NamedStyle.from_tree(fromstring(total))),
        ('Good', NamedStyle.from_tree(fromstring(good))),
        ('Bad', NamedStyle.from_tree(fromstring(bad))),
        ('Neutral', NamedStyle.from_tree(fromstring(neutral))),
        ('Accent1', NamedStyle.from_tree(fromstring(accent_1))),
        ('20 % - Accent1', NamedStyle.from_tree(fromstring(accent_1_20))),
        ('40 % - Accent1', NamedStyle.from_tree(fromstring(accent_1_40))),
        ('60 % - Accent1', NamedStyle.from_tree(fromstring(accent_1_60))),
        ('Accent2', NamedStyle.from_tree(fromstring(accent_2))),
        ('20 % - Accent2', NamedStyle.from_tree(fromstring(accent_2_20))),
        ('40 % - Accent2', NamedStyle.from_tree(fromstring(accent_2_40))),
        ('60 % - Accent2', NamedStyle.from_tree(fromstring(accent_2_60))),
        ('Accent3', NamedStyle.from_tree(fromstring(accent_3))),
        ('20 % - Accent3', NamedStyle.from_tree(fromstring(accent_3_20))),
        ('40 % - Accent3', NamedStyle.from_tree(fromstring(accent_3_40))),
        ('60 % - Accent3', NamedStyle.from_tree(fromstring(accent_3_60))),
        ('Accent4', NamedStyle.from_tree(fromstring(accent_4))),
        ('20 % - Accent4', NamedStyle.from_tree(fromstring(accent_4_20))),
        ('40 % - Accent4', NamedStyle.from_tree(fromstring(accent_4_40))),
        ('60 % - Accent4', NamedStyle.from_tree(fromstring(accent_4_60))),
        ('Accent5', NamedStyle.from_tree(fromstring(accent_5))),
        ('20 % - Accent5', NamedStyle.from_tree(fromstring(accent_5_20))),
        ('40 % - Accent5', NamedStyle.from_tree(fromstring(accent_5_40))),
        ('60 % - Accent5', NamedStyle.from_tree(fromstring(accent_5_60))),
        ('Accent6', NamedStyle.from_tree(fromstring(accent_6))),
        ('20 % - Accent6', NamedStyle.from_tree(fromstring(accent_6_20))),
        ('40 % - Accent6', NamedStyle.from_tree(fromstring(accent_6_40))),
        ('60 % - Accent6', NamedStyle.from_tree(fromstring(accent_6_60))),
        ('Explanatory Text', NamedStyle.from_tree(fromstring(explanatory))),
        ('Pandas', NamedStyle.from_tree(fromstring(pandas_highlight)))
    ]
)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v135\debugger.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Debugger
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime


class BreakpointId(str):
    '''
    Breakpoint identifier.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> BreakpointId:
        return cls(json)

    def __repr__(self):
        return 'BreakpointId({})'.format(super().__repr__())


class CallFrameId(str):
    '''
    Call frame identifier.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> CallFrameId:
        return cls(json)

    def __repr__(self):
        return 'CallFrameId({})'.format(super().__repr__())


@dataclass
class Location:
    '''
    Location in the source code.
    '''
    #: Script identifier as reported in the ``Debugger.scriptParsed``.
    script_id: runtime.ScriptId

    #: Line number in the script (0-based).
    line_number: int

    #: Column number in the script (0-based).
    column_number: typing.Optional[int] = None

    def to_json(self):
        json = dict()
        json['scriptId'] = self.script_id.to_json()
        json['lineNumber'] = self.line_number
        if self.column_number is not None:
            json['columnNumber'] = self.column_number
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            script_id=runtime.ScriptId.from_json(json['scriptId']),
            line_number=int(json['lineNumber']),
            column_number=int(json['columnNumber']) if 'columnNumber' in json else None,
        )


@dataclass
class ScriptPosition:
    '''
    Location in the source code.
    '''
    line_number: int

    column_number: int

    def to_json(self):
        json = dict()
        json['lineNumber'] = self.line_number
        json['columnNumber'] = self.column_number
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            line_number=int(json['lineNumber']),
            column_number=int(json['columnNumber']),
        )


@dataclass
class LocationRange:
    '''
    Location range within one script.
    '''
    script_id: runtime.ScriptId

    start: ScriptPosition

    end: ScriptPosition

    def to_json(self):
        json = dict()
        json['scriptId'] = self.script_id.to_json()
        json['start'] = self.start.to_json()
        json['end'] = self.end.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            script_id=runtime.ScriptId.from_json(json['scriptId']),
            start=ScriptPosition.from_json(json['start']),
            end=ScriptPosition.from_json(json['end']),
        )


@dataclass
class CallFrame:
    '''
    JavaScript call frame. Array of call frames form the call stack.
    '''
    #: Call frame identifier. This identifier is only valid while the virtual machine is paused.
    call_frame_id: CallFrameId

    #: Name of the JavaScript function called on this call frame.
    function_name: str

    #: Location in the source code.
    location: Location

    #: JavaScript script name or url.
    #: Deprecated in favor of using the ``location.scriptId`` to resolve the URL via a previously
    #: sent ``Debugger.scriptParsed`` event.
    url: str

    #: Scope chain for this call frame.
    scope_chain: typing.List[Scope]

    #: ``this`` object for this call frame.
    this: runtime.RemoteObject

    #: Location in the source code.
    function_location: typing.Optional[Location] = None

    #: The value being returned, if the function is at return point.
    return_value: typing.Optional[runtime.RemoteObject] = None

    #: Valid only while the VM is paused and indicates whether this frame
    #: can be restarted or not. Note that a ``true`` value here does not
    #: guarantee that Debugger#restartFrame with this CallFrameId will be
    #: successful, but it is very likely.
    can_be_restarted: typing.Optional[bool] = None

    def to_json(self):
        json = dict()
        json['callFrameId'] = self.call_frame_id.to_json()
        json['functionName'] = self.function_name
        json['location'] = self.location.to_json()
        json['url'] = self.url
        json['scopeChain'] = [i.to_json() for i in self.scope_chain]
        json['this'] = self.this.to_json()
        if self.function_location is not None:
            json['functionLocation'] = self.function_location.to_json()
        if self.return_value is not None:
            json['returnValue'] = self.return_value.to_json()
        if self.can_be_restarted is not None:
            json['canBeRestarted'] = self.can_be_restarted
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            call_frame_id=CallFrameId.from_json(json['callFrameId']),
            function_name=str(json['functionName']),
            location=Location.from_json(json['location']),
            url=str(json['url']),
            scope_chain=[Scope.from_json(i) for i in json['scopeChain']],
            this=runtime.RemoteObject.from_json(json['this']),
            function_location=Location.from_json(json['functionLocation']) if 'functionLocation' in json else None,
            return_value=runtime.RemoteObject.from_json(json['returnValue']) if 'returnValue' in json else None,
            can_be_restarted=bool(json['canBeRestarted']) if 'canBeRestarted' in json else None,
        )


@dataclass
class Scope:
    '''
    Scope description.
    '''
    #: Scope type.
    type_: str

    #: Object representing the scope. For ``global`` and ``with`` scopes it represents the actual
    #: object; for the rest of the scopes, it is artificial transient object enumerating scope
    #: variables as its properties.
    object_: runtime.RemoteObject

    name: typing.Optional[str] = None

    #: Location in the source code where scope starts
    start_location: typing.Optional[Location] = None

    #: Location in the source code where scope ends
    end_location: typing.Optional[Location] = None

    def to_json(self):
        json = dict()
        json['type'] = self.type_
        json['object'] = self.object_.to_json()
        if self.name is not None:
            json['name'] = self.name
        if self.start_location is not None:
            json['startLocation'] = self.start_location.to_json()
        if self.end_location is not None:
            json['endLocation'] = self.end_location.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            type_=str(json['type']),
            object_=runtime.RemoteObject.from_json(json['object']),
            name=str(json['name']) if 'name' in json else None,
            start_location=Location.from_json(json['startLocation']) if 'startLocation' in json else None,
            end_location=Location.from_json(json['endLocation']) if 'endLocation' in json else None,
        )


@dataclass
class SearchMatch:
    '''
    Search match for resource.
    '''
    #: Line number in resource content.
    line_number: float

    #: Line with match content.
    line_content: str

    def to_json(self):
        json = dict()
        json['lineNumber'] = self.line_number
        json['lineContent'] = self.line_content
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            line_number=float(json['lineNumber']),
            line_content=str(json['lineContent']),
        )


@dataclass
class BreakLocation:
    #: Script identifier as reported in the ``Debugger.scriptParsed``.
    script_id: runtime.ScriptId

    #: Line number in the script (0-based).
    line_number: int

    #: Column number in the script (0-based).
    column_number: typing.Optional[int] = None

    type_: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['scriptId'] = self.script_id.to_json()
        json['lineNumber'] = self.line_number
        if self.column_number is not None:
            json['columnNumber'] = self.column_number
        if self.type_ is not None:
            json['type'] = self.type_
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            script_id=runtime.ScriptId.from_json(json['scriptId']),
            line_number=int(json['lineNumber']),
            column_number=int(json['columnNumber']) if 'columnNumber' in json else None,
            type_=str(json['type']) if 'type' in json else None,
        )


@dataclass
class WasmDisassemblyChunk:
    #: The next chunk of disassembled lines.
    lines: typing.List[str]

    #: The bytecode offsets describing the start of each line.
    bytecode_offsets: typing.List[int]

    def to_json(self):
        json = dict()
        json['lines'] = [i for i in self.lines]
        json['bytecodeOffsets'] = [i for i in self.bytecode_offsets]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            lines=[str(i) for i in json['lines']],
            bytecode_offsets=[int(i) for i in json['bytecodeOffsets']],
        )


class ScriptLanguage(enum.Enum):
    '''
    Enum of possible script languages.
    '''
    JAVA_SCRIPT = "JavaScript"
    WEB_ASSEMBLY = "WebAssembly"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class DebugSymbols:
    '''
    Debug symbols available for a wasm script.
    '''
    #: Type of the debug symbols.
    type_: str

    #: URL of the external symbol source.
    external_url: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['type'] = self.type_
        if self.external_url is not None:
            json['externalURL'] = self.external_url
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            type_=str(json['type']),
            external_url=str(json['externalURL']) if 'externalURL' in json else None,
        )


@dataclass
class ResolvedBreakpoint:
    #: Breakpoint unique identifier.
    breakpoint_id: BreakpointId

    #: Actual breakpoint location.
    location: Location

    def to_json(self):
        json = dict()
        json['breakpointId'] = self.breakpoint_id.to_json()
        json['location'] = self.location.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            breakpoint_id=BreakpointId.from_json(json['breakpointId']),
            location=Location.from_json(json['location']),
        )


def continue_to_location(
        location: Location,
        target_call_frames: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Continues execution until specific location is reached.

    :param location: Location to continue to.
    :param target_call_frames: *(Optional)*
    '''
    params: T_JSON_DICT = dict()
    params['location'] = location.to_json()
    if target_call_frames is not None:
        params['targetCallFrames'] = target_call_frames
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.continueToLocation',
        'params': params,
    }
    json = yield cmd_dict


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables debugger for given page.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.disable',
    }
    json = yield cmd_dict


def enable(
        max_scripts_cache_size: typing.Optional[float] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,runtime.UniqueDebuggerId]:
    '''
    Enables debugger for the given page. Clients should not assume that the debugging has been
    enabled until the result for this command is received.

    :param max_scripts_cache_size: **(EXPERIMENTAL)** *(Optional)* The maximum size in bytes of collected scripts (not referenced by other heap objects) the debugger can hold. Puts no limit if parameter is omitted.
    :returns: Unique identifier of the debugger.
    '''
    params: T_JSON_DICT = dict()
    if max_scripts_cache_size is not None:
        params['maxScriptsCacheSize'] = max_scripts_cache_size
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.enable',
        'params': params,
    }
    json = yield cmd_dict
    return runtime.UniqueDebuggerId.from_json(json['debuggerId'])


def evaluate_on_call_frame(
        call_frame_id: CallFrameId,
        expression: str,
        object_group: typing.Optional[str] = None,
        include_command_line_api: typing.Optional[bool] = None,
        silent: typing.Optional[bool] = None,
        return_by_value: typing.Optional[bool] = None,
        generate_preview: typing.Optional[bool] = None,
        throw_on_side_effect: typing.Optional[bool] = None,
        timeout: typing.Optional[runtime.TimeDelta] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[runtime.RemoteObject, typing.Optional[runtime.ExceptionDetails]]]:
    '''
    Evaluates expression on a given call frame.

    :param call_frame_id: Call frame identifier to evaluate on.
    :param expression: Expression to evaluate.
    :param object_group: *(Optional)* String object group name to put result into (allows rapid releasing resulting object handles using ```releaseObjectGroup````).
    :param include_command_line_api: *(Optional)* Specifies whether command line API should be available to the evaluated expression, defaults to false.
    :param silent: *(Optional)* In silent mode exceptions thrown during evaluation are not reported and do not pause execution. Overrides ````setPauseOnException``` state.
    :param return_by_value: *(Optional)* Whether the result is expected to be a JSON object that should be sent by value.
    :param generate_preview: **(EXPERIMENTAL)** *(Optional)* Whether preview should be generated for the result.
    :param throw_on_side_effect: *(Optional)* Whether to throw an exception if side effect cannot be ruled out during evaluation.
    :param timeout: **(EXPERIMENTAL)** *(Optional)* Terminate execution after timing out (number of milliseconds).
    :returns: A tuple with the following items:

        0. **result** - Object wrapper for the evaluation result.
        1. **exceptionDetails** - *(Optional)* Exception details.
    '''
    params: T_JSON_DICT = dict()
    params['callFrameId'] = call_frame_id.to_json()
    params['expression'] = expression
    if object_group is not None:
        params['objectGroup'] = object_group
    if include_command_line_api is not None:
        params['includeCommandLineAPI'] = include_command_line_api
    if silent is not None:
        params['silent'] = silent
    if return_by_value is not None:
        params['returnByValue'] = return_by_value
    if generate_preview is not None:
        params['generatePreview'] = generate_preview
    if throw_on_side_effect is not None:
        params['throwOnSideEffect'] = throw_on_side_effect
    if timeout is not None:
        params['timeout'] = timeout.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.evaluateOnCallFrame',
        'params': params,
    }
    json = yield cmd_dict
    return (
        runtime.RemoteObject.from_json(json['result']),
        runtime.ExceptionDetails.from_json(json['exceptionDetails']) if 'exceptionDetails' in json else None
    )


def get_possible_breakpoints(
        start: Location,
        end: typing.Optional[Location] = None,
        restrict_to_function: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[BreakLocation]]:
    '''
    Returns possible locations for breakpoint. scriptId in start and end range locations should be
    the same.

    :param start: Start of range to search possible breakpoint locations in.
    :param end: *(Optional)* End of range to search possible breakpoint locations in (excluding). When not specified, end of scripts is used as end of range.
    :param restrict_to_function: *(Optional)* Only consider locations which are in the same (non-nested) function as start.
    :returns: List of the possible breakpoint locations.
    '''
    params: T_JSON_DICT = dict()
    params['start'] = start.to_json()
    if end is not None:
        params['end'] = end.to_json()
    if restrict_to_function is not None:
        params['restrictToFunction'] = restrict_to_function
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.getPossibleBreakpoints',
        'params': params,
    }
    json = yield cmd_dict
    return [BreakLocation.from_json(i) for i in json['locations']]


def get_script_source(
        script_id: runtime.ScriptId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[str, typing.Optional[str]]]:
    '''
    Returns source for the script with given id.

    :param script_id: Id of the script to get source for.
    :returns: A tuple with the following items:

        0. **scriptSource** - Script source (empty in case of Wasm bytecode).
        1. **bytecode** - *(Optional)* Wasm bytecode.
    '''
    params: T_JSON_DICT = dict()
    params['scriptId'] = script_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.getScriptSource',
        'params': params,
    }
    json = yield cmd_dict
    return (
        str(json['scriptSource']),
        str(json['bytecode']) if 'bytecode' in json else None
    )


def disassemble_wasm_module(
        script_id: runtime.ScriptId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[typing.Optional[str], int, typing.List[int], WasmDisassemblyChunk]]:
    '''


    **EXPERIMENTAL**

    :param script_id: Id of the script to disassemble
    :returns: A tuple with the following items:

        0. **streamId** - *(Optional)* For large modules, return a stream from which additional chunks of disassembly can be read successively.
        1. **totalNumberOfLines** - The total number of lines in the disassembly text.
        2. **functionBodyOffsets** - The offsets of all function bodies, in the format [start1, end1, start2, end2, ...] where all ends are exclusive.
        3. **chunk** - The first chunk of disassembly.
    '''
    params: T_JSON_DICT = dict()
    params['scriptId'] = script_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.disassembleWasmModule',
        'params': params,
    }
    json = yield cmd_dict
    return (
        str(json['streamId']) if 'streamId' in json else None,
        int(json['totalNumberOfLines']),
        [int(i) for i in json['functionBodyOffsets']],
        WasmDisassemblyChunk.from_json(json['chunk'])
    )


def next_wasm_disassembly_chunk(
        stream_id: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,WasmDisassemblyChunk]:
    '''
    Disassemble the next chunk of lines for the module corresponding to the
    stream. If disassembly is complete, this API will invalidate the streamId
    and return an empty chunk. Any subsequent calls for the now invalid stream
    will return errors.

    **EXPERIMENTAL**

    :param stream_id:
    :returns: The next chunk of disassembly.
    '''
    params: T_JSON_DICT = dict()
    params['streamId'] = stream_id
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.nextWasmDisassemblyChunk',
        'params': params,
    }
    json = yield cmd_dict
    return WasmDisassemblyChunk.from_json(json['chunk'])


def get_wasm_bytecode(
        script_id: runtime.ScriptId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,str]:
    '''
    This command is deprecated. Use getScriptSource instead.

    :param script_id: Id of the Wasm script to get source for.
    :returns: Script source.
    '''
    params: T_JSON_DICT = dict()
    params['scriptId'] = script_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.getWasmBytecode',
        'params': params,
    }
    json = yield cmd_dict
    return str(json['bytecode'])


def get_stack_trace(
        stack_trace_id: runtime.StackTraceId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,runtime.StackTrace]:
    '''
    Returns stack trace with given ``stackTraceId``.

    **EXPERIMENTAL**

    :param stack_trace_id:
    :returns: 
    '''
    params: T_JSON_DICT = dict()
    params['stackTraceId'] = stack_trace_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.getStackTrace',
        'params': params,
    }
    json = yield cmd_dict
    return runtime.StackTrace.from_json(json['stackTrace'])


def pause() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Stops on the next JavaScript statement.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.pause',
    }
    json = yield cmd_dict


def pause_on_async_call(
        parent_stack_trace_id: runtime.StackTraceId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''


    **EXPERIMENTAL**

    :param parent_stack_trace_id: Debugger will pause when async call with given stack trace is started.
    '''
    params: T_JSON_DICT = dict()
    params['parentStackTraceId'] = parent_stack_trace_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.pauseOnAsyncCall',
        'params': params,
    }
    json = yield cmd_dict


def remove_breakpoint(
        breakpoint_id: BreakpointId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Removes JavaScript breakpoint.

    :param breakpoint_id:
    '''
    params: T_JSON_DICT = dict()
    params['breakpointId'] = breakpoint_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.removeBreakpoint',
        'params': params,
    }
    json = yield cmd_dict


def restart_frame(
        call_frame_id: CallFrameId,
        mode: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[typing.List[CallFrame], typing.Optional[runtime.StackTrace], typing.Optional[runtime.StackTraceId]]]:
    '''
    Restarts particular call frame from the beginning. The old, deprecated
    behavior of ``restartFrame`` is to stay paused and allow further CDP commands
    after a restart was scheduled. This can cause problems with restarting, so
    we now continue execution immediatly after it has been scheduled until we
    reach the beginning of the restarted frame.

    To stay back-wards compatible, ``restartFrame`` now expects a ``mode``
    parameter to be present. If the ``mode`` parameter is missing, ``restartFrame``
    errors out.

    The various return values are deprecated and ``callFrames`` is always empty.
    Use the call frames from the ``Debugger#paused`` events instead, that fires
    once V8 pauses at the beginning of the restarted function.

    :param call_frame_id: Call frame identifier to evaluate on.
    :param mode: **(EXPERIMENTAL)** *(Optional)* The ```mode```` parameter must be present and set to 'StepInto', otherwise ````restartFrame``` will error out.
    :returns: A tuple with the following items:

        0. **callFrames** - New stack trace.
        1. **asyncStackTrace** - *(Optional)* Async stack trace, if any.
        2. **asyncStackTraceId** - *(Optional)* Async stack trace, if any.
    '''
    params: T_JSON_DICT = dict()
    params['callFrameId'] = call_frame_id.to_json()
    if mode is not None:
        params['mode'] = mode
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.restartFrame',
        'params': params,
    }
    json = yield cmd_dict
    return (
        [CallFrame.from_json(i) for i in json['callFrames']],
        runtime.StackTrace.from_json(json['asyncStackTrace']) if 'asyncStackTrace' in json else None,
        runtime.StackTraceId.from_json(json['asyncStackTraceId']) if 'asyncStackTraceId' in json else None
    )


def resume(
        terminate_on_resume: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Resumes JavaScript execution.

    :param terminate_on_resume: *(Optional)* Set to true to terminate execution upon resuming execution. In contrast to Runtime.terminateExecution, this will allows to execute further JavaScript (i.e. via evaluation) until execution of the paused code is actually resumed, at which point termination is triggered. If execution is currently not paused, this parameter has no effect.
    '''
    params: T_JSON_DICT = dict()
    if terminate_on_resume is not None:
        params['terminateOnResume'] = terminate_on_resume
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.resume',
        'params': params,
    }
    json = yield cmd_dict


def search_in_content(
        script_id: runtime.ScriptId,
        query: str,
        case_sensitive: typing.Optional[bool] = None,
        is_regex: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[SearchMatch]]:
    '''
    Searches for given string in script content.

    :param script_id: Id of the script to search in.
    :param query: String to search for.
    :param case_sensitive: *(Optional)* If true, search is case sensitive.
    :param is_regex: *(Optional)* If true, treats string parameter as regex.
    :returns: List of search matches.
    '''
    params: T_JSON_DICT = dict()
    params['scriptId'] = script_id.to_json()
    params['query'] = query
    if case_sensitive is not None:
        params['caseSensitive'] = case_sensitive
    if is_regex is not None:
        params['isRegex'] = is_regex
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.searchInContent',
        'params': params,
    }
    json = yield cmd_dict
    return [SearchMatch.from_json(i) for i in json['result']]


def set_async_call_stack_depth(
        max_depth: int
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables or disables async call stacks tracking.

    :param max_depth: Maximum depth of async call stacks. Setting to ```0``` will effectively disable collecting async call stacks (default).
    '''
    params: T_JSON_DICT = dict()
    params['maxDepth'] = max_depth
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.setAsyncCallStackDepth',
        'params': params,
    }
    json = yield cmd_dict


def set_blackbox_execution_contexts(
        unique_ids: typing.List[str]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Replace previous blackbox execution contexts with passed ones. Forces backend to skip
    stepping/pausing in scripts in these execution contexts. VM will try to leave blackboxed script by
    performing 'step in' several times, finally resorting to 'step out' if unsuccessful.

    **EXPERIMENTAL**

    :param unique_ids: Array of execution context unique ids for the debugger to ignore.
    '''
    params: T_JSON_DICT = dict()
    params['uniqueIds'] = [i for i in unique_ids]
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.setBlackboxExecutionContexts',
        'params': params,
    }
    json = yield cmd_dict


def set_blackbox_patterns(
        patterns: typing.List[str],
        skip_anonymous: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Replace previous blackbox patterns with passed ones. Forces backend to skip stepping/pausing in
    scripts with url matching one of the patterns. VM will try to leave blackboxed script by
    performing 'step in' several times, finally resorting to 'step out' if unsuccessful.

    **EXPERIMENTAL**

    :param patterns: Array of regexps that will be used to check script url for blackbox state.
    :param skip_anonymous: *(Optional)* If true, also ignore scripts with no source url.
    '''
    params: T_JSON_DICT = dict()
    params['patterns'] = [i for i in patterns]
    if skip_anonymous is not None:
        params['skipAnonymous'] = skip_anonymous
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.setBlackboxPatterns',
        'params': params,
    }
    json = yield cmd_dict


def set_blackboxed_ranges(
        script_id: runtime.ScriptId,
        positions: typing.List[ScriptPosition]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Makes backend skip steps in the script in blackboxed ranges. VM will try leave blacklisted
    scripts by performing 'step in' several times, finally resorting to 'step out' if unsuccessful.
    Positions array contains positions where blackbox state is changed. First interval isn't
    blackboxed. Array should be sorted.

    **EXPERIMENTAL**

    :param script_id: Id of the script.
    :param positions:
    '''
    params: T_JSON_DICT = dict()
    params['scriptId'] = script_id.to_json()
    params['positions'] = [i.to_json() for i in positions]
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.setBlackboxedRanges',
        'params': params,
    }
    json = yield cmd_dict


def set_breakpoint(
        location: Location,
        condition: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[BreakpointId, Location]]:
    '''
    Sets JavaScript breakpoint at a given location.

    :param location: Location to set breakpoint in.
    :param condition: *(Optional)* Expression to use as a breakpoint condition. When specified, debugger will only stop on the breakpoint if this expression evaluates to true.
    :returns: A tuple with the following items:

        0. **breakpointId** - Id of the created breakpoint for further reference.
        1. **actualLocation** - Location this breakpoint resolved into.
    '''
    params: T_JSON_DICT = dict()
    params['location'] = location.to_json()
    if condition is not None:
        params['condition'] = condition
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.setBreakpoint',
        'params': params,
    }
    json = yield cmd_dict
    return (
        BreakpointId.from_json(json['breakpointId']),
        Location.from_json(json['actualLocation'])
    )


def set_instrumentation_breakpoint(
        instrumentation: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,BreakpointId]:
    '''
    Sets instrumentation breakpoint.

    :param instrumentation: Instrumentation name.
    :returns: Id of the created breakpoint for further reference.
    '''
    params: T_JSON_DICT = dict()
    params['instrumentation'] = instrumentation
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.setInstrumentationBreakpoint',
        'params': params,
    }
    json = yield cmd_dict
    return BreakpointId.from_json(json['breakpointId'])


def set_breakpoint_by_url(
        line_number: int,
        url: typing.Optional[str] = None,
        url_regex: typing.Optional[str] = None,
        script_hash: typing.Optional[str] = None,
        column_number: typing.Optional[int] = None,
        condition: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[BreakpointId, typing.List[Location]]]:
    '''
    Sets JavaScript breakpoint at given location specified either by URL or URL regex. Once this
    command is issued, all existing parsed scripts will have breakpoints resolved and returned in
    ``locations`` property. Further matching script parsing will result in subsequent
    ``breakpointResolved`` events issued. This logical breakpoint will survive page reloads.

    :param line_number: Line number to set breakpoint at.
    :param url: *(Optional)* URL of the resources to set breakpoint on.
    :param url_regex: *(Optional)* Regex pattern for the URLs of the resources to set breakpoints on. Either ```url```` or ````urlRegex``` must be specified.
    :param script_hash: *(Optional)* Script hash of the resources to set breakpoint on.
    :param column_number: *(Optional)* Offset in the line to set breakpoint at.
    :param condition: *(Optional)* Expression to use as a breakpoint condition. When specified, debugger will only stop on the breakpoint if this expression evaluates to true.
    :returns: A tuple with the following items:

        0. **breakpointId** - Id of the created breakpoint for further reference.
        1. **locations** - List of the locations this breakpoint resolved into upon addition.
    '''
    params: T_JSON_DICT = dict()
    params['lineNumber'] = line_number
    if url is not None:
        params['url'] = url
    if url_regex is not None:
        params['urlRegex'] = url_regex
    if script_hash is not None:
        params['scriptHash'] = script_hash
    if column_number is not None:
        params['columnNumber'] = column_number
    if condition is not None:
        params['condition'] = condition
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.setBreakpointByUrl',
        'params': params,
    }
    json = yield cmd_dict
    return (
        BreakpointId.from_json(json['breakpointId']),
        [Location.from_json(i) for i in json['locations']]
    )


def set_breakpoint_on_function_call(
        object_id: runtime.RemoteObjectId,
        condition: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,BreakpointId]:
    '''
    Sets JavaScript breakpoint before each call to the given function.
    If another function was created from the same source as a given one,
    calling it will also trigger the breakpoint.

    **EXPERIMENTAL**

    :param object_id: Function object id.
    :param condition: *(Optional)* Expression to use as a breakpoint condition. When specified, debugger will stop on the breakpoint if this expression evaluates to true.
    :returns: Id of the created breakpoint for further reference.
    '''
    params: T_JSON_DICT = dict()
    params['objectId'] = object_id.to_json()
    if condition is not None:
        params['condition'] = condition
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.setBreakpointOnFunctionCall',
        'params': params,
    }
    json = yield cmd_dict
    return BreakpointId.from_json(json['breakpointId'])


def set_breakpoints_active(
        active: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Activates / deactivates all breakpoints on the page.

    :param active: New value for breakpoints active state.
    '''
    params: T_JSON_DICT = dict()
    params['active'] = active
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.setBreakpointsActive',
        'params': params,
    }
    json = yield cmd_dict


def set_pause_on_exceptions(
        state: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Defines pause on exceptions state. Can be set to stop on all exceptions, uncaught exceptions,
    or caught exceptions, no exceptions. Initial pause on exceptions state is ``none``.

    :param state: Pause on exceptions mode.
    '''
    params: T_JSON_DICT = dict()
    params['state'] = state
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.setPauseOnExceptions',
        'params': params,
    }
    json = yield cmd_dict


def set_return_value(
        new_value: runtime.CallArgument
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Changes return value in top frame. Available only at return break position.

    **EXPERIMENTAL**

    :param new_value: New return value.
    '''
    params: T_JSON_DICT = dict()
    params['newValue'] = new_value.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.setReturnValue',
        'params': params,
    }
    json = yield cmd_dict


def set_script_source(
        script_id: runtime.ScriptId,
        script_source: str,
        dry_run: typing.Optional[bool] = None,
        allow_top_frame_editing: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[typing.Optional[typing.List[CallFrame]], typing.Optional[bool], typing.Optional[runtime.StackTrace], typing.Optional[runtime.StackTraceId], str, typing.Optional[runtime.ExceptionDetails]]]:
    '''
    Edits JavaScript source live.

    In general, functions that are currently on the stack can not be edited with
    a single exception: If the edited function is the top-most stack frame and
    that is the only activation of that function on the stack. In this case
    the live edit will be successful and a ``Debugger.restartFrame`` for the
    top-most function is automatically triggered.

    :param script_id: Id of the script to edit.
    :param script_source: New content of the script.
    :param dry_run: *(Optional)* If true the change will not actually be applied. Dry run may be used to get result description without actually modifying the code.
    :param allow_top_frame_editing: **(EXPERIMENTAL)** *(Optional)* If true, then ```scriptSource```` is allowed to change the function on top of the stack as long as the top-most stack frame is the only activation of that function.
    :returns: A tuple with the following items:

        0. **callFrames** - *(Optional)* New stack trace in case editing has happened while VM was stopped.
        1. **stackChanged** - *(Optional)* Whether current call stack  was modified after applying the changes.
        2. **asyncStackTrace** - *(Optional)* Async stack trace, if any.
        3. **asyncStackTraceId** - *(Optional)* Async stack trace, if any.
        4. **status** - Whether the operation was successful or not. Only `` Ok`` denotes a successful live edit while the other enum variants denote why the live edit failed.
        5. **exceptionDetails** - *(Optional)* Exception details if any. Only present when `` status`` is `` CompileError`.
    '''
    params: T_JSON_DICT = dict()
    params['scriptId'] = script_id.to_json()
    params['scriptSource'] = script_source
    if dry_run is not None:
        params['dryRun'] = dry_run
    if allow_top_frame_editing is not None:
        params['allowTopFrameEditing'] = allow_top_frame_editing
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.setScriptSource',
        'params': params,
    }
    json = yield cmd_dict
    return (
        [CallFrame.from_json(i) for i in json['callFrames']] if 'callFrames' in json else None,
        bool(json['stackChanged']) if 'stackChanged' in json else None,
        runtime.StackTrace.from_json(json['asyncStackTrace']) if 'asyncStackTrace' in json else None,
        runtime.StackTraceId.from_json(json['asyncStackTraceId']) if 'asyncStackTraceId' in json else None,
        str(json['status']),
        runtime.ExceptionDetails.from_json(json['exceptionDetails']) if 'exceptionDetails' in json else None
    )


def set_skip_all_pauses(
        skip: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Makes page not interrupt on any pauses (breakpoint, exception, dom exception etc).

    :param skip: New value for skip pauses state.
    '''
    params: T_JSON_DICT = dict()
    params['skip'] = skip
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.setSkipAllPauses',
        'params': params,
    }
    json = yield cmd_dict


def set_variable_value(
        scope_number: int,
        variable_name: str,
        new_value: runtime.CallArgument,
        call_frame_id: CallFrameId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Changes value of variable in a callframe. Object-based scopes are not supported and must be
    mutated manually.

    :param scope_number: 0-based number of scope as was listed in scope chain. Only 'local', 'closure' and 'catch' scope types are allowed. Other scopes could be manipulated manually.
    :param variable_name: Variable name.
    :param new_value: New variable value.
    :param call_frame_id: Id of callframe that holds variable.
    '''
    params: T_JSON_DICT = dict()
    params['scopeNumber'] = scope_number
    params['variableName'] = variable_name
    params['newValue'] = new_value.to_json()
    params['callFrameId'] = call_frame_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.setVariableValue',
        'params': params,
    }
    json = yield cmd_dict


def step_into(
        break_on_async_call: typing.Optional[bool] = None,
        skip_list: typing.Optional[typing.List[LocationRange]] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Steps into the function call.

    :param break_on_async_call: **(EXPERIMENTAL)** *(Optional)* Debugger will pause on the execution of the first async task which was scheduled before next pause.
    :param skip_list: **(EXPERIMENTAL)** *(Optional)* The skipList specifies location ranges that should be skipped on step into.
    '''
    params: T_JSON_DICT = dict()
    if break_on_async_call is not None:
        params['breakOnAsyncCall'] = break_on_async_call
    if skip_list is not None:
        params['skipList'] = [i.to_json() for i in skip_list]
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.stepInto',
        'params': params,
    }
    json = yield cmd_dict


def step_out() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Steps out of the function call.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.stepOut',
    }
    json = yield cmd_dict


def step_over(
        skip_list: typing.Optional[typing.List[LocationRange]] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Steps over the statement.

    :param skip_list: **(EXPERIMENTAL)** *(Optional)* The skipList specifies location ranges that should be skipped on step over.
    '''
    params: T_JSON_DICT = dict()
    if skip_list is not None:
        params['skipList'] = [i.to_json() for i in skip_list]
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.stepOver',
        'params': params,
    }
    json = yield cmd_dict


@event_class('Debugger.breakpointResolved')
@dataclass
class BreakpointResolved:
    '''
    Fired when breakpoint is resolved to an actual script and location.
    Deprecated in favor of ``resolvedBreakpoints`` in the ``scriptParsed`` event.
    '''
    #: Breakpoint unique identifier.
    breakpoint_id: BreakpointId
    #: Actual breakpoint location.
    location: Location

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> BreakpointResolved:
        return cls(
            breakpoint_id=BreakpointId.from_json(json['breakpointId']),
            location=Location.from_json(json['location'])
        )


@event_class('Debugger.paused')
@dataclass
class Paused:
    '''
    Fired when the virtual machine stopped on breakpoint or exception or any other stop criteria.
    '''
    #: Call stack the virtual machine stopped on.
    call_frames: typing.List[CallFrame]
    #: Pause reason.
    reason: str
    #: Object containing break-specific auxiliary properties.
    data: typing.Optional[dict]
    #: Hit breakpoints IDs
    hit_breakpoints: typing.Optional[typing.List[str]]
    #: Async stack trace, if any.
    async_stack_trace: typing.Optional[runtime.StackTrace]
    #: Async stack trace, if any.
    async_stack_trace_id: typing.Optional[runtime.StackTraceId]
    #: Never present, will be removed.
    async_call_stack_trace_id: typing.Optional[runtime.StackTraceId]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> Paused:
        return cls(
            call_frames=[CallFrame.from_json(i) for i in json['callFrames']],
            reason=str(json['reason']),
            data=dict(json['data']) if 'data' in json else None,
            hit_breakpoints=[str(i) for i in json['hitBreakpoints']] if 'hitBreakpoints' in json else None,
            async_stack_trace=runtime.StackTrace.from_json(json['asyncStackTrace']) if 'asyncStackTrace' in json else None,
            async_stack_trace_id=runtime.StackTraceId.from_json(json['asyncStackTraceId']) if 'asyncStackTraceId' in json else None,
            async_call_stack_trace_id=runtime.StackTraceId.from_json(json['asyncCallStackTraceId']) if 'asyncCallStackTraceId' in json else None
        )


@event_class('Debugger.resumed')
@dataclass
class Resumed:
    '''
    Fired when the virtual machine resumed execution.
    '''


    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> Resumed:
        return cls(

        )


@event_class('Debugger.scriptFailedToParse')
@dataclass
class ScriptFailedToParse:
    '''
    Fired when virtual machine fails to parse the script.
    '''
    #: Identifier of the script parsed.
    script_id: runtime.ScriptId
    #: URL or name of the script parsed (if any).
    url: str
    #: Line offset of the script within the resource with given URL (for script tags).
    start_line: int
    #: Column offset of the script within the resource with given URL.
    start_column: int
    #: Last line of the script.
    end_line: int
    #: Length of the last line of the script.
    end_column: int
    #: Specifies script creation context.
    execution_context_id: runtime.ExecutionContextId
    #: Content hash of the script, SHA-256.
    hash_: str
    #: For Wasm modules, the content of the ``build_id`` custom section.
    build_id: str
    #: Embedder-specific auxiliary data likely matching {isDefault: boolean, type: 'default'``'isolated'``'worker', frameId: string}
    execution_context_aux_data: typing.Optional[dict]
    #: URL of source map associated with script (if any).
    source_map_url: typing.Optional[str]
    #: True, if this script has sourceURL.
    has_source_url: typing.Optional[bool]
    #: True, if this script is ES6 module.
    is_module: typing.Optional[bool]
    #: This script length.
    length: typing.Optional[int]
    #: JavaScript top stack frame of where the script parsed event was triggered if available.
    stack_trace: typing.Optional[runtime.StackTrace]
    #: If the scriptLanguage is WebAssembly, the code section offset in the module.
    code_offset: typing.Optional[int]
    #: The language of the script.
    script_language: typing.Optional[debugger.ScriptLanguage]
    #: The name the embedder supplied for this script.
    embedder_name: typing.Optional[str]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ScriptFailedToParse:
        return cls(
            script_id=runtime.ScriptId.from_json(json['scriptId']),
            url=str(json['url']),
            start_line=int(json['startLine']),
            start_column=int(json['startColumn']),
            end_line=int(json['endLine']),
            end_column=int(json['endColumn']),
            execution_context_id=runtime.ExecutionContextId.from_json(json['executionContextId']),
            hash_=str(json['hash']),
            build_id=str(json['buildId']),
            execution_context_aux_data=dict(json['executionContextAuxData']) if 'executionContextAuxData' in json else None,
            source_map_url=str(json['sourceMapURL']) if 'sourceMapURL' in json else None,
            has_source_url=bool(json['hasSourceURL']) if 'hasSourceURL' in json else None,
            is_module=bool(json['isModule']) if 'isModule' in json else None,
            length=int(json['length']) if 'length' in json else None,
            stack_trace=runtime.StackTrace.from_json(json['stackTrace']) if 'stackTrace' in json else None,
            code_offset=int(json['codeOffset']) if 'codeOffset' in json else None,
            script_language=debugger.ScriptLanguage.from_json(json['scriptLanguage']) if 'scriptLanguage' in json else None,
            embedder_name=str(json['embedderName']) if 'embedderName' in json else None
        )


@event_class('Debugger.scriptParsed')
@dataclass
class ScriptParsed:
    '''
    Fired when virtual machine parses script. This event is also fired for all known and uncollected
    scripts upon enabling debugger.
    '''
    #: Identifier of the script parsed.
    script_id: runtime.ScriptId
    #: URL or name of the script parsed (if any).
    url: str
    #: Line offset of the script within the resource with given URL (for script tags).
    start_line: int
    #: Column offset of the script within the resource with given URL.
    start_column: int
    #: Last line of the script.
    end_line: int
    #: Length of the last line of the script.
    end_column: int
    #: Specifies script creation context.
    execution_context_id: runtime.ExecutionContextId
    #: Content hash of the script, SHA-256.
    hash_: str
    #: For Wasm modules, the content of the ``build_id`` custom section.
    build_id: str
    #: Embedder-specific auxiliary data likely matching {isDefault: boolean, type: 'default'``'isolated'``'worker', frameId: string}
    execution_context_aux_data: typing.Optional[dict]
    #: True, if this script is generated as a result of the live edit operation.
    is_live_edit: typing.Optional[bool]
    #: URL of source map associated with script (if any).
    source_map_url: typing.Optional[str]
    #: True, if this script has sourceURL.
    has_source_url: typing.Optional[bool]
    #: True, if this script is ES6 module.
    is_module: typing.Optional[bool]
    #: This script length.
    length: typing.Optional[int]
    #: JavaScript top stack frame of where the script parsed event was triggered if available.
    stack_trace: typing.Optional[runtime.StackTrace]
    #: If the scriptLanguage is WebAssembly, the code section offset in the module.
    code_offset: typing.Optional[int]
    #: The language of the script.
    script_language: typing.Optional[debugger.ScriptLanguage]
    #: If the scriptLanguage is WebAssembly, the source of debug symbols for the module.
    debug_symbols: typing.Optional[typing.List[debugger.DebugSymbols]]
    #: The name the embedder supplied for this script.
    embedder_name: typing.Optional[str]
    #: The list of set breakpoints in this script if calls to ``setBreakpointByUrl``
    #: matches this script's URL or hash. Clients that use this list can ignore the
    #: ``breakpointResolved`` event. They are equivalent.
    resolved_breakpoints: typing.Optional[typing.List[ResolvedBreakpoint]]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ScriptParsed:
        return cls(
            script_id=runtime.ScriptId.from_json(json['scriptId']),
            url=str(json['url']),
            start_line=int(json['startLine']),
            start_column=int(json['startColumn']),
            end_line=int(json['endLine']),
            end_column=int(json['endColumn']),
            execution_context_id=runtime.ExecutionContextId.from_json(json['executionContextId']),
            hash_=str(json['hash']),
            build_id=str(json['buildId']),
            execution_context_aux_data=dict(json['executionContextAuxData']) if 'executionContextAuxData' in json else None,
            is_live_edit=bool(json['isLiveEdit']) if 'isLiveEdit' in json else None,
            source_map_url=str(json['sourceMapURL']) if 'sourceMapURL' in json else None,
            has_source_url=bool(json['hasSourceURL']) if 'hasSourceURL' in json else None,
            is_module=bool(json['isModule']) if 'isModule' in json else None,
            length=int(json['length']) if 'length' in json else None,
            stack_trace=runtime.StackTrace.from_json(json['stackTrace']) if 'stackTrace' in json else None,
            code_offset=int(json['codeOffset']) if 'codeOffset' in json else None,
            script_language=debugger.ScriptLanguage.from_json(json['scriptLanguage']) if 'scriptLanguage' in json else None,
            debug_symbols=[debugger.DebugSymbols.from_json(i) for i in json['debugSymbols']] if 'debugSymbols' in json else None,
            embedder_name=str(json['embedderName']) if 'embedderName' in json else None,
            resolved_breakpoints=[ResolvedBreakpoint.from_json(i) for i in json['resolvedBreakpoints']] if 'resolvedBreakpoints' in json else None
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v136\debugger.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Debugger
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime


class BreakpointId(str):
    '''
    Breakpoint identifier.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> BreakpointId:
        return cls(json)

    def __repr__(self):
        return 'BreakpointId({})'.format(super().__repr__())


class CallFrameId(str):
    '''
    Call frame identifier.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> CallFrameId:
        return cls(json)

    def __repr__(self):
        return 'CallFrameId({})'.format(super().__repr__())


@dataclass
class Location:
    '''
    Location in the source code.
    '''
    #: Script identifier as reported in the ``Debugger.scriptParsed``.
    script_id: runtime.ScriptId

    #: Line number in the script (0-based).
    line_number: int

    #: Column number in the script (0-based).
    column_number: typing.Optional[int] = None

    def to_json(self):
        json = dict()
        json['scriptId'] = self.script_id.to_json()
        json['lineNumber'] = self.line_number
        if self.column_number is not None:
            json['columnNumber'] = self.column_number
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            script_id=runtime.ScriptId.from_json(json['scriptId']),
            line_number=int(json['lineNumber']),
            column_number=int(json['columnNumber']) if 'columnNumber' in json else None,
        )


@dataclass
class ScriptPosition:
    '''
    Location in the source code.
    '''
    line_number: int

    column_number: int

    def to_json(self):
        json = dict()
        json['lineNumber'] = self.line_number
        json['columnNumber'] = self.column_number
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            line_number=int(json['lineNumber']),
            column_number=int(json['columnNumber']),
        )


@dataclass
class LocationRange:
    '''
    Location range within one script.
    '''
    script_id: runtime.ScriptId

    start: ScriptPosition

    end: ScriptPosition

    def to_json(self):
        json = dict()
        json['scriptId'] = self.script_id.to_json()
        json['start'] = self.start.to_json()
        json['end'] = self.end.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            script_id=runtime.ScriptId.from_json(json['scriptId']),
            start=ScriptPosition.from_json(json['start']),
            end=ScriptPosition.from_json(json['end']),
        )


@dataclass
class CallFrame:
    '''
    JavaScript call frame. Array of call frames form the call stack.
    '''
    #: Call frame identifier. This identifier is only valid while the virtual machine is paused.
    call_frame_id: CallFrameId

    #: Name of the JavaScript function called on this call frame.
    function_name: str

    #: Location in the source code.
    location: Location

    #: JavaScript script name or url.
    #: Deprecated in favor of using the ``location.scriptId`` to resolve the URL via a previously
    #: sent ``Debugger.scriptParsed`` event.
    url: str

    #: Scope chain for this call frame.
    scope_chain: typing.List[Scope]

    #: ``this`` object for this call frame.
    this: runtime.RemoteObject

    #: Location in the source code.
    function_location: typing.Optional[Location] = None

    #: The value being returned, if the function is at return point.
    return_value: typing.Optional[runtime.RemoteObject] = None

    #: Valid only while the VM is paused and indicates whether this frame
    #: can be restarted or not. Note that a ``true`` value here does not
    #: guarantee that Debugger#restartFrame with this CallFrameId will be
    #: successful, but it is very likely.
    can_be_restarted: typing.Optional[bool] = None

    def to_json(self):
        json = dict()
        json['callFrameId'] = self.call_frame_id.to_json()
        json['functionName'] = self.function_name
        json['location'] = self.location.to_json()
        json['url'] = self.url
        json['scopeChain'] = [i.to_json() for i in self.scope_chain]
        json['this'] = self.this.to_json()
        if self.function_location is not None:
            json['functionLocation'] = self.function_location.to_json()
        if self.return_value is not None:
            json['returnValue'] = self.return_value.to_json()
        if self.can_be_restarted is not None:
            json['canBeRestarted'] = self.can_be_restarted
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            call_frame_id=CallFrameId.from_json(json['callFrameId']),
            function_name=str(json['functionName']),
            location=Location.from_json(json['location']),
            url=str(json['url']),
            scope_chain=[Scope.from_json(i) for i in json['scopeChain']],
            this=runtime.RemoteObject.from_json(json['this']),
            function_location=Location.from_json(json['functionLocation']) if 'functionLocation' in json else None,
            return_value=runtime.RemoteObject.from_json(json['returnValue']) if 'returnValue' in json else None,
            can_be_restarted=bool(json['canBeRestarted']) if 'canBeRestarted' in json else None,
        )


@dataclass
class Scope:
    '''
    Scope description.
    '''
    #: Scope type.
    type_: str

    #: Object representing the scope. For ``global`` and ``with`` scopes it represents the actual
    #: object; for the rest of the scopes, it is artificial transient object enumerating scope
    #: variables as its properties.
    object_: runtime.RemoteObject

    name: typing.Optional[str] = None

    #: Location in the source code where scope starts
    start_location: typing.Optional[Location] = None

    #: Location in the source code where scope ends
    end_location: typing.Optional[Location] = None

    def to_json(self):
        json = dict()
        json['type'] = self.type_
        json['object'] = self.object_.to_json()
        if self.name is not None:
            json['name'] = self.name
        if self.start_location is not None:
            json['startLocation'] = self.start_location.to_json()
        if self.end_location is not None:
            json['endLocation'] = self.end_location.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            type_=str(json['type']),
            object_=runtime.RemoteObject.from_json(json['object']),
            name=str(json['name']) if 'name' in json else None,
            start_location=Location.from_json(json['startLocation']) if 'startLocation' in json else None,
            end_location=Location.from_json(json['endLocation']) if 'endLocation' in json else None,
        )


@dataclass
class SearchMatch:
    '''
    Search match for resource.
    '''
    #: Line number in resource content.
    line_number: float

    #: Line with match content.
    line_content: str

    def to_json(self):
        json = dict()
        json['lineNumber'] = self.line_number
        json['lineContent'] = self.line_content
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            line_number=float(json['lineNumber']),
            line_content=str(json['lineContent']),
        )


@dataclass
class BreakLocation:
    #: Script identifier as reported in the ``Debugger.scriptParsed``.
    script_id: runtime.ScriptId

    #: Line number in the script (0-based).
    line_number: int

    #: Column number in the script (0-based).
    column_number: typing.Optional[int] = None

    type_: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['scriptId'] = self.script_id.to_json()
        json['lineNumber'] = self.line_number
        if self.column_number is not None:
            json['columnNumber'] = self.column_number
        if self.type_ is not None:
            json['type'] = self.type_
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            script_id=runtime.ScriptId.from_json(json['scriptId']),
            line_number=int(json['lineNumber']),
            column_number=int(json['columnNumber']) if 'columnNumber' in json else None,
            type_=str(json['type']) if 'type' in json else None,
        )


@dataclass
class WasmDisassemblyChunk:
    #: The next chunk of disassembled lines.
    lines: typing.List[str]

    #: The bytecode offsets describing the start of each line.
    bytecode_offsets: typing.List[int]

    def to_json(self):
        json = dict()
        json['lines'] = [i for i in self.lines]
        json['bytecodeOffsets'] = [i for i in self.bytecode_offsets]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            lines=[str(i) for i in json['lines']],
            bytecode_offsets=[int(i) for i in json['bytecodeOffsets']],
        )


class ScriptLanguage(enum.Enum):
    '''
    Enum of possible script languages.
    '''
    JAVA_SCRIPT = "JavaScript"
    WEB_ASSEMBLY = "WebAssembly"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class DebugSymbols:
    '''
    Debug symbols available for a wasm script.
    '''
    #: Type of the debug symbols.
    type_: str

    #: URL of the external symbol source.
    external_url: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['type'] = self.type_
        if self.external_url is not None:
            json['externalURL'] = self.external_url
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            type_=str(json['type']),
            external_url=str(json['externalURL']) if 'externalURL' in json else None,
        )


@dataclass
class ResolvedBreakpoint:
    #: Breakpoint unique identifier.
    breakpoint_id: BreakpointId

    #: Actual breakpoint location.
    location: Location

    def to_json(self):
        json = dict()
        json['breakpointId'] = self.breakpoint_id.to_json()
        json['location'] = self.location.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            breakpoint_id=BreakpointId.from_json(json['breakpointId']),
            location=Location.from_json(json['location']),
        )


def continue_to_location(
        location: Location,
        target_call_frames: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Continues execution until specific location is reached.

    :param location: Location to continue to.
    :param target_call_frames: *(Optional)*
    '''
    params: T_JSON_DICT = dict()
    params['location'] = location.to_json()
    if target_call_frames is not None:
        params['targetCallFrames'] = target_call_frames
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.continueToLocation',
        'params': params,
    }
    json = yield cmd_dict


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables debugger for given page.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.disable',
    }
    json = yield cmd_dict


def enable(
        max_scripts_cache_size: typing.Optional[float] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,runtime.UniqueDebuggerId]:
    '''
    Enables debugger for the given page. Clients should not assume that the debugging has been
    enabled until the result for this command is received.

    :param max_scripts_cache_size: **(EXPERIMENTAL)** *(Optional)* The maximum size in bytes of collected scripts (not referenced by other heap objects) the debugger can hold. Puts no limit if parameter is omitted.
    :returns: Unique identifier of the debugger.
    '''
    params: T_JSON_DICT = dict()
    if max_scripts_cache_size is not None:
        params['maxScriptsCacheSize'] = max_scripts_cache_size
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.enable',
        'params': params,
    }
    json = yield cmd_dict
    return runtime.UniqueDebuggerId.from_json(json['debuggerId'])


def evaluate_on_call_frame(
        call_frame_id: CallFrameId,
        expression: str,
        object_group: typing.Optional[str] = None,
        include_command_line_api: typing.Optional[bool] = None,
        silent: typing.Optional[bool] = None,
        return_by_value: typing.Optional[bool] = None,
        generate_preview: typing.Optional[bool] = None,
        throw_on_side_effect: typing.Optional[bool] = None,
        timeout: typing.Optional[runtime.TimeDelta] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[runtime.RemoteObject, typing.Optional[runtime.ExceptionDetails]]]:
    '''
    Evaluates expression on a given call frame.

    :param call_frame_id: Call frame identifier to evaluate on.
    :param expression: Expression to evaluate.
    :param object_group: *(Optional)* String object group name to put result into (allows rapid releasing resulting object handles using ```releaseObjectGroup````).
    :param include_command_line_api: *(Optional)* Specifies whether command line API should be available to the evaluated expression, defaults to false.
    :param silent: *(Optional)* In silent mode exceptions thrown during evaluation are not reported and do not pause execution. Overrides ````setPauseOnException``` state.
    :param return_by_value: *(Optional)* Whether the result is expected to be a JSON object that should be sent by value.
    :param generate_preview: **(EXPERIMENTAL)** *(Optional)* Whether preview should be generated for the result.
    :param throw_on_side_effect: *(Optional)* Whether to throw an exception if side effect cannot be ruled out during evaluation.
    :param timeout: **(EXPERIMENTAL)** *(Optional)* Terminate execution after timing out (number of milliseconds).
    :returns: A tuple with the following items:

        0. **result** - Object wrapper for the evaluation result.
        1. **exceptionDetails** - *(Optional)* Exception details.
    '''
    params: T_JSON_DICT = dict()
    params['callFrameId'] = call_frame_id.to_json()
    params['expression'] = expression
    if object_group is not None:
        params['objectGroup'] = object_group
    if include_command_line_api is not None:
        params['includeCommandLineAPI'] = include_command_line_api
    if silent is not None:
        params['silent'] = silent
    if return_by_value is not None:
        params['returnByValue'] = return_by_value
    if generate_preview is not None:
        params['generatePreview'] = generate_preview
    if throw_on_side_effect is not None:
        params['throwOnSideEffect'] = throw_on_side_effect
    if timeout is not None:
        params['timeout'] = timeout.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.evaluateOnCallFrame',
        'params': params,
    }
    json = yield cmd_dict
    return (
        runtime.RemoteObject.from_json(json['result']),
        runtime.ExceptionDetails.from_json(json['exceptionDetails']) if 'exceptionDetails' in json else None
    )


def get_possible_breakpoints(
        start: Location,
        end: typing.Optional[Location] = None,
        restrict_to_function: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[BreakLocation]]:
    '''
    Returns possible locations for breakpoint. scriptId in start and end range locations should be
    the same.

    :param start: Start of range to search possible breakpoint locations in.
    :param end: *(Optional)* End of range to search possible breakpoint locations in (excluding). When not specified, end of scripts is used as end of range.
    :param restrict_to_function: *(Optional)* Only consider locations which are in the same (non-nested) function as start.
    :returns: List of the possible breakpoint locations.
    '''
    params: T_JSON_DICT = dict()
    params['start'] = start.to_json()
    if end is not None:
        params['end'] = end.to_json()
    if restrict_to_function is not None:
        params['restrictToFunction'] = restrict_to_function
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.getPossibleBreakpoints',
        'params': params,
    }
    json = yield cmd_dict
    return [BreakLocation.from_json(i) for i in json['locations']]


def get_script_source(
        script_id: runtime.ScriptId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[str, typing.Optional[str]]]:
    '''
    Returns source for the script with given id.

    :param script_id: Id of the script to get source for.
    :returns: A tuple with the following items:

        0. **scriptSource** - Script source (empty in case of Wasm bytecode).
        1. **bytecode** - *(Optional)* Wasm bytecode.
    '''
    params: T_JSON_DICT = dict()
    params['scriptId'] = script_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.getScriptSource',
        'params': params,
    }
    json = yield cmd_dict
    return (
        str(json['scriptSource']),
        str(json['bytecode']) if 'bytecode' in json else None
    )


def disassemble_wasm_module(
        script_id: runtime.ScriptId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[typing.Optional[str], int, typing.List[int], WasmDisassemblyChunk]]:
    '''


    **EXPERIMENTAL**

    :param script_id: Id of the script to disassemble
    :returns: A tuple with the following items:

        0. **streamId** - *(Optional)* For large modules, return a stream from which additional chunks of disassembly can be read successively.
        1. **totalNumberOfLines** - The total number of lines in the disassembly text.
        2. **functionBodyOffsets** - The offsets of all function bodies, in the format [start1, end1, start2, end2, ...] where all ends are exclusive.
        3. **chunk** - The first chunk of disassembly.
    '''
    params: T_JSON_DICT = dict()
    params['scriptId'] = script_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.disassembleWasmModule',
        'params': params,
    }
    json = yield cmd_dict
    return (
        str(json['streamId']) if 'streamId' in json else None,
        int(json['totalNumberOfLines']),
        [int(i) for i in json['functionBodyOffsets']],
        WasmDisassemblyChunk.from_json(json['chunk'])
    )


def next_wasm_disassembly_chunk(
        stream_id: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,WasmDisassemblyChunk]:
    '''
    Disassemble the next chunk of lines for the module corresponding to the
    stream. If disassembly is complete, this API will invalidate the streamId
    and return an empty chunk. Any subsequent calls for the now invalid stream
    will return errors.

    **EXPERIMENTAL**

    :param stream_id:
    :returns: The next chunk of disassembly.
    '''
    params: T_JSON_DICT = dict()
    params['streamId'] = stream_id
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.nextWasmDisassemblyChunk',
        'params': params,
    }
    json = yield cmd_dict
    return WasmDisassemblyChunk.from_json(json['chunk'])


def get_wasm_bytecode(
        script_id: runtime.ScriptId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,str]:
    '''
    This command is deprecated. Use getScriptSource instead.

    :param script_id: Id of the Wasm script to get source for.
    :returns: Script source.
    '''
    params: T_JSON_DICT = dict()
    params['scriptId'] = script_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.getWasmBytecode',
        'params': params,
    }
    json = yield cmd_dict
    return str(json['bytecode'])


def get_stack_trace(
        stack_trace_id: runtime.StackTraceId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,runtime.StackTrace]:
    '''
    Returns stack trace with given ``stackTraceId``.

    **EXPERIMENTAL**

    :param stack_trace_id:
    :returns: 
    '''
    params: T_JSON_DICT = dict()
    params['stackTraceId'] = stack_trace_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.getStackTrace',
        'params': params,
    }
    json = yield cmd_dict
    return runtime.StackTrace.from_json(json['stackTrace'])


def pause() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Stops on the next JavaScript statement.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.pause',
    }
    json = yield cmd_dict


def pause_on_async_call(
        parent_stack_trace_id: runtime.StackTraceId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''


    **EXPERIMENTAL**

    :param parent_stack_trace_id: Debugger will pause when async call with given stack trace is started.
    '''
    params: T_JSON_DICT = dict()
    params['parentStackTraceId'] = parent_stack_trace_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.pauseOnAsyncCall',
        'params': params,
    }
    json = yield cmd_dict


def remove_breakpoint(
        breakpoint_id: BreakpointId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Removes JavaScript breakpoint.

    :param breakpoint_id:
    '''
    params: T_JSON_DICT = dict()
    params['breakpointId'] = breakpoint_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.removeBreakpoint',
        'params': params,
    }
    json = yield cmd_dict


def restart_frame(
        call_frame_id: CallFrameId,
        mode: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[typing.List[CallFrame], typing.Optional[runtime.StackTrace], typing.Optional[runtime.StackTraceId]]]:
    '''
    Restarts particular call frame from the beginning. The old, deprecated
    behavior of ``restartFrame`` is to stay paused and allow further CDP commands
    after a restart was scheduled. This can cause problems with restarting, so
    we now continue execution immediatly after it has been scheduled until we
    reach the beginning of the restarted frame.

    To stay back-wards compatible, ``restartFrame`` now expects a ``mode``
    parameter to be present. If the ``mode`` parameter is missing, ``restartFrame``
    errors out.

    The various return values are deprecated and ``callFrames`` is always empty.
    Use the call frames from the ``Debugger#paused`` events instead, that fires
    once V8 pauses at the beginning of the restarted function.

    :param call_frame_id: Call frame identifier to evaluate on.
    :param mode: **(EXPERIMENTAL)** *(Optional)* The ```mode```` parameter must be present and set to 'StepInto', otherwise ````restartFrame``` will error out.
    :returns: A tuple with the following items:

        0. **callFrames** - New stack trace.
        1. **asyncStackTrace** - *(Optional)* Async stack trace, if any.
        2. **asyncStackTraceId** - *(Optional)* Async stack trace, if any.
    '''
    params: T_JSON_DICT = dict()
    params['callFrameId'] = call_frame_id.to_json()
    if mode is not None:
        params['mode'] = mode
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.restartFrame',
        'params': params,
    }
    json = yield cmd_dict
    return (
        [CallFrame.from_json(i) for i in json['callFrames']],
        runtime.StackTrace.from_json(json['asyncStackTrace']) if 'asyncStackTrace' in json else None,
        runtime.StackTraceId.from_json(json['asyncStackTraceId']) if 'asyncStackTraceId' in json else None
    )


def resume(
        terminate_on_resume: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Resumes JavaScript execution.

    :param terminate_on_resume: *(Optional)* Set to true to terminate execution upon resuming execution. In contrast to Runtime.terminateExecution, this will allows to execute further JavaScript (i.e. via evaluation) until execution of the paused code is actually resumed, at which point termination is triggered. If execution is currently not paused, this parameter has no effect.
    '''
    params: T_JSON_DICT = dict()
    if terminate_on_resume is not None:
        params['terminateOnResume'] = terminate_on_resume
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.resume',
        'params': params,
    }
    json = yield cmd_dict


def search_in_content(
        script_id: runtime.ScriptId,
        query: str,
        case_sensitive: typing.Optional[bool] = None,
        is_regex: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[SearchMatch]]:
    '''
    Searches for given string in script content.

    :param script_id: Id of the script to search in.
    :param query: String to search for.
    :param case_sensitive: *(Optional)* If true, search is case sensitive.
    :param is_regex: *(Optional)* If true, treats string parameter as regex.
    :returns: List of search matches.
    '''
    params: T_JSON_DICT = dict()
    params['scriptId'] = script_id.to_json()
    params['query'] = query
    if case_sensitive is not None:
        params['caseSensitive'] = case_sensitive
    if is_regex is not None:
        params['isRegex'] = is_regex
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.searchInContent',
        'params': params,
    }
    json = yield cmd_dict
    return [SearchMatch.from_json(i) for i in json['result']]


def set_async_call_stack_depth(
        max_depth: int
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables or disables async call stacks tracking.

    :param max_depth: Maximum depth of async call stacks. Setting to ```0``` will effectively disable collecting async call stacks (default).
    '''
    params: T_JSON_DICT = dict()
    params['maxDepth'] = max_depth
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.setAsyncCallStackDepth',
        'params': params,
    }
    json = yield cmd_dict


def set_blackbox_execution_contexts(
        unique_ids: typing.List[str]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Replace previous blackbox execution contexts with passed ones. Forces backend to skip
    stepping/pausing in scripts in these execution contexts. VM will try to leave blackboxed script by
    performing 'step in' several times, finally resorting to 'step out' if unsuccessful.

    **EXPERIMENTAL**

    :param unique_ids: Array of execution context unique ids for the debugger to ignore.
    '''
    params: T_JSON_DICT = dict()
    params['uniqueIds'] = [i for i in unique_ids]
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.setBlackboxExecutionContexts',
        'params': params,
    }
    json = yield cmd_dict


def set_blackbox_patterns(
        patterns: typing.List[str],
        skip_anonymous: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Replace previous blackbox patterns with passed ones. Forces backend to skip stepping/pausing in
    scripts with url matching one of the patterns. VM will try to leave blackboxed script by
    performing 'step in' several times, finally resorting to 'step out' if unsuccessful.

    **EXPERIMENTAL**

    :param patterns: Array of regexps that will be used to check script url for blackbox state.
    :param skip_anonymous: *(Optional)* If true, also ignore scripts with no source url.
    '''
    params: T_JSON_DICT = dict()
    params['patterns'] = [i for i in patterns]
    if skip_anonymous is not None:
        params['skipAnonymous'] = skip_anonymous
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.setBlackboxPatterns',
        'params': params,
    }
    json = yield cmd_dict


def set_blackboxed_ranges(
        script_id: runtime.ScriptId,
        positions: typing.List[ScriptPosition]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Makes backend skip steps in the script in blackboxed ranges. VM will try leave blacklisted
    scripts by performing 'step in' several times, finally resorting to 'step out' if unsuccessful.
    Positions array contains positions where blackbox state is changed. First interval isn't
    blackboxed. Array should be sorted.

    **EXPERIMENTAL**

    :param script_id: Id of the script.
    :param positions:
    '''
    params: T_JSON_DICT = dict()
    params['scriptId'] = script_id.to_json()
    params['positions'] = [i.to_json() for i in positions]
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.setBlackboxedRanges',
        'params': params,
    }
    json = yield cmd_dict


def set_breakpoint(
        location: Location,
        condition: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[BreakpointId, Location]]:
    '''
    Sets JavaScript breakpoint at a given location.

    :param location: Location to set breakpoint in.
    :param condition: *(Optional)* Expression to use as a breakpoint condition. When specified, debugger will only stop on the breakpoint if this expression evaluates to true.
    :returns: A tuple with the following items:

        0. **breakpointId** - Id of the created breakpoint for further reference.
        1. **actualLocation** - Location this breakpoint resolved into.
    '''
    params: T_JSON_DICT = dict()
    params['location'] = location.to_json()
    if condition is not None:
        params['condition'] = condition
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.setBreakpoint',
        'params': params,
    }
    json = yield cmd_dict
    return (
        BreakpointId.from_json(json['breakpointId']),
        Location.from_json(json['actualLocation'])
    )


def set_instrumentation_breakpoint(
        instrumentation: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,BreakpointId]:
    '''
    Sets instrumentation breakpoint.

    :param instrumentation: Instrumentation name.
    :returns: Id of the created breakpoint for further reference.
    '''
    params: T_JSON_DICT = dict()
    params['instrumentation'] = instrumentation
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.setInstrumentationBreakpoint',
        'params': params,
    }
    json = yield cmd_dict
    return BreakpointId.from_json(json['breakpointId'])


def set_breakpoint_by_url(
        line_number: int,
        url: typing.Optional[str] = None,
        url_regex: typing.Optional[str] = None,
        script_hash: typing.Optional[str] = None,
        column_number: typing.Optional[int] = None,
        condition: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[BreakpointId, typing.List[Location]]]:
    '''
    Sets JavaScript breakpoint at given location specified either by URL or URL regex. Once this
    command is issued, all existing parsed scripts will have breakpoints resolved and returned in
    ``locations`` property. Further matching script parsing will result in subsequent
    ``breakpointResolved`` events issued. This logical breakpoint will survive page reloads.

    :param line_number: Line number to set breakpoint at.
    :param url: *(Optional)* URL of the resources to set breakpoint on.
    :param url_regex: *(Optional)* Regex pattern for the URLs of the resources to set breakpoints on. Either ```url```` or ````urlRegex``` must be specified.
    :param script_hash: *(Optional)* Script hash of the resources to set breakpoint on.
    :param column_number: *(Optional)* Offset in the line to set breakpoint at.
    :param condition: *(Optional)* Expression to use as a breakpoint condition. When specified, debugger will only stop on the breakpoint if this expression evaluates to true.
    :returns: A tuple with the following items:

        0. **breakpointId** - Id of the created breakpoint for further reference.
        1. **locations** - List of the locations this breakpoint resolved into upon addition.
    '''
    params: T_JSON_DICT = dict()
    params['lineNumber'] = line_number
    if url is not None:
        params['url'] = url
    if url_regex is not None:
        params['urlRegex'] = url_regex
    if script_hash is not None:
        params['scriptHash'] = script_hash
    if column_number is not None:
        params['columnNumber'] = column_number
    if condition is not None:
        params['condition'] = condition
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.setBreakpointByUrl',
        'params': params,
    }
    json = yield cmd_dict
    return (
        BreakpointId.from_json(json['breakpointId']),
        [Location.from_json(i) for i in json['locations']]
    )


def set_breakpoint_on_function_call(
        object_id: runtime.RemoteObjectId,
        condition: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,BreakpointId]:
    '''
    Sets JavaScript breakpoint before each call to the given function.
    If another function was created from the same source as a given one,
    calling it will also trigger the breakpoint.

    **EXPERIMENTAL**

    :param object_id: Function object id.
    :param condition: *(Optional)* Expression to use as a breakpoint condition. When specified, debugger will stop on the breakpoint if this expression evaluates to true.
    :returns: Id of the created breakpoint for further reference.
    '''
    params: T_JSON_DICT = dict()
    params['objectId'] = object_id.to_json()
    if condition is not None:
        params['condition'] = condition
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.setBreakpointOnFunctionCall',
        'params': params,
    }
    json = yield cmd_dict
    return BreakpointId.from_json(json['breakpointId'])


def set_breakpoints_active(
        active: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Activates / deactivates all breakpoints on the page.

    :param active: New value for breakpoints active state.
    '''
    params: T_JSON_DICT = dict()
    params['active'] = active
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.setBreakpointsActive',
        'params': params,
    }
    json = yield cmd_dict


def set_pause_on_exceptions(
        state: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Defines pause on exceptions state. Can be set to stop on all exceptions, uncaught exceptions,
    or caught exceptions, no exceptions. Initial pause on exceptions state is ``none``.

    :param state: Pause on exceptions mode.
    '''
    params: T_JSON_DICT = dict()
    params['state'] = state
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.setPauseOnExceptions',
        'params': params,
    }
    json = yield cmd_dict


def set_return_value(
        new_value: runtime.CallArgument
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Changes return value in top frame. Available only at return break position.

    **EXPERIMENTAL**

    :param new_value: New return value.
    '''
    params: T_JSON_DICT = dict()
    params['newValue'] = new_value.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.setReturnValue',
        'params': params,
    }
    json = yield cmd_dict


def set_script_source(
        script_id: runtime.ScriptId,
        script_source: str,
        dry_run: typing.Optional[bool] = None,
        allow_top_frame_editing: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[typing.Optional[typing.List[CallFrame]], typing.Optional[bool], typing.Optional[runtime.StackTrace], typing.Optional[runtime.StackTraceId], str, typing.Optional[runtime.ExceptionDetails]]]:
    '''
    Edits JavaScript source live.

    In general, functions that are currently on the stack can not be edited with
    a single exception: If the edited function is the top-most stack frame and
    that is the only activation of that function on the stack. In this case
    the live edit will be successful and a ``Debugger.restartFrame`` for the
    top-most function is automatically triggered.

    :param script_id: Id of the script to edit.
    :param script_source: New content of the script.
    :param dry_run: *(Optional)* If true the change will not actually be applied. Dry run may be used to get result description without actually modifying the code.
    :param allow_top_frame_editing: **(EXPERIMENTAL)** *(Optional)* If true, then ```scriptSource```` is allowed to change the function on top of the stack as long as the top-most stack frame is the only activation of that function.
    :returns: A tuple with the following items:

        0. **callFrames** - *(Optional)* New stack trace in case editing has happened while VM was stopped.
        1. **stackChanged** - *(Optional)* Whether current call stack  was modified after applying the changes.
        2. **asyncStackTrace** - *(Optional)* Async stack trace, if any.
        3. **asyncStackTraceId** - *(Optional)* Async stack trace, if any.
        4. **status** - Whether the operation was successful or not. Only `` Ok`` denotes a successful live edit while the other enum variants denote why the live edit failed.
        5. **exceptionDetails** - *(Optional)* Exception details if any. Only present when `` status`` is `` CompileError`.
    '''
    params: T_JSON_DICT = dict()
    params['scriptId'] = script_id.to_json()
    params['scriptSource'] = script_source
    if dry_run is not None:
        params['dryRun'] = dry_run
    if allow_top_frame_editing is not None:
        params['allowTopFrameEditing'] = allow_top_frame_editing
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.setScriptSource',
        'params': params,
    }
    json = yield cmd_dict
    return (
        [CallFrame.from_json(i) for i in json['callFrames']] if 'callFrames' in json else None,
        bool(json['stackChanged']) if 'stackChanged' in json else None,
        runtime.StackTrace.from_json(json['asyncStackTrace']) if 'asyncStackTrace' in json else None,
        runtime.StackTraceId.from_json(json['asyncStackTraceId']) if 'asyncStackTraceId' in json else None,
        str(json['status']),
        runtime.ExceptionDetails.from_json(json['exceptionDetails']) if 'exceptionDetails' in json else None
    )


def set_skip_all_pauses(
        skip: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Makes page not interrupt on any pauses (breakpoint, exception, dom exception etc).

    :param skip: New value for skip pauses state.
    '''
    params: T_JSON_DICT = dict()
    params['skip'] = skip
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.setSkipAllPauses',
        'params': params,
    }
    json = yield cmd_dict


def set_variable_value(
        scope_number: int,
        variable_name: str,
        new_value: runtime.CallArgument,
        call_frame_id: CallFrameId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Changes value of variable in a callframe. Object-based scopes are not supported and must be
    mutated manually.

    :param scope_number: 0-based number of scope as was listed in scope chain. Only 'local', 'closure' and 'catch' scope types are allowed. Other scopes could be manipulated manually.
    :param variable_name: Variable name.
    :param new_value: New variable value.
    :param call_frame_id: Id of callframe that holds variable.
    '''
    params: T_JSON_DICT = dict()
    params['scopeNumber'] = scope_number
    params['variableName'] = variable_name
    params['newValue'] = new_value.to_json()
    params['callFrameId'] = call_frame_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.setVariableValue',
        'params': params,
    }
    json = yield cmd_dict


def step_into(
        break_on_async_call: typing.Optional[bool] = None,
        skip_list: typing.Optional[typing.List[LocationRange]] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Steps into the function call.

    :param break_on_async_call: **(EXPERIMENTAL)** *(Optional)* Debugger will pause on the execution of the first async task which was scheduled before next pause.
    :param skip_list: **(EXPERIMENTAL)** *(Optional)* The skipList specifies location ranges that should be skipped on step into.
    '''
    params: T_JSON_DICT = dict()
    if break_on_async_call is not None:
        params['breakOnAsyncCall'] = break_on_async_call
    if skip_list is not None:
        params['skipList'] = [i.to_json() for i in skip_list]
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.stepInto',
        'params': params,
    }
    json = yield cmd_dict


def step_out() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Steps out of the function call.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.stepOut',
    }
    json = yield cmd_dict


def step_over(
        skip_list: typing.Optional[typing.List[LocationRange]] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Steps over the statement.

    :param skip_list: **(EXPERIMENTAL)** *(Optional)* The skipList specifies location ranges that should be skipped on step over.
    '''
    params: T_JSON_DICT = dict()
    if skip_list is not None:
        params['skipList'] = [i.to_json() for i in skip_list]
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.stepOver',
        'params': params,
    }
    json = yield cmd_dict


@event_class('Debugger.breakpointResolved')
@dataclass
class BreakpointResolved:
    '''
    Fired when breakpoint is resolved to an actual script and location.
    Deprecated in favor of ``resolvedBreakpoints`` in the ``scriptParsed`` event.
    '''
    #: Breakpoint unique identifier.
    breakpoint_id: BreakpointId
    #: Actual breakpoint location.
    location: Location

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> BreakpointResolved:
        return cls(
            breakpoint_id=BreakpointId.from_json(json['breakpointId']),
            location=Location.from_json(json['location'])
        )


@event_class('Debugger.paused')
@dataclass
class Paused:
    '''
    Fired when the virtual machine stopped on breakpoint or exception or any other stop criteria.
    '''
    #: Call stack the virtual machine stopped on.
    call_frames: typing.List[CallFrame]
    #: Pause reason.
    reason: str
    #: Object containing break-specific auxiliary properties.
    data: typing.Optional[dict]
    #: Hit breakpoints IDs
    hit_breakpoints: typing.Optional[typing.List[str]]
    #: Async stack trace, if any.
    async_stack_trace: typing.Optional[runtime.StackTrace]
    #: Async stack trace, if any.
    async_stack_trace_id: typing.Optional[runtime.StackTraceId]
    #: Never present, will be removed.
    async_call_stack_trace_id: typing.Optional[runtime.StackTraceId]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> Paused:
        return cls(
            call_frames=[CallFrame.from_json(i) for i in json['callFrames']],
            reason=str(json['reason']),
            data=dict(json['data']) if 'data' in json else None,
            hit_breakpoints=[str(i) for i in json['hitBreakpoints']] if 'hitBreakpoints' in json else None,
            async_stack_trace=runtime.StackTrace.from_json(json['asyncStackTrace']) if 'asyncStackTrace' in json else None,
            async_stack_trace_id=runtime.StackTraceId.from_json(json['asyncStackTraceId']) if 'asyncStackTraceId' in json else None,
            async_call_stack_trace_id=runtime.StackTraceId.from_json(json['asyncCallStackTraceId']) if 'asyncCallStackTraceId' in json else None
        )


@event_class('Debugger.resumed')
@dataclass
class Resumed:
    '''
    Fired when the virtual machine resumed execution.
    '''


    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> Resumed:
        return cls(

        )


@event_class('Debugger.scriptFailedToParse')
@dataclass
class ScriptFailedToParse:
    '''
    Fired when virtual machine fails to parse the script.
    '''
    #: Identifier of the script parsed.
    script_id: runtime.ScriptId
    #: URL or name of the script parsed (if any).
    url: str
    #: Line offset of the script within the resource with given URL (for script tags).
    start_line: int
    #: Column offset of the script within the resource with given URL.
    start_column: int
    #: Last line of the script.
    end_line: int
    #: Length of the last line of the script.
    end_column: int
    #: Specifies script creation context.
    execution_context_id: runtime.ExecutionContextId
    #: Content hash of the script, SHA-256.
    hash_: str
    #: For Wasm modules, the content of the ``build_id`` custom section.
    build_id: str
    #: Embedder-specific auxiliary data likely matching {isDefault: boolean, type: 'default'``'isolated'``'worker', frameId: string}
    execution_context_aux_data: typing.Optional[dict]
    #: URL of source map associated with script (if any).
    source_map_url: typing.Optional[str]
    #: True, if this script has sourceURL.
    has_source_url: typing.Optional[bool]
    #: True, if this script is ES6 module.
    is_module: typing.Optional[bool]
    #: This script length.
    length: typing.Optional[int]
    #: JavaScript top stack frame of where the script parsed event was triggered if available.
    stack_trace: typing.Optional[runtime.StackTrace]
    #: If the scriptLanguage is WebAssembly, the code section offset in the module.
    code_offset: typing.Optional[int]
    #: The language of the script.
    script_language: typing.Optional[debugger.ScriptLanguage]
    #: The name the embedder supplied for this script.
    embedder_name: typing.Optional[str]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ScriptFailedToParse:
        return cls(
            script_id=runtime.ScriptId.from_json(json['scriptId']),
            url=str(json['url']),
            start_line=int(json['startLine']),
            start_column=int(json['startColumn']),
            end_line=int(json['endLine']),
            end_column=int(json['endColumn']),
            execution_context_id=runtime.ExecutionContextId.from_json(json['executionContextId']),
            hash_=str(json['hash']),
            build_id=str(json['buildId']),
            execution_context_aux_data=dict(json['executionContextAuxData']) if 'executionContextAuxData' in json else None,
            source_map_url=str(json['sourceMapURL']) if 'sourceMapURL' in json else None,
            has_source_url=bool(json['hasSourceURL']) if 'hasSourceURL' in json else None,
            is_module=bool(json['isModule']) if 'isModule' in json else None,
            length=int(json['length']) if 'length' in json else None,
            stack_trace=runtime.StackTrace.from_json(json['stackTrace']) if 'stackTrace' in json else None,
            code_offset=int(json['codeOffset']) if 'codeOffset' in json else None,
            script_language=debugger.ScriptLanguage.from_json(json['scriptLanguage']) if 'scriptLanguage' in json else None,
            embedder_name=str(json['embedderName']) if 'embedderName' in json else None
        )


@event_class('Debugger.scriptParsed')
@dataclass
class ScriptParsed:
    '''
    Fired when virtual machine parses script. This event is also fired for all known and uncollected
    scripts upon enabling debugger.
    '''
    #: Identifier of the script parsed.
    script_id: runtime.ScriptId
    #: URL or name of the script parsed (if any).
    url: str
    #: Line offset of the script within the resource with given URL (for script tags).
    start_line: int
    #: Column offset of the script within the resource with given URL.
    start_column: int
    #: Last line of the script.
    end_line: int
    #: Length of the last line of the script.
    end_column: int
    #: Specifies script creation context.
    execution_context_id: runtime.ExecutionContextId
    #: Content hash of the script, SHA-256.
    hash_: str
    #: For Wasm modules, the content of the ``build_id`` custom section.
    build_id: str
    #: Embedder-specific auxiliary data likely matching {isDefault: boolean, type: 'default'``'isolated'``'worker', frameId: string}
    execution_context_aux_data: typing.Optional[dict]
    #: True, if this script is generated as a result of the live edit operation.
    is_live_edit: typing.Optional[bool]
    #: URL of source map associated with script (if any).
    source_map_url: typing.Optional[str]
    #: True, if this script has sourceURL.
    has_source_url: typing.Optional[bool]
    #: True, if this script is ES6 module.
    is_module: typing.Optional[bool]
    #: This script length.
    length: typing.Optional[int]
    #: JavaScript top stack frame of where the script parsed event was triggered if available.
    stack_trace: typing.Optional[runtime.StackTrace]
    #: If the scriptLanguage is WebAssembly, the code section offset in the module.
    code_offset: typing.Optional[int]
    #: The language of the script.
    script_language: typing.Optional[debugger.ScriptLanguage]
    #: If the scriptLanguage is WebAssembly, the source of debug symbols for the module.
    debug_symbols: typing.Optional[typing.List[debugger.DebugSymbols]]
    #: The name the embedder supplied for this script.
    embedder_name: typing.Optional[str]
    #: The list of set breakpoints in this script if calls to ``setBreakpointByUrl``
    #: matches this script's URL or hash. Clients that use this list can ignore the
    #: ``breakpointResolved`` event. They are equivalent.
    resolved_breakpoints: typing.Optional[typing.List[ResolvedBreakpoint]]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ScriptParsed:
        return cls(
            script_id=runtime.ScriptId.from_json(json['scriptId']),
            url=str(json['url']),
            start_line=int(json['startLine']),
            start_column=int(json['startColumn']),
            end_line=int(json['endLine']),
            end_column=int(json['endColumn']),
            execution_context_id=runtime.ExecutionContextId.from_json(json['executionContextId']),
            hash_=str(json['hash']),
            build_id=str(json['buildId']),
            execution_context_aux_data=dict(json['executionContextAuxData']) if 'executionContextAuxData' in json else None,
            is_live_edit=bool(json['isLiveEdit']) if 'isLiveEdit' in json else None,
            source_map_url=str(json['sourceMapURL']) if 'sourceMapURL' in json else None,
            has_source_url=bool(json['hasSourceURL']) if 'hasSourceURL' in json else None,
            is_module=bool(json['isModule']) if 'isModule' in json else None,
            length=int(json['length']) if 'length' in json else None,
            stack_trace=runtime.StackTrace.from_json(json['stackTrace']) if 'stackTrace' in json else None,
            code_offset=int(json['codeOffset']) if 'codeOffset' in json else None,
            script_language=debugger.ScriptLanguage.from_json(json['scriptLanguage']) if 'scriptLanguage' in json else None,
            debug_symbols=[debugger.DebugSymbols.from_json(i) for i in json['debugSymbols']] if 'debugSymbols' in json else None,
            embedder_name=str(json['embedderName']) if 'embedderName' in json else None,
            resolved_breakpoints=[ResolvedBreakpoint.from_json(i) for i in json['resolvedBreakpoints']] if 'resolvedBreakpoints' in json else None
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v137\debugger.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Debugger
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime


class BreakpointId(str):
    '''
    Breakpoint identifier.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> BreakpointId:
        return cls(json)

    def __repr__(self):
        return 'BreakpointId({})'.format(super().__repr__())


class CallFrameId(str):
    '''
    Call frame identifier.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> CallFrameId:
        return cls(json)

    def __repr__(self):
        return 'CallFrameId({})'.format(super().__repr__())


@dataclass
class Location:
    '''
    Location in the source code.
    '''
    #: Script identifier as reported in the ``Debugger.scriptParsed``.
    script_id: runtime.ScriptId

    #: Line number in the script (0-based).
    line_number: int

    #: Column number in the script (0-based).
    column_number: typing.Optional[int] = None

    def to_json(self):
        json = dict()
        json['scriptId'] = self.script_id.to_json()
        json['lineNumber'] = self.line_number
        if self.column_number is not None:
            json['columnNumber'] = self.column_number
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            script_id=runtime.ScriptId.from_json(json['scriptId']),
            line_number=int(json['lineNumber']),
            column_number=int(json['columnNumber']) if 'columnNumber' in json else None,
        )


@dataclass
class ScriptPosition:
    '''
    Location in the source code.
    '''
    line_number: int

    column_number: int

    def to_json(self):
        json = dict()
        json['lineNumber'] = self.line_number
        json['columnNumber'] = self.column_number
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            line_number=int(json['lineNumber']),
            column_number=int(json['columnNumber']),
        )


@dataclass
class LocationRange:
    '''
    Location range within one script.
    '''
    script_id: runtime.ScriptId

    start: ScriptPosition

    end: ScriptPosition

    def to_json(self):
        json = dict()
        json['scriptId'] = self.script_id.to_json()
        json['start'] = self.start.to_json()
        json['end'] = self.end.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            script_id=runtime.ScriptId.from_json(json['scriptId']),
            start=ScriptPosition.from_json(json['start']),
            end=ScriptPosition.from_json(json['end']),
        )


@dataclass
class CallFrame:
    '''
    JavaScript call frame. Array of call frames form the call stack.
    '''
    #: Call frame identifier. This identifier is only valid while the virtual machine is paused.
    call_frame_id: CallFrameId

    #: Name of the JavaScript function called on this call frame.
    function_name: str

    #: Location in the source code.
    location: Location

    #: JavaScript script name or url.
    #: Deprecated in favor of using the ``location.scriptId`` to resolve the URL via a previously
    #: sent ``Debugger.scriptParsed`` event.
    url: str

    #: Scope chain for this call frame.
    scope_chain: typing.List[Scope]

    #: ``this`` object for this call frame.
    this: runtime.RemoteObject

    #: Location in the source code.
    function_location: typing.Optional[Location] = None

    #: The value being returned, if the function is at return point.
    return_value: typing.Optional[runtime.RemoteObject] = None

    #: Valid only while the VM is paused and indicates whether this frame
    #: can be restarted or not. Note that a ``true`` value here does not
    #: guarantee that Debugger#restartFrame with this CallFrameId will be
    #: successful, but it is very likely.
    can_be_restarted: typing.Optional[bool] = None

    def to_json(self):
        json = dict()
        json['callFrameId'] = self.call_frame_id.to_json()
        json['functionName'] = self.function_name
        json['location'] = self.location.to_json()
        json['url'] = self.url
        json['scopeChain'] = [i.to_json() for i in self.scope_chain]
        json['this'] = self.this.to_json()
        if self.function_location is not None:
            json['functionLocation'] = self.function_location.to_json()
        if self.return_value is not None:
            json['returnValue'] = self.return_value.to_json()
        if self.can_be_restarted is not None:
            json['canBeRestarted'] = self.can_be_restarted
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            call_frame_id=CallFrameId.from_json(json['callFrameId']),
            function_name=str(json['functionName']),
            location=Location.from_json(json['location']),
            url=str(json['url']),
            scope_chain=[Scope.from_json(i) for i in json['scopeChain']],
            this=runtime.RemoteObject.from_json(json['this']),
            function_location=Location.from_json(json['functionLocation']) if 'functionLocation' in json else None,
            return_value=runtime.RemoteObject.from_json(json['returnValue']) if 'returnValue' in json else None,
            can_be_restarted=bool(json['canBeRestarted']) if 'canBeRestarted' in json else None,
        )


@dataclass
class Scope:
    '''
    Scope description.
    '''
    #: Scope type.
    type_: str

    #: Object representing the scope. For ``global`` and ``with`` scopes it represents the actual
    #: object; for the rest of the scopes, it is artificial transient object enumerating scope
    #: variables as its properties.
    object_: runtime.RemoteObject

    name: typing.Optional[str] = None

    #: Location in the source code where scope starts
    start_location: typing.Optional[Location] = None

    #: Location in the source code where scope ends
    end_location: typing.Optional[Location] = None

    def to_json(self):
        json = dict()
        json['type'] = self.type_
        json['object'] = self.object_.to_json()
        if self.name is not None:
            json['name'] = self.name
        if self.start_location is not None:
            json['startLocation'] = self.start_location.to_json()
        if self.end_location is not None:
            json['endLocation'] = self.end_location.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            type_=str(json['type']),
            object_=runtime.RemoteObject.from_json(json['object']),
            name=str(json['name']) if 'name' in json else None,
            start_location=Location.from_json(json['startLocation']) if 'startLocation' in json else None,
            end_location=Location.from_json(json['endLocation']) if 'endLocation' in json else None,
        )


@dataclass
class SearchMatch:
    '''
    Search match for resource.
    '''
    #: Line number in resource content.
    line_number: float

    #: Line with match content.
    line_content: str

    def to_json(self):
        json = dict()
        json['lineNumber'] = self.line_number
        json['lineContent'] = self.line_content
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            line_number=float(json['lineNumber']),
            line_content=str(json['lineContent']),
        )


@dataclass
class BreakLocation:
    #: Script identifier as reported in the ``Debugger.scriptParsed``.
    script_id: runtime.ScriptId

    #: Line number in the script (0-based).
    line_number: int

    #: Column number in the script (0-based).
    column_number: typing.Optional[int] = None

    type_: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['scriptId'] = self.script_id.to_json()
        json['lineNumber'] = self.line_number
        if self.column_number is not None:
            json['columnNumber'] = self.column_number
        if self.type_ is not None:
            json['type'] = self.type_
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            script_id=runtime.ScriptId.from_json(json['scriptId']),
            line_number=int(json['lineNumber']),
            column_number=int(json['columnNumber']) if 'columnNumber' in json else None,
            type_=str(json['type']) if 'type' in json else None,
        )


@dataclass
class WasmDisassemblyChunk:
    #: The next chunk of disassembled lines.
    lines: typing.List[str]

    #: The bytecode offsets describing the start of each line.
    bytecode_offsets: typing.List[int]

    def to_json(self):
        json = dict()
        json['lines'] = [i for i in self.lines]
        json['bytecodeOffsets'] = [i for i in self.bytecode_offsets]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            lines=[str(i) for i in json['lines']],
            bytecode_offsets=[int(i) for i in json['bytecodeOffsets']],
        )


class ScriptLanguage(enum.Enum):
    '''
    Enum of possible script languages.
    '''
    JAVA_SCRIPT = "JavaScript"
    WEB_ASSEMBLY = "WebAssembly"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class DebugSymbols:
    '''
    Debug symbols available for a wasm script.
    '''
    #: Type of the debug symbols.
    type_: str

    #: URL of the external symbol source.
    external_url: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['type'] = self.type_
        if self.external_url is not None:
            json['externalURL'] = self.external_url
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            type_=str(json['type']),
            external_url=str(json['externalURL']) if 'externalURL' in json else None,
        )


@dataclass
class ResolvedBreakpoint:
    #: Breakpoint unique identifier.
    breakpoint_id: BreakpointId

    #: Actual breakpoint location.
    location: Location

    def to_json(self):
        json = dict()
        json['breakpointId'] = self.breakpoint_id.to_json()
        json['location'] = self.location.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            breakpoint_id=BreakpointId.from_json(json['breakpointId']),
            location=Location.from_json(json['location']),
        )


def continue_to_location(
        location: Location,
        target_call_frames: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Continues execution until specific location is reached.

    :param location: Location to continue to.
    :param target_call_frames: *(Optional)*
    '''
    params: T_JSON_DICT = dict()
    params['location'] = location.to_json()
    if target_call_frames is not None:
        params['targetCallFrames'] = target_call_frames
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.continueToLocation',
        'params': params,
    }
    json = yield cmd_dict


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables debugger for given page.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.disable',
    }
    json = yield cmd_dict


def enable(
        max_scripts_cache_size: typing.Optional[float] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,runtime.UniqueDebuggerId]:
    '''
    Enables debugger for the given page. Clients should not assume that the debugging has been
    enabled until the result for this command is received.

    :param max_scripts_cache_size: **(EXPERIMENTAL)** *(Optional)* The maximum size in bytes of collected scripts (not referenced by other heap objects) the debugger can hold. Puts no limit if parameter is omitted.
    :returns: Unique identifier of the debugger.
    '''
    params: T_JSON_DICT = dict()
    if max_scripts_cache_size is not None:
        params['maxScriptsCacheSize'] = max_scripts_cache_size
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.enable',
        'params': params,
    }
    json = yield cmd_dict
    return runtime.UniqueDebuggerId.from_json(json['debuggerId'])


def evaluate_on_call_frame(
        call_frame_id: CallFrameId,
        expression: str,
        object_group: typing.Optional[str] = None,
        include_command_line_api: typing.Optional[bool] = None,
        silent: typing.Optional[bool] = None,
        return_by_value: typing.Optional[bool] = None,
        generate_preview: typing.Optional[bool] = None,
        throw_on_side_effect: typing.Optional[bool] = None,
        timeout: typing.Optional[runtime.TimeDelta] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[runtime.RemoteObject, typing.Optional[runtime.ExceptionDetails]]]:
    '''
    Evaluates expression on a given call frame.

    :param call_frame_id: Call frame identifier to evaluate on.
    :param expression: Expression to evaluate.
    :param object_group: *(Optional)* String object group name to put result into (allows rapid releasing resulting object handles using ```releaseObjectGroup````).
    :param include_command_line_api: *(Optional)* Specifies whether command line API should be available to the evaluated expression, defaults to false.
    :param silent: *(Optional)* In silent mode exceptions thrown during evaluation are not reported and do not pause execution. Overrides ````setPauseOnException``` state.
    :param return_by_value: *(Optional)* Whether the result is expected to be a JSON object that should be sent by value.
    :param generate_preview: **(EXPERIMENTAL)** *(Optional)* Whether preview should be generated for the result.
    :param throw_on_side_effect: *(Optional)* Whether to throw an exception if side effect cannot be ruled out during evaluation.
    :param timeout: **(EXPERIMENTAL)** *(Optional)* Terminate execution after timing out (number of milliseconds).
    :returns: A tuple with the following items:

        0. **result** - Object wrapper for the evaluation result.
        1. **exceptionDetails** - *(Optional)* Exception details.
    '''
    params: T_JSON_DICT = dict()
    params['callFrameId'] = call_frame_id.to_json()
    params['expression'] = expression
    if object_group is not None:
        params['objectGroup'] = object_group
    if include_command_line_api is not None:
        params['includeCommandLineAPI'] = include_command_line_api
    if silent is not None:
        params['silent'] = silent
    if return_by_value is not None:
        params['returnByValue'] = return_by_value
    if generate_preview is not None:
        params['generatePreview'] = generate_preview
    if throw_on_side_effect is not None:
        params['throwOnSideEffect'] = throw_on_side_effect
    if timeout is not None:
        params['timeout'] = timeout.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.evaluateOnCallFrame',
        'params': params,
    }
    json = yield cmd_dict
    return (
        runtime.RemoteObject.from_json(json['result']),
        runtime.ExceptionDetails.from_json(json['exceptionDetails']) if 'exceptionDetails' in json else None
    )


def get_possible_breakpoints(
        start: Location,
        end: typing.Optional[Location] = None,
        restrict_to_function: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[BreakLocation]]:
    '''
    Returns possible locations for breakpoint. scriptId in start and end range locations should be
    the same.

    :param start: Start of range to search possible breakpoint locations in.
    :param end: *(Optional)* End of range to search possible breakpoint locations in (excluding). When not specified, end of scripts is used as end of range.
    :param restrict_to_function: *(Optional)* Only consider locations which are in the same (non-nested) function as start.
    :returns: List of the possible breakpoint locations.
    '''
    params: T_JSON_DICT = dict()
    params['start'] = start.to_json()
    if end is not None:
        params['end'] = end.to_json()
    if restrict_to_function is not None:
        params['restrictToFunction'] = restrict_to_function
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.getPossibleBreakpoints',
        'params': params,
    }
    json = yield cmd_dict
    return [BreakLocation.from_json(i) for i in json['locations']]


def get_script_source(
        script_id: runtime.ScriptId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[str, typing.Optional[str]]]:
    '''
    Returns source for the script with given id.

    :param script_id: Id of the script to get source for.
    :returns: A tuple with the following items:

        0. **scriptSource** - Script source (empty in case of Wasm bytecode).
        1. **bytecode** - *(Optional)* Wasm bytecode.
    '''
    params: T_JSON_DICT = dict()
    params['scriptId'] = script_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.getScriptSource',
        'params': params,
    }
    json = yield cmd_dict
    return (
        str(json['scriptSource']),
        str(json['bytecode']) if 'bytecode' in json else None
    )


def disassemble_wasm_module(
        script_id: runtime.ScriptId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[typing.Optional[str], int, typing.List[int], WasmDisassemblyChunk]]:
    '''


    **EXPERIMENTAL**

    :param script_id: Id of the script to disassemble
    :returns: A tuple with the following items:

        0. **streamId** - *(Optional)* For large modules, return a stream from which additional chunks of disassembly can be read successively.
        1. **totalNumberOfLines** - The total number of lines in the disassembly text.
        2. **functionBodyOffsets** - The offsets of all function bodies, in the format [start1, end1, start2, end2, ...] where all ends are exclusive.
        3. **chunk** - The first chunk of disassembly.
    '''
    params: T_JSON_DICT = dict()
    params['scriptId'] = script_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.disassembleWasmModule',
        'params': params,
    }
    json = yield cmd_dict
    return (
        str(json['streamId']) if 'streamId' in json else None,
        int(json['totalNumberOfLines']),
        [int(i) for i in json['functionBodyOffsets']],
        WasmDisassemblyChunk.from_json(json['chunk'])
    )


def next_wasm_disassembly_chunk(
        stream_id: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,WasmDisassemblyChunk]:
    '''
    Disassemble the next chunk of lines for the module corresponding to the
    stream. If disassembly is complete, this API will invalidate the streamId
    and return an empty chunk. Any subsequent calls for the now invalid stream
    will return errors.

    **EXPERIMENTAL**

    :param stream_id:
    :returns: The next chunk of disassembly.
    '''
    params: T_JSON_DICT = dict()
    params['streamId'] = stream_id
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.nextWasmDisassemblyChunk',
        'params': params,
    }
    json = yield cmd_dict
    return WasmDisassemblyChunk.from_json(json['chunk'])


def get_wasm_bytecode(
        script_id: runtime.ScriptId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,str]:
    '''
    This command is deprecated. Use getScriptSource instead.

    :param script_id: Id of the Wasm script to get source for.
    :returns: Script source.
    '''
    params: T_JSON_DICT = dict()
    params['scriptId'] = script_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.getWasmBytecode',
        'params': params,
    }
    json = yield cmd_dict
    return str(json['bytecode'])


def get_stack_trace(
        stack_trace_id: runtime.StackTraceId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,runtime.StackTrace]:
    '''
    Returns stack trace with given ``stackTraceId``.

    **EXPERIMENTAL**

    :param stack_trace_id:
    :returns: 
    '''
    params: T_JSON_DICT = dict()
    params['stackTraceId'] = stack_trace_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.getStackTrace',
        'params': params,
    }
    json = yield cmd_dict
    return runtime.StackTrace.from_json(json['stackTrace'])


def pause() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Stops on the next JavaScript statement.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.pause',
    }
    json = yield cmd_dict


def pause_on_async_call(
        parent_stack_trace_id: runtime.StackTraceId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''


    **EXPERIMENTAL**

    :param parent_stack_trace_id: Debugger will pause when async call with given stack trace is started.
    '''
    params: T_JSON_DICT = dict()
    params['parentStackTraceId'] = parent_stack_trace_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.pauseOnAsyncCall',
        'params': params,
    }
    json = yield cmd_dict


def remove_breakpoint(
        breakpoint_id: BreakpointId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Removes JavaScript breakpoint.

    :param breakpoint_id:
    '''
    params: T_JSON_DICT = dict()
    params['breakpointId'] = breakpoint_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.removeBreakpoint',
        'params': params,
    }
    json = yield cmd_dict


def restart_frame(
        call_frame_id: CallFrameId,
        mode: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[typing.List[CallFrame], typing.Optional[runtime.StackTrace], typing.Optional[runtime.StackTraceId]]]:
    '''
    Restarts particular call frame from the beginning. The old, deprecated
    behavior of ``restartFrame`` is to stay paused and allow further CDP commands
    after a restart was scheduled. This can cause problems with restarting, so
    we now continue execution immediatly after it has been scheduled until we
    reach the beginning of the restarted frame.

    To stay back-wards compatible, ``restartFrame`` now expects a ``mode``
    parameter to be present. If the ``mode`` parameter is missing, ``restartFrame``
    errors out.

    The various return values are deprecated and ``callFrames`` is always empty.
    Use the call frames from the ``Debugger#paused`` events instead, that fires
    once V8 pauses at the beginning of the restarted function.

    :param call_frame_id: Call frame identifier to evaluate on.
    :param mode: **(EXPERIMENTAL)** *(Optional)* The ```mode```` parameter must be present and set to 'StepInto', otherwise ````restartFrame``` will error out.
    :returns: A tuple with the following items:

        0. **callFrames** - New stack trace.
        1. **asyncStackTrace** - *(Optional)* Async stack trace, if any.
        2. **asyncStackTraceId** - *(Optional)* Async stack trace, if any.
    '''
    params: T_JSON_DICT = dict()
    params['callFrameId'] = call_frame_id.to_json()
    if mode is not None:
        params['mode'] = mode
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.restartFrame',
        'params': params,
    }
    json = yield cmd_dict
    return (
        [CallFrame.from_json(i) for i in json['callFrames']],
        runtime.StackTrace.from_json(json['asyncStackTrace']) if 'asyncStackTrace' in json else None,
        runtime.StackTraceId.from_json(json['asyncStackTraceId']) if 'asyncStackTraceId' in json else None
    )


def resume(
        terminate_on_resume: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Resumes JavaScript execution.

    :param terminate_on_resume: *(Optional)* Set to true to terminate execution upon resuming execution. In contrast to Runtime.terminateExecution, this will allows to execute further JavaScript (i.e. via evaluation) until execution of the paused code is actually resumed, at which point termination is triggered. If execution is currently not paused, this parameter has no effect.
    '''
    params: T_JSON_DICT = dict()
    if terminate_on_resume is not None:
        params['terminateOnResume'] = terminate_on_resume
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.resume',
        'params': params,
    }
    json = yield cmd_dict


def search_in_content(
        script_id: runtime.ScriptId,
        query: str,
        case_sensitive: typing.Optional[bool] = None,
        is_regex: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[SearchMatch]]:
    '''
    Searches for given string in script content.

    :param script_id: Id of the script to search in.
    :param query: String to search for.
    :param case_sensitive: *(Optional)* If true, search is case sensitive.
    :param is_regex: *(Optional)* If true, treats string parameter as regex.
    :returns: List of search matches.
    '''
    params: T_JSON_DICT = dict()
    params['scriptId'] = script_id.to_json()
    params['query'] = query
    if case_sensitive is not None:
        params['caseSensitive'] = case_sensitive
    if is_regex is not None:
        params['isRegex'] = is_regex
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.searchInContent',
        'params': params,
    }
    json = yield cmd_dict
    return [SearchMatch.from_json(i) for i in json['result']]


def set_async_call_stack_depth(
        max_depth: int
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables or disables async call stacks tracking.

    :param max_depth: Maximum depth of async call stacks. Setting to ```0``` will effectively disable collecting async call stacks (default).
    '''
    params: T_JSON_DICT = dict()
    params['maxDepth'] = max_depth
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.setAsyncCallStackDepth',
        'params': params,
    }
    json = yield cmd_dict


def set_blackbox_execution_contexts(
        unique_ids: typing.List[str]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Replace previous blackbox execution contexts with passed ones. Forces backend to skip
    stepping/pausing in scripts in these execution contexts. VM will try to leave blackboxed script by
    performing 'step in' several times, finally resorting to 'step out' if unsuccessful.

    **EXPERIMENTAL**

    :param unique_ids: Array of execution context unique ids for the debugger to ignore.
    '''
    params: T_JSON_DICT = dict()
    params['uniqueIds'] = [i for i in unique_ids]
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.setBlackboxExecutionContexts',
        'params': params,
    }
    json = yield cmd_dict


def set_blackbox_patterns(
        patterns: typing.List[str],
        skip_anonymous: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Replace previous blackbox patterns with passed ones. Forces backend to skip stepping/pausing in
    scripts with url matching one of the patterns. VM will try to leave blackboxed script by
    performing 'step in' several times, finally resorting to 'step out' if unsuccessful.

    **EXPERIMENTAL**

    :param patterns: Array of regexps that will be used to check script url for blackbox state.
    :param skip_anonymous: *(Optional)* If true, also ignore scripts with no source url.
    '''
    params: T_JSON_DICT = dict()
    params['patterns'] = [i for i in patterns]
    if skip_anonymous is not None:
        params['skipAnonymous'] = skip_anonymous
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.setBlackboxPatterns',
        'params': params,
    }
    json = yield cmd_dict


def set_blackboxed_ranges(
        script_id: runtime.ScriptId,
        positions: typing.List[ScriptPosition]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Makes backend skip steps in the script in blackboxed ranges. VM will try leave blacklisted
    scripts by performing 'step in' several times, finally resorting to 'step out' if unsuccessful.
    Positions array contains positions where blackbox state is changed. First interval isn't
    blackboxed. Array should be sorted.

    **EXPERIMENTAL**

    :param script_id: Id of the script.
    :param positions:
    '''
    params: T_JSON_DICT = dict()
    params['scriptId'] = script_id.to_json()
    params['positions'] = [i.to_json() for i in positions]
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.setBlackboxedRanges',
        'params': params,
    }
    json = yield cmd_dict


def set_breakpoint(
        location: Location,
        condition: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[BreakpointId, Location]]:
    '''
    Sets JavaScript breakpoint at a given location.

    :param location: Location to set breakpoint in.
    :param condition: *(Optional)* Expression to use as a breakpoint condition. When specified, debugger will only stop on the breakpoint if this expression evaluates to true.
    :returns: A tuple with the following items:

        0. **breakpointId** - Id of the created breakpoint for further reference.
        1. **actualLocation** - Location this breakpoint resolved into.
    '''
    params: T_JSON_DICT = dict()
    params['location'] = location.to_json()
    if condition is not None:
        params['condition'] = condition
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.setBreakpoint',
        'params': params,
    }
    json = yield cmd_dict
    return (
        BreakpointId.from_json(json['breakpointId']),
        Location.from_json(json['actualLocation'])
    )


def set_instrumentation_breakpoint(
        instrumentation: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,BreakpointId]:
    '''
    Sets instrumentation breakpoint.

    :param instrumentation: Instrumentation name.
    :returns: Id of the created breakpoint for further reference.
    '''
    params: T_JSON_DICT = dict()
    params['instrumentation'] = instrumentation
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.setInstrumentationBreakpoint',
        'params': params,
    }
    json = yield cmd_dict
    return BreakpointId.from_json(json['breakpointId'])


def set_breakpoint_by_url(
        line_number: int,
        url: typing.Optional[str] = None,
        url_regex: typing.Optional[str] = None,
        script_hash: typing.Optional[str] = None,
        column_number: typing.Optional[int] = None,
        condition: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[BreakpointId, typing.List[Location]]]:
    '''
    Sets JavaScript breakpoint at given location specified either by URL or URL regex. Once this
    command is issued, all existing parsed scripts will have breakpoints resolved and returned in
    ``locations`` property. Further matching script parsing will result in subsequent
    ``breakpointResolved`` events issued. This logical breakpoint will survive page reloads.

    :param line_number: Line number to set breakpoint at.
    :param url: *(Optional)* URL of the resources to set breakpoint on.
    :param url_regex: *(Optional)* Regex pattern for the URLs of the resources to set breakpoints on. Either ```url```` or ````urlRegex``` must be specified.
    :param script_hash: *(Optional)* Script hash of the resources to set breakpoint on.
    :param column_number: *(Optional)* Offset in the line to set breakpoint at.
    :param condition: *(Optional)* Expression to use as a breakpoint condition. When specified, debugger will only stop on the breakpoint if this expression evaluates to true.
    :returns: A tuple with the following items:

        0. **breakpointId** - Id of the created breakpoint for further reference.
        1. **locations** - List of the locations this breakpoint resolved into upon addition.
    '''
    params: T_JSON_DICT = dict()
    params['lineNumber'] = line_number
    if url is not None:
        params['url'] = url
    if url_regex is not None:
        params['urlRegex'] = url_regex
    if script_hash is not None:
        params['scriptHash'] = script_hash
    if column_number is not None:
        params['columnNumber'] = column_number
    if condition is not None:
        params['condition'] = condition
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.setBreakpointByUrl',
        'params': params,
    }
    json = yield cmd_dict
    return (
        BreakpointId.from_json(json['breakpointId']),
        [Location.from_json(i) for i in json['locations']]
    )


def set_breakpoint_on_function_call(
        object_id: runtime.RemoteObjectId,
        condition: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,BreakpointId]:
    '''
    Sets JavaScript breakpoint before each call to the given function.
    If another function was created from the same source as a given one,
    calling it will also trigger the breakpoint.

    **EXPERIMENTAL**

    :param object_id: Function object id.
    :param condition: *(Optional)* Expression to use as a breakpoint condition. When specified, debugger will stop on the breakpoint if this expression evaluates to true.
    :returns: Id of the created breakpoint for further reference.
    '''
    params: T_JSON_DICT = dict()
    params['objectId'] = object_id.to_json()
    if condition is not None:
        params['condition'] = condition
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.setBreakpointOnFunctionCall',
        'params': params,
    }
    json = yield cmd_dict
    return BreakpointId.from_json(json['breakpointId'])


def set_breakpoints_active(
        active: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Activates / deactivates all breakpoints on the page.

    :param active: New value for breakpoints active state.
    '''
    params: T_JSON_DICT = dict()
    params['active'] = active
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.setBreakpointsActive',
        'params': params,
    }
    json = yield cmd_dict


def set_pause_on_exceptions(
        state: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Defines pause on exceptions state. Can be set to stop on all exceptions, uncaught exceptions,
    or caught exceptions, no exceptions. Initial pause on exceptions state is ``none``.

    :param state: Pause on exceptions mode.
    '''
    params: T_JSON_DICT = dict()
    params['state'] = state
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.setPauseOnExceptions',
        'params': params,
    }
    json = yield cmd_dict


def set_return_value(
        new_value: runtime.CallArgument
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Changes return value in top frame. Available only at return break position.

    **EXPERIMENTAL**

    :param new_value: New return value.
    '''
    params: T_JSON_DICT = dict()
    params['newValue'] = new_value.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.setReturnValue',
        'params': params,
    }
    json = yield cmd_dict


def set_script_source(
        script_id: runtime.ScriptId,
        script_source: str,
        dry_run: typing.Optional[bool] = None,
        allow_top_frame_editing: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[typing.Optional[typing.List[CallFrame]], typing.Optional[bool], typing.Optional[runtime.StackTrace], typing.Optional[runtime.StackTraceId], str, typing.Optional[runtime.ExceptionDetails]]]:
    '''
    Edits JavaScript source live.

    In general, functions that are currently on the stack can not be edited with
    a single exception: If the edited function is the top-most stack frame and
    that is the only activation of that function on the stack. In this case
    the live edit will be successful and a ``Debugger.restartFrame`` for the
    top-most function is automatically triggered.

    :param script_id: Id of the script to edit.
    :param script_source: New content of the script.
    :param dry_run: *(Optional)* If true the change will not actually be applied. Dry run may be used to get result description without actually modifying the code.
    :param allow_top_frame_editing: **(EXPERIMENTAL)** *(Optional)* If true, then ```scriptSource```` is allowed to change the function on top of the stack as long as the top-most stack frame is the only activation of that function.
    :returns: A tuple with the following items:

        0. **callFrames** - *(Optional)* New stack trace in case editing has happened while VM was stopped.
        1. **stackChanged** - *(Optional)* Whether current call stack  was modified after applying the changes.
        2. **asyncStackTrace** - *(Optional)* Async stack trace, if any.
        3. **asyncStackTraceId** - *(Optional)* Async stack trace, if any.
        4. **status** - Whether the operation was successful or not. Only `` Ok`` denotes a successful live edit while the other enum variants denote why the live edit failed.
        5. **exceptionDetails** - *(Optional)* Exception details if any. Only present when `` status`` is `` CompileError`.
    '''
    params: T_JSON_DICT = dict()
    params['scriptId'] = script_id.to_json()
    params['scriptSource'] = script_source
    if dry_run is not None:
        params['dryRun'] = dry_run
    if allow_top_frame_editing is not None:
        params['allowTopFrameEditing'] = allow_top_frame_editing
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.setScriptSource',
        'params': params,
    }
    json = yield cmd_dict
    return (
        [CallFrame.from_json(i) for i in json['callFrames']] if 'callFrames' in json else None,
        bool(json['stackChanged']) if 'stackChanged' in json else None,
        runtime.StackTrace.from_json(json['asyncStackTrace']) if 'asyncStackTrace' in json else None,
        runtime.StackTraceId.from_json(json['asyncStackTraceId']) if 'asyncStackTraceId' in json else None,
        str(json['status']),
        runtime.ExceptionDetails.from_json(json['exceptionDetails']) if 'exceptionDetails' in json else None
    )


def set_skip_all_pauses(
        skip: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Makes page not interrupt on any pauses (breakpoint, exception, dom exception etc).

    :param skip: New value for skip pauses state.
    '''
    params: T_JSON_DICT = dict()
    params['skip'] = skip
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.setSkipAllPauses',
        'params': params,
    }
    json = yield cmd_dict


def set_variable_value(
        scope_number: int,
        variable_name: str,
        new_value: runtime.CallArgument,
        call_frame_id: CallFrameId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Changes value of variable in a callframe. Object-based scopes are not supported and must be
    mutated manually.

    :param scope_number: 0-based number of scope as was listed in scope chain. Only 'local', 'closure' and 'catch' scope types are allowed. Other scopes could be manipulated manually.
    :param variable_name: Variable name.
    :param new_value: New variable value.
    :param call_frame_id: Id of callframe that holds variable.
    '''
    params: T_JSON_DICT = dict()
    params['scopeNumber'] = scope_number
    params['variableName'] = variable_name
    params['newValue'] = new_value.to_json()
    params['callFrameId'] = call_frame_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.setVariableValue',
        'params': params,
    }
    json = yield cmd_dict


def step_into(
        break_on_async_call: typing.Optional[bool] = None,
        skip_list: typing.Optional[typing.List[LocationRange]] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Steps into the function call.

    :param break_on_async_call: **(EXPERIMENTAL)** *(Optional)* Debugger will pause on the execution of the first async task which was scheduled before next pause.
    :param skip_list: **(EXPERIMENTAL)** *(Optional)* The skipList specifies location ranges that should be skipped on step into.
    '''
    params: T_JSON_DICT = dict()
    if break_on_async_call is not None:
        params['breakOnAsyncCall'] = break_on_async_call
    if skip_list is not None:
        params['skipList'] = [i.to_json() for i in skip_list]
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.stepInto',
        'params': params,
    }
    json = yield cmd_dict


def step_out() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Steps out of the function call.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.stepOut',
    }
    json = yield cmd_dict


def step_over(
        skip_list: typing.Optional[typing.List[LocationRange]] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Steps over the statement.

    :param skip_list: **(EXPERIMENTAL)** *(Optional)* The skipList specifies location ranges that should be skipped on step over.
    '''
    params: T_JSON_DICT = dict()
    if skip_list is not None:
        params['skipList'] = [i.to_json() for i in skip_list]
    cmd_dict: T_JSON_DICT = {
        'method': 'Debugger.stepOver',
        'params': params,
    }
    json = yield cmd_dict


@event_class('Debugger.breakpointResolved')
@dataclass
class BreakpointResolved:
    '''
    Fired when breakpoint is resolved to an actual script and location.
    Deprecated in favor of ``resolvedBreakpoints`` in the ``scriptParsed`` event.
    '''
    #: Breakpoint unique identifier.
    breakpoint_id: BreakpointId
    #: Actual breakpoint location.
    location: Location

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> BreakpointResolved:
        return cls(
            breakpoint_id=BreakpointId.from_json(json['breakpointId']),
            location=Location.from_json(json['location'])
        )


@event_class('Debugger.paused')
@dataclass
class Paused:
    '''
    Fired when the virtual machine stopped on breakpoint or exception or any other stop criteria.
    '''
    #: Call stack the virtual machine stopped on.
    call_frames: typing.List[CallFrame]
    #: Pause reason.
    reason: str
    #: Object containing break-specific auxiliary properties.
    data: typing.Optional[dict]
    #: Hit breakpoints IDs
    hit_breakpoints: typing.Optional[typing.List[str]]
    #: Async stack trace, if any.
    async_stack_trace: typing.Optional[runtime.StackTrace]
    #: Async stack trace, if any.
    async_stack_trace_id: typing.Optional[runtime.StackTraceId]
    #: Never present, will be removed.
    async_call_stack_trace_id: typing.Optional[runtime.StackTraceId]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> Paused:
        return cls(
            call_frames=[CallFrame.from_json(i) for i in json['callFrames']],
            reason=str(json['reason']),
            data=dict(json['data']) if 'data' in json else None,
            hit_breakpoints=[str(i) for i in json['hitBreakpoints']] if 'hitBreakpoints' in json else None,
            async_stack_trace=runtime.StackTrace.from_json(json['asyncStackTrace']) if 'asyncStackTrace' in json else None,
            async_stack_trace_id=runtime.StackTraceId.from_json(json['asyncStackTraceId']) if 'asyncStackTraceId' in json else None,
            async_call_stack_trace_id=runtime.StackTraceId.from_json(json['asyncCallStackTraceId']) if 'asyncCallStackTraceId' in json else None
        )


@event_class('Debugger.resumed')
@dataclass
class Resumed:
    '''
    Fired when the virtual machine resumed execution.
    '''


    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> Resumed:
        return cls(

        )


@event_class('Debugger.scriptFailedToParse')
@dataclass
class ScriptFailedToParse:
    '''
    Fired when virtual machine fails to parse the script.
    '''
    #: Identifier of the script parsed.
    script_id: runtime.ScriptId
    #: URL or name of the script parsed (if any).
    url: str
    #: Line offset of the script within the resource with given URL (for script tags).
    start_line: int
    #: Column offset of the script within the resource with given URL.
    start_column: int
    #: Last line of the script.
    end_line: int
    #: Length of the last line of the script.
    end_column: int
    #: Specifies script creation context.
    execution_context_id: runtime.ExecutionContextId
    #: Content hash of the script, SHA-256.
    hash_: str
    #: For Wasm modules, the content of the ``build_id`` custom section. For JavaScript the ``debugId`` magic comment.
    build_id: str
    #: Embedder-specific auxiliary data likely matching {isDefault: boolean, type: 'default'``'isolated'``'worker', frameId: string}
    execution_context_aux_data: typing.Optional[dict]
    #: URL of source map associated with script (if any).
    source_map_url: typing.Optional[str]
    #: True, if this script has sourceURL.
    has_source_url: typing.Optional[bool]
    #: True, if this script is ES6 module.
    is_module: typing.Optional[bool]
    #: This script length.
    length: typing.Optional[int]
    #: JavaScript top stack frame of where the script parsed event was triggered if available.
    stack_trace: typing.Optional[runtime.StackTrace]
    #: If the scriptLanguage is WebAssembly, the code section offset in the module.
    code_offset: typing.Optional[int]
    #: The language of the script.
    script_language: typing.Optional[debugger.ScriptLanguage]
    #: The name the embedder supplied for this script.
    embedder_name: typing.Optional[str]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ScriptFailedToParse:
        return cls(
            script_id=runtime.ScriptId.from_json(json['scriptId']),
            url=str(json['url']),
            start_line=int(json['startLine']),
            start_column=int(json['startColumn']),
            end_line=int(json['endLine']),
            end_column=int(json['endColumn']),
            execution_context_id=runtime.ExecutionContextId.from_json(json['executionContextId']),
            hash_=str(json['hash']),
            build_id=str(json['buildId']),
            execution_context_aux_data=dict(json['executionContextAuxData']) if 'executionContextAuxData' in json else None,
            source_map_url=str(json['sourceMapURL']) if 'sourceMapURL' in json else None,
            has_source_url=bool(json['hasSourceURL']) if 'hasSourceURL' in json else None,
            is_module=bool(json['isModule']) if 'isModule' in json else None,
            length=int(json['length']) if 'length' in json else None,
            stack_trace=runtime.StackTrace.from_json(json['stackTrace']) if 'stackTrace' in json else None,
            code_offset=int(json['codeOffset']) if 'codeOffset' in json else None,
            script_language=debugger.ScriptLanguage.from_json(json['scriptLanguage']) if 'scriptLanguage' in json else None,
            embedder_name=str(json['embedderName']) if 'embedderName' in json else None
        )


@event_class('Debugger.scriptParsed')
@dataclass
class ScriptParsed:
    '''
    Fired when virtual machine parses script. This event is also fired for all known and uncollected
    scripts upon enabling debugger.
    '''
    #: Identifier of the script parsed.
    script_id: runtime.ScriptId
    #: URL or name of the script parsed (if any).
    url: str
    #: Line offset of the script within the resource with given URL (for script tags).
    start_line: int
    #: Column offset of the script within the resource with given URL.
    start_column: int
    #: Last line of the script.
    end_line: int
    #: Length of the last line of the script.
    end_column: int
    #: Specifies script creation context.
    execution_context_id: runtime.ExecutionContextId
    #: Content hash of the script, SHA-256.
    hash_: str
    #: For Wasm modules, the content of the ``build_id`` custom section. For JavaScript the ``debugId`` magic comment.
    build_id: str
    #: Embedder-specific auxiliary data likely matching {isDefault: boolean, type: 'default'``'isolated'``'worker', frameId: string}
    execution_context_aux_data: typing.Optional[dict]
    #: True, if this script is generated as a result of the live edit operation.
    is_live_edit: typing.Optional[bool]
    #: URL of source map associated with script (if any).
    source_map_url: typing.Optional[str]
    #: True, if this script has sourceURL.
    has_source_url: typing.Optional[bool]
    #: True, if this script is ES6 module.
    is_module: typing.Optional[bool]
    #: This script length.
    length: typing.Optional[int]
    #: JavaScript top stack frame of where the script parsed event was triggered if available.
    stack_trace: typing.Optional[runtime.StackTrace]
    #: If the scriptLanguage is WebAssembly, the code section offset in the module.
    code_offset: typing.Optional[int]
    #: The language of the script.
    script_language: typing.Optional[debugger.ScriptLanguage]
    #: If the scriptLanguage is WebAssembly, the source of debug symbols for the module.
    debug_symbols: typing.Optional[typing.List[debugger.DebugSymbols]]
    #: The name the embedder supplied for this script.
    embedder_name: typing.Optional[str]
    #: The list of set breakpoints in this script if calls to ``setBreakpointByUrl``
    #: matches this script's URL or hash. Clients that use this list can ignore the
    #: ``breakpointResolved`` event. They are equivalent.
    resolved_breakpoints: typing.Optional[typing.List[ResolvedBreakpoint]]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ScriptParsed:
        return cls(
            script_id=runtime.ScriptId.from_json(json['scriptId']),
            url=str(json['url']),
            start_line=int(json['startLine']),
            start_column=int(json['startColumn']),
            end_line=int(json['endLine']),
            end_column=int(json['endColumn']),
            execution_context_id=runtime.ExecutionContextId.from_json(json['executionContextId']),
            hash_=str(json['hash']),
            build_id=str(json['buildId']),
            execution_context_aux_data=dict(json['executionContextAuxData']) if 'executionContextAuxData' in json else None,
            is_live_edit=bool(json['isLiveEdit']) if 'isLiveEdit' in json else None,
            source_map_url=str(json['sourceMapURL']) if 'sourceMapURL' in json else None,
            has_source_url=bool(json['hasSourceURL']) if 'hasSourceURL' in json else None,
            is_module=bool(json['isModule']) if 'isModule' in json else None,
            length=int(json['length']) if 'length' in json else None,
            stack_trace=runtime.StackTrace.from_json(json['stackTrace']) if 'stackTrace' in json else None,
            code_offset=int(json['codeOffset']) if 'codeOffset' in json else None,
            script_language=debugger.ScriptLanguage.from_json(json['scriptLanguage']) if 'scriptLanguage' in json else None,
            debug_symbols=[debugger.DebugSymbols.from_json(i) for i in json['debugSymbols']] if 'debugSymbols' in json else None,
            embedder_name=str(json['embedderName']) if 'embedderName' in json else None,
            resolved_breakpoints=[ResolvedBreakpoint.from_json(i) for i in json['resolvedBreakpoints']] if 'resolvedBreakpoints' in json else None
        )