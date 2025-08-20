
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\urllib3\contrib\_securetransport\low_level.py ===
"""
Low-level helpers for the SecureTransport bindings.

These are Python functions that are not directly related to the high-level APIs
but are necessary to get them to work. They include a whole bunch of low-level
CoreFoundation messing about and memory management. The concerns in this module
are almost entirely about trying to avoid memory leaks and providing
appropriate and useful assistance to the higher-level code.
"""
import base64
import ctypes
import itertools
import os
import re
import ssl
import struct
import tempfile

from .bindings import CFConst, CoreFoundation, Security

# This regular expression is used to grab PEM data out of a PEM bundle.
_PEM_CERTS_RE = re.compile(
    b"-----BEGIN CERTIFICATE-----\n(.*?)\n-----END CERTIFICATE-----", re.DOTALL
)


def _cf_data_from_bytes(bytestring):
    """
    Given a bytestring, create a CFData object from it. This CFData object must
    be CFReleased by the caller.
    """
    return CoreFoundation.CFDataCreate(
        CoreFoundation.kCFAllocatorDefault, bytestring, len(bytestring)
    )


def _cf_dictionary_from_tuples(tuples):
    """
    Given a list of Python tuples, create an associated CFDictionary.
    """
    dictionary_size = len(tuples)

    # We need to get the dictionary keys and values out in the same order.
    keys = (t[0] for t in tuples)
    values = (t[1] for t in tuples)
    cf_keys = (CoreFoundation.CFTypeRef * dictionary_size)(*keys)
    cf_values = (CoreFoundation.CFTypeRef * dictionary_size)(*values)

    return CoreFoundation.CFDictionaryCreate(
        CoreFoundation.kCFAllocatorDefault,
        cf_keys,
        cf_values,
        dictionary_size,
        CoreFoundation.kCFTypeDictionaryKeyCallBacks,
        CoreFoundation.kCFTypeDictionaryValueCallBacks,
    )


def _cfstr(py_bstr):
    """
    Given a Python binary data, create a CFString.
    The string must be CFReleased by the caller.
    """
    c_str = ctypes.c_char_p(py_bstr)
    cf_str = CoreFoundation.CFStringCreateWithCString(
        CoreFoundation.kCFAllocatorDefault,
        c_str,
        CFConst.kCFStringEncodingUTF8,
    )
    return cf_str


def _create_cfstring_array(lst):
    """
    Given a list of Python binary data, create an associated CFMutableArray.
    The array must be CFReleased by the caller.

    Raises an ssl.SSLError on failure.
    """
    cf_arr = None
    try:
        cf_arr = CoreFoundation.CFArrayCreateMutable(
            CoreFoundation.kCFAllocatorDefault,
            0,
            ctypes.byref(CoreFoundation.kCFTypeArrayCallBacks),
        )
        if not cf_arr:
            raise MemoryError("Unable to allocate memory!")
        for item in lst:
            cf_str = _cfstr(item)
            if not cf_str:
                raise MemoryError("Unable to allocate memory!")
            try:
                CoreFoundation.CFArrayAppendValue(cf_arr, cf_str)
            finally:
                CoreFoundation.CFRelease(cf_str)
    except BaseException as e:
        if cf_arr:
            CoreFoundation.CFRelease(cf_arr)
        raise ssl.SSLError("Unable to allocate array: %s" % (e,))
    return cf_arr


def _cf_string_to_unicode(value):
    """
    Creates a Unicode string from a CFString object. Used entirely for error
    reporting.

    Yes, it annoys me quite a lot that this function is this complex.
    """
    value_as_void_p = ctypes.cast(value, ctypes.POINTER(ctypes.c_void_p))

    string = CoreFoundation.CFStringGetCStringPtr(
        value_as_void_p, CFConst.kCFStringEncodingUTF8
    )
    if string is None:
        buffer = ctypes.create_string_buffer(1024)
        result = CoreFoundation.CFStringGetCString(
            value_as_void_p, buffer, 1024, CFConst.kCFStringEncodingUTF8
        )
        if not result:
            raise OSError("Error copying C string from CFStringRef")
        string = buffer.value
    if string is not None:
        string = string.decode("utf-8")
    return string


def _assert_no_error(error, exception_class=None):
    """
    Checks the return code and throws an exception if there is an error to
    report
    """
    if error == 0:
        return

    cf_error_string = Security.SecCopyErrorMessageString(error, None)
    output = _cf_string_to_unicode(cf_error_string)
    CoreFoundation.CFRelease(cf_error_string)

    if output is None or output == u"":
        output = u"OSStatus %s" % error

    if exception_class is None:
        exception_class = ssl.SSLError

    raise exception_class(output)


def _cert_array_from_pem(pem_bundle):
    """
    Given a bundle of certs in PEM format, turns them into a CFArray of certs
    that can be used to validate a cert chain.
    """
    # Normalize the PEM bundle's line endings.
    pem_bundle = pem_bundle.replace(b"\r\n", b"\n")

    der_certs = [
        base64.b64decode(match.group(1)) for match in _PEM_CERTS_RE.finditer(pem_bundle)
    ]
    if not der_certs:
        raise ssl.SSLError("No root certificates specified")

    cert_array = CoreFoundation.CFArrayCreateMutable(
        CoreFoundation.kCFAllocatorDefault,
        0,
        ctypes.byref(CoreFoundation.kCFTypeArrayCallBacks),
    )
    if not cert_array:
        raise ssl.SSLError("Unable to allocate memory!")

    try:
        for der_bytes in der_certs:
            certdata = _cf_data_from_bytes(der_bytes)
            if not certdata:
                raise ssl.SSLError("Unable to allocate memory!")
            cert = Security.SecCertificateCreateWithData(
                CoreFoundation.kCFAllocatorDefault, certdata
            )
            CoreFoundation.CFRelease(certdata)
            if not cert:
                raise ssl.SSLError("Unable to build cert object!")

            CoreFoundation.CFArrayAppendValue(cert_array, cert)
            CoreFoundation.CFRelease(cert)
    except Exception:
        # We need to free the array before the exception bubbles further.
        # We only want to do that if an error occurs: otherwise, the caller
        # should free.
        CoreFoundation.CFRelease(cert_array)
        raise

    return cert_array


def _is_cert(item):
    """
    Returns True if a given CFTypeRef is a certificate.
    """
    expected = Security.SecCertificateGetTypeID()
    return CoreFoundation.CFGetTypeID(item) == expected


def _is_identity(item):
    """
    Returns True if a given CFTypeRef is an identity.
    """
    expected = Security.SecIdentityGetTypeID()
    return CoreFoundation.CFGetTypeID(item) == expected


def _temporary_keychain():
    """
    This function creates a temporary Mac keychain that we can use to work with
    credentials. This keychain uses a one-time password and a temporary file to
    store the data. We expect to have one keychain per socket. The returned
    SecKeychainRef must be freed by the caller, including calling
    SecKeychainDelete.

    Returns a tuple of the SecKeychainRef and the path to the temporary
    directory that contains it.
    """
    # Unfortunately, SecKeychainCreate requires a path to a keychain. This
    # means we cannot use mkstemp to use a generic temporary file. Instead,
    # we're going to create a temporary directory and a filename to use there.
    # This filename will be 8 random bytes expanded into base64. We also need
    # some random bytes to password-protect the keychain we're creating, so we
    # ask for 40 random bytes.
    random_bytes = os.urandom(40)
    filename = base64.b16encode(random_bytes[:8]).decode("utf-8")
    password = base64.b16encode(random_bytes[8:])  # Must be valid UTF-8
    tempdirectory = tempfile.mkdtemp()

    keychain_path = os.path.join(tempdirectory, filename).encode("utf-8")

    # We now want to create the keychain itself.
    keychain = Security.SecKeychainRef()
    status = Security.SecKeychainCreate(
        keychain_path, len(password), password, False, None, ctypes.byref(keychain)
    )
    _assert_no_error(status)

    # Having created the keychain, we want to pass it off to the caller.
    return keychain, tempdirectory


def _load_items_from_file(keychain, path):
    """
    Given a single file, loads all the trust objects from it into arrays and
    the keychain.
    Returns a tuple of lists: the first list is a list of identities, the
    second a list of certs.
    """
    certificates = []
    identities = []
    result_array = None

    with open(path, "rb") as f:
        raw_filedata = f.read()

    try:
        filedata = CoreFoundation.CFDataCreate(
            CoreFoundation.kCFAllocatorDefault, raw_filedata, len(raw_filedata)
        )
        result_array = CoreFoundation.CFArrayRef()
        result = Security.SecItemImport(
            filedata,  # cert data
            None,  # Filename, leaving it out for now
            None,  # What the type of the file is, we don't care
            None,  # what's in the file, we don't care
            0,  # import flags
            None,  # key params, can include passphrase in the future
            keychain,  # The keychain to insert into
            ctypes.byref(result_array),  # Results
        )
        _assert_no_error(result)

        # A CFArray is not very useful to us as an intermediary
        # representation, so we are going to extract the objects we want
        # and then free the array. We don't need to keep hold of keys: the
        # keychain already has them!
        result_count = CoreFoundation.CFArrayGetCount(result_array)
        for index in range(result_count):
            item = CoreFoundation.CFArrayGetValueAtIndex(result_array, index)
            item = ctypes.cast(item, CoreFoundation.CFTypeRef)

            if _is_cert(item):
                CoreFoundation.CFRetain(item)
                certificates.append(item)
            elif _is_identity(item):
                CoreFoundation.CFRetain(item)
                identities.append(item)
    finally:
        if result_array:
            CoreFoundation.CFRelease(result_array)

        CoreFoundation.CFRelease(filedata)

    return (identities, certificates)


def _load_client_cert_chain(keychain, *paths):
    """
    Load certificates and maybe keys from a number of files. Has the end goal
    of returning a CFArray containing one SecIdentityRef, and then zero or more
    SecCertificateRef objects, suitable for use as a client certificate trust
    chain.
    """
    # Ok, the strategy.
    #
    # This relies on knowing that macOS will not give you a SecIdentityRef
    # unless you have imported a key into a keychain. This is a somewhat
    # artificial limitation of macOS (for example, it doesn't necessarily
    # affect iOS), but there is nothing inside Security.framework that lets you
    # get a SecIdentityRef without having a key in a keychain.
    #
    # So the policy here is we take all the files and iterate them in order.
    # Each one will use SecItemImport to have one or more objects loaded from
    # it. We will also point at a keychain that macOS can use to work with the
    # private key.
    #
    # Once we have all the objects, we'll check what we actually have. If we
    # already have a SecIdentityRef in hand, fab: we'll use that. Otherwise,
    # we'll take the first certificate (which we assume to be our leaf) and
    # ask the keychain to give us a SecIdentityRef with that cert's associated
    # key.
    #
    # We'll then return a CFArray containing the trust chain: one
    # SecIdentityRef and then zero-or-more SecCertificateRef objects. The
    # responsibility for freeing this CFArray will be with the caller. This
    # CFArray must remain alive for the entire connection, so in practice it
    # will be stored with a single SSLSocket, along with the reference to the
    # keychain.
    certificates = []
    identities = []

    # Filter out bad paths.
    paths = (path for path in paths if path)

    try:
        for file_path in paths:
            new_identities, new_certs = _load_items_from_file(keychain, file_path)
            identities.extend(new_identities)
            certificates.extend(new_certs)

        # Ok, we have everything. The question is: do we have an identity? If
        # not, we want to grab one from the first cert we have.
        if not identities:
            new_identity = Security.SecIdentityRef()
            status = Security.SecIdentityCreateWithCertificate(
                keychain, certificates[0], ctypes.byref(new_identity)
            )
            _assert_no_error(status)
            identities.append(new_identity)

            # We now want to release the original certificate, as we no longer
            # need it.
            CoreFoundation.CFRelease(certificates.pop(0))

        # We now need to build a new CFArray that holds the trust chain.
        trust_chain = CoreFoundation.CFArrayCreateMutable(
            CoreFoundation.kCFAllocatorDefault,
            0,
            ctypes.byref(CoreFoundation.kCFTypeArrayCallBacks),
        )
        for item in itertools.chain(identities, certificates):
            # ArrayAppendValue does a CFRetain on the item. That's fine,
            # because the finally block will release our other refs to them.
            CoreFoundation.CFArrayAppendValue(trust_chain, item)

        return trust_chain
    finally:
        for obj in itertools.chain(identities, certificates):
            CoreFoundation.CFRelease(obj)


TLS_PROTOCOL_VERSIONS = {
    "SSLv2": (0, 2),
    "SSLv3": (3, 0),
    "TLSv1": (3, 1),
    "TLSv1.1": (3, 2),
    "TLSv1.2": (3, 3),
}


def _build_tls_unknown_ca_alert(version):
    """
    Builds a TLS alert record for an unknown CA.
    """
    ver_maj, ver_min = TLS_PROTOCOL_VERSIONS[version]
    severity_fatal = 0x02
    description_unknown_ca = 0x30
    msg = struct.pack(">BB", severity_fatal, description_unknown_ca)
    msg_len = len(msg)
    record_type_alert = 0x15
    record = struct.pack(">BBBH", record_type_alert, ver_maj, ver_min, msg_len) + msg
    return record

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\ntheory\elliptic_curve.py ===
from sympy.core.numbers import oo
from sympy.core.symbol import symbols
from sympy.polys.domains import FiniteField, QQ, RationalField, FF
from sympy.polys.polytools import Poly
from sympy.solvers.solvers import solve
from sympy.utilities.iterables import is_sequence
from sympy.utilities.misc import as_int
from .factor_ import divisors
from .residue_ntheory import polynomial_congruence


class EllipticCurve:
    """
    Create the following Elliptic Curve over domain.

    `y^{2} + a_{1} x y + a_{3} y = x^{3} + a_{2} x^{2} + a_{4} x + a_{6}`

    The default domain is ``QQ``. If no coefficient ``a1``, ``a2``, ``a3``,
    is given then it creates a curve with the following form:

    `y^{2} = x^{3} + a_{4} x + a_{6}`

    Examples
    ========

    References
    ==========

    .. [1] J. Silverman "A Friendly Introduction to Number Theory" Third Edition
    .. [2] https://mathworld.wolfram.com/EllipticDiscriminant.html
    .. [3] G. Hardy, E. Wright "An Introduction to the Theory of Numbers" Sixth Edition

    """

    def __init__(self, a4, a6, a1=0, a2=0, a3=0, modulus=0):
        if modulus == 0:
            domain = QQ
        else:
            domain = FF(modulus)
        a1, a2, a3, a4, a6 = map(domain.convert, (a1, a2, a3, a4, a6))
        self._domain = domain
        self.modulus = modulus
        # Calculate discriminant
        b2 = a1**2 + 4 * a2
        b4 = 2 * a4 + a1 * a3
        b6 = a3**2 + 4 * a6
        b8 = a1**2 * a6 + 4 * a2 * a6 - a1 * a3 * a4 + a2 * a3**2 - a4**2
        self._b2, self._b4, self._b6, self._b8 = b2, b4, b6, b8
        self._discrim = -b2**2 * b8 - 8 * b4**3 - 27 * b6**2 + 9 * b2 * b4 * b6
        self._a1 = a1
        self._a2 = a2
        self._a3 = a3
        self._a4 = a4
        self._a6 = a6
        x, y, z = symbols('x y z')
        self.x, self.y, self.z = x, y, z
        self._poly = Poly(y**2*z + a1*x*y*z + a3*y*z**2 - x**3 - a2*x**2*z - a4*x*z**2 - a6*z**3, domain=domain)
        if isinstance(self._domain, FiniteField):
            self._rank = 0
        elif isinstance(self._domain, RationalField):
            self._rank = None

    def __call__(self, x, y, z=1):
        return EllipticCurvePoint(x, y, z, self)

    def __contains__(self, point):
        if is_sequence(point):
            if len(point) == 2:
                z1 = 1
            else:
                z1 = point[2]
            x1, y1 = point[:2]
        elif isinstance(point, EllipticCurvePoint):
            x1, y1, z1 = point.x, point.y, point.z
        else:
            raise ValueError('Invalid point.')
        if self.characteristic == 0 and z1 == 0:
            return True
        return self._poly.subs({self.x: x1, self.y: y1, self.z: z1}) == 0

    def __repr__(self):
        return self._poly.__repr__()

    def minimal(self):
        """
        Return minimal Weierstrass equation.

        Examples
        ========

        >>> from sympy.ntheory.elliptic_curve import EllipticCurve

        >>> e1 = EllipticCurve(-10, -20, 0, -1, 1)
        >>> e1.minimal()
        Poly(-x**3 + 13392*x*z**2 + y**2*z + 1080432*z**3, x, y, z, domain='QQ')

        """
        char = self.characteristic
        if char == 2:
            return self
        if char == 3:
            return EllipticCurve(self._b4/2, self._b6/4, a2=self._b2/4, modulus=self.modulus)
        c4 = self._b2**2 - 24*self._b4
        c6 = -self._b2**3 + 36*self._b2*self._b4 - 216*self._b6
        return EllipticCurve(-27*c4, -54*c6, modulus=self.modulus)

    def points(self):
        """
        Return points of curve over Finite Field.

        Examples
        ========

        >>> from sympy.ntheory.elliptic_curve import EllipticCurve
        >>> e2 = EllipticCurve(1, 1, 1, 1, 1, modulus=5)
        >>> e2.points()
        {(0, 2), (1, 4), (2, 0), (2, 2), (3, 0), (3, 1), (4, 0)}

        """

        char = self.characteristic
        all_pt = set()
        if char >= 1:
            for i in range(char):
                congruence_eq = self._poly.subs({self.x: i, self.z: 1}).expr
                sol = polynomial_congruence(congruence_eq, char)
                all_pt.update((i, num) for num in sol)
            return all_pt
        else:
            raise ValueError("Infinitely many points")

    def points_x(self, x):
        """Returns points on the curve for the given x-coordinate."""
        pt = []
        if self._domain == QQ:
            for y in solve(self._poly.subs(self.x, x)):
                pt.append((x, y))
        else:
            congruence_eq = self._poly.subs({self.x: x, self.z: 1}).expr
            for y in polynomial_congruence(congruence_eq, self.characteristic):
                pt.append((x, y))
        return pt

    def torsion_points(self):
        """
        Return torsion points of curve over Rational number.

        Return point objects those are finite order.
        According to Nagell-Lutz theorem, torsion point p(x, y)
        x and y are integers, either y = 0 or y**2 is divisor
        of discriminent. According to Mazur's theorem, there are
        at most 15 points in torsion collection.

        Examples
        ========

        >>> from sympy.ntheory.elliptic_curve import EllipticCurve
        >>> e2 = EllipticCurve(-43, 166)
        >>> sorted(e2.torsion_points())
        [(-5, -16), (-5, 16), O, (3, -8), (3, 8), (11, -32), (11, 32)]

        """
        if self.characteristic > 0:
            raise ValueError("No torsion point for Finite Field.")
        l = [EllipticCurvePoint.point_at_infinity(self)]
        for xx in solve(self._poly.subs({self.y: 0, self.z: 1})):
            if xx.is_rational:
                l.append(self(xx, 0))
        for i in divisors(self.discriminant, generator=True):
            j = int(i**.5)
            if j**2 == i:
                for xx in solve(self._poly.subs({self.y: j, self.z: 1})):
                    if not xx.is_rational:
                        continue
                    p = self(xx, j)
                    if p.order() != oo:
                        l.extend([p, -p])
        return l

    @property
    def characteristic(self):
        """
        Return domain characteristic.

        Examples
        ========

        >>> from sympy.ntheory.elliptic_curve import EllipticCurve
        >>> e2 = EllipticCurve(-43, 166)
        >>> e2.characteristic
        0

        """
        return self._domain.characteristic()

    @property
    def discriminant(self):
        """
        Return curve discriminant.

        Examples
        ========

        >>> from sympy.ntheory.elliptic_curve import EllipticCurve
        >>> e2 = EllipticCurve(0, 17)
        >>> e2.discriminant
        -124848

        """
        return int(self._discrim)

    @property
    def is_singular(self):
        """
        Return True if curve discriminant is equal to zero.
        """
        return self.discriminant == 0

    @property
    def j_invariant(self):
        """
        Return curve j-invariant.

        Examples
        ========

        >>> from sympy.ntheory.elliptic_curve import EllipticCurve
        >>> e1 = EllipticCurve(-2, 0, 0, 1, 1)
        >>> e1.j_invariant
        1404928/389

        """
        c4 = self._b2**2 - 24*self._b4
        return self._domain.to_sympy(c4**3 / self._discrim)

    @property
    def order(self):
        """
        Number of points in Finite field.

        Examples
        ========

        >>> from sympy.ntheory.elliptic_curve import EllipticCurve
        >>> e2 = EllipticCurve(1, 0, modulus=19)
        >>> e2.order
        19

        """
        if self.characteristic == 0:
            raise NotImplementedError("Still not implemented")
        return len(self.points())

    @property
    def rank(self):
        """
        Number of independent points of infinite order.

        For Finite field, it must be 0.
        """
        if self._rank is not None:
            return self._rank
        raise NotImplementedError("Still not implemented")


class EllipticCurvePoint:
    """
    Point of Elliptic Curve

    Examples
    ========

    >>> from sympy.ntheory.elliptic_curve import EllipticCurve
    >>> e1 = EllipticCurve(-17, 16)
    >>> p1 = e1(0, -4, 1)
    >>> p2 = e1(1, 0)
    >>> p1 + p2
    (15, -56)
    >>> e3 = EllipticCurve(-1, 9)
    >>> e3(1, -3) * 3
    (664/169, 17811/2197)
    >>> (e3(1, -3) * 3).order()
    oo
    >>> e2 = EllipticCurve(-2, 0, 0, 1, 1)
    >>> p = e2(-1,1)
    >>> q = e2(0, -1)
    >>> p+q
    (4, 8)
    >>> p-q
    (1, 0)
    >>> 3*p-5*q
    (328/361, -2800/6859)
    """

    @staticmethod
    def point_at_infinity(curve):
        return EllipticCurvePoint(0, 1, 0, curve)

    def __init__(self, x, y, z, curve):
        dom = curve._domain.convert
        self.x = dom(x)
        self.y = dom(y)
        self.z = dom(z)
        self._curve = curve
        self._domain = self._curve._domain
        if not self._curve.__contains__(self):
            raise ValueError("The curve does not contain this point")

    def __add__(self, p):
        if self.z == 0:
            return p
        if p.z == 0:
            return self
        x1, y1 = self.x/self.z, self.y/self.z
        x2, y2 = p.x/p.z, p.y/p.z
        a1 = self._curve._a1
        a2 = self._curve._a2
        a3 = self._curve._a3
        a4 = self._curve._a4
        a6 = self._curve._a6
        if x1 != x2:
            slope = (y1 - y2) / (x1 - x2)
            yint = (y1 * x2 - y2 * x1) / (x2 - x1)
        else:
            if (y1 + y2) == 0:
                return self.point_at_infinity(self._curve)
            slope = (3 * x1**2 + 2*a2*x1 + a4 - a1*y1) / (a1 * x1 + a3 + 2 * y1)
            yint = (-x1**3 + a4*x1 + 2*a6 - a3*y1) / (a1*x1 + a3 + 2*y1)
        x3 = slope**2 + a1*slope - a2 - x1 - x2
        y3 = -(slope + a1) * x3 - yint - a3
        return self._curve(x3, y3, 1)

    def __lt__(self, other):
        return (self.x, self.y, self.z) < (other.x, other.y, other.z)

    def __mul__(self, n):
        n = as_int(n)
        r = self.point_at_infinity(self._curve)
        if n == 0:
            return r
        if n < 0:
            return -self * -n
        p = self
        while n:
            if n & 1:
                r = r + p
            n >>= 1
            p = p + p
        return r

    def __rmul__(self, n):
        return self * n

    def __neg__(self):
        return EllipticCurvePoint(self.x, -self.y - self._curve._a1*self.x - self._curve._a3, self.z, self._curve)

    def __repr__(self):
        if self.z == 0:
            return 'O'
        dom = self._curve._domain
        try:
            return '({}, {})'.format(dom.to_sympy(self.x), dom.to_sympy(self.y))
        except TypeError:
            pass
        return '({}, {})'.format(self.x, self.y)

    def __sub__(self, other):
        return self.__add__(-other)

    def order(self):
        """
        Return point order n where nP = 0.

        """
        if self.z == 0:
            return 1
        if self.y == 0:  # P = -P
            return 2
        p = self * 2
        if p.y == -self.y:  # 2P = -P
            return 3
        i = 2
        if self._domain != QQ:
            while int(p.x) == p.x and int(p.y) == p.y:
                p = self + p
                i += 1
                if p.z == 0:
                    return i
            return oo
        while p.x.numerator == p.x and p.y.numerator == p.y:
            p = self + p
            i += 1
            if i > 12:
                return oo
            if p.z == 0:
                return i
        return oo

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\charset_normalizer\cd.py ===
from __future__ import annotations

import importlib
from codecs import IncrementalDecoder
from collections import Counter
from functools import lru_cache
from typing import Counter as TypeCounter

from .constant import (
    FREQUENCIES,
    KO_NAMES,
    LANGUAGE_SUPPORTED_COUNT,
    TOO_SMALL_SEQUENCE,
    ZH_NAMES,
)
from .md import is_suspiciously_successive_range
from .models import CoherenceMatches
from .utils import (
    is_accentuated,
    is_latin,
    is_multi_byte_encoding,
    is_unicode_range_secondary,
    unicode_range,
)


def encoding_unicode_range(iana_name: str) -> list[str]:
    """
    Return associated unicode ranges in a single byte code page.
    """
    if is_multi_byte_encoding(iana_name):
        raise OSError("Function not supported on multi-byte code page")

    decoder = importlib.import_module(f"encodings.{iana_name}").IncrementalDecoder

    p: IncrementalDecoder = decoder(errors="ignore")
    seen_ranges: dict[str, int] = {}
    character_count: int = 0

    for i in range(0x40, 0xFF):
        chunk: str = p.decode(bytes([i]))

        if chunk:
            character_range: str | None = unicode_range(chunk)

            if character_range is None:
                continue

            if is_unicode_range_secondary(character_range) is False:
                if character_range not in seen_ranges:
                    seen_ranges[character_range] = 0
                seen_ranges[character_range] += 1
                character_count += 1

    return sorted(
        [
            character_range
            for character_range in seen_ranges
            if seen_ranges[character_range] / character_count >= 0.15
        ]
    )


def unicode_range_languages(primary_range: str) -> list[str]:
    """
    Return inferred languages used with a unicode range.
    """
    languages: list[str] = []

    for language, characters in FREQUENCIES.items():
        for character in characters:
            if unicode_range(character) == primary_range:
                languages.append(language)
                break

    return languages


@lru_cache()
def encoding_languages(iana_name: str) -> list[str]:
    """
    Single-byte encoding language association. Some code page are heavily linked to particular language(s).
    This function does the correspondence.
    """
    unicode_ranges: list[str] = encoding_unicode_range(iana_name)
    primary_range: str | None = None

    for specified_range in unicode_ranges:
        if "Latin" not in specified_range:
            primary_range = specified_range
            break

    if primary_range is None:
        return ["Latin Based"]

    return unicode_range_languages(primary_range)


@lru_cache()
def mb_encoding_languages(iana_name: str) -> list[str]:
    """
    Multi-byte encoding language association. Some code page are heavily linked to particular language(s).
    This function does the correspondence.
    """
    if (
        iana_name.startswith("shift_")
        or iana_name.startswith("iso2022_jp")
        or iana_name.startswith("euc_j")
        or iana_name == "cp932"
    ):
        return ["Japanese"]
    if iana_name.startswith("gb") or iana_name in ZH_NAMES:
        return ["Chinese"]
    if iana_name.startswith("iso2022_kr") or iana_name in KO_NAMES:
        return ["Korean"]

    return []


@lru_cache(maxsize=LANGUAGE_SUPPORTED_COUNT)
def get_target_features(language: str) -> tuple[bool, bool]:
    """
    Determine main aspects from a supported language if it contains accents and if is pure Latin.
    """
    target_have_accents: bool = False
    target_pure_latin: bool = True

    for character in FREQUENCIES[language]:
        if not target_have_accents and is_accentuated(character):
            target_have_accents = True
        if target_pure_latin and is_latin(character) is False:
            target_pure_latin = False

    return target_have_accents, target_pure_latin


def alphabet_languages(
    characters: list[str], ignore_non_latin: bool = False
) -> list[str]:
    """
    Return associated languages associated to given characters.
    """
    languages: list[tuple[str, float]] = []

    source_have_accents = any(is_accentuated(character) for character in characters)

    for language, language_characters in FREQUENCIES.items():
        target_have_accents, target_pure_latin = get_target_features(language)

        if ignore_non_latin and target_pure_latin is False:
            continue

        if target_have_accents is False and source_have_accents:
            continue

        character_count: int = len(language_characters)

        character_match_count: int = len(
            [c for c in language_characters if c in characters]
        )

        ratio: float = character_match_count / character_count

        if ratio >= 0.2:
            languages.append((language, ratio))

    languages = sorted(languages, key=lambda x: x[1], reverse=True)

    return [compatible_language[0] for compatible_language in languages]


def characters_popularity_compare(
    language: str, ordered_characters: list[str]
) -> float:
    """
    Determine if a ordered characters list (by occurrence from most appearance to rarest) match a particular language.
    The result is a ratio between 0. (absolutely no correspondence) and 1. (near perfect fit).
    Beware that is function is not strict on the match in order to ease the detection. (Meaning close match is 1.)
    """
    if language not in FREQUENCIES:
        raise ValueError(f"{language} not available")

    character_approved_count: int = 0
    FREQUENCIES_language_set = set(FREQUENCIES[language])

    ordered_characters_count: int = len(ordered_characters)
    target_language_characters_count: int = len(FREQUENCIES[language])

    large_alphabet: bool = target_language_characters_count > 26

    for character, character_rank in zip(
        ordered_characters, range(0, ordered_characters_count)
    ):
        if character not in FREQUENCIES_language_set:
            continue

        character_rank_in_language: int = FREQUENCIES[language].index(character)
        expected_projection_ratio: float = (
            target_language_characters_count / ordered_characters_count
        )
        character_rank_projection: int = int(character_rank * expected_projection_ratio)

        if (
            large_alphabet is False
            and abs(character_rank_projection - character_rank_in_language) > 4
        ):
            continue

        if (
            large_alphabet is True
            and abs(character_rank_projection - character_rank_in_language)
            < target_language_characters_count / 3
        ):
            character_approved_count += 1
            continue

        characters_before_source: list[str] = FREQUENCIES[language][
            0:character_rank_in_language
        ]
        characters_after_source: list[str] = FREQUENCIES[language][
            character_rank_in_language:
        ]
        characters_before: list[str] = ordered_characters[0:character_rank]
        characters_after: list[str] = ordered_characters[character_rank:]

        before_match_count: int = len(
            set(characters_before) & set(characters_before_source)
        )

        after_match_count: int = len(
            set(characters_after) & set(characters_after_source)
        )

        if len(characters_before_source) == 0 and before_match_count <= 4:
            character_approved_count += 1
            continue

        if len(characters_after_source) == 0 and after_match_count <= 4:
            character_approved_count += 1
            continue

        if (
            before_match_count / len(characters_before_source) >= 0.4
            or after_match_count / len(characters_after_source) >= 0.4
        ):
            character_approved_count += 1
            continue

    return character_approved_count / len(ordered_characters)


def alpha_unicode_split(decoded_sequence: str) -> list[str]:
    """
    Given a decoded text sequence, return a list of str. Unicode range / alphabet separation.
    Ex. a text containing English/Latin with a bit a Hebrew will return two items in the resulting list;
    One containing the latin letters and the other hebrew.
    """
    layers: dict[str, str] = {}

    for character in decoded_sequence:
        if character.isalpha() is False:
            continue

        character_range: str | None = unicode_range(character)

        if character_range is None:
            continue

        layer_target_range: str | None = None

        for discovered_range in layers:
            if (
                is_suspiciously_successive_range(discovered_range, character_range)
                is False
            ):
                layer_target_range = discovered_range
                break

        if layer_target_range is None:
            layer_target_range = character_range

        if layer_target_range not in layers:
            layers[layer_target_range] = character.lower()
            continue

        layers[layer_target_range] += character.lower()

    return list(layers.values())


def merge_coherence_ratios(results: list[CoherenceMatches]) -> CoherenceMatches:
    """
    This function merge results previously given by the function coherence_ratio.
    The return type is the same as coherence_ratio.
    """
    per_language_ratios: dict[str, list[float]] = {}
    for result in results:
        for sub_result in result:
            language, ratio = sub_result
            if language not in per_language_ratios:
                per_language_ratios[language] = [ratio]
                continue
            per_language_ratios[language].append(ratio)

    merge = [
        (
            language,
            round(
                sum(per_language_ratios[language]) / len(per_language_ratios[language]),
                4,
            ),
        )
        for language in per_language_ratios
    ]

    return sorted(merge, key=lambda x: x[1], reverse=True)


def filter_alt_coherence_matches(results: CoherenceMatches) -> CoherenceMatches:
    """
    We shall NOT return "English—" in CoherenceMatches because it is an alternative
    of "English". This function only keeps the best match and remove the em-dash in it.
    """
    index_results: dict[str, list[float]] = dict()

    for result in results:
        language, ratio = result
        no_em_name: str = language.replace("—", "")

        if no_em_name not in index_results:
            index_results[no_em_name] = []

        index_results[no_em_name].append(ratio)

    if any(len(index_results[e]) > 1 for e in index_results):
        filtered_results: CoherenceMatches = []

        for language in index_results:
            filtered_results.append((language, max(index_results[language])))

        return filtered_results

    return results


@lru_cache(maxsize=2048)
def coherence_ratio(
    decoded_sequence: str, threshold: float = 0.1, lg_inclusion: str | None = None
) -> CoherenceMatches:
    """
    Detect ANY language that can be identified in given sequence. The sequence will be analysed by layers.
    A layer = Character extraction by alphabets/ranges.
    """

    results: list[tuple[str, float]] = []
    ignore_non_latin: bool = False

    sufficient_match_count: int = 0

    lg_inclusion_list = lg_inclusion.split(",") if lg_inclusion is not None else []
    if "Latin Based" in lg_inclusion_list:
        ignore_non_latin = True
        lg_inclusion_list.remove("Latin Based")

    for layer in alpha_unicode_split(decoded_sequence):
        sequence_frequencies: TypeCounter[str] = Counter(layer)
        most_common = sequence_frequencies.most_common()

        character_count: int = sum(o for c, o in most_common)

        if character_count <= TOO_SMALL_SEQUENCE:
            continue

        popular_character_ordered: list[str] = [c for c, o in most_common]

        for language in lg_inclusion_list or alphabet_languages(
            popular_character_ordered, ignore_non_latin
        ):
            ratio: float = characters_popularity_compare(
                language, popular_character_ordered
            )

            if ratio < threshold:
                continue
            elif ratio >= 0.8:
                sufficient_match_count += 1

            results.append((language, round(ratio, 4)))

            if sufficient_match_count >= 3:
                break

    return sorted(
        filter_alt_coherence_matches(results), key=lambda x: x[1], reverse=True
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\markupsafe\__init__.py ===
from __future__ import annotations

import collections.abc as cabc
import string
import typing as t

try:
    from ._speedups import _escape_inner
except ImportError:
    from ._native import _escape_inner

if t.TYPE_CHECKING:
    import typing_extensions as te


class _HasHTML(t.Protocol):
    def __html__(self, /) -> str: ...


class _TPEscape(t.Protocol):
    def __call__(self, s: t.Any, /) -> Markup: ...


def escape(s: t.Any, /) -> Markup:
    """Replace the characters ``&``, ``<``, ``>``, ``'``, and ``"`` in
    the string with HTML-safe sequences. Use this if you need to display
    text that might contain such characters in HTML.

    If the object has an ``__html__`` method, it is called and the
    return value is assumed to already be safe for HTML.

    :param s: An object to be converted to a string and escaped.
    :return: A :class:`Markup` string with the escaped text.
    """
    # If the object is already a plain string, skip __html__ check and string
    # conversion. This is the most common use case.
    # Use type(s) instead of s.__class__ because a proxy object may be reporting
    # the __class__ of the proxied value.
    if type(s) is str:
        return Markup(_escape_inner(s))

    if hasattr(s, "__html__"):
        return Markup(s.__html__())

    return Markup(_escape_inner(str(s)))


def escape_silent(s: t.Any | None, /) -> Markup:
    """Like :func:`escape` but treats ``None`` as the empty string.
    Useful with optional values, as otherwise you get the string
    ``'None'`` when the value is ``None``.

    >>> escape(None)
    Markup('None')
    >>> escape_silent(None)
    Markup('')
    """
    if s is None:
        return Markup()

    return escape(s)


def soft_str(s: t.Any, /) -> str:
    """Convert an object to a string if it isn't already. This preserves
    a :class:`Markup` string rather than converting it back to a basic
    string, so it will still be marked as safe and won't be escaped
    again.

    >>> value = escape("<User 1>")
    >>> value
    Markup('&lt;User 1&gt;')
    >>> escape(str(value))
    Markup('&amp;lt;User 1&amp;gt;')
    >>> escape(soft_str(value))
    Markup('&lt;User 1&gt;')
    """
    if not isinstance(s, str):
        return str(s)

    return s


class Markup(str):
    """A string that is ready to be safely inserted into an HTML or XML
    document, either because it was escaped or because it was marked
    safe.

    Passing an object to the constructor converts it to text and wraps
    it to mark it safe without escaping. To escape the text, use the
    :meth:`escape` class method instead.

    >>> Markup("Hello, <em>World</em>!")
    Markup('Hello, <em>World</em>!')
    >>> Markup(42)
    Markup('42')
    >>> Markup.escape("Hello, <em>World</em>!")
    Markup('Hello &lt;em&gt;World&lt;/em&gt;!')

    This implements the ``__html__()`` interface that some frameworks
    use. Passing an object that implements ``__html__()`` will wrap the
    output of that method, marking it safe.

    >>> class Foo:
    ...     def __html__(self):
    ...         return '<a href="/foo">foo</a>'
    ...
    >>> Markup(Foo())
    Markup('<a href="/foo">foo</a>')

    This is a subclass of :class:`str`. It has the same methods, but
    escapes their arguments and returns a ``Markup`` instance.

    >>> Markup("<em>%s</em>") % ("foo & bar",)
    Markup('<em>foo &amp; bar</em>')
    >>> Markup("<em>Hello</em> ") + "<foo>"
    Markup('<em>Hello</em> &lt;foo&gt;')
    """

    __slots__ = ()

    def __new__(
        cls, object: t.Any = "", encoding: str | None = None, errors: str = "strict"
    ) -> te.Self:
        if hasattr(object, "__html__"):
            object = object.__html__()

        if encoding is None:
            return super().__new__(cls, object)

        return super().__new__(cls, object, encoding, errors)

    def __html__(self, /) -> te.Self:
        return self

    def __add__(self, value: str | _HasHTML, /) -> te.Self:
        if isinstance(value, str) or hasattr(value, "__html__"):
            return self.__class__(super().__add__(self.escape(value)))

        return NotImplemented

    def __radd__(self, value: str | _HasHTML, /) -> te.Self:
        if isinstance(value, str) or hasattr(value, "__html__"):
            return self.escape(value).__add__(self)

        return NotImplemented

    def __mul__(self, value: t.SupportsIndex, /) -> te.Self:
        return self.__class__(super().__mul__(value))

    def __rmul__(self, value: t.SupportsIndex, /) -> te.Self:
        return self.__class__(super().__mul__(value))

    def __mod__(self, value: t.Any, /) -> te.Self:
        if isinstance(value, tuple):
            # a tuple of arguments, each wrapped
            value = tuple(_MarkupEscapeHelper(x, self.escape) for x in value)
        elif hasattr(type(value), "__getitem__") and not isinstance(value, str):
            # a mapping of arguments, wrapped
            value = _MarkupEscapeHelper(value, self.escape)
        else:
            # a single argument, wrapped with the helper and a tuple
            value = (_MarkupEscapeHelper(value, self.escape),)

        return self.__class__(super().__mod__(value))

    def __repr__(self, /) -> str:
        return f"{self.__class__.__name__}({super().__repr__()})"

    def join(self, iterable: cabc.Iterable[str | _HasHTML], /) -> te.Self:
        return self.__class__(super().join(map(self.escape, iterable)))

    def split(  # type: ignore[override]
        self, /, sep: str | None = None, maxsplit: t.SupportsIndex = -1
    ) -> list[te.Self]:
        return [self.__class__(v) for v in super().split(sep, maxsplit)]

    def rsplit(  # type: ignore[override]
        self, /, sep: str | None = None, maxsplit: t.SupportsIndex = -1
    ) -> list[te.Self]:
        return [self.__class__(v) for v in super().rsplit(sep, maxsplit)]

    def splitlines(  # type: ignore[override]
        self, /, keepends: bool = False
    ) -> list[te.Self]:
        return [self.__class__(v) for v in super().splitlines(keepends)]

    def unescape(self, /) -> str:
        """Convert escaped markup back into a text string. This replaces
        HTML entities with the characters they represent.

        >>> Markup("Main &raquo; <em>About</em>").unescape()
        'Main » <em>About</em>'
        """
        from html import unescape

        return unescape(str(self))

    def striptags(self, /) -> str:
        """:meth:`unescape` the markup, remove tags, and normalize
        whitespace to single spaces.

        >>> Markup("Main &raquo;\t<em>About</em>").striptags()
        'Main » About'
        """
        value = str(self)

        # Look for comments then tags separately. Otherwise, a comment that
        # contains a tag would end early, leaving some of the comment behind.

        # keep finding comment start marks
        while (start := value.find("<!--")) != -1:
            # find a comment end mark beyond the start, otherwise stop
            if (end := value.find("-->", start)) == -1:
                break

            value = f"{value[:start]}{value[end + 3:]}"

        # remove tags using the same method
        while (start := value.find("<")) != -1:
            if (end := value.find(">", start)) == -1:
                break

            value = f"{value[:start]}{value[end + 1:]}"

        # collapse spaces
        value = " ".join(value.split())
        return self.__class__(value).unescape()

    @classmethod
    def escape(cls, s: t.Any, /) -> te.Self:
        """Escape a string. Calls :func:`escape` and ensures that for
        subclasses the correct type is returned.
        """
        rv = escape(s)

        if rv.__class__ is not cls:
            return cls(rv)

        return rv  # type: ignore[return-value]

    def __getitem__(self, key: t.SupportsIndex | slice, /) -> te.Self:
        return self.__class__(super().__getitem__(key))

    def capitalize(self, /) -> te.Self:
        return self.__class__(super().capitalize())

    def title(self, /) -> te.Self:
        return self.__class__(super().title())

    def lower(self, /) -> te.Self:
        return self.__class__(super().lower())

    def upper(self, /) -> te.Self:
        return self.__class__(super().upper())

    def replace(self, old: str, new: str, count: t.SupportsIndex = -1, /) -> te.Self:
        return self.__class__(super().replace(old, self.escape(new), count))

    def ljust(self, width: t.SupportsIndex, fillchar: str = " ", /) -> te.Self:
        return self.__class__(super().ljust(width, self.escape(fillchar)))

    def rjust(self, width: t.SupportsIndex, fillchar: str = " ", /) -> te.Self:
        return self.__class__(super().rjust(width, self.escape(fillchar)))

    def lstrip(self, chars: str | None = None, /) -> te.Self:
        return self.__class__(super().lstrip(chars))

    def rstrip(self, chars: str | None = None, /) -> te.Self:
        return self.__class__(super().rstrip(chars))

    def center(self, width: t.SupportsIndex, fillchar: str = " ", /) -> te.Self:
        return self.__class__(super().center(width, self.escape(fillchar)))

    def strip(self, chars: str | None = None, /) -> te.Self:
        return self.__class__(super().strip(chars))

    def translate(
        self,
        table: cabc.Mapping[int, str | int | None],  # type: ignore[override]
        /,
    ) -> str:
        return self.__class__(super().translate(table))

    def expandtabs(self, /, tabsize: t.SupportsIndex = 8) -> te.Self:
        return self.__class__(super().expandtabs(tabsize))

    def swapcase(self, /) -> te.Self:
        return self.__class__(super().swapcase())

    def zfill(self, width: t.SupportsIndex, /) -> te.Self:
        return self.__class__(super().zfill(width))

    def casefold(self, /) -> te.Self:
        return self.__class__(super().casefold())

    def removeprefix(self, prefix: str, /) -> te.Self:
        return self.__class__(super().removeprefix(prefix))

    def removesuffix(self, suffix: str) -> te.Self:
        return self.__class__(super().removesuffix(suffix))

    def partition(self, sep: str, /) -> tuple[te.Self, te.Self, te.Self]:
        left, sep, right = super().partition(sep)
        cls = self.__class__
        return cls(left), cls(sep), cls(right)

    def rpartition(self, sep: str, /) -> tuple[te.Self, te.Self, te.Self]:
        left, sep, right = super().rpartition(sep)
        cls = self.__class__
        return cls(left), cls(sep), cls(right)

    def format(self, *args: t.Any, **kwargs: t.Any) -> te.Self:
        formatter = EscapeFormatter(self.escape)
        return self.__class__(formatter.vformat(self, args, kwargs))

    def format_map(
        self,
        mapping: cabc.Mapping[str, t.Any],  # type: ignore[override]
        /,
    ) -> te.Self:
        formatter = EscapeFormatter(self.escape)
        return self.__class__(formatter.vformat(self, (), mapping))

    def __html_format__(self, format_spec: str, /) -> te.Self:
        if format_spec:
            raise ValueError("Unsupported format specification for Markup.")

        return self


class EscapeFormatter(string.Formatter):
    __slots__ = ("escape",)

    def __init__(self, escape: _TPEscape) -> None:
        self.escape: _TPEscape = escape
        super().__init__()

    def format_field(self, value: t.Any, format_spec: str) -> str:
        if hasattr(value, "__html_format__"):
            rv = value.__html_format__(format_spec)
        elif hasattr(value, "__html__"):
            if format_spec:
                raise ValueError(
                    f"Format specifier {format_spec} given, but {type(value)} does not"
                    " define __html_format__. A class that defines __html__ must define"
                    " __html_format__ to work with format specifiers."
                )
            rv = value.__html__()
        else:
            # We need to make sure the format spec is str here as
            # otherwise the wrong callback methods are invoked.
            rv = super().format_field(value, str(format_spec))
        return str(self.escape(rv))


class _MarkupEscapeHelper:
    """Helper for :meth:`Markup.__mod__`."""

    __slots__ = ("obj", "escape")

    def __init__(self, obj: t.Any, escape: _TPEscape) -> None:
        self.obj: t.Any = obj
        self.escape: _TPEscape = escape

    def __getitem__(self, key: t.Any, /) -> te.Self:
        return self.__class__(self.obj[key], self.escape)

    def __str__(self, /) -> str:
        return str(self.escape(self.obj))

    def __repr__(self, /) -> str:
        return str(self.escape(repr(self.obj)))

    def __int__(self, /) -> int:
        return int(self.obj)

    def __float__(self, /) -> float:
        return float(self.obj)


def __getattr__(name: str) -> t.Any:
    if name == "__version__":
        import importlib.metadata
        import warnings

        warnings.warn(
            "The '__version__' attribute is deprecated and will be removed in"
            " MarkupSafe 3.1. Use feature detection, or"
            ' `importlib.metadata.version("markupsafe")`, instead.',
            stacklevel=2,
        )
        return importlib.metadata.version("markupsafe")

    raise AttributeError(name)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\large_model_exporter.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""
Export LLM to onnx
"""

import argparse
import inspect
import math
import os
import tempfile
from pathlib import Path

import onnx
import torch
import transformers
from torch import nn


def disable_huggingface_init():
    """do not init model twice as it slow initialization"""

    torch.nn.init.kaiming_uniform_ = lambda x, *args, **kwargs: x
    torch.nn.init.uniform_ = lambda x, *args, **kwargs: x
    torch.nn.init.normal_ = lambda x, *args, **kwargs: x
    torch.nn.init.constant_ = lambda x, *args, **kwargs: x
    torch.nn.init.xavier_uniform_ = lambda x, *args, **kwargs: x
    torch.nn.init.xavier_normal_ = lambda x, *args, **kwargs: x
    torch.nn.init.kaiming_normal_ = lambda x, *args, **kwargs: x
    torch.nn.init.orthogonal_ = lambda x, *args, **kwargs: x


def get_model_parameter_size(model: nn.Module):
    """to calculate how much memory this model needs"""
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    return all_size


def initialize_model_and_sample_inputs(hf_model: str, cache_dir: str | None, tokenizer=None):
    """
    get the pretrained torch model from hugginface,
    and sample model-inputs
    """

    disable_huggingface_init()

    model = transformers.AutoModelForCausalLM.from_pretrained(  # type: ignore
        hf_model, torch_dtype=torch.float16, cache_dir=cache_dir, trust_remote_code=True
    )
    if tokenizer is None:
        tokenizer = hf_model
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer)  # type: ignore

    sample_inputs = tuple(tokenizer("Hello, my dog is cute", return_tensors="pt").values())
    return model, sample_inputs


def auto_pipeline_parallel(model: nn.Module, gpulist: list, sample_inputs: tuple):
    """Make the model executable across multiple GPUs."""

    def input_gpu_device_hook(mod, inputs, kwargs):
        modifyed_inputs = []
        first_dev = None
        for layer_input in inputs:
            if type(layer_input) is not torch.Tensor:
                modifyed_inputs.append(layer_input)
            elif hasattr(mod, "weight"):
                modifyed_inputs.append(layer_input.to(mod.weight.device))
            elif hasattr(mod, "parameters"):
                device = next(mod.parameters(), layer_input).device
                modifyed_inputs.append(layer_input.to(device))
            elif hasattr(next(mod.children(), None), "weight"):
                modifyed_inputs.append(layer_input.to(next(mod.children()).weight.device))
            elif first_dev is not None and layer_input.device != first_dev:
                modifyed_inputs.append(layer_input.to(first_dev))
            else:
                modifyed_inputs.append(layer_input)
            if first_dev is None:
                first_dev = modifyed_inputs[0].device
        for key, value in kwargs.items():
            if type(value) is torch.Tensor:
                kwargs[key] = value.to(first_dev)

        return (tuple(modifyed_inputs), kwargs)

    def move_layer_to_device_rurc(mod, dev):
        mod.to(dev)
        for layer in mod.named_children():
            move_layer_to_device_rurc(layer[1], dev)

    model = model.half()
    all_hooks = []
    all_hooks.append(model.register_forward_pre_hook(input_gpu_device_hook, with_kwargs=True))
    pre_fix = next(iter(model.named_children()))[0]
    for top_name, top_module in model.named_children():
        for name, module in top_module.named_children():
            all_hooks.append(module.register_forward_pre_hook(input_gpu_device_hook, with_kwargs=True))
            if type(module) in [torch.nn.ModuleList]:
                num_layers_on_each_gpu = math.floor(len(module) / len(gpulist))
                for idx, attn_layer in enumerate(module):
                    all_hooks.append(attn_layer.register_forward_pre_hook(input_gpu_device_hook, with_kwargs=True))

                    to_dev = gpulist[min(idx // num_layers_on_each_gpu, len(gpulist))]
                    attn_layer.to(to_dev)
                    move_layer_to_device_rurc(attn_layer, to_dev)
                    print(f"move {pre_fix}.{name}.{idx} to {to_dev}")
            else:
                module.to(gpulist[0])
                print(f"move {pre_fix}.{name} to {gpulist[0]}")
        if len(list(top_module.named_children())) == 0:
            top_module.to(gpulist[0])
            print(f"move {top_name} to {gpulist[0]}")

    with torch.no_grad():
        model(sample_inputs[0], attention_mask=sample_inputs[1])
    return model


def retrieve_onnx_inputs(model: nn.Module, sample_inputs: tuple, with_past: bool):
    """
    auto retrieve onnx inputs from torch model as we can't enumlate all possibilities
    for all models
    """
    user_inputs = []

    def hook_for_inputs(_, inputs, kwargs):
        user_inputs.append((inputs, kwargs))
        return user_inputs[0]

    hook_handle = model.register_forward_pre_hook(hook_for_inputs, with_kwargs=True)

    forward_params = inspect.signature(model.forward).parameters
    input_keys = list(forward_params.keys())
    default_values = [forward_params.get(key).default for key in input_keys]
    out = model(sample_inputs[0], attention_mask=sample_inputs[1])
    hook_handle.remove()
    user_inputs = user_inputs[0]
    onnx_inputs = default_values
    for idx, _val in enumerate(user_inputs[0]):
        onnx_inputs[idx] = user_inputs[0][idx]
    for key, value in user_inputs[1].items():
        idx = input_keys.index(key)
        onnx_inputs[idx] = value
    for idx, (key, value) in enumerate(zip(input_keys, onnx_inputs, strict=False)):
        if type(value) is torch.Tensor:
            value.to(model.device)
        if "use_cache" in key:
            onnx_inputs[idx] = with_past
            out = model(sample_inputs[0], attention_mask=sample_inputs[1], use_cache=with_past) if with_past else out

    return input_keys, onnx_inputs, out.past_key_values


def move_to_appropriate_device(model: nn.Module, sample_inputs_tp: tuple) -> nn.Module:
    """
    According to the model size, we will upload it to
    CPU if has no GPU or enough GPU memory,
    Single GPU if has only one GPU in local or model size is enough to fit one GPU
    Multiple GPU if there is more than one gpu in local and model is too large
    """
    total_mem_per_cpu = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024

    print(f"Model_Size = {get_model_parameter_size(model) / 1024} GB")
    print(f"total_mem_per_cpu = {total_mem_per_cpu / 1024} GB")
    if get_model_parameter_size(model) > total_mem_per_cpu * 0.45:
        device_collection = [torch.device(i) for i in range(torch.cuda.device_count())]
        if len(device_collection) > 1:
            print(
                f"{len(device_collection)} GPUs are used to export onnx, \
                   Please set CUDA_VISIBLE_DEVICES to use specific GPU group"
            )
            model = auto_pipeline_parallel(model, device_collection, sample_inputs_tp)
        else:
            print("!!!! convert model to float and export onnx using CPU")
            model = model.cpu().float()
    else:
        print("Export model on a single GPU")
        model = model.cuda().half()
    return model


def adapt_inputs_to_device(sample_inputs: tuple, device: torch.device) -> tuple:
    """move inputs to device"""
    sample_inputs_ = []
    for sample_int in sample_inputs:
        if isinstance(sample_int, torch.Tensor):
            sample_inputs_.append(sample_int.to(device))
        else:
            sample_inputs_.append(sample_int)
    return tuple(sample_inputs_)


def fetch_onnx_inputs_outputs_name(
    model: nn.Module,
    onnx_inputs: list,
    torch_input_names: tuple,
    past_key_values: tuple,
    with_past: bool,
    input_with_past: bool,
):
    """fetch onnx inputs and outputs name"""
    num_of_past_key = 0
    kv_cache_axis = {0: "batch_size"}
    # try get num_of_past_key and shape of past_key_value
    if past_key_values is not None:
        num_of_past_key = len(past_key_values)
        seq_index = (torch.tensor(past_key_values[0][0].shape) == onnx_inputs[0].shape[-1]).nonzero().view(-1)
        assert seq_index.numel() == 1
        kv_cache_axis = {0: "batch_size", seq_index.item(): "seq_len"}

    if not num_of_past_key:
        num_of_past_key = model.config.num_hidden_layers

    # filter out constant inputs
    onnx_inp_names = tuple(
        [torch_input_names[i] for i in range(len(torch_input_names)) if isinstance(onnx_inputs[i], torch.Tensor)]
    )
    assert "input_ids" in onnx_inp_names and "attention_mask" in onnx_inp_names, (
        "input_ids and attention_mask must be existed in inputs"
    )
    onnx_out_names = ("logits",)
    onnx_dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
    }
    # add dyanmic dimensions for the unkonw inputs
    for idx, name in enumerate(onnx_inp_names):
        if name not in onnx_dynamic_axes:
            unknown_dims = {i: f"{idx}__unknown_dims__{i}" for i in range(onnx_inputs[idx].dim())}
            onnx_dynamic_axes[name] = unknown_dims
    if input_with_past:
        for i in range(num_of_past_key):
            onnx_inp_names += (f"past_key_values.{i}.key",)
            onnx_inp_names += (f"past_key_values.{i}.value",)

            onnx_dynamic_axes[onnx_inp_names[-1]] = kv_cache_axis
            onnx_dynamic_axes[onnx_inp_names[-2]] = kv_cache_axis

    if with_past or input_with_past:
        for i in range(num_of_past_key):
            onnx_out_names += (f"present.{i}.key",)
            onnx_out_names += (f"present.{i}.value",)

    for idx, name in enumerate(torch_input_names):
        if input_with_past:
            if name == "past_key_values":
                onnx_inputs[idx] = past_key_values
            elif name == "attention_mask":
                attn_mask = onnx_inputs[idx]
                onnx_inputs[idx] = torch.cat(
                    (attn_mask, torch.ones((attn_mask.shape[0], 1), device=attn_mask.device, dtype=attn_mask.dtype)),
                    dim=1,
                )
            elif name == "input_ids":
                input_ids = onnx_inputs[idx]
                onnx_inputs[idx] = input_ids[:, -1:]

    return onnx_inp_names, onnx_out_names, onnx_dynamic_axes


def do_export_internal(model: nn.Module, onnx_io_tuple: tuple, onnx_inputs: tuple, onnx_path: Path, opset: int):
    """do export with torch.onnx.export"""
    onnx_model_name = onnx_path.name
    onnx_inp_names, onnx_out_names, onnx_dynamic_axes = onnx_io_tuple
    # two step to export onnx
    # 1. export onnx with lots of pieces of weights
    # 2. save all weights to external data
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_onnx = os.path.join(tmpdirname, "tmp.onnx")

        torch.onnx.export(
            model=model,
            args=tuple(onnx_inputs),
            f=tmp_onnx,
            verbose=False,
            opset_version=opset,
            input_names=onnx_inp_names,
            output_names=onnx_out_names,
            dynamic_axes=onnx_dynamic_axes,
        )

        onnx_path.unlink(missing_ok=True)
        (onnx_path.parent / f"{onnx_model_name}_ext.data").unlink(missing_ok=True)

        onnx_model = onnx.load(str(tmp_onnx))
        onnx.save_model(
            onnx_model,
            str(onnx_path),
            save_as_external_data=(len(os.listdir(tmpdirname)) > 1),
            all_tensors_to_one_file=True,
            location=f"{onnx_model_name}_ext.data",
            size_threshold=1024,
            convert_attribute=False,
        )


@torch.no_grad()
def export_onnx(hf_model: str, cache_dir: str | None, onnx_path_str: str, with_past: bool, opset: int):
    """
    do export
    model: torch model
    onnx_path: where the onnx model saved to
    sample_inputs_tp: inputs for torch model
    """
    model, sample_inputs_tp = initialize_model_and_sample_inputs(hf_model, cache_dir)

    model = move_to_appropriate_device(model, sample_inputs_tp)

    sample_inputs = adapt_inputs_to_device(sample_inputs_tp, next(model.parameters()).device)

    # input_keys would be usesful if the model has some special inputs
    input_keys, onnx_inputs, past_key_value = retrieve_onnx_inputs(model, sample_inputs, with_past)

    onnx_io_tuple = fetch_onnx_inputs_outputs_name(model, onnx_inputs, input_keys, past_key_value, with_past, False)

    onnx_model_name = "model.onnx"
    onnx_path: Path = Path(onnx_path_str).absolute()
    if onnx_path.suffix != ".onnx":
        onnx_path = onnx_path / onnx_model_name

    do_export_internal(model, onnx_io_tuple, onnx_inputs, onnx_path, opset)
    if not with_past:
        return

    onnx_io_tuple = fetch_onnx_inputs_outputs_name(model, onnx_inputs, input_keys, past_key_value, with_past, True)

    onnx_model_name = "model_with_past.onnx"
    onnx_path = onnx_path.parent / onnx_model_name

    do_export_internal(model, onnx_io_tuple, onnx_inputs, onnx_path, opset)


def parse_arguments():
    """arguments parsing."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model",
        required=True,
        type=str,
        default=["meta-llama/Llama-2-70b-hf"],
        help="Pre-trained models in huggingface model hub",
    )
    parser.add_argument(
        "-s",
        "--saved_path",
        required=False,
        type=str,
        default="./onnx_models/",
        help="where the onnx model will be saved",
    )
    parser.add_argument(
        "--cache_dir",
        required=False,
        type=str,
        default=None,
        help=("cache directly of huggingface, by setting this to avoid useless downloading if you have one"),
    )
    parser.add_argument(
        "--with_past",
        action="store_true",
        default=False,
        help=("The tool will export onnx without past-key-value by default"),
    )
    parser.add_argument(
        "--opset",
        required=False,
        type=int,
        default=17,
        help=(
            "the opset to save onnx model, \
              try to increase it if this opset doens't have new features you want"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    export_onnx(args.model, args.cache_dir, args.saved_path, args.with_past, args.opset)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\stable_diffusion\engine_builder_tensorrt.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Modified from TensorRT demo diffusion, which has the following license:
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
# --------------------------------------------------------------------------

import gc
import os
import pathlib
from collections import OrderedDict

import numpy as np
import tensorrt as trt
import torch
from cuda import cudart
from diffusion_models import PipelineInfo
from engine_builder import EngineBuilder, EngineType
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import (
    CreateConfig,
    ModifyNetworkOutputs,
    Profile,
    engine_from_bytes,
    engine_from_network,
    network_from_onnx_path,
    save_engine,
)

# Map of numpy dtype -> torch dtype
numpy_to_torch_dtype_dict = {
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
}


def _cuda_assert(cuda_ret):
    err = cuda_ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"CUDA ERROR: {err}, error code reference: https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t"
        )
    if len(cuda_ret) > 1:
        return cuda_ret[1]
    return None


class TensorrtEngine:
    def __init__(
        self,
        engine_path,
    ):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()
        self.cuda_graph_instance = None

    def __del__(self):
        del self.engine
        del self.context
        del self.buffers
        del self.tensors

    def build(
        self,
        onnx_path,
        fp16,
        input_profile=None,
        enable_all_tactics=False,
        timing_cache=None,
        update_output_names=None,
    ):
        print(f"Building TensorRT engine for {onnx_path}: {self.engine_path}")
        p = Profile()
        if input_profile:
            for name, dims in input_profile.items():
                assert len(dims) == 3
                p.add(name, min=dims[0], opt=dims[1], max=dims[2])

        config_kwargs = {}
        if not enable_all_tactics:
            config_kwargs["tactic_sources"] = []

        network = network_from_onnx_path(onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM])
        if update_output_names:
            print(f"Updating network outputs to {update_output_names}")
            network = ModifyNetworkOutputs(network, update_output_names)
        engine = engine_from_network(
            network,
            config=CreateConfig(
                fp16=fp16, refittable=False, profiles=[p], load_timing_cache=timing_cache, **config_kwargs
            ),
            save_timing_cache=timing_cache,
        )
        save_engine(engine, path=self.engine_path)

    def load(self):
        print(f"Loading TensorRT engine: {self.engine_path}")
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))

    def activate(self, reuse_device_memory=None):
        if reuse_device_memory:
            self.context = self.engine.create_execution_context_without_device_memory()
            self.context.device_memory = reuse_device_memory
        else:
            self.context = self.engine.create_execution_context()

    def allocate_buffers(self, shape_dict=None, device="cuda"):
        for idx in range(self.engine.num_io_tensors):
            binding = self.engine[idx]
            if shape_dict and binding in shape_dict:
                shape = shape_dict[binding]
            else:
                shape = self.engine.get_binding_shape(binding)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            if self.engine.binding_is_input(binding):
                self.context.set_binding_shape(idx, shape)
            tensor = torch.empty(tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]).to(device=device)
            self.tensors[binding] = tensor

    def infer(self, feed_dict, stream, use_cuda_graph=False):
        for name, buf in feed_dict.items():
            self.tensors[name].copy_(buf)

        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        if use_cuda_graph:
            if self.cuda_graph_instance is not None:
                _cuda_assert(cudart.cudaGraphLaunch(self.cuda_graph_instance, stream))
                _cuda_assert(cudart.cudaStreamSynchronize(stream))
            else:
                # do inference before CUDA graph capture
                noerror = self.context.execute_async_v3(stream)
                if not noerror:
                    raise ValueError("ERROR: inference failed.")
                # capture cuda graph
                _cuda_assert(
                    cudart.cudaStreamBeginCapture(stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
                )
                self.context.execute_async_v3(stream)
                self.graph = _cuda_assert(cudart.cudaStreamEndCapture(stream))

                from cuda import nvrtc

                result, major, minor = nvrtc.nvrtcVersion()
                assert result == nvrtc.nvrtcResult(0)
                if major < 12:
                    self.cuda_graph_instance = _cuda_assert(
                        cudart.cudaGraphInstantiate(self.graph, b"", 0)
                    )  # cuda < 12
                else:
                    self.cuda_graph_instance = _cuda_assert(cudart.cudaGraphInstantiate(self.graph, 0))  # cuda >= 12
        else:
            noerror = self.context.execute_async_v3(stream)
            if not noerror:
                raise ValueError("ERROR: inference failed.")

        return self.tensors


class TensorrtEngineBuilder(EngineBuilder):
    """
    Helper class to hide the detail of TensorRT Engine from pipeline.
    """

    def __init__(
        self,
        pipeline_info: PipelineInfo,
        max_batch_size=16,
        device="cuda",
        use_cuda_graph=False,
    ):
        """
        Initializes the ONNX Runtime TensorRT ExecutionProvider Engine Builder.

        Args:
            pipeline_info (PipelineInfo):
                Version and Type of pipeline.
            max_batch_size (int):
                Maximum batch size for dynamic batch engine.
            device (str):
                device to run.
            use_cuda_graph (bool):
                Use CUDA graph to capture engine execution and then launch inference
        """
        super().__init__(
            EngineType.TRT,
            pipeline_info,
            max_batch_size=max_batch_size,
            device=device,
            use_cuda_graph=use_cuda_graph,
        )

        self.stream = None
        self.shared_device_memory = None

    def load_resources(self, image_height, image_width, batch_size):
        super().load_resources(image_height, image_width, batch_size)

        self.stream = _cuda_assert(cudart.cudaStreamCreate())

    def teardown(self):
        super().teardown()

        if self.shared_device_memory:
            cudart.cudaFree(self.shared_device_memory)

        cudart.cudaStreamDestroy(self.stream)
        del self.stream

    def load_engines(
        self,
        engine_dir,
        framework_model_dir,
        onnx_dir,
        onnx_opset,
        opt_batch_size,
        opt_image_height,
        opt_image_width,
        static_batch=False,
        static_shape=True,
        enable_all_tactics=False,
        timing_cache=None,
    ):
        """
        Build and load engines for TensorRT accelerated inference.
        Export ONNX models first, if applicable.

        Args:
            engine_dir (str):
                Directory to write the TensorRT engines.
            framework_model_dir (str):
                Directory to write the framework model ckpt.
            onnx_dir (str):
                Directory to write the ONNX models.
            onnx_opset (int):
                ONNX opset version to export the models.
            opt_batch_size (int):
                Batch size to optimize for during engine building.
            opt_image_height (int):
                Image height to optimize for during engine building. Must be a multiple of 8.
            opt_image_width (int):
                Image width to optimize for during engine building. Must be a multiple of 8.
            static_batch (bool):
                Build engine only for specified opt_batch_size.
            static_shape (bool):
                Build engine only for specified opt_image_height & opt_image_width. Default = True.
            enable_all_tactics (bool):
                Enable all tactic sources during TensorRT engine builds.
            timing_cache (str):
                Path to the timing cache to accelerate build or None
        """
        # Create directory
        for directory in [engine_dir, onnx_dir]:
            if not os.path.exists(directory):
                print(f"[I] Create directory: {directory}")
                pathlib.Path(directory).mkdir(parents=True)

        self.load_models(framework_model_dir)

        # Load lora only when we need export text encoder or UNet to ONNX.
        load_lora = False
        if self.pipeline_info.lora_weights:
            for model_name, model_obj in self.models.items():
                if model_name not in ["clip", "clip2", "unet", "unetxl"]:
                    continue
                profile_id = model_obj.get_profile_id(
                    opt_batch_size, opt_image_height, opt_image_width, static_batch, static_shape
                )
                engine_path = self.get_engine_path(engine_dir, model_name, profile_id)
                if not os.path.exists(engine_path):
                    onnx_path = self.get_onnx_path(model_name, onnx_dir, opt=False)
                    onnx_opt_path = self.get_onnx_path(model_name, onnx_dir, opt=True)
                    if not os.path.exists(onnx_opt_path):
                        if not os.path.exists(onnx_path):
                            load_lora = True
                            break

        # Export models to ONNX
        self.disable_torch_spda()
        pipe = self.load_pipeline_with_lora() if load_lora else None

        for model_name, model_obj in self.models.items():
            if model_name == "vae" and self.vae_torch_fallback:
                continue
            profile_id = model_obj.get_profile_id(
                opt_batch_size, opt_image_height, opt_image_width, static_batch, static_shape
            )
            engine_path = self.get_engine_path(engine_dir, model_name, profile_id)
            if not os.path.exists(engine_path):
                onnx_path = self.get_onnx_path(model_name, onnx_dir, opt=False)
                onnx_opt_path = self.get_onnx_path(model_name, onnx_dir, opt=True)
                if not os.path.exists(onnx_opt_path):
                    if not os.path.exists(onnx_path):
                        print(f"Exporting model: {onnx_path}")
                        model = self.get_or_load_model(pipe, model_name, model_obj, framework_model_dir)

                        with torch.inference_mode(), torch.autocast("cuda"):
                            inputs = model_obj.get_sample_input(1, opt_image_height, opt_image_width)
                            torch.onnx.export(
                                model,
                                inputs,
                                onnx_path,
                                export_params=True,
                                opset_version=onnx_opset,
                                do_constant_folding=True,
                                input_names=model_obj.get_input_names(),
                                output_names=model_obj.get_output_names(),
                                dynamic_axes=model_obj.get_dynamic_axes(),
                            )
                        del model
                        torch.cuda.empty_cache()
                        gc.collect()
                    else:
                        print(f"Found cached model: {onnx_path}")

                    # Optimize onnx
                    if not os.path.exists(onnx_opt_path):
                        print(f"Generating optimizing model: {onnx_opt_path}")
                        model_obj.optimize_trt(onnx_path, onnx_opt_path)
                    else:
                        print(f"Found cached optimized model: {onnx_opt_path} ")
        self.enable_torch_spda()

        # Build TensorRT engines
        for model_name, model_obj in self.models.items():
            if model_name == "vae" and self.vae_torch_fallback:
                continue
            profile_id = model_obj.get_profile_id(
                opt_batch_size, opt_image_height, opt_image_width, static_batch, static_shape
            )
            engine_path = self.get_engine_path(engine_dir, model_name, profile_id)
            engine = TensorrtEngine(engine_path)
            onnx_opt_path = self.get_onnx_path(model_name, onnx_dir, opt=True)

            if not os.path.exists(engine.engine_path):
                engine.build(
                    onnx_opt_path,
                    fp16=True,
                    input_profile=model_obj.get_input_profile(
                        opt_batch_size,
                        opt_image_height,
                        opt_image_width,
                        static_batch,
                        static_shape,
                    ),
                    enable_all_tactics=enable_all_tactics,
                    timing_cache=timing_cache,
                    update_output_names=None,
                )
            self.engines[model_name] = engine

        # Load TensorRT engines
        for model_name in self.models:
            if model_name == "vae" and self.vae_torch_fallback:
                continue
            self.engines[model_name].load()

    def max_device_memory(self):
        max_device_memory = 0
        for engine in self.engines.values():
            max_device_memory = max(max_device_memory, engine.engine.device_memory_size)
        return max_device_memory

    def activate_engines(self, shared_device_memory=None):
        if shared_device_memory is None:
            max_device_memory = self.max_device_memory()
            _, shared_device_memory = cudart.cudaMalloc(max_device_memory)
        self.shared_device_memory = shared_device_memory
        # Load and activate TensorRT engines
        for engine in self.engines.values():
            engine.activate(reuse_device_memory=self.shared_device_memory)

    def run_engine(self, model_name, feed_dict):
        return self.engines[model_name].infer(feed_dict, self.stream, use_cuda_graph=self.use_cuda_graph)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v135\heap_profiler.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: HeapProfiler (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime


class HeapSnapshotObjectId(str):
    '''
    Heap snapshot object id.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> HeapSnapshotObjectId:
        return cls(json)

    def __repr__(self):
        return 'HeapSnapshotObjectId({})'.format(super().__repr__())


@dataclass
class SamplingHeapProfileNode:
    '''
    Sampling Heap Profile node. Holds callsite information, allocation statistics and child nodes.
    '''
    #: Function location.
    call_frame: runtime.CallFrame

    #: Allocations size in bytes for the node excluding children.
    self_size: float

    #: Node id. Ids are unique across all profiles collected between startSampling and stopSampling.
    id_: int

    #: Child nodes.
    children: typing.List[SamplingHeapProfileNode]

    def to_json(self):
        json = dict()
        json['callFrame'] = self.call_frame.to_json()
        json['selfSize'] = self.self_size
        json['id'] = self.id_
        json['children'] = [i.to_json() for i in self.children]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            call_frame=runtime.CallFrame.from_json(json['callFrame']),
            self_size=float(json['selfSize']),
            id_=int(json['id']),
            children=[SamplingHeapProfileNode.from_json(i) for i in json['children']],
        )


@dataclass
class SamplingHeapProfileSample:
    '''
    A single sample from a sampling profile.
    '''
    #: Allocation size in bytes attributed to the sample.
    size: float

    #: Id of the corresponding profile tree node.
    node_id: int

    #: Time-ordered sample ordinal number. It is unique across all profiles retrieved
    #: between startSampling and stopSampling.
    ordinal: float

    def to_json(self):
        json = dict()
        json['size'] = self.size
        json['nodeId'] = self.node_id
        json['ordinal'] = self.ordinal
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            size=float(json['size']),
            node_id=int(json['nodeId']),
            ordinal=float(json['ordinal']),
        )


@dataclass
class SamplingHeapProfile:
    '''
    Sampling profile.
    '''
    head: SamplingHeapProfileNode

    samples: typing.List[SamplingHeapProfileSample]

    def to_json(self):
        json = dict()
        json['head'] = self.head.to_json()
        json['samples'] = [i.to_json() for i in self.samples]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            head=SamplingHeapProfileNode.from_json(json['head']),
            samples=[SamplingHeapProfileSample.from_json(i) for i in json['samples']],
        )


def add_inspected_heap_object(
        heap_object_id: HeapSnapshotObjectId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables console to refer to the node with given id via $x (see Command Line API for more details
    $x functions).

    :param heap_object_id: Heap snapshot object id to be accessible by means of $x command line API.
    '''
    params: T_JSON_DICT = dict()
    params['heapObjectId'] = heap_object_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'HeapProfiler.addInspectedHeapObject',
        'params': params,
    }
    json = yield cmd_dict


def collect_garbage() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:

    cmd_dict: T_JSON_DICT = {
        'method': 'HeapProfiler.collectGarbage',
    }
    json = yield cmd_dict


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:

    cmd_dict: T_JSON_DICT = {
        'method': 'HeapProfiler.disable',
    }
    json = yield cmd_dict


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:

    cmd_dict: T_JSON_DICT = {
        'method': 'HeapProfiler.enable',
    }
    json = yield cmd_dict


def get_heap_object_id(
        object_id: runtime.RemoteObjectId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,HeapSnapshotObjectId]:
    '''
    :param object_id: Identifier of the object to get heap object id for.
    :returns: Id of the heap snapshot object corresponding to the passed remote object id.
    '''
    params: T_JSON_DICT = dict()
    params['objectId'] = object_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'HeapProfiler.getHeapObjectId',
        'params': params,
    }
    json = yield cmd_dict
    return HeapSnapshotObjectId.from_json(json['heapSnapshotObjectId'])


def get_object_by_heap_object_id(
        object_id: HeapSnapshotObjectId,
        object_group: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,runtime.RemoteObject]:
    '''
    :param object_id:
    :param object_group: *(Optional)* Symbolic group name that can be used to release multiple objects.
    :returns: Evaluation result.
    '''
    params: T_JSON_DICT = dict()
    params['objectId'] = object_id.to_json()
    if object_group is not None:
        params['objectGroup'] = object_group
    cmd_dict: T_JSON_DICT = {
        'method': 'HeapProfiler.getObjectByHeapObjectId',
        'params': params,
    }
    json = yield cmd_dict
    return runtime.RemoteObject.from_json(json['result'])


def get_sampling_profile() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,SamplingHeapProfile]:
    '''


    :returns: Return the sampling profile being collected.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'HeapProfiler.getSamplingProfile',
    }
    json = yield cmd_dict
    return SamplingHeapProfile.from_json(json['profile'])


def start_sampling(
        sampling_interval: typing.Optional[float] = None,
        include_objects_collected_by_major_gc: typing.Optional[bool] = None,
        include_objects_collected_by_minor_gc: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param sampling_interval: *(Optional)* Average sample interval in bytes. Poisson distribution is used for the intervals. The default value is 32768 bytes.
    :param include_objects_collected_by_major_gc: *(Optional)* By default, the sampling heap profiler reports only objects which are still alive when the profile is returned via getSamplingProfile or stopSampling, which is useful for determining what functions contribute the most to steady-state memory usage. This flag instructs the sampling heap profiler to also include information about objects discarded by major GC, which will show which functions cause large temporary memory usage or long GC pauses.
    :param include_objects_collected_by_minor_gc: *(Optional)* By default, the sampling heap profiler reports only objects which are still alive when the profile is returned via getSamplingProfile or stopSampling, which is useful for determining what functions contribute the most to steady-state memory usage. This flag instructs the sampling heap profiler to also include information about objects discarded by minor GC, which is useful when tuning a latency-sensitive application for minimal GC activity.
    '''
    params: T_JSON_DICT = dict()
    if sampling_interval is not None:
        params['samplingInterval'] = sampling_interval
    if include_objects_collected_by_major_gc is not None:
        params['includeObjectsCollectedByMajorGC'] = include_objects_collected_by_major_gc
    if include_objects_collected_by_minor_gc is not None:
        params['includeObjectsCollectedByMinorGC'] = include_objects_collected_by_minor_gc
    cmd_dict: T_JSON_DICT = {
        'method': 'HeapProfiler.startSampling',
        'params': params,
    }
    json = yield cmd_dict


def start_tracking_heap_objects(
        track_allocations: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param track_allocations: *(Optional)*
    '''
    params: T_JSON_DICT = dict()
    if track_allocations is not None:
        params['trackAllocations'] = track_allocations
    cmd_dict: T_JSON_DICT = {
        'method': 'HeapProfiler.startTrackingHeapObjects',
        'params': params,
    }
    json = yield cmd_dict


def stop_sampling() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,SamplingHeapProfile]:
    '''


    :returns: Recorded sampling heap profile.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'HeapProfiler.stopSampling',
    }
    json = yield cmd_dict
    return SamplingHeapProfile.from_json(json['profile'])


def stop_tracking_heap_objects(
        report_progress: typing.Optional[bool] = None,
        treat_global_objects_as_roots: typing.Optional[bool] = None,
        capture_numeric_value: typing.Optional[bool] = None,
        expose_internals: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param report_progress: *(Optional)* If true 'reportHeapSnapshotProgress' events will be generated while snapshot is being taken when the tracking is stopped.
    :param treat_global_objects_as_roots: *(Optional)* Deprecated in favor of ```exposeInternals```.
    :param capture_numeric_value: *(Optional)* If true, numerical values are included in the snapshot
    :param expose_internals: **(EXPERIMENTAL)** *(Optional)* If true, exposes internals of the snapshot.
    '''
    params: T_JSON_DICT = dict()
    if report_progress is not None:
        params['reportProgress'] = report_progress
    if treat_global_objects_as_roots is not None:
        params['treatGlobalObjectsAsRoots'] = treat_global_objects_as_roots
    if capture_numeric_value is not None:
        params['captureNumericValue'] = capture_numeric_value
    if expose_internals is not None:
        params['exposeInternals'] = expose_internals
    cmd_dict: T_JSON_DICT = {
        'method': 'HeapProfiler.stopTrackingHeapObjects',
        'params': params,
    }
    json = yield cmd_dict


def take_heap_snapshot(
        report_progress: typing.Optional[bool] = None,
        treat_global_objects_as_roots: typing.Optional[bool] = None,
        capture_numeric_value: typing.Optional[bool] = None,
        expose_internals: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param report_progress: *(Optional)* If true 'reportHeapSnapshotProgress' events will be generated while snapshot is being taken.
    :param treat_global_objects_as_roots: *(Optional)* If true, a raw snapshot without artificial roots will be generated. Deprecated in favor of ```exposeInternals```.
    :param capture_numeric_value: *(Optional)* If true, numerical values are included in the snapshot
    :param expose_internals: **(EXPERIMENTAL)** *(Optional)* If true, exposes internals of the snapshot.
    '''
    params: T_JSON_DICT = dict()
    if report_progress is not None:
        params['reportProgress'] = report_progress
    if treat_global_objects_as_roots is not None:
        params['treatGlobalObjectsAsRoots'] = treat_global_objects_as_roots
    if capture_numeric_value is not None:
        params['captureNumericValue'] = capture_numeric_value
    if expose_internals is not None:
        params['exposeInternals'] = expose_internals
    cmd_dict: T_JSON_DICT = {
        'method': 'HeapProfiler.takeHeapSnapshot',
        'params': params,
    }
    json = yield cmd_dict


@event_class('HeapProfiler.addHeapSnapshotChunk')
@dataclass
class AddHeapSnapshotChunk:
    chunk: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AddHeapSnapshotChunk:
        return cls(
            chunk=str(json['chunk'])
        )


@event_class('HeapProfiler.heapStatsUpdate')
@dataclass
class HeapStatsUpdate:
    '''
    If heap objects tracking has been started then backend may send update for one or more fragments
    '''
    #: An array of triplets. Each triplet describes a fragment. The first integer is the fragment
    #: index, the second integer is a total count of objects for the fragment, the third integer is
    #: a total size of the objects for the fragment.
    stats_update: typing.List[int]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> HeapStatsUpdate:
        return cls(
            stats_update=[int(i) for i in json['statsUpdate']]
        )


@event_class('HeapProfiler.lastSeenObjectId')
@dataclass
class LastSeenObjectId:
    '''
    If heap objects tracking has been started then backend regularly sends a current value for last
    seen object id and corresponding timestamp. If the were changes in the heap since last event
    then one or more heapStatsUpdate events will be sent before a new lastSeenObjectId event.
    '''
    last_seen_object_id: int
    timestamp: float

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> LastSeenObjectId:
        return cls(
            last_seen_object_id=int(json['lastSeenObjectId']),
            timestamp=float(json['timestamp'])
        )


@event_class('HeapProfiler.reportHeapSnapshotProgress')
@dataclass
class ReportHeapSnapshotProgress:
    done: int
    total: int
    finished: typing.Optional[bool]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ReportHeapSnapshotProgress:
        return cls(
            done=int(json['done']),
            total=int(json['total']),
            finished=bool(json['finished']) if 'finished' in json else None
        )


@event_class('HeapProfiler.resetProfiles')
@dataclass
class ResetProfiles:


    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ResetProfiles:
        return cls(

        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v136\heap_profiler.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: HeapProfiler (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime


class HeapSnapshotObjectId(str):
    '''
    Heap snapshot object id.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> HeapSnapshotObjectId:
        return cls(json)

    def __repr__(self):
        return 'HeapSnapshotObjectId({})'.format(super().__repr__())


@dataclass
class SamplingHeapProfileNode:
    '''
    Sampling Heap Profile node. Holds callsite information, allocation statistics and child nodes.
    '''
    #: Function location.
    call_frame: runtime.CallFrame

    #: Allocations size in bytes for the node excluding children.
    self_size: float

    #: Node id. Ids are unique across all profiles collected between startSampling and stopSampling.
    id_: int

    #: Child nodes.
    children: typing.List[SamplingHeapProfileNode]

    def to_json(self):
        json = dict()
        json['callFrame'] = self.call_frame.to_json()
        json['selfSize'] = self.self_size
        json['id'] = self.id_
        json['children'] = [i.to_json() for i in self.children]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            call_frame=runtime.CallFrame.from_json(json['callFrame']),
            self_size=float(json['selfSize']),
            id_=int(json['id']),
            children=[SamplingHeapProfileNode.from_json(i) for i in json['children']],
        )


@dataclass
class SamplingHeapProfileSample:
    '''
    A single sample from a sampling profile.
    '''
    #: Allocation size in bytes attributed to the sample.
    size: float

    #: Id of the corresponding profile tree node.
    node_id: int

    #: Time-ordered sample ordinal number. It is unique across all profiles retrieved
    #: between startSampling and stopSampling.
    ordinal: float

    def to_json(self):
        json = dict()
        json['size'] = self.size
        json['nodeId'] = self.node_id
        json['ordinal'] = self.ordinal
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            size=float(json['size']),
            node_id=int(json['nodeId']),
            ordinal=float(json['ordinal']),
        )


@dataclass
class SamplingHeapProfile:
    '''
    Sampling profile.
    '''
    head: SamplingHeapProfileNode

    samples: typing.List[SamplingHeapProfileSample]

    def to_json(self):
        json = dict()
        json['head'] = self.head.to_json()
        json['samples'] = [i.to_json() for i in self.samples]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            head=SamplingHeapProfileNode.from_json(json['head']),
            samples=[SamplingHeapProfileSample.from_json(i) for i in json['samples']],
        )


def add_inspected_heap_object(
        heap_object_id: HeapSnapshotObjectId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables console to refer to the node with given id via $x (see Command Line API for more details
    $x functions).

    :param heap_object_id: Heap snapshot object id to be accessible by means of $x command line API.
    '''
    params: T_JSON_DICT = dict()
    params['heapObjectId'] = heap_object_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'HeapProfiler.addInspectedHeapObject',
        'params': params,
    }
    json = yield cmd_dict


def collect_garbage() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:

    cmd_dict: T_JSON_DICT = {
        'method': 'HeapProfiler.collectGarbage',
    }
    json = yield cmd_dict


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:

    cmd_dict: T_JSON_DICT = {
        'method': 'HeapProfiler.disable',
    }
    json = yield cmd_dict


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:

    cmd_dict: T_JSON_DICT = {
        'method': 'HeapProfiler.enable',
    }
    json = yield cmd_dict


def get_heap_object_id(
        object_id: runtime.RemoteObjectId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,HeapSnapshotObjectId]:
    '''
    :param object_id: Identifier of the object to get heap object id for.
    :returns: Id of the heap snapshot object corresponding to the passed remote object id.
    '''
    params: T_JSON_DICT = dict()
    params['objectId'] = object_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'HeapProfiler.getHeapObjectId',
        'params': params,
    }
    json = yield cmd_dict
    return HeapSnapshotObjectId.from_json(json['heapSnapshotObjectId'])


def get_object_by_heap_object_id(
        object_id: HeapSnapshotObjectId,
        object_group: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,runtime.RemoteObject]:
    '''
    :param object_id:
    :param object_group: *(Optional)* Symbolic group name that can be used to release multiple objects.
    :returns: Evaluation result.
    '''
    params: T_JSON_DICT = dict()
    params['objectId'] = object_id.to_json()
    if object_group is not None:
        params['objectGroup'] = object_group
    cmd_dict: T_JSON_DICT = {
        'method': 'HeapProfiler.getObjectByHeapObjectId',
        'params': params,
    }
    json = yield cmd_dict
    return runtime.RemoteObject.from_json(json['result'])


def get_sampling_profile() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,SamplingHeapProfile]:
    '''


    :returns: Return the sampling profile being collected.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'HeapProfiler.getSamplingProfile',
    }
    json = yield cmd_dict
    return SamplingHeapProfile.from_json(json['profile'])


def start_sampling(
        sampling_interval: typing.Optional[float] = None,
        include_objects_collected_by_major_gc: typing.Optional[bool] = None,
        include_objects_collected_by_minor_gc: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param sampling_interval: *(Optional)* Average sample interval in bytes. Poisson distribution is used for the intervals. The default value is 32768 bytes.
    :param include_objects_collected_by_major_gc: *(Optional)* By default, the sampling heap profiler reports only objects which are still alive when the profile is returned via getSamplingProfile or stopSampling, which is useful for determining what functions contribute the most to steady-state memory usage. This flag instructs the sampling heap profiler to also include information about objects discarded by major GC, which will show which functions cause large temporary memory usage or long GC pauses.
    :param include_objects_collected_by_minor_gc: *(Optional)* By default, the sampling heap profiler reports only objects which are still alive when the profile is returned via getSamplingProfile or stopSampling, which is useful for determining what functions contribute the most to steady-state memory usage. This flag instructs the sampling heap profiler to also include information about objects discarded by minor GC, which is useful when tuning a latency-sensitive application for minimal GC activity.
    '''
    params: T_JSON_DICT = dict()
    if sampling_interval is not None:
        params['samplingInterval'] = sampling_interval
    if include_objects_collected_by_major_gc is not None:
        params['includeObjectsCollectedByMajorGC'] = include_objects_collected_by_major_gc
    if include_objects_collected_by_minor_gc is not None:
        params['includeObjectsCollectedByMinorGC'] = include_objects_collected_by_minor_gc
    cmd_dict: T_JSON_DICT = {
        'method': 'HeapProfiler.startSampling',
        'params': params,
    }
    json = yield cmd_dict


def start_tracking_heap_objects(
        track_allocations: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param track_allocations: *(Optional)*
    '''
    params: T_JSON_DICT = dict()
    if track_allocations is not None:
        params['trackAllocations'] = track_allocations
    cmd_dict: T_JSON_DICT = {
        'method': 'HeapProfiler.startTrackingHeapObjects',
        'params': params,
    }
    json = yield cmd_dict


def stop_sampling() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,SamplingHeapProfile]:
    '''


    :returns: Recorded sampling heap profile.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'HeapProfiler.stopSampling',
    }
    json = yield cmd_dict
    return SamplingHeapProfile.from_json(json['profile'])


def stop_tracking_heap_objects(
        report_progress: typing.Optional[bool] = None,
        treat_global_objects_as_roots: typing.Optional[bool] = None,
        capture_numeric_value: typing.Optional[bool] = None,
        expose_internals: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param report_progress: *(Optional)* If true 'reportHeapSnapshotProgress' events will be generated while snapshot is being taken when the tracking is stopped.
    :param treat_global_objects_as_roots: *(Optional)* Deprecated in favor of ```exposeInternals```.
    :param capture_numeric_value: *(Optional)* If true, numerical values are included in the snapshot
    :param expose_internals: **(EXPERIMENTAL)** *(Optional)* If true, exposes internals of the snapshot.
    '''
    params: T_JSON_DICT = dict()
    if report_progress is not None:
        params['reportProgress'] = report_progress
    if treat_global_objects_as_roots is not None:
        params['treatGlobalObjectsAsRoots'] = treat_global_objects_as_roots
    if capture_numeric_value is not None:
        params['captureNumericValue'] = capture_numeric_value
    if expose_internals is not None:
        params['exposeInternals'] = expose_internals
    cmd_dict: T_JSON_DICT = {
        'method': 'HeapProfiler.stopTrackingHeapObjects',
        'params': params,
    }
    json = yield cmd_dict


def take_heap_snapshot(
        report_progress: typing.Optional[bool] = None,
        treat_global_objects_as_roots: typing.Optional[bool] = None,
        capture_numeric_value: typing.Optional[bool] = None,
        expose_internals: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param report_progress: *(Optional)* If true 'reportHeapSnapshotProgress' events will be generated while snapshot is being taken.
    :param treat_global_objects_as_roots: *(Optional)* If true, a raw snapshot without artificial roots will be generated. Deprecated in favor of ```exposeInternals```.
    :param capture_numeric_value: *(Optional)* If true, numerical values are included in the snapshot
    :param expose_internals: **(EXPERIMENTAL)** *(Optional)* If true, exposes internals of the snapshot.
    '''
    params: T_JSON_DICT = dict()
    if report_progress is not None:
        params['reportProgress'] = report_progress
    if treat_global_objects_as_roots is not None:
        params['treatGlobalObjectsAsRoots'] = treat_global_objects_as_roots
    if capture_numeric_value is not None:
        params['captureNumericValue'] = capture_numeric_value
    if expose_internals is not None:
        params['exposeInternals'] = expose_internals
    cmd_dict: T_JSON_DICT = {
        'method': 'HeapProfiler.takeHeapSnapshot',
        'params': params,
    }
    json = yield cmd_dict


@event_class('HeapProfiler.addHeapSnapshotChunk')
@dataclass
class AddHeapSnapshotChunk:
    chunk: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AddHeapSnapshotChunk:
        return cls(
            chunk=str(json['chunk'])
        )


@event_class('HeapProfiler.heapStatsUpdate')
@dataclass
class HeapStatsUpdate:
    '''
    If heap objects tracking has been started then backend may send update for one or more fragments
    '''
    #: An array of triplets. Each triplet describes a fragment. The first integer is the fragment
    #: index, the second integer is a total count of objects for the fragment, the third integer is
    #: a total size of the objects for the fragment.
    stats_update: typing.List[int]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> HeapStatsUpdate:
        return cls(
            stats_update=[int(i) for i in json['statsUpdate']]
        )


@event_class('HeapProfiler.lastSeenObjectId')
@dataclass
class LastSeenObjectId:
    '''
    If heap objects tracking has been started then backend regularly sends a current value for last
    seen object id and corresponding timestamp. If the were changes in the heap since last event
    then one or more heapStatsUpdate events will be sent before a new lastSeenObjectId event.
    '''
    last_seen_object_id: int
    timestamp: float

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> LastSeenObjectId:
        return cls(
            last_seen_object_id=int(json['lastSeenObjectId']),
            timestamp=float(json['timestamp'])
        )


@event_class('HeapProfiler.reportHeapSnapshotProgress')
@dataclass
class ReportHeapSnapshotProgress:
    done: int
    total: int
    finished: typing.Optional[bool]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ReportHeapSnapshotProgress:
        return cls(
            done=int(json['done']),
            total=int(json['total']),
            finished=bool(json['finished']) if 'finished' in json else None
        )


@event_class('HeapProfiler.resetProfiles')
@dataclass
class ResetProfiles:


    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ResetProfiles:
        return cls(

        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v137\heap_profiler.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: HeapProfiler (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime


class HeapSnapshotObjectId(str):
    '''
    Heap snapshot object id.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> HeapSnapshotObjectId:
        return cls(json)

    def __repr__(self):
        return 'HeapSnapshotObjectId({})'.format(super().__repr__())


@dataclass
class SamplingHeapProfileNode:
    '''
    Sampling Heap Profile node. Holds callsite information, allocation statistics and child nodes.
    '''
    #: Function location.
    call_frame: runtime.CallFrame

    #: Allocations size in bytes for the node excluding children.
    self_size: float

    #: Node id. Ids are unique across all profiles collected between startSampling and stopSampling.
    id_: int

    #: Child nodes.
    children: typing.List[SamplingHeapProfileNode]

    def to_json(self):
        json = dict()
        json['callFrame'] = self.call_frame.to_json()
        json['selfSize'] = self.self_size
        json['id'] = self.id_
        json['children'] = [i.to_json() for i in self.children]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            call_frame=runtime.CallFrame.from_json(json['callFrame']),
            self_size=float(json['selfSize']),
            id_=int(json['id']),
            children=[SamplingHeapProfileNode.from_json(i) for i in json['children']],
        )


@dataclass
class SamplingHeapProfileSample:
    '''
    A single sample from a sampling profile.
    '''
    #: Allocation size in bytes attributed to the sample.
    size: float

    #: Id of the corresponding profile tree node.
    node_id: int

    #: Time-ordered sample ordinal number. It is unique across all profiles retrieved
    #: between startSampling and stopSampling.
    ordinal: float

    def to_json(self):
        json = dict()
        json['size'] = self.size
        json['nodeId'] = self.node_id
        json['ordinal'] = self.ordinal
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            size=float(json['size']),
            node_id=int(json['nodeId']),
            ordinal=float(json['ordinal']),
        )


@dataclass
class SamplingHeapProfile:
    '''
    Sampling profile.
    '''
    head: SamplingHeapProfileNode

    samples: typing.List[SamplingHeapProfileSample]

    def to_json(self):
        json = dict()
        json['head'] = self.head.to_json()
        json['samples'] = [i.to_json() for i in self.samples]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            head=SamplingHeapProfileNode.from_json(json['head']),
            samples=[SamplingHeapProfileSample.from_json(i) for i in json['samples']],
        )


def add_inspected_heap_object(
        heap_object_id: HeapSnapshotObjectId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables console to refer to the node with given id via $x (see Command Line API for more details
    $x functions).

    :param heap_object_id: Heap snapshot object id to be accessible by means of $x command line API.
    '''
    params: T_JSON_DICT = dict()
    params['heapObjectId'] = heap_object_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'HeapProfiler.addInspectedHeapObject',
        'params': params,
    }
    json = yield cmd_dict


def collect_garbage() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:

    cmd_dict: T_JSON_DICT = {
        'method': 'HeapProfiler.collectGarbage',
    }
    json = yield cmd_dict


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:

    cmd_dict: T_JSON_DICT = {
        'method': 'HeapProfiler.disable',
    }
    json = yield cmd_dict


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:

    cmd_dict: T_JSON_DICT = {
        'method': 'HeapProfiler.enable',
    }
    json = yield cmd_dict


def get_heap_object_id(
        object_id: runtime.RemoteObjectId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,HeapSnapshotObjectId]:
    '''
    :param object_id: Identifier of the object to get heap object id for.
    :returns: Id of the heap snapshot object corresponding to the passed remote object id.
    '''
    params: T_JSON_DICT = dict()
    params['objectId'] = object_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'HeapProfiler.getHeapObjectId',
        'params': params,
    }
    json = yield cmd_dict
    return HeapSnapshotObjectId.from_json(json['heapSnapshotObjectId'])


def get_object_by_heap_object_id(
        object_id: HeapSnapshotObjectId,
        object_group: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,runtime.RemoteObject]:
    '''
    :param object_id:
    :param object_group: *(Optional)* Symbolic group name that can be used to release multiple objects.
    :returns: Evaluation result.
    '''
    params: T_JSON_DICT = dict()
    params['objectId'] = object_id.to_json()
    if object_group is not None:
        params['objectGroup'] = object_group
    cmd_dict: T_JSON_DICT = {
        'method': 'HeapProfiler.getObjectByHeapObjectId',
        'params': params,
    }
    json = yield cmd_dict
    return runtime.RemoteObject.from_json(json['result'])


def get_sampling_profile() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,SamplingHeapProfile]:
    '''


    :returns: Return the sampling profile being collected.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'HeapProfiler.getSamplingProfile',
    }
    json = yield cmd_dict
    return SamplingHeapProfile.from_json(json['profile'])


def start_sampling(
        sampling_interval: typing.Optional[float] = None,
        include_objects_collected_by_major_gc: typing.Optional[bool] = None,
        include_objects_collected_by_minor_gc: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param sampling_interval: *(Optional)* Average sample interval in bytes. Poisson distribution is used for the intervals. The default value is 32768 bytes.
    :param include_objects_collected_by_major_gc: *(Optional)* By default, the sampling heap profiler reports only objects which are still alive when the profile is returned via getSamplingProfile or stopSampling, which is useful for determining what functions contribute the most to steady-state memory usage. This flag instructs the sampling heap profiler to also include information about objects discarded by major GC, which will show which functions cause large temporary memory usage or long GC pauses.
    :param include_objects_collected_by_minor_gc: *(Optional)* By default, the sampling heap profiler reports only objects which are still alive when the profile is returned via getSamplingProfile or stopSampling, which is useful for determining what functions contribute the most to steady-state memory usage. This flag instructs the sampling heap profiler to also include information about objects discarded by minor GC, which is useful when tuning a latency-sensitive application for minimal GC activity.
    '''
    params: T_JSON_DICT = dict()
    if sampling_interval is not None:
        params['samplingInterval'] = sampling_interval
    if include_objects_collected_by_major_gc is not None:
        params['includeObjectsCollectedByMajorGC'] = include_objects_collected_by_major_gc
    if include_objects_collected_by_minor_gc is not None:
        params['includeObjectsCollectedByMinorGC'] = include_objects_collected_by_minor_gc
    cmd_dict: T_JSON_DICT = {
        'method': 'HeapProfiler.startSampling',
        'params': params,
    }
    json = yield cmd_dict


def start_tracking_heap_objects(
        track_allocations: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param track_allocations: *(Optional)*
    '''
    params: T_JSON_DICT = dict()
    if track_allocations is not None:
        params['trackAllocations'] = track_allocations
    cmd_dict: T_JSON_DICT = {
        'method': 'HeapProfiler.startTrackingHeapObjects',
        'params': params,
    }
    json = yield cmd_dict


def stop_sampling() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,SamplingHeapProfile]:
    '''


    :returns: Recorded sampling heap profile.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'HeapProfiler.stopSampling',
    }
    json = yield cmd_dict
    return SamplingHeapProfile.from_json(json['profile'])


def stop_tracking_heap_objects(
        report_progress: typing.Optional[bool] = None,
        treat_global_objects_as_roots: typing.Optional[bool] = None,
        capture_numeric_value: typing.Optional[bool] = None,
        expose_internals: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param report_progress: *(Optional)* If true 'reportHeapSnapshotProgress' events will be generated while snapshot is being taken when the tracking is stopped.
    :param treat_global_objects_as_roots: *(Optional)* Deprecated in favor of ```exposeInternals```.
    :param capture_numeric_value: *(Optional)* If true, numerical values are included in the snapshot
    :param expose_internals: **(EXPERIMENTAL)** *(Optional)* If true, exposes internals of the snapshot.
    '''
    params: T_JSON_DICT = dict()
    if report_progress is not None:
        params['reportProgress'] = report_progress
    if treat_global_objects_as_roots is not None:
        params['treatGlobalObjectsAsRoots'] = treat_global_objects_as_roots
    if capture_numeric_value is not None:
        params['captureNumericValue'] = capture_numeric_value
    if expose_internals is not None:
        params['exposeInternals'] = expose_internals
    cmd_dict: T_JSON_DICT = {
        'method': 'HeapProfiler.stopTrackingHeapObjects',
        'params': params,
    }
    json = yield cmd_dict


def take_heap_snapshot(
        report_progress: typing.Optional[bool] = None,
        treat_global_objects_as_roots: typing.Optional[bool] = None,
        capture_numeric_value: typing.Optional[bool] = None,
        expose_internals: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param report_progress: *(Optional)* If true 'reportHeapSnapshotProgress' events will be generated while snapshot is being taken.
    :param treat_global_objects_as_roots: *(Optional)* If true, a raw snapshot without artificial roots will be generated. Deprecated in favor of ```exposeInternals```.
    :param capture_numeric_value: *(Optional)* If true, numerical values are included in the snapshot
    :param expose_internals: **(EXPERIMENTAL)** *(Optional)* If true, exposes internals of the snapshot.
    '''
    params: T_JSON_DICT = dict()
    if report_progress is not None:
        params['reportProgress'] = report_progress
    if treat_global_objects_as_roots is not None:
        params['treatGlobalObjectsAsRoots'] = treat_global_objects_as_roots
    if capture_numeric_value is not None:
        params['captureNumericValue'] = capture_numeric_value
    if expose_internals is not None:
        params['exposeInternals'] = expose_internals
    cmd_dict: T_JSON_DICT = {
        'method': 'HeapProfiler.takeHeapSnapshot',
        'params': params,
    }
    json = yield cmd_dict


@event_class('HeapProfiler.addHeapSnapshotChunk')
@dataclass
class AddHeapSnapshotChunk:
    chunk: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AddHeapSnapshotChunk:
        return cls(
            chunk=str(json['chunk'])
        )


@event_class('HeapProfiler.heapStatsUpdate')
@dataclass
class HeapStatsUpdate:
    '''
    If heap objects tracking has been started then backend may send update for one or more fragments
    '''
    #: An array of triplets. Each triplet describes a fragment. The first integer is the fragment
    #: index, the second integer is a total count of objects for the fragment, the third integer is
    #: a total size of the objects for the fragment.
    stats_update: typing.List[int]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> HeapStatsUpdate:
        return cls(
            stats_update=[int(i) for i in json['statsUpdate']]
        )


@event_class('HeapProfiler.lastSeenObjectId')
@dataclass
class LastSeenObjectId:
    '''
    If heap objects tracking has been started then backend regularly sends a current value for last
    seen object id and corresponding timestamp. If the were changes in the heap since last event
    then one or more heapStatsUpdate events will be sent before a new lastSeenObjectId event.
    '''
    last_seen_object_id: int
    timestamp: float

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> LastSeenObjectId:
        return cls(
            last_seen_object_id=int(json['lastSeenObjectId']),
            timestamp=float(json['timestamp'])
        )


@event_class('HeapProfiler.reportHeapSnapshotProgress')
@dataclass
class ReportHeapSnapshotProgress:
    done: int
    total: int
    finished: typing.Optional[bool]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ReportHeapSnapshotProgress:
        return cls(
            done=int(json['done']),
            total=int(json['total']),
            finished=bool(json['finished']) if 'finished' in json else None
        )


@event_class('HeapProfiler.resetProfiles')
@dataclass
class ResetProfiles:


    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ResetProfiles:
        return cls(

        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\agca\ideals.py ===
"""Computations with ideals of polynomial rings."""

from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polyutils import IntegerPowerable


class Ideal(IntegerPowerable):
    """
    Abstract base class for ideals.

    Do not instantiate - use explicit constructors in the ring class instead:

    >>> from sympy import QQ
    >>> from sympy.abc import x
    >>> QQ.old_poly_ring(x).ideal(x+1)
    <x + 1>

    Attributes

    - ring - the ring this ideal belongs to

    Non-implemented methods:

    - _contains_elem
    - _contains_ideal
    - _quotient
    - _intersect
    - _union
    - _product
    - is_whole_ring
    - is_zero
    - is_prime, is_maximal, is_primary, is_radical
    - is_principal
    - height, depth
    - radical

    Methods that likely should be overridden in subclasses:

    - reduce_element
    """

    def _contains_elem(self, x):
        """Implementation of element containment."""
        raise NotImplementedError

    def _contains_ideal(self, I):
        """Implementation of ideal containment."""
        raise NotImplementedError

    def _quotient(self, J):
        """Implementation of ideal quotient."""
        raise NotImplementedError

    def _intersect(self, J):
        """Implementation of ideal intersection."""
        raise NotImplementedError

    def is_whole_ring(self):
        """Return True if ``self`` is the whole ring."""
        raise NotImplementedError

    def is_zero(self):
        """Return True if ``self`` is the zero ideal."""
        raise NotImplementedError

    def _equals(self, J):
        """Implementation of ideal equality."""
        return self._contains_ideal(J) and J._contains_ideal(self)

    def is_prime(self):
        """Return True if ``self`` is a prime ideal."""
        raise NotImplementedError

    def is_maximal(self):
        """Return True if ``self`` is a maximal ideal."""
        raise NotImplementedError

    def is_radical(self):
        """Return True if ``self`` is a radical ideal."""
        raise NotImplementedError

    def is_primary(self):
        """Return True if ``self`` is a primary ideal."""
        raise NotImplementedError

    def is_principal(self):
        """Return True if ``self`` is a principal ideal."""
        raise NotImplementedError

    def radical(self):
        """Compute the radical of ``self``."""
        raise NotImplementedError

    def depth(self):
        """Compute the depth of ``self``."""
        raise NotImplementedError

    def height(self):
        """Compute the height of ``self``."""
        raise NotImplementedError

    # TODO more

    # non-implemented methods end here

    def __init__(self, ring):
        self.ring = ring

    def _check_ideal(self, J):
        """Helper to check ``J`` is an ideal of our ring."""
        if not isinstance(J, Ideal) or J.ring != self.ring:
            raise ValueError(
                'J must be an ideal of %s, got %s' % (self.ring, J))

    def contains(self, elem):
        """
        Return True if ``elem`` is an element of this ideal.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).ideal(x+1, x-1).contains(3)
        True
        >>> QQ.old_poly_ring(x).ideal(x**2, x**3).contains(x)
        False
        """
        return self._contains_elem(self.ring.convert(elem))

    def subset(self, other):
        """
        Returns True if ``other`` is is a subset of ``self``.

        Here ``other`` may be an ideal.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> I = QQ.old_poly_ring(x).ideal(x+1)
        >>> I.subset([x**2 - 1, x**2 + 2*x + 1])
        True
        >>> I.subset([x**2 + 1, x + 1])
        False
        >>> I.subset(QQ.old_poly_ring(x).ideal(x**2 - 1))
        True
        """
        if isinstance(other, Ideal):
            return self._contains_ideal(other)
        return all(self._contains_elem(x) for x in other)

    def quotient(self, J, **opts):
        r"""
        Compute the ideal quotient of ``self`` by ``J``.

        That is, if ``self`` is the ideal `I`, compute the set
        `I : J = \{x \in R | xJ \subset I \}`.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy import QQ
        >>> R = QQ.old_poly_ring(x, y)
        >>> R.ideal(x*y).quotient(R.ideal(x))
        <y>
        """
        self._check_ideal(J)
        return self._quotient(J, **opts)

    def intersect(self, J):
        """
        Compute the intersection of self with ideal J.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy import QQ
        >>> R = QQ.old_poly_ring(x, y)
        >>> R.ideal(x).intersect(R.ideal(y))
        <x*y>
        """
        self._check_ideal(J)
        return self._intersect(J)

    def saturate(self, J):
        r"""
        Compute the ideal saturation of ``self`` by ``J``.

        That is, if ``self`` is the ideal `I`, compute the set
        `I : J^\infty = \{x \in R | xJ^n \subset I \text{ for some } n\}`.
        """
        raise NotImplementedError
        # Note this can be implemented using repeated quotient

    def union(self, J):
        """
        Compute the ideal generated by the union of ``self`` and ``J``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).ideal(x**2 - 1).union(QQ.old_poly_ring(x).ideal((x+1)**2)) == QQ.old_poly_ring(x).ideal(x+1)
        True
        """
        self._check_ideal(J)
        return self._union(J)

    def product(self, J):
        r"""
        Compute the ideal product of ``self`` and ``J``.

        That is, compute the ideal generated by products `xy`, for `x` an element
        of ``self`` and `y \in J`.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x, y).ideal(x).product(QQ.old_poly_ring(x, y).ideal(y))
        <x*y>
        """
        self._check_ideal(J)
        return self._product(J)

    def reduce_element(self, x):
        """
        Reduce the element ``x`` of our ring modulo the ideal ``self``.

        Here "reduce" has no specific meaning: it could return a unique normal
        form, simplify the expression a bit, or just do nothing.
        """
        return x

    def __add__(self, e):
        if not isinstance(e, Ideal):
            R = self.ring.quotient_ring(self)
            if isinstance(e, R.dtype):
                return e
            if isinstance(e, R.ring.dtype):
                return R(e)
            return R.convert(e)
        self._check_ideal(e)
        return self.union(e)

    __radd__ = __add__

    def __mul__(self, e):
        if not isinstance(e, Ideal):
            try:
                e = self.ring.ideal(e)
            except CoercionFailed:
                return NotImplemented
        self._check_ideal(e)
        return self.product(e)

    __rmul__ = __mul__

    def _zeroth_power(self):
        return self.ring.ideal(1)

    def _first_power(self):
        # Raising to any power but 1 returns a new instance. So we mult by 1
        # here so that the first power is no exception.
        return self * 1

    def __eq__(self, e):
        if not isinstance(e, Ideal) or e.ring != self.ring:
            return False
        return self._equals(e)

    def __ne__(self, e):
        return not (self == e)


class ModuleImplementedIdeal(Ideal):
    """
    Ideal implementation relying on the modules code.

    Attributes:

    - _module - the underlying module
    """

    def __init__(self, ring, module):
        Ideal.__init__(self, ring)
        self._module = module

    def _contains_elem(self, x):
        return self._module.contains([x])

    def _contains_ideal(self, J):
        if not isinstance(J, ModuleImplementedIdeal):
            raise NotImplementedError
        return self._module.is_submodule(J._module)

    def _intersect(self, J):
        if not isinstance(J, ModuleImplementedIdeal):
            raise NotImplementedError
        return self.__class__(self.ring, self._module.intersect(J._module))

    def _quotient(self, J, **opts):
        if not isinstance(J, ModuleImplementedIdeal):
            raise NotImplementedError
        return self._module.module_quotient(J._module, **opts)

    def _union(self, J):
        if not isinstance(J, ModuleImplementedIdeal):
            raise NotImplementedError
        return self.__class__(self.ring, self._module.union(J._module))

    @property
    def gens(self):
        """
        Return generators for ``self``.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.abc import x, y
        >>> list(QQ.old_poly_ring(x, y).ideal(x, y, x**2 + y).gens)
        [DMP_Python([[1], []], QQ), DMP_Python([[1, 0]], QQ), DMP_Python([[1], [], [1, 0]], QQ)]
        """
        return (x[0] for x in self._module.gens)

    def is_zero(self):
        """
        Return True if ``self`` is the zero ideal.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).ideal(x).is_zero()
        False
        >>> QQ.old_poly_ring(x).ideal().is_zero()
        True
        """
        return self._module.is_zero()

    def is_whole_ring(self):
        """
        Return True if ``self`` is the whole ring, i.e. one generator is a unit.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ, ilex
        >>> QQ.old_poly_ring(x).ideal(x).is_whole_ring()
        False
        >>> QQ.old_poly_ring(x).ideal(3).is_whole_ring()
        True
        >>> QQ.old_poly_ring(x, order=ilex).ideal(2 + x).is_whole_ring()
        True
        """
        return self._module.is_full_module()

    def __repr__(self):
        from sympy.printing.str import sstr
        gens = [self.ring.to_sympy(x) for [x] in self._module.gens]
        return '<' + ','.join(sstr(g) for g in gens) + '>'

    # NOTE this is the only method using the fact that the module is a SubModule
    def _product(self, J):
        if not isinstance(J, ModuleImplementedIdeal):
            raise NotImplementedError
        return self.__class__(self.ring, self._module.submodule(
            *[[x*y] for [x] in self._module.gens for [y] in J._module.gens]))

    def in_terms_of_generators(self, e):
        """
        Express ``e`` in terms of the generators of ``self``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> I = QQ.old_poly_ring(x).ideal(x**2 + 1, x)
        >>> I.in_terms_of_generators(1)  # doctest: +SKIP
        [DMP_Python([1], QQ), DMP_Python([-1, 0], QQ)]
        """
        return self._module.in_terms_of_generators([e])

    def reduce_element(self, x, **options):
        return self._module.reduce_element([x], **options)[0]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\series\limits.py ===
from sympy.calculus.accumulationbounds import AccumBounds
from sympy.core import S, Symbol, Add, sympify, Expr, PoleError, Mul
from sympy.core.exprtools import factor_terms
from sympy.core.numbers import Float, _illegal
from sympy.core.function import AppliedUndef
from sympy.core.symbol import Dummy
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import (Abs, sign, arg, re)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.special.gamma_functions import gamma
from sympy.polys import PolynomialError, factor
from sympy.series.order import Order
from .gruntz import gruntz

def limit(e, z, z0, dir="+"):
    """Computes the limit of ``e(z)`` at the point ``z0``.

    Parameters
    ==========

    e : expression, the limit of which is to be taken

    z : symbol representing the variable in the limit.
        Other symbols are treated as constants. Multivariate limits
        are not supported.

    z0 : the value toward which ``z`` tends. Can be any expression,
        including ``oo`` and ``-oo``.

    dir : string, optional (default: "+")
        The limit is bi-directional if ``dir="+-"``, from the right
        (z->z0+) if ``dir="+"``, and from the left (z->z0-) if
        ``dir="-"``. For infinite ``z0`` (``oo`` or ``-oo``), the ``dir``
        argument is determined from the direction of the infinity
        (i.e., ``dir="-"`` for ``oo``).

    Examples
    ========

    >>> from sympy import limit, sin, oo
    >>> from sympy.abc import x
    >>> limit(sin(x)/x, x, 0)
    1
    >>> limit(1/x, x, 0) # default dir='+'
    oo
    >>> limit(1/x, x, 0, dir="-")
    -oo
    >>> limit(1/x, x, 0, dir='+-')
    zoo
    >>> limit(1/x, x, oo)
    0

    Notes
    =====

    First we try some heuristics for easy and frequent cases like "x", "1/x",
    "x**2" and similar, so that it's fast. For all other cases, we use the
    Gruntz algorithm (see the gruntz() function).

    See Also
    ========

     limit_seq : returns the limit of a sequence.
    """

    return Limit(e, z, z0, dir).doit(deep=False)


def heuristics(e, z, z0, dir):
    """Computes the limit of an expression term-wise.
    Parameters are the same as for the ``limit`` function.
    Works with the arguments of expression ``e`` one by one, computing
    the limit of each and then combining the results. This approach
    works only for simple limits, but it is fast.
    """

    rv = None
    if z0 is S.Infinity:
        rv = limit(e.subs(z, 1/z), z, S.Zero, "+")
        if isinstance(rv, Limit):
            return
    elif (e.is_Mul or e.is_Add or e.is_Pow or (e.is_Function and not isinstance(e, AppliedUndef))):
        r = []
        from sympy.simplify.simplify import together
        for a in e.args:
            l = limit(a, z, z0, dir)
            if l.has(S.Infinity) and l.is_finite is None:
                if isinstance(e, Add):
                    m = factor_terms(e)
                    if not isinstance(m, Mul): # try together
                        m = together(m)
                    if not isinstance(m, Mul): # try factor if the previous methods failed
                        m = factor(e)
                    if isinstance(m, Mul):
                        return heuristics(m, z, z0, dir)
                    return
                return
            elif isinstance(l, Limit):
                return
            elif l is S.NaN:
                return
            else:
                r.append(l)
        if r:
            rv = e.func(*r)
            if rv is S.NaN and e.is_Mul and any(isinstance(rr, AccumBounds) for rr in r):
                r2 = []
                e2 = []
                for ii, rval in enumerate(r):
                    if isinstance(rval, AccumBounds):
                        r2.append(rval)
                    else:
                        e2.append(e.args[ii])

                if len(e2) > 0:
                    e3 = Mul(*e2).simplify()
                    l = limit(e3, z, z0, dir)
                    rv = l * Mul(*r2)

            if rv is S.NaN:
                try:
                    from sympy.simplify.ratsimp import ratsimp
                    rat_e = ratsimp(e)
                except PolynomialError:
                    return
                if rat_e is S.NaN or rat_e == e:
                    return
                return limit(rat_e, z, z0, dir)
    return rv


class Limit(Expr):
    """Represents an unevaluated limit.

    Examples
    ========

    >>> from sympy import Limit, sin
    >>> from sympy.abc import x
    >>> Limit(sin(x)/x, x, 0)
    Limit(sin(x)/x, x, 0, dir='+')
    >>> Limit(1/x, x, 0, dir="-")
    Limit(1/x, x, 0, dir='-')

    """

    def __new__(cls, e, z, z0, dir="+"):
        e = sympify(e)
        z = sympify(z)
        z0 = sympify(z0)

        if z0 in (S.Infinity, S.ImaginaryUnit*S.Infinity):
            dir = "-"
        elif z0 in (S.NegativeInfinity, S.ImaginaryUnit*S.NegativeInfinity):
            dir = "+"

        if(z0.has(z)):
            raise NotImplementedError("Limits approaching a variable point are"
                    " not supported (%s -> %s)" % (z, z0))
        if isinstance(dir, str):
            dir = Symbol(dir)
        elif not isinstance(dir, Symbol):
            raise TypeError("direction must be of type basestring or "
                    "Symbol, not %s" % type(dir))
        if str(dir) not in ('+', '-', '+-'):
            raise ValueError("direction must be one of '+', '-' "
                    "or '+-', not %s" % dir)

        obj = Expr.__new__(cls)
        obj._args = (e, z, z0, dir)
        return obj


    @property
    def free_symbols(self):
        e = self.args[0]
        isyms = e.free_symbols
        isyms.difference_update(self.args[1].free_symbols)
        isyms.update(self.args[2].free_symbols)
        return isyms


    def pow_heuristics(self, e):
        _, z, z0, _ = self.args
        b1, e1 = e.base, e.exp
        if not b1.has(z):
            res = limit(e1*log(b1), z, z0)
            return exp(res)

        ex_lim = limit(e1, z, z0)
        base_lim = limit(b1, z, z0)

        if base_lim is S.One:
            if ex_lim in (S.Infinity, S.NegativeInfinity):
                res = limit(e1*(b1 - 1), z, z0)
                return exp(res)
        if base_lim is S.NegativeInfinity and ex_lim is S.Infinity:
            return S.ComplexInfinity


    def doit(self, **hints):
        """Evaluates the limit.

        Parameters
        ==========

        deep : bool, optional (default: True)
            Invoke the ``doit`` method of the expressions involved before
            taking the limit.

        hints : optional keyword arguments
            To be passed to ``doit`` methods; only used if deep is True.
        """

        e, z, z0, dir = self.args

        if str(dir) == '+-':
            r = limit(e, z, z0, dir='+')
            l = limit(e, z, z0, dir='-')
            if isinstance(r, Limit) and isinstance(l, Limit):
                if r.args[0] == l.args[0]:
                    return self
            if r == l:
                return l
            if r.is_infinite and l.is_infinite:
                return S.ComplexInfinity
            raise ValueError("The limit does not exist since "
                             "left hand limit = %s and right hand limit = %s"
                             % (l, r))

        if z0 is S.ComplexInfinity:
            raise NotImplementedError("Limits at complex "
                                    "infinity are not implemented")

        if z0.is_infinite:
            cdir = sign(z0)
            cdir = cdir/abs(cdir)
            e = e.subs(z, cdir*z)
            dir = "-"
            z0 = S.Infinity

        if hints.get('deep', True):
            e = e.doit(**hints)
            z = z.doit(**hints)
            z0 = z0.doit(**hints)

        if e == z:
            return z0

        if not e.has(z):
            return e

        if z0 is S.NaN:
            return S.NaN

        if e.has(*_illegal):
            return self

        if e.is_Order:
            return Order(limit(e.expr, z, z0), *e.args[1:])

        cdir = S.Zero
        if str(dir) == "+":
            cdir = S.One
        elif str(dir) == "-":
            cdir = S.NegativeOne

        def set_signs(expr):
            if not expr.args:
                return expr
            newargs = tuple(set_signs(arg) for arg in expr.args)
            if newargs != expr.args:
                expr = expr.func(*newargs)
            abs_flag = isinstance(expr, Abs)
            arg_flag = isinstance(expr, arg)
            sign_flag = isinstance(expr, sign)
            if abs_flag or sign_flag or arg_flag:
                try:
                    sig = limit(expr.args[0], z, z0, dir)
                    if sig.is_zero:
                        sig = limit(1/expr.args[0], z, z0, dir)
                except NotImplementedError:
                    return expr
                else:
                    if sig.is_extended_real:
                        if (sig < 0) == True:
                            return (-expr.args[0] if abs_flag else
                                    S.NegativeOne if sign_flag else S.Pi)
                        elif (sig > 0) == True:
                            return (expr.args[0] if abs_flag else
                                    S.One if sign_flag else S.Zero)
            return expr

        if e.has(Float):
            # Convert floats like 0.5 to exact SymPy numbers like S.Half, to
            # prevent rounding errors which can lead to unexpected execution
            # of conditional blocks that work on comparisons
            # Also see comments in https://github.com/sympy/sympy/issues/19453
            from sympy.simplify.simplify import nsimplify
            e = nsimplify(e)
        e = set_signs(e)


        if e.is_meromorphic(z, z0):
            if z0 is S.Infinity:
                newe = e.subs(z, 1/z)
                # cdir changes sign as oo- should become 0+
                cdir = -cdir
            else:
                newe = e.subs(z, z + z0)
            try:
                coeff, ex = newe.leadterm(z, cdir=cdir)
            except ValueError:
                pass
            else:
                if ex > 0:
                    return S.Zero
                elif ex == 0:
                    return coeff
                if cdir == 1 or not(int(ex) & 1):
                    return S.Infinity*sign(coeff)
                elif cdir == -1:
                    return S.NegativeInfinity*sign(coeff)
                else:
                    return S.ComplexInfinity

        if z0 is S.Infinity:
            if e.is_Mul:
                e = factor_terms(e)
            dummy = Dummy('z', positive=z.is_positive, negative=z.is_negative, real=z.is_real)
            newe = e.subs(z, 1/dummy)
            # cdir changes sign as oo- should become 0+
            cdir = -cdir
            newz = dummy
        else:
            newe = e.subs(z, z + z0)
            newz = z
        try:
            coeff, ex = newe.leadterm(newz, cdir=cdir)
        except (ValueError, NotImplementedError, PoleError):
            # The NotImplementedError catching is for custom functions
            from sympy.simplify.powsimp import powsimp
            e = powsimp(e)
            if e.is_Pow:
                r = self.pow_heuristics(e)
                if r is not None:
                    return r
            try:
                coeff = newe.as_leading_term(newz, cdir=cdir)
                if coeff != newe and (coeff.has(exp) or coeff.has(S.Exp1)):
                    return gruntz(coeff, newz, 0, "-" if re(cdir).is_negative else "+")
            except (ValueError, NotImplementedError, PoleError):
                pass
        else:
            if isinstance(coeff, AccumBounds) and ex == S.Zero:
                return coeff
            if coeff.has(S.Infinity, S.NegativeInfinity, S.ComplexInfinity, S.NaN):
                return self
            if not coeff.has(newz):
                if ex.is_positive:
                    return S.Zero
                elif ex == 0:
                    return coeff
                elif ex.is_negative:
                    if cdir == 1:
                        return S.Infinity*sign(coeff)
                    elif cdir == -1:
                        return S.NegativeInfinity*sign(coeff)*S.NegativeOne**(S.One + ex)
                    else:
                        return S.ComplexInfinity
                else:
                    raise NotImplementedError("Not sure of sign of %s" % ex)

        # gruntz fails on factorials but works with the gamma function
        # If no factorial term is present, e should remain unchanged.
        # factorial is defined to be zero for negative inputs (which
        # differs from gamma) so only rewrite for non-negative z0.
        if z0.is_extended_nonnegative:
            e = e.rewrite(factorial, gamma)

        l = None

        try:
            r = gruntz(e, z, z0, dir)
            if r is S.NaN or l is S.NaN:
                raise PoleError()
        except (PoleError, ValueError):
            if l is not None:
                raise
            r = heuristics(e, z, z0, dir)
            if r is None:
                return self

        return r

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\solvers\ode\subscheck.py ===
from sympy.core import S, Pow
from sympy.core.function import (Derivative, AppliedUndef, diff)
from sympy.core.relational import Equality, Eq
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify

from sympy.logic.boolalg import BooleanAtom
from sympy.functions import exp
from sympy.series import Order
from sympy.simplify.simplify import simplify, posify, besselsimp
from sympy.simplify.trigsimp import trigsimp
from sympy.simplify.sqrtdenest import sqrtdenest
from sympy.solvers import solve
from sympy.solvers.deutils import _preprocess, ode_order
from sympy.utilities.iterables import iterable, is_sequence


def sub_func_doit(eq, func, new):
    r"""
    When replacing the func with something else, we usually want the
    derivative evaluated, so this function helps in making that happen.

    Examples
    ========

    >>> from sympy import Derivative, symbols, Function
    >>> from sympy.solvers.ode.subscheck import sub_func_doit
    >>> x, z = symbols('x, z')
    >>> y = Function('y')

    >>> sub_func_doit(3*Derivative(y(x), x) - 1, y(x), x)
    2

    >>> sub_func_doit(x*Derivative(y(x), x) - y(x)**2 + y(x), y(x),
    ... 1/(x*(z + 1/x)))
    x*(-1/(x**2*(z + 1/x)) + 1/(x**3*(z + 1/x)**2)) + 1/(x*(z + 1/x))
    ...- 1/(x**2*(z + 1/x)**2)
    """
    reps= {func: new}
    for d in eq.atoms(Derivative):
        if d.expr == func:
            reps[d] = new.diff(*d.variable_count)
        else:
            reps[d] = d.xreplace({func: new}).doit(deep=False)
    return eq.xreplace(reps)


def checkodesol(ode, sol, func=None, order='auto', solve_for_func=True):
    r"""
    Substitutes ``sol`` into ``ode`` and checks that the result is ``0``.

    This works when ``func`` is one function, like `f(x)` or a list of
    functions like `[f(x), g(x)]` when `ode` is a system of ODEs.  ``sol`` can
    be a single solution or a list of solutions.  Each solution may be an
    :py:class:`~sympy.core.relational.Equality` that the solution satisfies,
    e.g. ``Eq(f(x), C1), Eq(f(x) + C1, 0)``; or simply an
    :py:class:`~sympy.core.expr.Expr`, e.g. ``f(x) - C1``. In most cases it
    will not be necessary to explicitly identify the function, but if the
    function cannot be inferred from the original equation it can be supplied
    through the ``func`` argument.

    If a sequence of solutions is passed, the same sort of container will be
    used to return the result for each solution.

    It tries the following methods, in order, until it finds zero equivalence:

    1. Substitute the solution for `f` in the original equation.  This only
       works if ``ode`` is solved for `f`.  It will attempt to solve it first
       unless ``solve_for_func == False``.
    2. Take `n` derivatives of the solution, where `n` is the order of
       ``ode``, and check to see if that is equal to the solution.  This only
       works on exact ODEs.
    3. Take the 1st, 2nd, ..., `n`\th derivatives of the solution, each time
       solving for the derivative of `f` of that order (this will always be
       possible because `f` is a linear operator). Then back substitute each
       derivative into ``ode`` in reverse order.

    This function returns a tuple.  The first item in the tuple is ``True`` if
    the substitution results in ``0``, and ``False`` otherwise. The second
    item in the tuple is what the substitution results in.  It should always
    be ``0`` if the first item is ``True``. Sometimes this function will
    return ``False`` even when an expression is identically equal to ``0``.
    This happens when :py:meth:`~sympy.simplify.simplify.simplify` does not
    reduce the expression to ``0``.  If an expression returned by this
    function vanishes identically, then ``sol`` really is a solution to
    the ``ode``.

    If this function seems to hang, it is probably because of a hard
    simplification.

    To use this function to test, test the first item of the tuple.

    Examples
    ========

    >>> from sympy import (Eq, Function, checkodesol, symbols,
    ...     Derivative, exp)
    >>> x, C1, C2 = symbols('x,C1,C2')
    >>> f, g = symbols('f g', cls=Function)
    >>> checkodesol(f(x).diff(x), Eq(f(x), C1))
    (True, 0)
    >>> assert checkodesol(f(x).diff(x), C1)[0]
    >>> assert not checkodesol(f(x).diff(x), x)[0]
    >>> checkodesol(f(x).diff(x, 2), x**2)
    (False, 2)

    >>> eqs = [Eq(Derivative(f(x), x), f(x)), Eq(Derivative(g(x), x), g(x))]
    >>> sol = [Eq(f(x), C1*exp(x)), Eq(g(x), C2*exp(x))]
    >>> checkodesol(eqs, sol)
    (True, [0, 0])

    """
    if iterable(ode):
        return checksysodesol(ode, sol, func=func)

    if not isinstance(ode, Equality):
        ode = Eq(ode, 0)
    if func is None:
        try:
            _, func = _preprocess(ode.lhs)
        except ValueError:
            funcs = [s.atoms(AppliedUndef) for s in (
                sol if is_sequence(sol, set) else [sol])]
            funcs = set().union(*funcs)
            if len(funcs) != 1:
                raise ValueError(
                    'must pass func arg to checkodesol for this case.')
            func = funcs.pop()
    if not isinstance(func, AppliedUndef) or len(func.args) != 1:
        raise ValueError(
            "func must be a function of one variable, not %s" % func)
    if is_sequence(sol, set):
        return type(sol)([checkodesol(ode, i, order=order, solve_for_func=solve_for_func) for i in sol])

    if not isinstance(sol, Equality):
        sol = Eq(func, sol)
    elif sol.rhs == func:
        sol = sol.reversed

    if order == 'auto':
        order = ode_order(ode, func)
    solved = sol.lhs == func and not sol.rhs.has(func)
    if solve_for_func and not solved:
        rhs = solve(sol, func)
        if rhs:
            eqs = [Eq(func, t) for t in rhs]
            if len(rhs) == 1:
                eqs = eqs[0]
            return checkodesol(ode, eqs, order=order,
                solve_for_func=False)

    x = func.args[0]

    # Handle series solutions here
    if sol.has(Order):
        assert sol.lhs == func
        Oterm = sol.rhs.getO()
        solrhs = sol.rhs.removeO()

        Oexpr = Oterm.expr
        assert isinstance(Oexpr, Pow)
        sorder = Oexpr.exp
        assert Oterm == Order(x**sorder)

        odesubs = (ode.lhs-ode.rhs).subs(func, solrhs).doit().expand()

        neworder = Order(x**(sorder - order))
        odesubs = odesubs + neworder
        assert odesubs.getO() == neworder
        residual = odesubs.removeO()

        return (residual == 0, residual)

    s = True
    testnum = 0
    while s:
        if testnum == 0:
            # First pass, try substituting a solved solution directly into the
            # ODE. This has the highest chance of succeeding.
            ode_diff = ode.lhs - ode.rhs

            if sol.lhs == func:
                s = sub_func_doit(ode_diff, func, sol.rhs)
                s = besselsimp(s)
            else:
                testnum += 1
                continue
            ss = simplify(s.rewrite(exp))
            if ss:
                # with the new numer_denom in power.py, if we do a simple
                # expansion then testnum == 0 verifies all solutions.
                s = ss.expand(force=True)
            else:
                s = 0
            testnum += 1
        elif testnum == 1:
            # Second pass. If we cannot substitute f, try seeing if the nth
            # derivative is equal, this will only work for odes that are exact,
            # by definition.
            s = simplify(
                trigsimp(diff(sol.lhs, x, order) - diff(sol.rhs, x, order)) -
                trigsimp(ode.lhs) + trigsimp(ode.rhs))
            # s2 = simplify(
            #     diff(sol.lhs, x, order) - diff(sol.rhs, x, order) - \
            #     ode.lhs + ode.rhs)
            testnum += 1
        elif testnum == 2:
            # Third pass. Try solving for df/dx and substituting that into the
            # ODE. Thanks to Chris Smith for suggesting this method.  Many of
            # the comments below are his, too.
            # The method:
            # - Take each of 1..n derivatives of the solution.
            # - Solve each nth derivative for d^(n)f/dx^(n)
            #   (the differential of that order)
            # - Back substitute into the ODE in decreasing order
            #   (i.e., n, n-1, ...)
            # - Check the result for zero equivalence
            if sol.lhs == func and not sol.rhs.has(func):
                diffsols = {0: sol.rhs}
            elif sol.rhs == func and not sol.lhs.has(func):
                diffsols = {0: sol.lhs}
            else:
                diffsols = {}
            sol = sol.lhs - sol.rhs
            for i in range(1, order + 1):
                # Differentiation is a linear operator, so there should always
                # be 1 solution. Nonetheless, we test just to make sure.
                # We only need to solve once.  After that, we automatically
                # have the solution to the differential in the order we want.
                if i == 1:
                    ds = sol.diff(x)
                    try:
                        sdf = solve(ds, func.diff(x, i))
                        if not sdf:
                            raise NotImplementedError
                    except NotImplementedError:
                        testnum += 1
                        break
                    else:
                        diffsols[i] = sdf[0]
                else:
                    # This is what the solution says df/dx should be.
                    diffsols[i] = diffsols[i - 1].diff(x)

            # Make sure the above didn't fail.
            if testnum > 2:
                continue
            else:
                # Substitute it into ODE to check for self consistency.
                lhs, rhs = ode.lhs, ode.rhs
                for i in range(order, -1, -1):
                    if i == 0 and 0 not in diffsols:
                        # We can only substitute f(x) if the solution was
                        # solved for f(x).
                        break
                    lhs = sub_func_doit(lhs, func.diff(x, i), diffsols[i])
                    rhs = sub_func_doit(rhs, func.diff(x, i), diffsols[i])
                    ode_or_bool = Eq(lhs, rhs)
                    ode_or_bool = simplify(ode_or_bool)

                    if isinstance(ode_or_bool, (bool, BooleanAtom)):
                        if ode_or_bool:
                            lhs = rhs = S.Zero
                    else:
                        lhs = ode_or_bool.lhs
                        rhs = ode_or_bool.rhs
                # No sense in overworking simplify -- just prove that the
                # numerator goes to zero
                num = trigsimp((lhs - rhs).as_numer_denom()[0])
                # since solutions are obtained using force=True we test
                # using the same level of assumptions
                ## replace function with dummy so assumptions will work
                _func = Dummy('func')
                num = num.subs(func, _func)
                ## posify the expression
                num, reps = posify(num)
                s = simplify(num).xreplace(reps).xreplace({_func: func})
                testnum += 1
        else:
            break

    if not s:
        return (True, s)
    elif s is True:  # The code above never was able to change s
        raise NotImplementedError("Unable to test if " + str(sol) +
            " is a solution to " + str(ode) + ".")
    else:
        return (False, s)


def checksysodesol(eqs, sols, func=None):
    r"""
    Substitutes corresponding ``sols`` for each functions into each ``eqs`` and
    checks that the result of substitutions for each equation is ``0``. The
    equations and solutions passed can be any iterable.

    This only works when each ``sols`` have one function only, like `x(t)` or `y(t)`.
    For each function, ``sols`` can have a single solution or a list of solutions.
    In most cases it will not be necessary to explicitly identify the function,
    but if the function cannot be inferred from the original equation it
    can be supplied through the ``func`` argument.

    When a sequence of equations is passed, the same sequence is used to return
    the result for each equation with each function substituted with corresponding
    solutions.

    It tries the following method to find zero equivalence for each equation:

    Substitute the solutions for functions, like `x(t)` and `y(t)` into the
    original equations containing those functions.
    This function returns a tuple.  The first item in the tuple is ``True`` if
    the substitution results for each equation is ``0``, and ``False`` otherwise.
    The second item in the tuple is what the substitution results in.  Each element
    of the ``list`` should always be ``0`` corresponding to each equation if the
    first item is ``True``. Note that sometimes this function may return ``False``,
    but with an expression that is identically equal to ``0``, instead of returning
    ``True``.  This is because :py:meth:`~sympy.simplify.simplify.simplify` cannot
    reduce the expression to ``0``.  If an expression returned by each function
    vanishes identically, then ``sols`` really is a solution to ``eqs``.

    If this function seems to hang, it is probably because of a difficult simplification.

    Examples
    ========

    >>> from sympy import Eq, diff, symbols, sin, cos, exp, sqrt, S, Function
    >>> from sympy.solvers.ode.subscheck import checksysodesol
    >>> C1, C2 = symbols('C1:3')
    >>> t = symbols('t')
    >>> x, y = symbols('x, y', cls=Function)
    >>> eq = (Eq(diff(x(t),t), x(t) + y(t) + 17), Eq(diff(y(t),t), -2*x(t) + y(t) + 12))
    >>> sol = [Eq(x(t), (C1*sin(sqrt(2)*t) + C2*cos(sqrt(2)*t))*exp(t) - S(5)/3),
    ... Eq(y(t), (sqrt(2)*C1*cos(sqrt(2)*t) - sqrt(2)*C2*sin(sqrt(2)*t))*exp(t) - S(46)/3)]
    >>> checksysodesol(eq, sol)
    (True, [0, 0])
    >>> eq = (Eq(diff(x(t),t),x(t)*y(t)**4), Eq(diff(y(t),t),y(t)**3))
    >>> sol = [Eq(x(t), C1*exp(-1/(4*(C2 + t)))), Eq(y(t), -sqrt(2)*sqrt(-1/(C2 + t))/2),
    ... Eq(x(t), C1*exp(-1/(4*(C2 + t)))), Eq(y(t), sqrt(2)*sqrt(-1/(C2 + t))/2)]
    >>> checksysodesol(eq, sol)
    (True, [0, 0])

    """
    def _sympify(eq):
        return list(map(sympify, eq if iterable(eq) else [eq]))
    eqs = _sympify(eqs)
    for i in range(len(eqs)):
        if isinstance(eqs[i], Equality):
            eqs[i] = eqs[i].lhs - eqs[i].rhs
    if func is None:
        funcs = []
        for eq in eqs:
            derivs = eq.atoms(Derivative)
            func = set().union(*[d.atoms(AppliedUndef) for d in derivs])
            funcs.extend(func)
        funcs = list(set(funcs))
    if not all(isinstance(func, AppliedUndef) and len(func.args) == 1 for func in funcs)\
    and len({func.args for func in funcs})!=1:
        raise ValueError("func must be a function of one variable, not %s" % func)
    for sol in sols:
        if len(sol.atoms(AppliedUndef)) != 1:
            raise ValueError("solutions should have one function only")
    if len(funcs) != len({sol.lhs for sol in sols}):
        raise ValueError("number of solutions provided does not match the number of equations")
    dictsol = {}
    for sol in sols:
        func = list(sol.atoms(AppliedUndef))[0]
        if sol.rhs == func:
            sol = sol.reversed
        solved = sol.lhs == func and not sol.rhs.has(func)
        if not solved:
            rhs = solve(sol, func)
            if not rhs:
                raise NotImplementedError
        else:
            rhs = sol.rhs
        dictsol[func] = rhs
    checkeq = []
    for eq in eqs:
        for func in funcs:
            eq = sub_func_doit(eq, func, dictsol[func])
        ss = simplify(eq)
        if ss != 0:
            eq = ss.expand(force=True)
            if eq != 0:
                eq = sqrtdenest(eq).simplify()
        else:
            eq = 0
        checkeq.append(eq)
    if len(set(checkeq)) == 1 and list(set(checkeq))[0] == 0:
        return (True, checkeq)
    else:
        return (False, checkeq)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\testing\pytest.py ===
"""py.test hacks to support XFAIL/XPASS"""

import platform
import sys
import re
import functools
import os
import contextlib
import warnings
import inspect
import pathlib
from typing import Any, Callable

from sympy.utilities.exceptions import SymPyDeprecationWarning
# Imported here for backwards compatibility. Note: do not import this from
# here in library code (importing sympy.pytest in library code will break the
# pytest integration).
from sympy.utilities.exceptions import ignore_warnings # noqa:F401

ON_CI = os.getenv('CI', None) == "true"

try:
    import pytest
    USE_PYTEST = getattr(sys, '_running_pytest', False)
except ImportError:
    USE_PYTEST = False

IS_WASM: bool = sys.platform == 'emscripten' or platform.machine() in ["wasm32", "wasm64"]

raises: Callable[[Any, Any], Any]
XFAIL: Callable[[Any], Any]
skip: Callable[[Any], Any]
SKIP: Callable[[Any], Any]
slow: Callable[[Any], Any]
tooslow: Callable[[Any], Any]
nocache_fail: Callable[[Any], Any]


if USE_PYTEST:
    raises = pytest.raises
    skip = pytest.skip
    XFAIL = pytest.mark.xfail
    SKIP = pytest.mark.skip
    slow = pytest.mark.slow
    tooslow = pytest.mark.tooslow
    nocache_fail = pytest.mark.nocache_fail
    from _pytest.outcomes import Failed

else:
    # Not using pytest so define the things that would have been imported from
    # there.

    # _pytest._code.code.ExceptionInfo
    class ExceptionInfo:
        def __init__(self, value):
            self.value = value

        def __repr__(self):
            return "<ExceptionInfo {!r}>".format(self.value)


    def raises(expectedException, code=None):
        """
        Tests that ``code`` raises the exception ``expectedException``.

        ``code`` may be a callable, such as a lambda expression or function
        name.

        If ``code`` is not given or None, ``raises`` will return a context
        manager for use in ``with`` statements; the code to execute then
        comes from the scope of the ``with``.

        ``raises()`` does nothing if the callable raises the expected exception,
        otherwise it raises an AssertionError.

        Examples
        ========

        >>> from sympy.testing.pytest import raises

        >>> raises(ZeroDivisionError, lambda: 1/0)
        <ExceptionInfo ZeroDivisionError(...)>
        >>> raises(ZeroDivisionError, lambda: 1/2)
        Traceback (most recent call last):
        ...
        Failed: DID NOT RAISE

        >>> with raises(ZeroDivisionError):
        ...     n = 1/0
        >>> with raises(ZeroDivisionError):
        ...     n = 1/2
        Traceback (most recent call last):
        ...
        Failed: DID NOT RAISE

        Note that you cannot test multiple statements via
        ``with raises``:

        >>> with raises(ZeroDivisionError):
        ...     n = 1/0    # will execute and raise, aborting the ``with``
        ...     n = 9999/0 # never executed

        This is just what ``with`` is supposed to do: abort the
        contained statement sequence at the first exception and let
        the context manager deal with the exception.

        To test multiple statements, you'll need a separate ``with``
        for each:

        >>> with raises(ZeroDivisionError):
        ...     n = 1/0    # will execute and raise
        >>> with raises(ZeroDivisionError):
        ...     n = 9999/0 # will also execute and raise

        """
        if code is None:
            return RaisesContext(expectedException)
        elif callable(code):
            try:
                code()
            except expectedException as e:
                return ExceptionInfo(e)
            raise Failed("DID NOT RAISE")
        elif isinstance(code, str):
            raise TypeError(
                '\'raises(xxx, "code")\' has been phased out; '
                'change \'raises(xxx, "expression")\' '
                'to \'raises(xxx, lambda: expression)\', '
                '\'raises(xxx, "statement")\' '
                'to \'with raises(xxx): statement\'')
        else:
            raise TypeError(
                'raises() expects a callable for the 2nd argument.')

    class RaisesContext:
        def __init__(self, expectedException):
            self.expectedException = expectedException

        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc_value, traceback):
            if exc_type is None:
                raise Failed("DID NOT RAISE")
            return issubclass(exc_type, self.expectedException)

    class XFail(Exception):
        pass

    class XPass(Exception):
        pass

    class Skipped(Exception):
        pass

    class Failed(Exception):  # type: ignore
        pass

    def XFAIL(func):
        def wrapper():
            try:
                func()
            except Exception as e:
                message = str(e)
                if message != "Timeout":
                    raise XFail(func.__name__)
                else:
                    raise Skipped("Timeout")
            raise XPass(func.__name__)

        wrapper = functools.update_wrapper(wrapper, func)
        return wrapper

    def skip(str):
        raise Skipped(str)

    def SKIP(reason):
        """Similar to ``skip()``, but this is a decorator. """
        def wrapper(func):
            def func_wrapper():
                raise Skipped(reason)

            func_wrapper = functools.update_wrapper(func_wrapper, func)
            return func_wrapper

        return wrapper

    def slow(func):
        func._slow = True

        def func_wrapper():
            func()

        func_wrapper = functools.update_wrapper(func_wrapper, func)
        func_wrapper.__wrapped__ = func
        return func_wrapper

    def tooslow(func):
        func._slow = True
        func._tooslow = True

        def func_wrapper():
            skip("Too slow")

        func_wrapper = functools.update_wrapper(func_wrapper, func)
        func_wrapper.__wrapped__ = func
        return func_wrapper

    def nocache_fail(func):
        "Dummy decorator for marking tests that fail when cache is disabled"
        return func

@contextlib.contextmanager
def warns(warningcls, *, match='', test_stacklevel=True):
    '''
    Like raises but tests that warnings are emitted.

    >>> from sympy.testing.pytest import warns
    >>> import warnings

    >>> with warns(UserWarning):
    ...     warnings.warn('deprecated', UserWarning, stacklevel=2)

    >>> with warns(UserWarning):
    ...     pass
    Traceback (most recent call last):
    ...
    Failed: DID NOT WARN. No warnings of type UserWarning\
    was emitted. The list of emitted warnings is: [].

    ``test_stacklevel`` makes it check that the ``stacklevel`` parameter to
    ``warn()`` is set so that the warning shows the user line of code (the
    code under the warns() context manager). Set this to False if this is
    ambiguous or if the context manager does not test the direct user code
    that emits the warning.

    If the warning is a ``SymPyDeprecationWarning``, this additionally tests
    that the ``active_deprecations_target`` is a real target in the
    ``active-deprecations.md`` file.

    '''
    # Absorbs all warnings in warnrec
    with warnings.catch_warnings(record=True) as warnrec:
        # Any warning other than the one we are looking for is an error
        warnings.simplefilter("error")
        warnings.filterwarnings("always", category=warningcls)
        # Now run the test
        yield warnrec

    # Raise if expected warning not found
    if not any(issubclass(w.category, warningcls) for w in warnrec):
        msg = ('Failed: DID NOT WARN.'
               ' No warnings of type %s was emitted.'
               ' The list of emitted warnings is: %s.'
               ) % (warningcls, [w.message for w in warnrec])
        raise Failed(msg)

    # We don't include the match in the filter above because it would then
    # fall to the error filter, so we instead manually check that it matches
    # here
    for w in warnrec:
        # Should always be true due to the filters above
        assert issubclass(w.category, warningcls)
        if not re.compile(match, re.IGNORECASE).match(str(w.message)):
            raise Failed(f"Failed: WRONG MESSAGE. A warning with of the correct category ({warningcls.__name__}) was issued, but it did not match the given match regex ({match!r})")

    if test_stacklevel:
        for f in inspect.stack():
            thisfile = f.filename
            file = os.path.split(thisfile)[1]
            if file.startswith('test_'):
                break
            elif file == 'doctest.py':
                # skip the stacklevel testing in the doctests of this
                # function
                return
        else:
            raise RuntimeError("Could not find the file for the given warning to test the stacklevel")
        for w in warnrec:
            if w.filename != thisfile:
                msg = f'''\
Failed: Warning has the wrong stacklevel. The warning stacklevel needs to be
set so that the line of code shown in the warning message is user code that
calls the deprecated code (the current stacklevel is showing code from
{w.filename} (line {w.lineno}), expected {thisfile})'''.replace('\n', ' ')
                raise Failed(msg)

    if warningcls == SymPyDeprecationWarning:
        this_file = pathlib.Path(__file__)
        active_deprecations_file = (this_file.parent.parent.parent / 'doc' /
                                    'src' / 'explanation' /
                                    'active-deprecations.md')
        if not active_deprecations_file.exists():
            # We can only test that the active_deprecations_target works if we are
            # in the git repo.
            return
        targets = []
        for w in warnrec:
            targets.append(w.message.active_deprecations_target)
        text = pathlib.Path(active_deprecations_file).read_text(encoding="utf-8")
        for target in targets:
            if f'({target})=' not in text:
                raise Failed(f"The active deprecations target {target!r} does not appear to be a valid target in the active-deprecations.md file ({active_deprecations_file}).")

def _both_exp_pow(func):
    """
    Decorator used to run the test twice: the first time `e^x` is represented
    as ``Pow(E, x)``, the second time as ``exp(x)`` (exponential object is not
    a power).

    This is a temporary trick helping to manage the elimination of the class
    ``exp`` in favor of a replacement by ``Pow(E, ...)``.
    """
    from sympy.core.parameters import _exp_is_pow

    def func_wrap():
        with _exp_is_pow(True):
            func()
        with _exp_is_pow(False):
            func()

    wrapper = functools.update_wrapper(func_wrap, func)
    return wrapper


@contextlib.contextmanager
def warns_deprecated_sympy():
    '''
    Shorthand for ``warns(SymPyDeprecationWarning)``

    This is the recommended way to test that ``SymPyDeprecationWarning`` is
    emitted for deprecated features in SymPy. To test for other warnings use
    ``warns``. To suppress warnings without asserting that they are emitted
    use ``ignore_warnings``.

    .. note::

       ``warns_deprecated_sympy()`` is only intended for internal use in the
       SymPy test suite to test that a deprecation warning triggers properly.
       All other code in the SymPy codebase, including documentation examples,
       should not use deprecated behavior.

       If you are a user of SymPy and you want to disable
       SymPyDeprecationWarnings, use ``warnings`` filters (see
       :ref:`silencing-sympy-deprecation-warnings`).

    >>> from sympy.testing.pytest import warns_deprecated_sympy
    >>> from sympy.utilities.exceptions import sympy_deprecation_warning
    >>> with warns_deprecated_sympy():
    ...     sympy_deprecation_warning("Don't use",
    ...        deprecated_since_version="1.0",
    ...        active_deprecations_target="active-deprecations")

    >>> with warns_deprecated_sympy():
    ...     pass
    Traceback (most recent call last):
    ...
    Failed: DID NOT WARN. No warnings of type \
    SymPyDeprecationWarning was emitted. The list of emitted warnings is: [].

    .. note::

       Sometimes the stacklevel test will fail because the same warning is
       emitted multiple times. In this case, you can use
       :func:`sympy.utilities.exceptions.ignore_warnings` in the code to
       prevent the ``SymPyDeprecationWarning`` from being emitted again
       recursively. In rare cases it is impossible to have a consistent
       ``stacklevel`` for deprecation warnings because different ways of
       calling a function will produce different call stacks.. In those cases,
       use ``warns(SymPyDeprecationWarning)`` instead.

    See Also
    ========
    sympy.utilities.exceptions.SymPyDeprecationWarning
    sympy.utilities.exceptions.sympy_deprecation_warning
    sympy.utilities.decorator.deprecated

    '''
    with warns(SymPyDeprecationWarning):
        yield


def skip_under_pyodide(message):
    """Decorator to skip a test if running under Pyodide/WASM."""
    def decorator(test_func):
        @functools.wraps(test_func)
        def test_wrapper():
            if IS_WASM:
                skip(message)
            return test_func()
        return test_wrapper
    return decorator

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\commands\list.py ===
import json
import logging
from email.parser import Parser
from optparse import Values
from typing import TYPE_CHECKING, Generator, List, Optional, Sequence, Tuple, cast

from pip._vendor.packaging.utils import canonicalize_name
from pip._vendor.packaging.version import Version

from pip._internal.cli import cmdoptions
from pip._internal.cli.index_command import IndexGroupCommand
from pip._internal.cli.status_codes import SUCCESS
from pip._internal.exceptions import CommandError
from pip._internal.metadata import BaseDistribution, get_environment
from pip._internal.models.selection_prefs import SelectionPreferences
from pip._internal.utils.compat import stdlib_pkgs
from pip._internal.utils.misc import tabulate, write_output

if TYPE_CHECKING:
    from pip._internal.index.package_finder import PackageFinder
    from pip._internal.network.session import PipSession

    class _DistWithLatestInfo(BaseDistribution):
        """Give the distribution object a couple of extra fields.

        These will be populated during ``get_outdated()``. This is dirty but
        makes the rest of the code much cleaner.
        """

        latest_version: Version
        latest_filetype: str

    _ProcessedDists = Sequence[_DistWithLatestInfo]


logger = logging.getLogger(__name__)


class ListCommand(IndexGroupCommand):
    """
    List installed packages, including editables.

    Packages are listed in a case-insensitive sorted order.
    """

    ignore_require_venv = True
    usage = """
      %prog [options]"""

    def add_options(self) -> None:
        self.cmd_opts.add_option(
            "-o",
            "--outdated",
            action="store_true",
            default=False,
            help="List outdated packages",
        )
        self.cmd_opts.add_option(
            "-u",
            "--uptodate",
            action="store_true",
            default=False,
            help="List uptodate packages",
        )
        self.cmd_opts.add_option(
            "-e",
            "--editable",
            action="store_true",
            default=False,
            help="List editable projects.",
        )
        self.cmd_opts.add_option(
            "-l",
            "--local",
            action="store_true",
            default=False,
            help=(
                "If in a virtualenv that has global access, do not list "
                "globally-installed packages."
            ),
        )
        self.cmd_opts.add_option(
            "--user",
            dest="user",
            action="store_true",
            default=False,
            help="Only output packages installed in user-site.",
        )
        self.cmd_opts.add_option(cmdoptions.list_path())
        self.cmd_opts.add_option(
            "--pre",
            action="store_true",
            default=False,
            help=(
                "Include pre-release and development versions. By default, "
                "pip only finds stable versions."
            ),
        )

        self.cmd_opts.add_option(
            "--format",
            action="store",
            dest="list_format",
            default="columns",
            choices=("columns", "freeze", "json"),
            help=(
                "Select the output format among: columns (default), freeze, or json. "
                "The 'freeze' format cannot be used with the --outdated option."
            ),
        )

        self.cmd_opts.add_option(
            "--not-required",
            action="store_true",
            dest="not_required",
            help="List packages that are not dependencies of installed packages.",
        )

        self.cmd_opts.add_option(
            "--exclude-editable",
            action="store_false",
            dest="include_editable",
            help="Exclude editable package from output.",
        )
        self.cmd_opts.add_option(
            "--include-editable",
            action="store_true",
            dest="include_editable",
            help="Include editable package in output.",
            default=True,
        )
        self.cmd_opts.add_option(cmdoptions.list_exclude())
        index_opts = cmdoptions.make_option_group(cmdoptions.index_group, self.parser)

        self.parser.insert_option_group(0, index_opts)
        self.parser.insert_option_group(0, self.cmd_opts)

    def handle_pip_version_check(self, options: Values) -> None:
        if options.outdated or options.uptodate:
            super().handle_pip_version_check(options)

    def _build_package_finder(
        self, options: Values, session: "PipSession"
    ) -> "PackageFinder":
        """
        Create a package finder appropriate to this list command.
        """
        # Lazy import the heavy index modules as most list invocations won't need 'em.
        from pip._internal.index.collector import LinkCollector
        from pip._internal.index.package_finder import PackageFinder

        link_collector = LinkCollector.create(session, options=options)

        # Pass allow_yanked=False to ignore yanked versions.
        selection_prefs = SelectionPreferences(
            allow_yanked=False,
            allow_all_prereleases=options.pre,
        )

        return PackageFinder.create(
            link_collector=link_collector,
            selection_prefs=selection_prefs,
        )

    def run(self, options: Values, args: List[str]) -> int:
        if options.outdated and options.uptodate:
            raise CommandError("Options --outdated and --uptodate cannot be combined.")

        if options.outdated and options.list_format == "freeze":
            raise CommandError(
                "List format 'freeze' cannot be used with the --outdated option."
            )

        cmdoptions.check_list_path_option(options)

        skip = set(stdlib_pkgs)
        if options.excludes:
            skip.update(canonicalize_name(n) for n in options.excludes)

        packages: _ProcessedDists = [
            cast("_DistWithLatestInfo", d)
            for d in get_environment(options.path).iter_installed_distributions(
                local_only=options.local,
                user_only=options.user,
                editables_only=options.editable,
                include_editables=options.include_editable,
                skip=skip,
            )
        ]

        # get_not_required must be called firstly in order to find and
        # filter out all dependencies correctly. Otherwise a package
        # can't be identified as requirement because some parent packages
        # could be filtered out before.
        if options.not_required:
            packages = self.get_not_required(packages, options)

        if options.outdated:
            packages = self.get_outdated(packages, options)
        elif options.uptodate:
            packages = self.get_uptodate(packages, options)

        self.output_package_listing(packages, options)
        return SUCCESS

    def get_outdated(
        self, packages: "_ProcessedDists", options: Values
    ) -> "_ProcessedDists":
        return [
            dist
            for dist in self.iter_packages_latest_infos(packages, options)
            if dist.latest_version > dist.version
        ]

    def get_uptodate(
        self, packages: "_ProcessedDists", options: Values
    ) -> "_ProcessedDists":
        return [
            dist
            for dist in self.iter_packages_latest_infos(packages, options)
            if dist.latest_version == dist.version
        ]

    def get_not_required(
        self, packages: "_ProcessedDists", options: Values
    ) -> "_ProcessedDists":
        dep_keys = {
            canonicalize_name(dep.name)
            for dist in packages
            for dep in (dist.iter_dependencies() or ())
        }

        # Create a set to remove duplicate packages, and cast it to a list
        # to keep the return type consistent with get_outdated and
        # get_uptodate
        return list({pkg for pkg in packages if pkg.canonical_name not in dep_keys})

    def iter_packages_latest_infos(
        self, packages: "_ProcessedDists", options: Values
    ) -> Generator["_DistWithLatestInfo", None, None]:
        with self._build_session(options) as session:
            finder = self._build_package_finder(options, session)

            def latest_info(
                dist: "_DistWithLatestInfo",
            ) -> Optional["_DistWithLatestInfo"]:
                all_candidates = finder.find_all_candidates(dist.canonical_name)
                if not options.pre:
                    # Remove prereleases
                    all_candidates = [
                        candidate
                        for candidate in all_candidates
                        if not candidate.version.is_prerelease
                    ]

                evaluator = finder.make_candidate_evaluator(
                    project_name=dist.canonical_name,
                )
                best_candidate = evaluator.sort_best_candidate(all_candidates)
                if best_candidate is None:
                    return None

                remote_version = best_candidate.version
                if best_candidate.link.is_wheel:
                    typ = "wheel"
                else:
                    typ = "sdist"
                dist.latest_version = remote_version
                dist.latest_filetype = typ
                return dist

            for dist in map(latest_info, packages):
                if dist is not None:
                    yield dist

    def output_package_listing(
        self, packages: "_ProcessedDists", options: Values
    ) -> None:
        packages = sorted(
            packages,
            key=lambda dist: dist.canonical_name,
        )
        if options.list_format == "columns" and packages:
            data, header = format_for_columns(packages, options)
            self.output_package_listing_columns(data, header)
        elif options.list_format == "freeze":
            for dist in packages:
                if options.verbose >= 1:
                    write_output(
                        "%s==%s (%s)", dist.raw_name, dist.version, dist.location
                    )
                else:
                    write_output("%s==%s", dist.raw_name, dist.version)
        elif options.list_format == "json":
            write_output(format_for_json(packages, options))

    def output_package_listing_columns(
        self, data: List[List[str]], header: List[str]
    ) -> None:
        # insert the header first: we need to know the size of column names
        if len(data) > 0:
            data.insert(0, header)

        pkg_strings, sizes = tabulate(data)

        # Create and add a separator.
        if len(data) > 0:
            pkg_strings.insert(1, " ".join("-" * x for x in sizes))

        for val in pkg_strings:
            write_output(val)


def format_for_columns(
    pkgs: "_ProcessedDists", options: Values
) -> Tuple[List[List[str]], List[str]]:
    """
    Convert the package data into something usable
    by output_package_listing_columns.
    """
    header = ["Package", "Version"]

    running_outdated = options.outdated
    if running_outdated:
        header.extend(["Latest", "Type"])

    def wheel_build_tag(dist: BaseDistribution) -> Optional[str]:
        try:
            wheel_file = dist.read_text("WHEEL")
        except FileNotFoundError:
            return None
        return Parser().parsestr(wheel_file).get("Build")

    build_tags = [wheel_build_tag(p) for p in pkgs]
    has_build_tags = any(build_tags)
    if has_build_tags:
        header.append("Build")

    if options.verbose >= 1:
        header.append("Location")
    if options.verbose >= 1:
        header.append("Installer")

    has_editables = any(x.editable for x in pkgs)
    if has_editables:
        header.append("Editable project location")

    data = []
    for i, proj in enumerate(pkgs):
        # if we're working on the 'outdated' list, separate out the
        # latest_version and type
        row = [proj.raw_name, proj.raw_version]

        if running_outdated:
            row.append(str(proj.latest_version))
            row.append(proj.latest_filetype)

        if has_build_tags:
            row.append(build_tags[i] or "")

        if has_editables:
            row.append(proj.editable_project_location or "")

        if options.verbose >= 1:
            row.append(proj.location or "")
        if options.verbose >= 1:
            row.append(proj.installer)

        data.append(row)

    return data, header


def format_for_json(packages: "_ProcessedDists", options: Values) -> str:
    data = []
    for dist in packages:
        info = {
            "name": dist.raw_name,
            "version": str(dist.version),
        }
        if options.verbose >= 1:
            info["location"] = dist.location or ""
            info["installer"] = dist.installer
        if options.outdated:
            info["latest_version"] = str(dist.latest_version)
            info["latest_filetype"] = dist.latest_filetype
        editable_project_location = dist.editable_project_location
        if editable_project_location:
            info["editable_project_location"] = editable_project_location
        data.append(info)
    return json.dumps(data)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\_add_newdocs_scalars.py ===
"""
This file is separate from ``_add_newdocs.py`` so that it can be mocked out by
our sphinx ``conf.py`` during doc builds, where we want to avoid showing
platform-dependent information.
"""
import os
import sys

from numpy._core import dtype
from numpy._core import numerictypes as _numerictypes
from numpy._core.function_base import add_newdoc

##############################################################################
#
# Documentation for concrete scalar classes
#
##############################################################################

def numeric_type_aliases(aliases):
    def type_aliases_gen():
        for alias, doc in aliases:
            try:
                alias_type = getattr(_numerictypes, alias)
            except AttributeError:
                # The set of aliases that actually exist varies between platforms
                pass
            else:
                yield (alias_type, alias, doc)
    return list(type_aliases_gen())


possible_aliases = numeric_type_aliases([
    ('int8', '8-bit signed integer (``-128`` to ``127``)'),
    ('int16', '16-bit signed integer (``-32_768`` to ``32_767``)'),
    ('int32', '32-bit signed integer (``-2_147_483_648`` to ``2_147_483_647``)'),
    ('int64', '64-bit signed integer (``-9_223_372_036_854_775_808`` to ``9_223_372_036_854_775_807``)'),
    ('intp', 'Signed integer large enough to fit pointer, compatible with C ``intptr_t``'),
    ('uint8', '8-bit unsigned integer (``0`` to ``255``)'),
    ('uint16', '16-bit unsigned integer (``0`` to ``65_535``)'),
    ('uint32', '32-bit unsigned integer (``0`` to ``4_294_967_295``)'),
    ('uint64', '64-bit unsigned integer (``0`` to ``18_446_744_073_709_551_615``)'),
    ('uintp', 'Unsigned integer large enough to fit pointer, compatible with C ``uintptr_t``'),
    ('float16', '16-bit-precision floating-point number type: sign bit, 5 bits exponent, 10 bits mantissa'),
    ('float32', '32-bit-precision floating-point number type: sign bit, 8 bits exponent, 23 bits mantissa'),
    ('float64', '64-bit precision floating-point number type: sign bit, 11 bits exponent, 52 bits mantissa'),
    ('float96', '96-bit extended-precision floating-point number type'),
    ('float128', '128-bit extended-precision floating-point number type'),
    ('complex64', 'Complex number type composed of 2 32-bit-precision floating-point numbers'),
    ('complex128', 'Complex number type composed of 2 64-bit-precision floating-point numbers'),
    ('complex192', 'Complex number type composed of 2 96-bit extended-precision floating-point numbers'),
    ('complex256', 'Complex number type composed of 2 128-bit extended-precision floating-point numbers'),
    ])


def _get_platform_and_machine():
    try:
        system, _, _, _, machine = os.uname()
    except AttributeError:
        system = sys.platform
        if system == 'win32':
            machine = os.environ.get('PROCESSOR_ARCHITEW6432', '') \
                    or os.environ.get('PROCESSOR_ARCHITECTURE', '')
        else:
            machine = 'unknown'
    return system, machine


_system, _machine = _get_platform_and_machine()
_doc_alias_string = f":Alias on this platform ({_system} {_machine}):"


def add_newdoc_for_scalar_type(obj, fixed_aliases, doc):
    # note: `:field: value` is rST syntax which renders as field lists.
    o = getattr(_numerictypes, obj)

    character_code = dtype(o).char
    canonical_name_doc = "" if obj == o.__name__ else \
                        f":Canonical name: `numpy.{obj}`\n    "
    if fixed_aliases:
        alias_doc = ''.join(f":Alias: `numpy.{alias}`\n    "
                            for alias in fixed_aliases)
    else:
        alias_doc = ''
    alias_doc += ''.join(f"{_doc_alias_string} `numpy.{alias}`: {doc}.\n    "
                         for (alias_type, alias, doc) in possible_aliases if alias_type is o)

    docstring = f"""
    {doc.strip()}

    :Character code: ``'{character_code}'``
    {canonical_name_doc}{alias_doc}
    """

    add_newdoc('numpy._core.numerictypes', obj, docstring)


_bool_docstring = (
    """
    Boolean type (True or False), stored as a byte.

    .. warning::

       The :class:`bool` type is not a subclass of the :class:`int_` type
       (the :class:`bool` is not even a number type). This is different
       than Python's default implementation of :class:`bool` as a
       sub-class of :class:`int`.
    """
)

add_newdoc_for_scalar_type('bool', [], _bool_docstring)

add_newdoc_for_scalar_type('bool_', [], _bool_docstring)

add_newdoc_for_scalar_type('byte', [],
    """
    Signed integer type, compatible with C ``char``.
    """)

add_newdoc_for_scalar_type('short', [],
    """
    Signed integer type, compatible with C ``short``.
    """)

add_newdoc_for_scalar_type('intc', [],
    """
    Signed integer type, compatible with C ``int``.
    """)

# TODO: These docs probably need an if to highlight the default rather than
#       the C-types (and be correct).
add_newdoc_for_scalar_type('int_', [],
    """
    Default signed integer type, 64bit on 64bit systems and 32bit on 32bit
    systems.
    """)

add_newdoc_for_scalar_type('longlong', [],
    """
    Signed integer type, compatible with C ``long long``.
    """)

add_newdoc_for_scalar_type('ubyte', [],
    """
    Unsigned integer type, compatible with C ``unsigned char``.
    """)

add_newdoc_for_scalar_type('ushort', [],
    """
    Unsigned integer type, compatible with C ``unsigned short``.
    """)

add_newdoc_for_scalar_type('uintc', [],
    """
    Unsigned integer type, compatible with C ``unsigned int``.
    """)

add_newdoc_for_scalar_type('uint', [],
    """
    Unsigned signed integer type, 64bit on 64bit systems and 32bit on 32bit
    systems.
    """)

add_newdoc_for_scalar_type('ulonglong', [],
    """
    Signed integer type, compatible with C ``unsigned long long``.
    """)

add_newdoc_for_scalar_type('half', [],
    """
    Half-precision floating-point number type.
    """)

add_newdoc_for_scalar_type('single', [],
    """
    Single-precision floating-point number type, compatible with C ``float``.
    """)

add_newdoc_for_scalar_type('double', [],
    """
    Double-precision floating-point number type, compatible with Python
    :class:`float` and C ``double``.
    """)

add_newdoc_for_scalar_type('longdouble', [],
    """
    Extended-precision floating-point number type, compatible with C
    ``long double`` but not necessarily with IEEE 754 quadruple-precision.
    """)

add_newdoc_for_scalar_type('csingle', [],
    """
    Complex number type composed of two single-precision floating-point
    numbers.
    """)

add_newdoc_for_scalar_type('cdouble', [],
    """
    Complex number type composed of two double-precision floating-point
    numbers, compatible with Python :class:`complex`.
    """)

add_newdoc_for_scalar_type('clongdouble', [],
    """
    Complex number type composed of two extended-precision floating-point
    numbers.
    """)

add_newdoc_for_scalar_type('object_', [],
    """
    Any Python object.
    """)

add_newdoc_for_scalar_type('str_', [],
    r"""
    A unicode string.

    This type strips trailing null codepoints.

    >>> s = np.str_("abc\x00")
    >>> s
    'abc'

    Unlike the builtin :class:`str`, this supports the
    :ref:`python:bufferobjects`, exposing its contents as UCS4:

    >>> m = memoryview(np.str_("abc"))
    >>> m.format
    '3w'
    >>> m.tobytes()
    b'a\x00\x00\x00b\x00\x00\x00c\x00\x00\x00'
    """)

add_newdoc_for_scalar_type('bytes_', [],
    r"""
    A byte string.

    When used in arrays, this type strips trailing null bytes.
    """)

add_newdoc_for_scalar_type('void', [],
    r"""
    np.void(length_or_data, /, dtype=None)

    Create a new structured or unstructured void scalar.

    Parameters
    ----------
    length_or_data : int, array-like, bytes-like, object
       One of multiple meanings (see notes).  The length or
       bytes data of an unstructured void.  Or alternatively,
       the data to be stored in the new scalar when `dtype`
       is provided.
       This can be an array-like, in which case an array may
       be returned.
    dtype : dtype, optional
       If provided the dtype of the new scalar.  This dtype must
       be "void" dtype (i.e. a structured or unstructured void,
       see also :ref:`defining-structured-types`).

       .. versionadded:: 1.24

    Notes
    -----
    For historical reasons and because void scalars can represent both
    arbitrary byte data and structured dtypes, the void constructor
    has three calling conventions:

    1. ``np.void(5)`` creates a ``dtype="V5"`` scalar filled with five
       ``\0`` bytes.  The 5 can be a Python or NumPy integer.
    2. ``np.void(b"bytes-like")`` creates a void scalar from the byte string.
       The dtype itemsize will match the byte string length, here ``"V10"``.
    3. When a ``dtype=`` is passed the call is roughly the same as an
       array creation.  However, a void scalar rather than array is returned.

    Please see the examples which show all three different conventions.

    Examples
    --------
    >>> np.void(5)
    np.void(b'\x00\x00\x00\x00\x00')
    >>> np.void(b'abcd')
    np.void(b'\x61\x62\x63\x64')
    >>> np.void((3.2, b'eggs'), dtype="d,S5")
    np.void((3.2, b'eggs'), dtype=[('f0', '<f8'), ('f1', 'S5')])
    >>> np.void(3, dtype=[('x', np.int8), ('y', np.int8)])
    np.void((3, 3), dtype=[('x', 'i1'), ('y', 'i1')])

    """)

add_newdoc_for_scalar_type('datetime64', [],
    """
    If created from a 64-bit integer, it represents an offset from
    ``1970-01-01T00:00:00``.
    If created from string, the string can be in ISO 8601 date
    or datetime format.

    When parsing a string to create a datetime object, if the string contains
    a trailing timezone (A 'Z' or a timezone offset), the timezone will be
    dropped and a User Warning is given.

    Datetime64 objects should be considered to be UTC and therefore have an
    offset of +0000.

    >>> np.datetime64(10, 'Y')
    np.datetime64('1980')
    >>> np.datetime64('1980', 'Y')
    np.datetime64('1980')
    >>> np.datetime64(10, 'D')
    np.datetime64('1970-01-11')

    See :ref:`arrays.datetime` for more information.
    """)

add_newdoc_for_scalar_type('timedelta64', [],
    """
    A timedelta stored as a 64-bit integer.

    See :ref:`arrays.datetime` for more information.
    """)

add_newdoc('numpy._core.numerictypes', "integer", ('is_integer',
    """
    integer.is_integer() -> bool

    Return ``True`` if the number is finite with integral value.

    .. versionadded:: 1.22

    Examples
    --------
    >>> import numpy as np
    >>> np.int64(-2).is_integer()
    True
    >>> np.uint32(5).is_integer()
    True
    """))

# TODO: work out how to put this on the base class, np.floating
for float_name in ('half', 'single', 'double', 'longdouble'):
    add_newdoc('numpy._core.numerictypes', float_name, ('as_integer_ratio',
        f"""
        {float_name}.as_integer_ratio() -> (int, int)

        Return a pair of integers, whose ratio is exactly equal to the original
        floating point number, and with a positive denominator.
        Raise `OverflowError` on infinities and a `ValueError` on NaNs.

        >>> np.{float_name}(10.0).as_integer_ratio()
        (10, 1)
        >>> np.{float_name}(0.0).as_integer_ratio()
        (0, 1)
        >>> np.{float_name}(-.25).as_integer_ratio()
        (-1, 4)
        """))

    add_newdoc('numpy._core.numerictypes', float_name, ('is_integer',
        f"""
        {float_name}.is_integer() -> bool

        Return ``True`` if the floating point number is finite with integral
        value, and ``False`` otherwise.

        .. versionadded:: 1.22

        Examples
        --------
        >>> np.{float_name}(-2.0).is_integer()
        True
        >>> np.{float_name}(3.2).is_integer()
        False
        """))

for int_name in ('int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32',
        'int64', 'uint64', 'int64', 'uint64', 'int64', 'uint64'):
    # Add negative examples for signed cases by checking typecode
    add_newdoc('numpy._core.numerictypes', int_name, ('bit_count',
        f"""
        {int_name}.bit_count() -> int

        Computes the number of 1-bits in the absolute value of the input.
        Analogous to the builtin `int.bit_count` or ``popcount`` in C++.

        Examples
        --------
        >>> np.{int_name}(127).bit_count()
        7""" +
        (f"""
        >>> np.{int_name}(-127).bit_count()
        7
        """ if dtype(int_name).char.islower() else "")))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\worksheet\_writer.py ===
# Copyright (c) 2010-2024 openpyxl

import atexit
from collections import defaultdict
from io import BytesIO
import os
from tempfile import NamedTemporaryFile
from warnings import warn

from openpyxl.xml.functions import xmlfile
from openpyxl.xml.constants import SHEET_MAIN_NS

from openpyxl.comments.comment_sheet import CommentRecord
from openpyxl.packaging.relationship import Relationship, RelationshipList
from openpyxl.styles.differential import DifferentialStyle

from .dimensions import SheetDimension
from .hyperlink import HyperlinkList
from .merge import MergeCell, MergeCells
from .related import Related
from .table import TablePartList

from openpyxl.cell._writer import write_cell


ALL_TEMP_FILES = []

@atexit.register
def _openpyxl_shutdown():
    for path in ALL_TEMP_FILES:
        if os.path.exists(path):
            os.remove(path)


def create_temporary_file(suffix=''):
    fobj = NamedTemporaryFile(mode='w+', suffix=suffix,
                              prefix='openpyxl.', delete=False)
    filename = fobj.name
    fobj.close()
    ALL_TEMP_FILES.append(filename)
    return filename


class WorksheetWriter:


    def __init__(self, ws, out=None):
        self.ws = ws
        self.ws._hyperlinks = []
        self.ws._comments = []
        if out is None:
            out = create_temporary_file()
        self.out = out
        self._rels = RelationshipList()
        self.xf = self.get_stream()
        next(self.xf) # start generator


    def write_properties(self):
        props = self.ws.sheet_properties
        self.xf.send(props.to_tree())


    def write_dimensions(self):
        """
        Write worksheet size if known
        """
        ref = getattr(self.ws, 'calculate_dimension', None)
        if ref:
            dim = SheetDimension(ref())
            self.xf.send(dim.to_tree())


    def write_format(self):
        self.ws.sheet_format.outlineLevelCol = self.ws.column_dimensions.max_outline
        fmt = self.ws.sheet_format
        self.xf.send(fmt.to_tree())


    def write_views(self):
        views = self.ws.views
        self.xf.send(views.to_tree())


    def write_cols(self):
        cols = self.ws.column_dimensions
        self.xf.send(cols.to_tree())


    def write_top(self):
        """
        Write all elements up to rows:
        properties
        dimensions
        views
        format
        cols
        """
        self.write_properties()
        self.write_dimensions()
        self.write_views()
        self.write_format()
        self.write_cols()


    def rows(self):
        """Return all rows, and any cells that they contain"""
        # order cells by row
        rows = defaultdict(list)
        for (row, col), cell in sorted(self.ws._cells.items()):
            rows[row].append(cell)

        # add empty rows if styling has been applied
        for row in self.ws.row_dimensions.keys() - rows.keys():
            rows[row] = []

        return sorted(rows.items())


    def write_rows(self):
        xf = self.xf.send(True)

        with xf.element("sheetData"):
            for row_idx, row in self.rows():
                self.write_row(xf, row, row_idx)

        self.xf.send(None) # return control to generator


    def write_row(self, xf, row, row_idx):
        attrs = {'r': f"{row_idx}"}
        dims = self.ws.row_dimensions
        attrs.update(dims.get(row_idx, {}))

        with xf.element("row", attrs):

            for cell in row:
                if cell._comment is not None:
                    comment = CommentRecord.from_cell(cell)
                    self.ws._comments.append(comment)
                if (
                    cell._value is None
                    and not cell.has_style
                    and not cell._comment
                    ):
                    continue
                write_cell(xf, self.ws, cell, cell.has_style)


    def write_protection(self):
        prot = self.ws.protection
        if prot:
            self.xf.send(prot.to_tree())


    def write_scenarios(self):
        scenarios = self.ws.scenarios
        if scenarios:
            self.xf.send(scenarios.to_tree())


    def write_filter(self):
        flt = self.ws.auto_filter
        if flt:
            self.xf.send(flt.to_tree())


    def write_sort(self):
        """
        As per discusion with the OOXML Working Group global sort state is not required.
        openpyxl never reads it from existing files
        """
        pass


    def write_merged_cells(self):
        merged = self.ws.merged_cells
        if merged:
            cells = [MergeCell(str(ref)) for ref in self.ws.merged_cells]
            self.xf.send(MergeCells(mergeCell=cells).to_tree())


    def write_formatting(self):
        df = DifferentialStyle()
        wb = self.ws.parent
        for cf in self.ws.conditional_formatting:
            for rule in cf.rules:
                if rule.dxf and rule.dxf != df:
                    rule.dxfId = wb._differential_styles.add(rule.dxf)
            self.xf.send(cf.to_tree())


    def write_validations(self):
        dv = self.ws.data_validations
        if dv:
            self.xf.send(dv.to_tree())


    def write_hyperlinks(self):

        links = self.ws._hyperlinks

        for link in links:
            if link.target:
                rel = Relationship(type="hyperlink", TargetMode="External", Target=link.target)
                self._rels.append(rel)
                link.id = rel.id

        if links:
            self.xf.send(HyperlinkList(links).to_tree())


    def write_print(self):
        print_options = self.ws.print_options
        if print_options:
            self.xf.send(print_options.to_tree())


    def write_margins(self):
        margins = self.ws.page_margins
        if margins:
            self.xf.send(margins.to_tree())


    def write_page(self):
        setup = self.ws.page_setup
        if setup:
            self.xf.send(setup.to_tree())


    def write_header(self):
        hf = self.ws.HeaderFooter
        if hf:
            self.xf.send(hf.to_tree())


    def write_breaks(self):
        brks = (self.ws.row_breaks, self.ws.col_breaks)
        for brk in brks:
            if brk:
                self.xf.send(brk.to_tree())


    def write_drawings(self):
        if self.ws._charts or self.ws._images:
            rel = Relationship(type="drawing", Target="")
            self._rels.append(rel)
            drawing = Related()
            drawing.id = rel.id
            self.xf.send(drawing.to_tree("drawing"))


    def write_legacy(self):
        """
        Comments & VBA controls use VML and require an additional element
        that is no longer in the specification.
        """
        if (self.ws.legacy_drawing is not None or self.ws._comments):
            legacy = Related(id="anysvml")
            self.xf.send(legacy.to_tree("legacyDrawing"))


    def write_tables(self):
        tables = TablePartList()

        for table in self.ws.tables.values():
            if not table.tableColumns:
                table._initialise_columns()
                if table.headerRowCount:
                    try:
                        row = self.ws[table.ref][0]
                        for cell, col in zip(row, table.tableColumns):
                            if cell.data_type != "s":
                                warn("File may not be readable: column headings must be strings.")
                            col.name = str(cell.value)
                    except TypeError:
                        warn("Column headings are missing, file may not be readable")
            rel = Relationship(Type=table._rel_type, Target="")
            self._rels.append(rel)
            table._rel_id = rel.Id
            tables.append(Related(id=rel.Id))

        if tables:
            self.xf.send(tables.to_tree())


    def get_stream(self):
        with xmlfile(self.out) as xf:
            with xf.element("worksheet", xmlns=SHEET_MAIN_NS):
                try:
                    while True:
                        el = (yield)
                        if el is True:
                            yield xf
                        elif el is None: # et_xmlfile chokes
                            continue
                        else:
                            xf.write(el)
                except GeneratorExit:
                    pass


    def write_tail(self):
        """
        Write all elements after the rows
        calc properties
        protection
        protected ranges #
        scenarios
        filters
        sorts # always ignored
        data consolidation #
        custom views #
        merged cells
        phonetic properties #
        conditional formatting
        data validation
        hyperlinks
        print options
        page margins
        page setup
        header
        row breaks
        col breaks
        custom properties #
        cell watches #
        ignored errors #
        smart tags #
        drawing
        drawingHF #
        background #
        OLE objects #
        controls #
        web publishing #
        tables
        """
        self.write_protection()
        self.write_scenarios()
        self.write_filter()
        self.write_merged_cells()
        self.write_formatting()
        self.write_validations()
        self.write_hyperlinks()
        self.write_print()
        self.write_margins()
        self.write_page()
        self.write_header()
        self.write_breaks()
        self.write_drawings()
        self.write_legacy()
        self.write_tables()


    def write(self):
        """
        High level
        """
        self.write_top()
        self.write_rows()
        self.write_tail()
        self.close()


    def close(self):
        """
        Close the context manager
        """
        if self.xf:
            self.xf.close()


    def read(self):
        """
        Close the context manager and return serialised XML
        """
        self.close()
        if isinstance(self.out, BytesIO):
            return self.out.getvalue()
        with open(self.out, "rb") as src:
            out = src.read()

        return out


    def cleanup(self):
        """
        Remove tempfile
        """
        os.remove(self.out)
        ALL_TEMP_FILES.remove(self.out)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\event\registry.py ===
# event/registry.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

"""Provides managed registration services on behalf of :func:`.listen`
arguments.

By "managed registration", we mean that event listening functions and
other objects can be added to various collections in such a way that their
membership in all those collections can be revoked at once, based on
an equivalent :class:`._EventKey`.

"""
from __future__ import annotations

import collections
import types
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Deque
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union
import weakref

from .. import exc
from .. import util

if typing.TYPE_CHECKING:
    from .attr import RefCollection
    from .base import dispatcher

_ListenerFnType = Callable[..., Any]
_ListenerFnKeyType = Union[int, Tuple[int, int]]
_EventKeyTupleType = Tuple[int, str, _ListenerFnKeyType]


_ET = TypeVar("_ET", bound="EventTarget")


class EventTarget:
    """represents an event target, that is, something we can listen on
    either with that target as a class or as an instance.

    Examples include:  Connection, Mapper, Table, Session,
    InstrumentedAttribute, Engine, Pool, Dialect.

    """

    __slots__ = ()

    dispatch: dispatcher[Any]


_RefCollectionToListenerType = Dict[
    "weakref.ref[RefCollection[Any]]",
    "weakref.ref[_ListenerFnType]",
]

_key_to_collection: Dict[_EventKeyTupleType, _RefCollectionToListenerType] = (
    collections.defaultdict(dict)
)
"""
Given an original listen() argument, can locate all
listener collections and the listener fn contained

(target, identifier, fn) -> {
                            ref(listenercollection) -> ref(listener_fn)
                            ref(listenercollection) -> ref(listener_fn)
                            ref(listenercollection) -> ref(listener_fn)
                        }
"""

_ListenerToEventKeyType = Dict[
    "weakref.ref[_ListenerFnType]",
    _EventKeyTupleType,
]
_collection_to_key: Dict[
    weakref.ref[RefCollection[Any]],
    _ListenerToEventKeyType,
] = collections.defaultdict(dict)
"""
Given a _ListenerCollection or _ClsLevelListener, can locate
all the original listen() arguments and the listener fn contained

ref(listenercollection) -> {
                            ref(listener_fn) -> (target, identifier, fn),
                            ref(listener_fn) -> (target, identifier, fn),
                            ref(listener_fn) -> (target, identifier, fn),
                        }
"""


def _collection_gced(ref: weakref.ref[Any]) -> None:
    # defaultdict, so can't get a KeyError
    if not _collection_to_key or ref not in _collection_to_key:
        return

    ref = cast("weakref.ref[RefCollection[EventTarget]]", ref)

    listener_to_key = _collection_to_key.pop(ref)
    for key in listener_to_key.values():
        if key in _key_to_collection:
            # defaultdict, so can't get a KeyError
            dispatch_reg = _key_to_collection[key]
            dispatch_reg.pop(ref)
            if not dispatch_reg:
                _key_to_collection.pop(key)


def _stored_in_collection(
    event_key: _EventKey[_ET], owner: RefCollection[_ET]
) -> bool:
    key = event_key._key

    dispatch_reg = _key_to_collection[key]

    owner_ref = owner.ref
    listen_ref = weakref.ref(event_key._listen_fn)

    if owner_ref in dispatch_reg:
        return False

    dispatch_reg[owner_ref] = listen_ref

    listener_to_key = _collection_to_key[owner_ref]
    listener_to_key[listen_ref] = key

    return True


def _removed_from_collection(
    event_key: _EventKey[_ET], owner: RefCollection[_ET]
) -> None:
    key = event_key._key

    dispatch_reg = _key_to_collection[key]

    listen_ref = weakref.ref(event_key._listen_fn)

    owner_ref = owner.ref
    dispatch_reg.pop(owner_ref, None)
    if not dispatch_reg:
        del _key_to_collection[key]

    if owner_ref in _collection_to_key:
        listener_to_key = _collection_to_key[owner_ref]
        # see #12216 - this guards against a removal that already occurred
        # here. however, I cannot come up with a test that shows any negative
        # side effects occurring from this removal happening, even though an
        # event key may still be referenced from a clsleveldispatch here
        listener_to_key.pop(listen_ref, None)


def _stored_in_collection_multi(
    newowner: RefCollection[_ET],
    oldowner: RefCollection[_ET],
    elements: Iterable[_ListenerFnType],
) -> None:
    if not elements:
        return

    oldowner_ref = oldowner.ref
    newowner_ref = newowner.ref

    old_listener_to_key = _collection_to_key[oldowner_ref]
    new_listener_to_key = _collection_to_key[newowner_ref]

    for listen_fn in elements:
        listen_ref = weakref.ref(listen_fn)
        try:
            key = old_listener_to_key[listen_ref]
        except KeyError:
            # can occur during interpreter shutdown.
            # see #6740
            continue

        try:
            dispatch_reg = _key_to_collection[key]
        except KeyError:
            continue

        if newowner_ref in dispatch_reg:
            assert dispatch_reg[newowner_ref] == listen_ref
        else:
            dispatch_reg[newowner_ref] = listen_ref

        new_listener_to_key[listen_ref] = key


def _clear(
    owner: RefCollection[_ET],
    elements: Iterable[_ListenerFnType],
) -> None:
    if not elements:
        return

    owner_ref = owner.ref
    listener_to_key = _collection_to_key[owner_ref]
    for listen_fn in elements:
        listen_ref = weakref.ref(listen_fn)
        key = listener_to_key[listen_ref]
        dispatch_reg = _key_to_collection[key]
        dispatch_reg.pop(owner_ref, None)

        if not dispatch_reg:
            del _key_to_collection[key]


class _EventKey(Generic[_ET]):
    """Represent :func:`.listen` arguments."""

    __slots__ = (
        "target",
        "identifier",
        "fn",
        "fn_key",
        "fn_wrap",
        "dispatch_target",
    )

    target: _ET
    identifier: str
    fn: _ListenerFnType
    fn_key: _ListenerFnKeyType
    dispatch_target: Any
    _fn_wrap: Optional[_ListenerFnType]

    def __init__(
        self,
        target: _ET,
        identifier: str,
        fn: _ListenerFnType,
        dispatch_target: Any,
        _fn_wrap: Optional[_ListenerFnType] = None,
    ):
        self.target = target
        self.identifier = identifier
        self.fn = fn
        if isinstance(fn, types.MethodType):
            self.fn_key = id(fn.__func__), id(fn.__self__)
        else:
            self.fn_key = id(fn)
        self.fn_wrap = _fn_wrap
        self.dispatch_target = dispatch_target

    @property
    def _key(self) -> _EventKeyTupleType:
        return (id(self.target), self.identifier, self.fn_key)

    def with_wrapper(self, fn_wrap: _ListenerFnType) -> _EventKey[_ET]:
        if fn_wrap is self._listen_fn:
            return self
        else:
            return _EventKey(
                self.target,
                self.identifier,
                self.fn,
                self.dispatch_target,
                _fn_wrap=fn_wrap,
            )

    def with_dispatch_target(self, dispatch_target: Any) -> _EventKey[_ET]:
        if dispatch_target is self.dispatch_target:
            return self
        else:
            return _EventKey(
                self.target,
                self.identifier,
                self.fn,
                dispatch_target,
                _fn_wrap=self.fn_wrap,
            )

    def listen(self, *args: Any, **kw: Any) -> None:
        once = kw.pop("once", False)
        once_unless_exception = kw.pop("_once_unless_exception", False)
        named = kw.pop("named", False)

        target, identifier, fn = (
            self.dispatch_target,
            self.identifier,
            self._listen_fn,
        )

        dispatch_collection = getattr(target.dispatch, identifier)

        adjusted_fn = dispatch_collection._adjust_fn_spec(fn, named)

        self = self.with_wrapper(adjusted_fn)

        stub_function = getattr(
            self.dispatch_target.dispatch._events, self.identifier
        )
        if hasattr(stub_function, "_sa_warn"):
            stub_function._sa_warn()

        if once or once_unless_exception:
            self.with_wrapper(
                util.only_once(
                    self._listen_fn, retry_on_exception=once_unless_exception
                )
            ).listen(*args, **kw)
        else:
            self.dispatch_target.dispatch._listen(self, *args, **kw)

    def remove(self) -> None:
        key = self._key

        if key not in _key_to_collection:
            raise exc.InvalidRequestError(
                "No listeners found for event %s / %r / %s "
                % (self.target, self.identifier, self.fn)
            )

        dispatch_reg = _key_to_collection.pop(key)

        for collection_ref, listener_ref in dispatch_reg.items():
            collection = collection_ref()
            listener_fn = listener_ref()
            if collection is not None and listener_fn is not None:
                collection.remove(self.with_wrapper(listener_fn))

    def contains(self) -> bool:
        """Return True if this event key is registered to listen."""
        return self._key in _key_to_collection

    def base_listen(
        self,
        propagate: bool = False,
        insert: bool = False,
        named: bool = False,
        retval: Optional[bool] = None,
        asyncio: bool = False,
    ) -> None:
        target, identifier = self.dispatch_target, self.identifier

        dispatch_collection = getattr(target.dispatch, identifier)

        for_modify = dispatch_collection.for_modify(target.dispatch)
        if asyncio:
            for_modify._set_asyncio()

        if insert:
            for_modify.insert(self, propagate)
        else:
            for_modify.append(self, propagate)

    @property
    def _listen_fn(self) -> _ListenerFnType:
        return self.fn_wrap or self.fn

    def append_to_list(
        self,
        owner: RefCollection[_ET],
        list_: Deque[_ListenerFnType],
    ) -> bool:
        if _stored_in_collection(self, owner):
            list_.append(self._listen_fn)
            return True
        else:
            return False

    def remove_from_list(
        self,
        owner: RefCollection[_ET],
        list_: Deque[_ListenerFnType],
    ) -> None:
        _removed_from_collection(self, owner)
        list_.remove(self._listen_fn)

    def prepend_to_list(
        self,
        owner: RefCollection[_ET],
        list_: Deque[_ListenerFnType],
    ) -> bool:
        if _stored_in_collection(self, owner):
            list_.appendleft(self._listen_fn)
            return True
        else:
            return False

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\assumptions\predicates\order.py ===
from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher


class NegativePredicate(Predicate):
    r"""
    Negative number predicate.

    Explanation
    ===========

    ``Q.negative(x)`` is true iff ``x`` is a real number and :math:`x < 0`, that is,
    it is in the interval :math:`(-\infty, 0)`.  Note in particular that negative
    infinity is not negative.

    A few important facts about negative numbers:

    - Note that ``Q.nonnegative`` and ``~Q.negative`` are *not* the same
        thing. ``~Q.negative(x)`` simply means that ``x`` is not negative,
        whereas ``Q.nonnegative(x)`` means that ``x`` is real and not
        negative, i.e., ``Q.nonnegative(x)`` is logically equivalent to
        ``Q.zero(x) | Q.positive(x)``.  So for example, ``~Q.negative(I)`` is
        true, whereas ``Q.nonnegative(I)`` is false.

    - See the documentation of ``Q.real`` for more information about
        related facts.

    Examples
    ========

    >>> from sympy import Q, ask, symbols, I
    >>> x = symbols('x')
    >>> ask(Q.negative(x), Q.real(x) & ~Q.positive(x) & ~Q.zero(x))
    True
    >>> ask(Q.negative(-1))
    True
    >>> ask(Q.nonnegative(I))
    False
    >>> ask(~Q.negative(I))
    True

    """
    name = 'negative'
    handler = Dispatcher(
        "NegativeHandler",
        doc=("Handler for Q.negative. Test that an expression is strictly less"
        " than zero.")
    )


class NonNegativePredicate(Predicate):
    """
    Nonnegative real number predicate.

    Explanation
    ===========

    ``ask(Q.nonnegative(x))`` is true iff ``x`` belongs to the set of
    positive numbers including zero.

    - Note that ``Q.nonnegative`` and ``~Q.negative`` are *not* the same
        thing. ``~Q.negative(x)`` simply means that ``x`` is not negative,
        whereas ``Q.nonnegative(x)`` means that ``x`` is real and not
        negative, i.e., ``Q.nonnegative(x)`` is logically equivalent to
        ``Q.zero(x) | Q.positive(x)``.  So for example, ``~Q.negative(I)`` is
        true, whereas ``Q.nonnegative(I)`` is false.

    Examples
    ========

    >>> from sympy import Q, ask, I
    >>> ask(Q.nonnegative(1))
    True
    >>> ask(Q.nonnegative(0))
    True
    >>> ask(Q.nonnegative(-1))
    False
    >>> ask(Q.nonnegative(I))
    False
    >>> ask(Q.nonnegative(-I))
    False

    """
    name = 'nonnegative'
    handler = Dispatcher(
        "NonNegativeHandler",
        doc=("Handler for Q.nonnegative.")
    )


class NonZeroPredicate(Predicate):
    """
    Nonzero real number predicate.

    Explanation
    ===========

    ``ask(Q.nonzero(x))`` is true iff ``x`` is real and ``x`` is not zero.  Note in
    particular that ``Q.nonzero(x)`` is false if ``x`` is not real.  Use
    ``~Q.zero(x)`` if you want the negation of being zero without any real
    assumptions.

    A few important facts about nonzero numbers:

    - ``Q.nonzero`` is logically equivalent to ``Q.positive | Q.negative``.

    - See the documentation of ``Q.real`` for more information about
        related facts.

    Examples
    ========

    >>> from sympy import Q, ask, symbols, I, oo
    >>> x = symbols('x')
    >>> print(ask(Q.nonzero(x), ~Q.zero(x)))
    None
    >>> ask(Q.nonzero(x), Q.positive(x))
    True
    >>> ask(Q.nonzero(x), Q.zero(x))
    False
    >>> ask(Q.nonzero(0))
    False
    >>> ask(Q.nonzero(I))
    False
    >>> ask(~Q.zero(I))
    True
    >>> ask(Q.nonzero(oo))
    False

    """
    name = 'nonzero'
    handler = Dispatcher(
        "NonZeroHandler",
        doc=("Handler for key 'nonzero'. Test that an expression is not identically"
        " zero.")
    )


class ZeroPredicate(Predicate):
    """
    Zero number predicate.

    Explanation
    ===========

    ``ask(Q.zero(x))`` is true iff the value of ``x`` is zero.

    Examples
    ========

    >>> from sympy import ask, Q, oo, symbols
    >>> x, y = symbols('x, y')
    >>> ask(Q.zero(0))
    True
    >>> ask(Q.zero(1/oo))
    True
    >>> print(ask(Q.zero(0*oo)))
    None
    >>> ask(Q.zero(1))
    False
    >>> ask(Q.zero(x*y), Q.zero(x) | Q.zero(y))
    True

    """
    name = 'zero'
    handler = Dispatcher(
        "ZeroHandler",
        doc="Handler for key 'zero'."
    )


class NonPositivePredicate(Predicate):
    """
    Nonpositive real number predicate.

    Explanation
    ===========

    ``ask(Q.nonpositive(x))`` is true iff ``x`` belongs to the set of
    negative numbers including zero.

    - Note that ``Q.nonpositive`` and ``~Q.positive`` are *not* the same
        thing. ``~Q.positive(x)`` simply means that ``x`` is not positive,
        whereas ``Q.nonpositive(x)`` means that ``x`` is real and not
        positive, i.e., ``Q.nonpositive(x)`` is logically equivalent to
        `Q.negative(x) | Q.zero(x)``.  So for example, ``~Q.positive(I)`` is
        true, whereas ``Q.nonpositive(I)`` is false.

    Examples
    ========

    >>> from sympy import Q, ask, I

    >>> ask(Q.nonpositive(-1))
    True
    >>> ask(Q.nonpositive(0))
    True
    >>> ask(Q.nonpositive(1))
    False
    >>> ask(Q.nonpositive(I))
    False
    >>> ask(Q.nonpositive(-I))
    False

    """
    name = 'nonpositive'
    handler = Dispatcher(
        "NonPositiveHandler",
        doc="Handler for key 'nonpositive'."
    )


class PositivePredicate(Predicate):
    r"""
    Positive real number predicate.

    Explanation
    ===========

    ``Q.positive(x)`` is true iff ``x`` is real and `x > 0`, that is if ``x``
    is in the interval `(0, \infty)`.  In particular, infinity is not
    positive.

    A few important facts about positive numbers:

    - Note that ``Q.nonpositive`` and ``~Q.positive`` are *not* the same
        thing. ``~Q.positive(x)`` simply means that ``x`` is not positive,
        whereas ``Q.nonpositive(x)`` means that ``x`` is real and not
        positive, i.e., ``Q.nonpositive(x)`` is logically equivalent to
        `Q.negative(x) | Q.zero(x)``.  So for example, ``~Q.positive(I)`` is
        true, whereas ``Q.nonpositive(I)`` is false.

    - See the documentation of ``Q.real`` for more information about
        related facts.

    Examples
    ========

    >>> from sympy import Q, ask, symbols, I
    >>> x = symbols('x')
    >>> ask(Q.positive(x), Q.real(x) & ~Q.negative(x) & ~Q.zero(x))
    True
    >>> ask(Q.positive(1))
    True
    >>> ask(Q.nonpositive(I))
    False
    >>> ask(~Q.positive(I))
    True

    """
    name = 'positive'
    handler = Dispatcher(
        "PositiveHandler",
        doc=("Handler for key 'positive'. Test that an expression is strictly"
        " greater than zero.")
    )


class ExtendedPositivePredicate(Predicate):
    r"""
    Positive extended real number predicate.

    Explanation
    ===========

    ``Q.extended_positive(x)`` is true iff ``x`` is extended real and
    `x > 0`, that is if ``x`` is in the interval `(0, \infty]`.

    Examples
    ========

    >>> from sympy import ask, I, oo, Q
    >>> ask(Q.extended_positive(1))
    True
    >>> ask(Q.extended_positive(oo))
    True
    >>> ask(Q.extended_positive(I))
    False

    """
    name = 'extended_positive'
    handler = Dispatcher("ExtendedPositiveHandler")


class ExtendedNegativePredicate(Predicate):
    r"""
    Negative extended real number predicate.

    Explanation
    ===========

    ``Q.extended_negative(x)`` is true iff ``x`` is extended real and
    `x < 0`, that is if ``x`` is in the interval `[-\infty, 0)`.

    Examples
    ========

    >>> from sympy import ask, I, oo, Q
    >>> ask(Q.extended_negative(-1))
    True
    >>> ask(Q.extended_negative(-oo))
    True
    >>> ask(Q.extended_negative(-I))
    False

    """
    name = 'extended_negative'
    handler = Dispatcher("ExtendedNegativeHandler")


class ExtendedNonZeroPredicate(Predicate):
    """
    Nonzero extended real number predicate.

    Explanation
    ===========

    ``ask(Q.extended_nonzero(x))`` is true iff ``x`` is extended real and
    ``x`` is not zero.

    Examples
    ========

    >>> from sympy import ask, I, oo, Q
    >>> ask(Q.extended_nonzero(-1))
    True
    >>> ask(Q.extended_nonzero(oo))
    True
    >>> ask(Q.extended_nonzero(I))
    False

    """
    name = 'extended_nonzero'
    handler = Dispatcher("ExtendedNonZeroHandler")


class ExtendedNonPositivePredicate(Predicate):
    """
    Nonpositive extended real number predicate.

    Explanation
    ===========

    ``ask(Q.extended_nonpositive(x))`` is true iff ``x`` is extended real and
    ``x`` is not positive.

    Examples
    ========

    >>> from sympy import ask, I, oo, Q
    >>> ask(Q.extended_nonpositive(-1))
    True
    >>> ask(Q.extended_nonpositive(oo))
    False
    >>> ask(Q.extended_nonpositive(0))
    True
    >>> ask(Q.extended_nonpositive(I))
    False

    """
    name = 'extended_nonpositive'
    handler = Dispatcher("ExtendedNonPositiveHandler")


class ExtendedNonNegativePredicate(Predicate):
    """
    Nonnegative extended real number predicate.

    Explanation
    ===========

    ``ask(Q.extended_nonnegative(x))`` is true iff ``x`` is extended real and
    ``x`` is not negative.

    Examples
    ========

    >>> from sympy import ask, I, oo, Q
    >>> ask(Q.extended_nonnegative(-1))
    False
    >>> ask(Q.extended_nonnegative(oo))
    True
    >>> ask(Q.extended_nonnegative(0))
    True
    >>> ask(Q.extended_nonnegative(I))
    False

    """
    name = 'extended_nonnegative'
    handler = Dispatcher("ExtendedNonNegativeHandler")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\preview.py ===
import os
from os.path import join
import shutil
import tempfile
from pathlib import Path

try:
    from subprocess import STDOUT, CalledProcessError, check_output
except ImportError:
    pass

from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.misc import debug
from .latex import latex

__doctest_requires__ = {('preview',): ['pyglet']}


def _check_output_no_window(*args, **kwargs):
    # Avoid showing a cmd.exe window when running this
    # on Windows
    if os.name == 'nt':
        creation_flag = 0x08000000 # CREATE_NO_WINDOW
    else:
        creation_flag = 0 # Default value
    return check_output(*args, creationflags=creation_flag, **kwargs)


def system_default_viewer(fname, fmt):
    """ Open fname with the default system viewer.

    In practice, it is impossible for python to know when the system viewer is
    done. For this reason, we ensure the passed file will not be deleted under
    it, and this function does not attempt to block.
    """
    # copy to a new temporary file that will not be deleted
    with tempfile.NamedTemporaryFile(prefix='sympy-preview-',
                                     suffix=os.path.splitext(fname)[1],
                                     delete=False) as temp_f:
        with open(fname, 'rb') as f:
            shutil.copyfileobj(f, temp_f)

    import platform
    if platform.system() == 'Darwin':
        import subprocess
        subprocess.call(('open', temp_f.name))
    elif platform.system() == 'Windows':
        os.startfile(temp_f.name)
    else:
        import subprocess
        subprocess.call(('xdg-open', temp_f.name))


def pyglet_viewer(fname, fmt):
    try:
        from pyglet import window, image, gl
        from pyglet.window import key
        from pyglet.image.codecs import ImageDecodeException
    except ImportError:
        raise ImportError("pyglet is required for preview.\n visit https://pyglet.org/")

    try:
        img = image.load(fname)
    except ImageDecodeException:
        raise ValueError("pyglet preview does not work for '{}' files.".format(fmt))

    offset = 25

    config = gl.Config(double_buffer=False)
    win = window.Window(
        width=img.width + 2*offset,
        height=img.height + 2*offset,
        caption="SymPy",
        resizable=False,
        config=config
    )

    win.set_vsync(False)

    try:
        def on_close():
            win.has_exit = True

        win.on_close = on_close

        def on_key_press(symbol, modifiers):
            if symbol in [key.Q, key.ESCAPE]:
                on_close()

        win.on_key_press = on_key_press

        def on_expose():
            gl.glClearColor(1.0, 1.0, 1.0, 1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)

            img.blit(
                (win.width - img.width) / 2,
                (win.height - img.height) / 2
            )

        win.on_expose = on_expose

        while not win.has_exit:
            win.dispatch_events()
            win.flip()
    except KeyboardInterrupt:
        pass

    win.close()


def _get_latex_main(expr, *, preamble=None, packages=(), extra_preamble=None,
                    euler=True, fontsize=None, **latex_settings):
    """
    Generate string of a LaTeX document rendering ``expr``.
    """
    if preamble is None:
        actual_packages = packages + ("amsmath", "amsfonts")
        if euler:
            actual_packages += ("euler",)
        package_includes = "\n" + "\n".join(["\\usepackage{%s}" % p
                                             for p in actual_packages])
        if extra_preamble:
            package_includes += extra_preamble

        if not fontsize:
            fontsize = "12pt"
        elif isinstance(fontsize, int):
            fontsize = "{}pt".format(fontsize)
        preamble = r"""\documentclass[varwidth,%s]{standalone}
%s

\begin{document}
""" % (fontsize, package_includes)
    else:
        if packages or extra_preamble:
            raise ValueError("The \"packages\" or \"extra_preamble\" keywords"
                             "must not be set if a "
                             "custom LaTeX preamble was specified")

    if isinstance(expr, str):
        latex_string = expr
    else:
        latex_string = ('$\\displaystyle ' +
                        latex(expr, mode='plain', **latex_settings) +
                        '$')

    return preamble + '\n' + latex_string + '\n\n' + r"\end{document}"


@doctest_depends_on(exe=('latex', 'dvipng'), modules=('pyglet',),
            disable_viewers=('evince', 'gimp', 'superior-dvi-viewer'))
def preview(expr, output='png', viewer=None, euler=True, packages=(),
            filename=None, outputbuffer=None, preamble=None, dvioptions=None,
            outputTexFile=None, extra_preamble=None, fontsize=None,
            **latex_settings):
    r"""
    View expression or LaTeX markup in PNG, DVI, PostScript or PDF form.

    If the expr argument is an expression, it will be exported to LaTeX and
    then compiled using the available TeX distribution.  The first argument,
    'expr', may also be a LaTeX string.  The function will then run the
    appropriate viewer for the given output format or use the user defined
    one. By default png output is generated.

    By default pretty Euler fonts are used for typesetting (they were used to
    typeset the well known "Concrete Mathematics" book). For that to work, you
    need the 'eulervm.sty' LaTeX style (in Debian/Ubuntu, install the
    texlive-fonts-extra package). If you prefer default AMS fonts or your
    system lacks 'eulervm' LaTeX package then unset the 'euler' keyword
    argument.

    To use viewer auto-detection, lets say for 'png' output, issue

    >>> from sympy import symbols, preview, Symbol
    >>> x, y = symbols("x,y")

    >>> preview(x + y, output='png')

    This will choose 'pyglet' by default. To select a different one, do

    >>> preview(x + y, output='png', viewer='gimp')

    The 'png' format is considered special. For all other formats the rules
    are slightly different. As an example we will take 'dvi' output format. If
    you would run

    >>> preview(x + y, output='dvi')

    then 'view' will look for available 'dvi' viewers on your system
    (predefined in the function, so it will try evince, first, then kdvi and
    xdvi). If nothing is found, it will fall back to using a system file
    association (via ``open`` and ``xdg-open``). To always use your system file
    association without searching for the above readers, use

    >>> from sympy.printing.preview import system_default_viewer
    >>> preview(x + y, output='dvi', viewer=system_default_viewer)

    If this still does not find the viewer you want, it can be set explicitly.

    >>> preview(x + y, output='dvi', viewer='superior-dvi-viewer')

    This will skip auto-detection and will run user specified
    'superior-dvi-viewer'. If ``view`` fails to find it on your system it will
    gracefully raise an exception.

    You may also enter ``'file'`` for the viewer argument. Doing so will cause
    this function to return a file object in read-only mode, if ``filename``
    is unset. However, if it was set, then 'preview' writes the generated
    file to this filename instead.

    There is also support for writing to a ``io.BytesIO`` like object, which
    needs to be passed to the ``outputbuffer`` argument.

    >>> from io import BytesIO
    >>> obj = BytesIO()
    >>> preview(x + y, output='png', viewer='BytesIO',
    ...         outputbuffer=obj)

    The LaTeX preamble can be customized by setting the 'preamble' keyword
    argument. This can be used, e.g., to set a different font size, use a
    custom documentclass or import certain set of LaTeX packages.

    >>> preamble = "\\documentclass[10pt]{article}\n" \
    ...            "\\usepackage{amsmath,amsfonts}\\begin{document}"
    >>> preview(x + y, output='png', preamble=preamble)

    It is also possible to use the standard preamble and provide additional
    information to the preamble using the ``extra_preamble`` keyword argument.

    >>> from sympy import sin
    >>> extra_preamble = "\\renewcommand{\\sin}{\\cos}"
    >>> preview(sin(x), output='png', extra_preamble=extra_preamble)

    If the value of 'output' is different from 'dvi' then command line
    options can be set ('dvioptions' argument) for the execution of the
    'dvi'+output conversion tool. These options have to be in the form of a
    list of strings (see ``subprocess.Popen``).

    Additional keyword args will be passed to the :func:`~sympy.printing.latex.latex` call,
    e.g., the ``symbol_names`` flag.

    >>> phidd = Symbol('phidd')
    >>> preview(phidd, symbol_names={phidd: r'\ddot{\varphi}'})

    For post-processing the generated TeX File can be written to a file by
    passing the desired filename to the 'outputTexFile' keyword
    argument. To write the TeX code to a file named
    ``"sample.tex"`` and run the default png viewer to display the resulting
    bitmap, do

    >>> preview(x + y, outputTexFile="sample.tex")


    """
    # pyglet is the default for png
    if viewer is None and output == "png":
        try:
            import pyglet  # noqa: F401
        except ImportError:
            pass
        else:
            viewer = pyglet_viewer

    # look up a known application
    if viewer is None:
        # sorted in order from most pretty to most ugly
        # very discussable, but indeed 'gv' looks awful :)
        candidates = {
            "dvi": [ "evince", "okular", "kdvi", "xdvi" ],
            "ps": [ "evince", "okular", "gsview", "gv" ],
            "pdf": [ "evince", "okular", "kpdf", "acroread", "xpdf", "gv" ],
        }

        for candidate in candidates.get(output, []):
            path = shutil.which(candidate)
            if path is not None:
                viewer = path
                break

    # otherwise, use the system default for file association
    if viewer is None:
        viewer = system_default_viewer

    if viewer == "file":
        if filename is None:
            raise ValueError("filename has to be specified if viewer=\"file\"")
    elif viewer == "BytesIO":
        if outputbuffer is None:
            raise ValueError("outputbuffer has to be a BytesIO "
                             "compatible object if viewer=\"BytesIO\"")
    elif not callable(viewer) and not shutil.which(viewer):
        raise OSError("Unrecognized viewer: %s" % viewer)

    latex_main = _get_latex_main(expr, preamble=preamble, packages=packages,
                                 euler=euler, extra_preamble=extra_preamble,
                                 fontsize=fontsize, **latex_settings)

    debug("Latex code:")
    debug(latex_main)
    with tempfile.TemporaryDirectory() as workdir:
        Path(join(workdir, 'texput.tex')).write_text(latex_main, encoding='utf-8')

        if outputTexFile is not None:
            shutil.copyfile(join(workdir, 'texput.tex'), outputTexFile)

        if not shutil.which('latex'):
            raise RuntimeError("latex program is not installed")

        try:
            _check_output_no_window(
                ['latex', '-halt-on-error', '-interaction=nonstopmode',
                 'texput.tex'],
                cwd=workdir,
                stderr=STDOUT)
        except CalledProcessError as e:
            raise RuntimeError(
                "'latex' exited abnormally with the following output:\n%s" %
                e.output)

        src = "texput.%s" % (output)

        if output != "dvi":
            # in order of preference
            commandnames = {
                "ps": ["dvips"],
                "pdf": ["dvipdfmx", "dvipdfm", "dvipdf"],
                "png": ["dvipng"],
                "svg": ["dvisvgm"],
            }
            try:
                cmd_variants = commandnames[output]
            except KeyError:
                raise ValueError("Invalid output format: %s" % output) from None

            # find an appropriate command
            for cmd_variant in cmd_variants:
                cmd_path = shutil.which(cmd_variant)
                if cmd_path:
                    cmd = [cmd_path]
                    break
            else:
                if len(cmd_variants) > 1:
                    raise RuntimeError("None of %s are installed" % ", ".join(cmd_variants))
                else:
                    raise RuntimeError("%s is not installed" % cmd_variants[0])

            defaultoptions = {
                "dvipng": ["-T", "tight", "-z", "9", "--truecolor"],
                "dvisvgm": ["--no-fonts"],
            }

            commandend = {
                "dvips": ["-o", src, "texput.dvi"],
                "dvipdf": ["texput.dvi", src],
                "dvipdfm": ["-o", src, "texput.dvi"],
                "dvipdfmx": ["-o", src, "texput.dvi"],
                "dvipng": ["-o", src, "texput.dvi"],
                "dvisvgm": ["-o", src, "texput.dvi"],
            }

            if dvioptions is not None:
                cmd.extend(dvioptions)
            else:
                cmd.extend(defaultoptions.get(cmd_variant, []))
            cmd.extend(commandend[cmd_variant])

            try:
                _check_output_no_window(cmd, cwd=workdir, stderr=STDOUT)
            except CalledProcessError as e:
                raise RuntimeError(
                    "'%s' exited abnormally with the following output:\n%s" %
                    (' '.join(cmd), e.output))


        if viewer == "file":
            shutil.move(join(workdir, src), filename)
        elif viewer == "BytesIO":
            s = Path(join(workdir, src)).read_bytes()
            outputbuffer.write(s)
        elif callable(viewer):
            viewer(join(workdir, src), fmt=output)
        else:
            try:
                _check_output_no_window(
                    [viewer, src], cwd=workdir, stderr=STDOUT)
            except CalledProcessError as e:
                raise RuntimeError(
                    "'%s %s' exited abnormally with the following output:\n%s" %
                    (viewer, src, e.output))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_core\_tests\test_parking_lot.py ===
from __future__ import annotations

import re
from typing import TypeVar

import pytest

import trio
from trio.lowlevel import (
    add_parking_lot_breaker,
    current_task,
    remove_parking_lot_breaker,
)
from trio.testing import Matcher, RaisesGroup

from ... import _core
from ...testing import wait_all_tasks_blocked
from .._parking_lot import ParkingLot
from .tutil import check_sequence_matches

T = TypeVar("T")


async def test_parking_lot_basic() -> None:
    record = []

    async def waiter(i: int, lot: ParkingLot) -> None:
        record.append(f"sleep {i}")
        await lot.park()
        record.append(f"wake {i}")

    async with _core.open_nursery() as nursery:
        lot = ParkingLot()
        assert not lot
        assert len(lot) == 0
        assert lot.statistics().tasks_waiting == 0
        for i in range(3):
            nursery.start_soon(waiter, i, lot)
        await wait_all_tasks_blocked()
        assert len(record) == 3
        assert bool(lot)
        assert len(lot) == 3
        assert lot.statistics().tasks_waiting == 3
        lot.unpark_all()
        assert lot.statistics().tasks_waiting == 0
        await wait_all_tasks_blocked()
        assert len(record) == 6

    check_sequence_matches(
        record,
        [{"sleep 0", "sleep 1", "sleep 2"}, {"wake 0", "wake 1", "wake 2"}],
    )

    async with _core.open_nursery() as nursery:
        record = []
        for i in range(3):
            nursery.start_soon(waiter, i, lot)
            await wait_all_tasks_blocked()
        assert len(record) == 3
        for _ in range(3):
            lot.unpark()
            await wait_all_tasks_blocked()
        # 1-by-1 wakeups are strict FIFO
        assert record == [
            "sleep 0",
            "sleep 1",
            "sleep 2",
            "wake 0",
            "wake 1",
            "wake 2",
        ]

    # It's legal (but a no-op) to try and unpark while there's nothing parked
    lot.unpark()
    lot.unpark(count=1)
    lot.unpark(count=100)

    # Check unpark with count
    async with _core.open_nursery() as nursery:
        record = []
        for i in range(3):
            nursery.start_soon(waiter, i, lot)
            await wait_all_tasks_blocked()
        lot.unpark(count=2)
        await wait_all_tasks_blocked()
        check_sequence_matches(
            record,
            ["sleep 0", "sleep 1", "sleep 2", {"wake 0", "wake 1"}],
        )
        lot.unpark_all()

    with pytest.raises(
        ValueError,
        match=r"^Cannot pop a non-integer number of tasks\.$",
    ):
        lot.unpark(count=1.5)


async def cancellable_waiter(
    name: T,
    lot: ParkingLot,
    scopes: dict[T, _core.CancelScope],
    record: list[str],
) -> None:
    with _core.CancelScope() as scope:
        scopes[name] = scope
        record.append(f"sleep {name}")
        try:
            await lot.park()
        except _core.Cancelled:
            record.append(f"cancelled {name}")
        else:
            record.append(f"wake {name}")


async def test_parking_lot_cancel() -> None:
    record: list[str] = []
    scopes: dict[int, _core.CancelScope] = {}

    async with _core.open_nursery() as nursery:
        lot = ParkingLot()
        nursery.start_soon(cancellable_waiter, 1, lot, scopes, record)
        await wait_all_tasks_blocked()
        nursery.start_soon(cancellable_waiter, 2, lot, scopes, record)
        await wait_all_tasks_blocked()
        nursery.start_soon(cancellable_waiter, 3, lot, scopes, record)
        await wait_all_tasks_blocked()
        assert len(record) == 3

        scopes[2].cancel()
        await wait_all_tasks_blocked()
        assert len(record) == 4
        lot.unpark_all()
        await wait_all_tasks_blocked()
        assert len(record) == 6

    check_sequence_matches(
        record,
        ["sleep 1", "sleep 2", "sleep 3", "cancelled 2", {"wake 1", "wake 3"}],
    )


async def test_parking_lot_repark() -> None:
    record: list[str] = []
    scopes: dict[int, _core.CancelScope] = {}
    lot1 = ParkingLot()
    lot2 = ParkingLot()

    with pytest.raises(TypeError):
        lot1.repark([])  # type: ignore[arg-type]

    async with _core.open_nursery() as nursery:
        nursery.start_soon(cancellable_waiter, 1, lot1, scopes, record)
        await wait_all_tasks_blocked()
        nursery.start_soon(cancellable_waiter, 2, lot1, scopes, record)
        await wait_all_tasks_blocked()
        nursery.start_soon(cancellable_waiter, 3, lot1, scopes, record)
        await wait_all_tasks_blocked()
        assert len(record) == 3

        assert len(lot1) == 3
        lot1.repark(lot2)
        assert len(lot1) == 2
        assert len(lot2) == 1
        lot2.unpark_all()
        await wait_all_tasks_blocked()
        assert len(record) == 4
        assert record == ["sleep 1", "sleep 2", "sleep 3", "wake 1"]

        lot1.repark_all(lot2)
        assert len(lot1) == 0
        assert len(lot2) == 2

        scopes[2].cancel()
        await wait_all_tasks_blocked()
        assert len(lot2) == 1
        assert record == [
            "sleep 1",
            "sleep 2",
            "sleep 3",
            "wake 1",
            "cancelled 2",
        ]

        lot2.unpark_all()
        await wait_all_tasks_blocked()
        assert record == [
            "sleep 1",
            "sleep 2",
            "sleep 3",
            "wake 1",
            "cancelled 2",
            "wake 3",
        ]


async def test_parking_lot_repark_with_count() -> None:
    record: list[str] = []
    scopes: dict[int, _core.CancelScope] = {}
    lot1 = ParkingLot()
    lot2 = ParkingLot()
    async with _core.open_nursery() as nursery:
        nursery.start_soon(cancellable_waiter, 1, lot1, scopes, record)
        await wait_all_tasks_blocked()
        nursery.start_soon(cancellable_waiter, 2, lot1, scopes, record)
        await wait_all_tasks_blocked()
        nursery.start_soon(cancellable_waiter, 3, lot1, scopes, record)
        await wait_all_tasks_blocked()
        assert len(record) == 3

        assert len(lot1) == 3
        assert len(lot2) == 0
        lot1.repark(lot2, count=2)
        assert len(lot1) == 1
        assert len(lot2) == 2
        while lot2:
            lot2.unpark()
            await wait_all_tasks_blocked()
        assert record == [
            "sleep 1",
            "sleep 2",
            "sleep 3",
            "wake 1",
            "wake 2",
        ]
        lot1.unpark_all()


async def dummy_task(
    task_status: _core.TaskStatus[_core.Task] = trio.TASK_STATUS_IGNORED,
) -> None:
    task_status.started(_core.current_task())
    await trio.sleep_forever()


async def test_parking_lot_breaker_basic() -> None:
    """Test basic functionality for breaking lots."""
    lot = ParkingLot()
    task = current_task()

    # defaults to current task
    lot.break_lot()
    assert lot.broken_by == [task]

    # breaking the lot again with the same task appends another copy in `broken_by`
    lot.break_lot()
    assert lot.broken_by == [task, task]

    # trying to park in broken lot errors
    broken_by_str = re.escape(str([task, task]))
    with pytest.raises(
        _core.BrokenResourceError,
        match=f"^Attempted to park in parking lot broken by {broken_by_str}$",
    ):
        await lot.park()


async def test_parking_lot_break_parking_tasks() -> None:
    """Checks that tasks currently waiting to park raise an error when the breaker exits."""

    async def bad_parker(lot: ParkingLot, scope: _core.CancelScope) -> None:
        add_parking_lot_breaker(current_task(), lot)
        with scope:
            await trio.sleep_forever()

    lot = ParkingLot()
    cs = _core.CancelScope()

    # check that parked task errors
    with RaisesGroup(
        Matcher(_core.BrokenResourceError, match="^Parking lot broken by"),
    ):
        async with _core.open_nursery() as nursery:
            nursery.start_soon(bad_parker, lot, cs)
            await wait_all_tasks_blocked()

            nursery.start_soon(lot.park)
            await wait_all_tasks_blocked()

            cs.cancel()


async def test_parking_lot_breaker_registration() -> None:
    lot = ParkingLot()
    task = current_task()

    with pytest.raises(
        RuntimeError,
        match="Attempted to remove task as breaker for a lot it is not registered for",
    ):
        remove_parking_lot_breaker(task, lot)

    # check that a task can be registered as breaker for the same lot multiple times
    add_parking_lot_breaker(task, lot)
    add_parking_lot_breaker(task, lot)
    remove_parking_lot_breaker(task, lot)
    remove_parking_lot_breaker(task, lot)

    with pytest.raises(
        RuntimeError,
        match="Attempted to remove task as breaker for a lot it is not registered for",
    ):
        remove_parking_lot_breaker(task, lot)

    # registering a task as breaker on an already broken lot is fine
    lot.break_lot()
    child_task: _core.Task | None = None
    async with trio.open_nursery() as nursery:
        child_task = await nursery.start(dummy_task)
        assert isinstance(child_task, _core.Task)
        add_parking_lot_breaker(child_task, lot)
        nursery.cancel_scope.cancel()
    assert lot.broken_by == [task, child_task]

    # manually breaking a lot with an already exited task is fine
    lot = ParkingLot()
    lot.break_lot(child_task)
    assert lot.broken_by == [child_task]


async def test_parking_lot_breaker_rebreak() -> None:
    lot = ParkingLot()
    task = current_task()
    lot.break_lot()

    # breaking an already broken lot with a different task is allowed
    # The nursery is only to create a task we can pass to lot.break_lot
    async with trio.open_nursery() as nursery:
        child_task = await nursery.start(dummy_task)
        lot.break_lot(child_task)
        nursery.cancel_scope.cancel()

    assert lot.broken_by == [task, child_task]


async def test_parking_lot_multiple_breakers_exit() -> None:
    # register multiple tasks as lot breakers, then have them all exit
    lot = ParkingLot()
    async with trio.open_nursery() as nursery:
        child_task1 = await nursery.start(dummy_task)
        child_task2 = await nursery.start(dummy_task)
        child_task3 = await nursery.start(dummy_task)
        assert isinstance(child_task1, _core.Task)
        assert isinstance(child_task2, _core.Task)
        assert isinstance(child_task3, _core.Task)
        add_parking_lot_breaker(child_task1, lot)
        add_parking_lot_breaker(child_task2, lot)
        add_parking_lot_breaker(child_task3, lot)
        nursery.cancel_scope.cancel()

    # I think the order is guaranteed currently, but doesn't hurt to be safe.
    assert set(lot.broken_by) == {child_task1, child_task2, child_task3}


async def test_parking_lot_breaker_register_exited_task() -> None:
    lot = ParkingLot()
    child_task: _core.Task | None = None
    async with trio.open_nursery() as nursery:
        value = await nursery.start(dummy_task)
        assert isinstance(value, _core.Task)
        child_task = value
        nursery.cancel_scope.cancel()
    # trying to register an exited task as lot breaker errors
    with pytest.raises(
        trio.BrokenResourceError,
        match=r"^Attempted to add already exited task as lot breaker.$",
    ):
        add_parking_lot_breaker(child_task, lot)


async def test_parking_lot_break_itself() -> None:
    """Break a parking lot, where the breakee is parked.
    Doing this is weird, but should probably be supported.
    """

    async def return_me_and_park(
        lot: ParkingLot,
        *,
        task_status: _core.TaskStatus[_core.Task] = trio.TASK_STATUS_IGNORED,
    ) -> None:
        task_status.started(_core.current_task())
        await lot.park()

    lot = ParkingLot()
    with RaisesGroup(
        Matcher(_core.BrokenResourceError, match="^Parking lot broken by"),
    ):
        async with _core.open_nursery() as nursery:
            child_task = await nursery.start(return_me_and_park, lot)
            lot.break_lot(child_task)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\quantization\qdq_loss_debug.py ===
# --------------------------------------------------------------------------
# Copyright (c) Microsoft, Intel Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""Utilities to run a given ONNX model, while saving input/output tensors of
eligible operator nodes.

A use case is to debug quantization induced accuracy drop. An AI engineer can
run the original float32 model and the quantized model with the same inputs,
then compare the corresponding activations between the two models to find
where the divergence is.

Example Usage:

```python
    class ExampleDataReader(CalibrationDataReader):
        def __init__(self):
            ...
        def get_next(self):
            ...

    input_data_reader = ExampleDataReader()

    augmented_model_path = str(Path(self._tmp_model_dir.name).joinpath("augmented_model.onnx"))
    modify_model_output_intermediate_tensors (path_to_onnx_model, augmented_model_path)

    tensor_dict = collect_activations(augmented_model_path, input_data_reader)
```

`tensor_dict` points to a dictionary where the keys are tensor names and each value
is a list of tensors, one from each model run

"""

import logging
import math
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy
import onnx
from onnx import helper, numpy_helper

import onnxruntime

from .calibrate import CalibraterBase, CalibrationDataReader
from .onnx_model import ONNXModel
from .quant_utils import (
    DEQUANT_OP_NAME,
    DEQUANT_OUTPUT_SUFFIX,
    QUANT_INPUT_SUFFIX,
    TENSOR_NAME_QUANT_SUFFIX,
    find_by_name,
    load_model_with_shape_infer,
)

_TENSOR_SAVE_POSTFIX = "_ReshapedSavedOutput"
_TENSOR_SAVE_POSTFIX_LEN = len(_TENSOR_SAVE_POSTFIX)


def modify_model_output_intermediate_tensors(
    input_model_path: str | Path,
    output_model_path: str | Path,
    op_types_for_saving: Sequence[str] | None = None,
    save_as_external_data: bool = False,
) -> None:
    """Augment a given ONNX model to save node input/output tensors.

    Add all input/output tensors of operator nodes to model outputs
    so that their values can be retrieved for debugging purposes.

    Args:
        input_model: the path to load the model.
        op_types_for_saving: Operator types for which the
                input/output should be saved. By default, saving all the
                float32/float16 tensors.

    Returns:
        The augmented ONNX model
    """

    if op_types_for_saving is None:
        op_types_for_saving = []
    saver = CalibraterBase(input_model_path, op_types_to_calibrate=op_types_for_saving)
    model_to_augment = saver.model
    tensors, value_infos = saver.select_tensors_to_calibrate(model_to_augment)
    reshape_shape_name = "LinearReshape_" + str(time.time())
    reshape_shape = numpy_helper.from_array(numpy.array([-1], dtype=numpy.int64), reshape_shape_name)
    model_to_augment.graph.initializer.append(reshape_shape)

    for tensor_name in tensors:
        reshape_output = tensor_name + _TENSOR_SAVE_POSTFIX
        reshape_node = onnx.helper.make_node(
            "Reshape",
            inputs=[tensor_name, reshape_shape_name],
            outputs=[reshape_output],
            name=reshape_output,
        )
        model_to_augment.graph.node.append(reshape_node)
        reshape_output_value_info = helper.make_tensor_value_info(
            reshape_output, value_infos[tensor_name].type.tensor_type.elem_type, [-1]
        )
        model_to_augment.graph.output.append(reshape_output_value_info)

    onnx.save(
        model_to_augment,
        output_model_path,
        save_as_external_data=save_as_external_data,
    )


def collect_activations(
    augmented_model: str,
    input_reader: CalibrationDataReader,
    session_options=None,
    execution_providers: Sequence[str] | None = None,
) -> dict[str, list[numpy.ndarray]]:
    """Run augmented model and collect activations tensors.

    Args:
        augmented_model: Path to augmented model created by modify_model_output_intermediate_tensors ()
        input_reader: Logic for reading input for the model, augmented model have the same
            input with the original model.
        session_options: Optional OnnxRuntime session options for controlling model run.
            By default graph optimization is turned off
        execution_providers: Collection of execution providers for running the model.
            Only CPU EP is used by default.

    Returns:
        A dictionary where the key is tensor name and values are list of tensors from each batch
    """

    if session_options is None:
        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    if execution_providers is None:
        execution_providers = ["CPUExecutionProvider"]

    inference_session = onnxruntime.InferenceSession(
        augmented_model,
        sess_options=session_options,
        providers=execution_providers,
    )

    intermediate_outputs = []
    for input_d in input_reader:
        intermediate_outputs.append(inference_session.run(None, input_d))
    if not intermediate_outputs:
        raise RuntimeError("No data is collected while running augmented model!")

    output_dict = {}
    output_info = inference_session.get_outputs()
    for batch in intermediate_outputs:
        for output, output_data in zip(output_info, batch, strict=False):
            if output.name.endswith(_TENSOR_SAVE_POSTFIX):
                output_name = output.name[:-_TENSOR_SAVE_POSTFIX_LEN]
                output_dict.setdefault(output_name, []).append(output_data)

    return output_dict


_POST_QDQ_POSTFIX1 = DEQUANT_OUTPUT_SUFFIX + "_1"


def _add_pre_post_qdq_pair(
    qdq_cmp: dict[str, dict[str, Sequence[numpy.ndarray]]],
    activation_name: str,
    pre_qdq_tensors: Sequence[numpy.ndarray] | None,
    post_qdq_tensors: Sequence[numpy.ndarray] | None,
) -> None:
    if post_qdq_tensors is not None and pre_qdq_tensors is not None:
        qdq_cmp[activation_name] = {}
        qdq_cmp[activation_name]["pre_qdq"] = pre_qdq_tensors
        qdq_cmp[activation_name]["post_qdq"] = post_qdq_tensors


def create_activation_matching(
    qdq_activations: dict[str, Sequence[numpy.ndarray]],
    float_activations: dict[str, Sequence[numpy.ndarray]] | None = None,
) -> dict[str, dict[str, Sequence[numpy.ndarray]]]:
    """Comparing activation values to help debugging accuracy loss due to quantization.

    This functions takes saved activations from the QDQ model and (optionally) the
    float point model, and provides a data structure for comparing:
        * from the qdq model, activation values before and after QDQ operation
        * across both models, activations from the orignal model vs the corresponding
          activations in the QDQ model

    Arg:
        qdq_activations: Output of `collect_activations`. This must be from a quantized
            model with QDQ format.
        float_activations: Output of `collect_activations`. This must be from the float
            point model.

    Returns:
        Dict for comparing pre and post quantized activation tensors. E.g.
        ```
        qdq_cmp = cmp_qdq_input_output(qdq_activations)
        print(qdq_cmp['activation1']['pre_qdq'][0])
        print(qdq_cmp['activation1'][`post_qdq'][0])


        qdq_cmp = cmp_qdq_input_output(qdq_activations, float_activations)
        print(qdq_cmp['activation1']['float'][0])
        print(qdq_cmp['activation1']['pre_qdq'][0])
        print(qdq_cmp['activation1'][`post_qdq'][0])
        ```
    """

    qdq_cmp: dict[str, dict[str, Sequence[numpy.ndarray]]] = {}
    for tensor_name, tensors in qdq_activations.items():
        if tensor_name.endswith(QUANT_INPUT_SUFFIX):
            pre_name = tensor_name[: -len(QUANT_INPUT_SUFFIX)]
            post_qdq_tensors = qdq_activations.get(pre_name)
            pre_qdq_tensors = tensors
            _add_pre_post_qdq_pair(qdq_cmp, pre_name, pre_qdq_tensors, post_qdq_tensors)
        elif tensor_name.endswith(DEQUANT_OUTPUT_SUFFIX):
            pre_name = tensor_name[: -len(DEQUANT_OUTPUT_SUFFIX)]
            pre_qdq_tensors = qdq_activations.get(pre_name)
            post_qdq_tensors = tensors
            _add_pre_post_qdq_pair(qdq_cmp, pre_name, pre_qdq_tensors, post_qdq_tensors)
        elif tensor_name.endswith(_POST_QDQ_POSTFIX1):
            pre_name = tensor_name[: -len(_POST_QDQ_POSTFIX1)]
            pre_qdq_tensors = qdq_activations.get(pre_name)
            post_qdq_tensors = tensors
            _add_pre_post_qdq_pair(qdq_cmp, pre_name, pre_qdq_tensors, post_qdq_tensors)

    if not float_activations:
        return qdq_cmp

    for act_name, act_values in qdq_cmp.items():
        float_acts = float_activations.get(act_name)
        if float_acts is not None:
            act_values["float"] = float_acts

    return qdq_cmp


def _run_dequantize_linear(
    weight_tensor: numpy.ndarray, weight_scale: numpy.ndarray, weight_zp: numpy.ndarray, channel_axis: int
) -> numpy.ndarray | None:
    assert weight_scale.shape == weight_zp.shape
    if weight_zp.size == 1:
        return (weight_tensor - weight_zp) * weight_scale

    assert weight_zp.ndim == 1
    reshape_dims = list(weight_tensor.shape)  # deep copy
    reshape_dims[channel_axis] = 1  # only one per channel for reshape
    channel_count = weight_tensor.shape[channel_axis]
    dequantized_weights = None
    for i in range(channel_count):
        per_channel_data = weight_tensor.take(i, channel_axis)
        dequantized_per_channel_data = (per_channel_data - weight_zp[i]) * weight_scale[i]
        if i == 0:
            dequantized_weights = numpy.asarray(dequantized_per_channel_data).reshape(reshape_dims)
        else:
            channel_weights = numpy.asarray(dequantized_per_channel_data).reshape(reshape_dims)
            dequantized_weights = numpy.concatenate((dequantized_weights, channel_weights), channel_axis)

    if dequantized_weights is None:
        return None

    dequantized_weights.reshape(weight_tensor.shape)
    return dequantized_weights


def create_weight_matching(float_model_path: str, qdq_model_path: str) -> dict[str, dict[str, numpy.ndarray]]:
    """Comparing weight values to help debugging accuracy loss due to quantization.

    This functions takes the float model and the qdq model, and provides a data structure for comparing
    their corresponding weights to locate quantization errors

    Arg:
        float_model_path: Path points to the float point model.
        qdq_model_path: Path points to the qdq model.

    Returns:
        Dict for comparing weight tensors. E.g.
        ```
        qdq_weight_cmp = create_weight_matching(float_model, qdq_model)
        print(qdq_weight_cmp['activation1']['float'])
        print(qdq_weight_cmp['activation1']['dequantized'])
        ```
    """
    float_onnx_model = ONNXModel(load_model_with_shape_infer(Path(float_model_path)))
    qdq_onnx_model = ONNXModel(load_model_with_shape_infer(Path(qdq_model_path)))

    matched_weights: dict[str, dict[str, numpy.ndarray]] = {}
    initializers = qdq_onnx_model.initializer()
    for node in qdq_onnx_model.nodes():
        if node.op_type != DEQUANT_OP_NAME:
            continue  # Only care about DQ node
        weight_name: str = node.input[0]
        weight_values = find_by_name(weight_name, initializers)
        if not weight_values:
            continue  # Only care about DQ node with const inputs
        if not weight_name.endswith(TENSOR_NAME_QUANT_SUFFIX):
            logging.error(f"Model Error in '{qdq_model_path}': Dequantized tensor name '{weight_name}' not recognized!")
            continue

        axis = -1
        for attr in node.attribute:
            if attr.name == "axis":
                axis = attr.i

        weight_tensor = numpy_helper.to_array(weight_values)
        weight_scale = numpy_helper.to_array(find_by_name(node.input[1], initializers))
        if len(node.input) > 2:
            weight_zp = numpy_helper.to_array(find_by_name(node.input[2], initializers))
        else:
            weight_zp = numpy.zeros(weight_scale.shape, dtype=numpy.int32)

        # Perform dequantization:
        if weight_scale.size == weight_zp.size == 1:
            # Avoids the confusion between a scaler and a tensor of one element.
            weight_scale = weight_scale.reshape(())
            weight_zp = weight_zp.reshape(())
        if weight_scale.shape != weight_zp.shape:
            raise RuntimeError(
                f"scale and zero_point must have the same shape but {weight_scale.shape} != {weight_zp.shape}"
            )
        weight_quant = _run_dequantize_linear(weight_tensor, weight_scale, weight_zp, channel_axis=axis)
        weight_name = weight_name[: -len(TENSOR_NAME_QUANT_SUFFIX)]
        if weight_quant is None:
            logging.error(f"Model Error in '{qdq_model_path}': '{weight_name}' per-channel quantization on 0 channel")
            continue

        float_values = find_by_name(weight_name, float_onnx_model.initializer())
        if not float_values:
            logging.error(f"Model Error in '{float_model_path}': weight tensor '{weight_name}' not found!")
            continue
        weight_float = numpy_helper.to_array(float_values)
        matched_weights[weight_name] = {"float": weight_float, "dequantized": weight_quant}

    return matched_weights


def compute_signal_to_quantization_noice_ratio(
    x: Sequence[numpy.ndarray] | numpy.ndarray, y: Sequence[numpy.ndarray] | numpy.ndarray
) -> float:
    if isinstance(x, numpy.ndarray):
        xlist = [x]
    else:
        xlist = x
    if isinstance(y, numpy.ndarray):
        ylist = [y]
    else:
        ylist = y
    if len(xlist) != len(ylist):
        raise RuntimeError("Unequal number of tensors to compare!")

    left = numpy.concatenate(xlist).flatten()
    right = numpy.concatenate(ylist).flatten()

    epsilon = numpy.finfo("float").eps
    tensor_norm = max(numpy.linalg.norm(left), epsilon)
    diff_norm = max(numpy.linalg.norm(left - right), epsilon)
    res = tensor_norm / diff_norm
    return 20 * math.log10(res)


def compute_weight_error(
    weights_match: dict[str, dict[str, numpy.ndarray]],
    err_func: Callable[[numpy.ndarray, numpy.ndarray], float] = compute_signal_to_quantization_noice_ratio,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for weight_name, weight_match in weights_match.items():
        result[weight_name] = err_func(weight_match["float"], weight_match["dequantized"])
    return result


def compute_activation_error(
    activations_match: dict[str, dict[str, Sequence[numpy.ndarray]]],
    err_func: Callable[
        [Sequence[numpy.ndarray], Sequence[numpy.ndarray]], float
    ] = compute_signal_to_quantization_noice_ratio,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name, match in activations_match.items():
        err_result: dict[str, float] = {}
        err_result["qdq_err"] = err_func(match["pre_qdq"], match["post_qdq"])
        float_activation = match["float"]
        if float_activation:
            err_result["xmodel_err"] = err_func(float_activation, match["post_qdq"])
        result[name] = err_result
    return result

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\io\json\_table_schema.py ===
"""
Table Schema builders

https://specs.frictionlessdata.io/table-schema/
"""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)
import warnings

from pandas._libs import lib
from pandas._libs.json import ujson_loads
from pandas._libs.tslibs import timezones
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_string_dtype,
)
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    DatetimeTZDtype,
    ExtensionDtype,
    PeriodDtype,
)

from pandas import DataFrame
import pandas.core.common as com

from pandas.tseries.frequencies import to_offset

if TYPE_CHECKING:
    from pandas._typing import (
        DtypeObj,
        JSONSerializable,
    )

    from pandas import Series
    from pandas.core.indexes.multi import MultiIndex


TABLE_SCHEMA_VERSION = "1.4.0"


def as_json_table_type(x: DtypeObj) -> str:
    """
    Convert a NumPy / pandas type to its corresponding json_table.

    Parameters
    ----------
    x : np.dtype or ExtensionDtype

    Returns
    -------
    str
        the Table Schema data types

    Notes
    -----
    This table shows the relationship between NumPy / pandas dtypes,
    and Table Schema dtypes.

    ==============  =================
    Pandas type     Table Schema type
    ==============  =================
    int64           integer
    float64         number
    bool            boolean
    datetime64[ns]  datetime
    timedelta64[ns] duration
    object          str
    categorical     any
    =============== =================
    """
    if is_integer_dtype(x):
        return "integer"
    elif is_bool_dtype(x):
        return "boolean"
    elif is_numeric_dtype(x):
        return "number"
    elif lib.is_np_dtype(x, "M") or isinstance(x, (DatetimeTZDtype, PeriodDtype)):
        return "datetime"
    elif lib.is_np_dtype(x, "m"):
        return "duration"
    elif isinstance(x, ExtensionDtype):
        return "any"
    elif is_string_dtype(x):
        return "string"
    else:
        return "any"


def set_default_names(data):
    """Sets index names to 'index' for regular, or 'level_x' for Multi"""
    if com.all_not_none(*data.index.names):
        nms = data.index.names
        if len(nms) == 1 and data.index.name == "index":
            warnings.warn(
                "Index name of 'index' is not round-trippable.",
                stacklevel=find_stack_level(),
            )
        elif len(nms) > 1 and any(x.startswith("level_") for x in nms):
            warnings.warn(
                "Index names beginning with 'level_' are not round-trippable.",
                stacklevel=find_stack_level(),
            )
        return data

    data = data.copy()
    if data.index.nlevels > 1:
        data.index.names = com.fill_missing_names(data.index.names)
    else:
        data.index.name = data.index.name or "index"
    return data


def convert_pandas_type_to_json_field(arr) -> dict[str, JSONSerializable]:
    dtype = arr.dtype
    name: JSONSerializable
    if arr.name is None:
        name = "values"
    else:
        name = arr.name
    field: dict[str, JSONSerializable] = {
        "name": name,
        "type": as_json_table_type(dtype),
    }

    if isinstance(dtype, CategoricalDtype):
        cats = dtype.categories
        ordered = dtype.ordered

        field["constraints"] = {"enum": list(cats)}
        field["ordered"] = ordered
    elif isinstance(dtype, PeriodDtype):
        field["freq"] = dtype.freq.freqstr
    elif isinstance(dtype, DatetimeTZDtype):
        if timezones.is_utc(dtype.tz):
            # timezone.utc has no "zone" attr
            field["tz"] = "UTC"
        else:
            # error: "tzinfo" has no attribute "zone"
            field["tz"] = dtype.tz.zone  # type: ignore[attr-defined]
    elif isinstance(dtype, ExtensionDtype):
        field["extDtype"] = dtype.name
    return field


def convert_json_field_to_pandas_type(field) -> str | CategoricalDtype:
    """
    Converts a JSON field descriptor into its corresponding NumPy / pandas type

    Parameters
    ----------
    field
        A JSON field descriptor

    Returns
    -------
    dtype

    Raises
    ------
    ValueError
        If the type of the provided field is unknown or currently unsupported

    Examples
    --------
    >>> convert_json_field_to_pandas_type({"name": "an_int", "type": "integer"})
    'int64'

    >>> convert_json_field_to_pandas_type(
    ...     {
    ...         "name": "a_categorical",
    ...         "type": "any",
    ...         "constraints": {"enum": ["a", "b", "c"]},
    ...         "ordered": True,
    ...     }
    ... )
    CategoricalDtype(categories=['a', 'b', 'c'], ordered=True, categories_dtype=object)

    >>> convert_json_field_to_pandas_type({"name": "a_datetime", "type": "datetime"})
    'datetime64[ns]'

    >>> convert_json_field_to_pandas_type(
    ...     {"name": "a_datetime_with_tz", "type": "datetime", "tz": "US/Central"}
    ... )
    'datetime64[ns, US/Central]'
    """
    typ = field["type"]
    if typ == "string":
        return "object"
    elif typ == "integer":
        return field.get("extDtype", "int64")
    elif typ == "number":
        return field.get("extDtype", "float64")
    elif typ == "boolean":
        return field.get("extDtype", "bool")
    elif typ == "duration":
        return "timedelta64"
    elif typ == "datetime":
        if field.get("tz"):
            return f"datetime64[ns, {field['tz']}]"
        elif field.get("freq"):
            # GH#9586 rename frequency M to ME for offsets
            offset = to_offset(field["freq"])
            freq_n, freq_name = offset.n, offset.name
            freq = freq_to_period_freqstr(freq_n, freq_name)
            # GH#47747 using datetime over period to minimize the change surface
            return f"period[{freq}]"
        else:
            return "datetime64[ns]"
    elif typ == "any":
        if "constraints" in field and "ordered" in field:
            return CategoricalDtype(
                categories=field["constraints"]["enum"], ordered=field["ordered"]
            )
        elif "extDtype" in field:
            return registry.find(field["extDtype"])
        else:
            return "object"

    raise ValueError(f"Unsupported or invalid field type: {typ}")


def build_table_schema(
    data: DataFrame | Series,
    index: bool = True,
    primary_key: bool | None = None,
    version: bool = True,
) -> dict[str, JSONSerializable]:
    """
    Create a Table schema from ``data``.

    Parameters
    ----------
    data : Series, DataFrame
    index : bool, default True
        Whether to include ``data.index`` in the schema.
    primary_key : bool or None, default True
        Column names to designate as the primary key.
        The default `None` will set `'primaryKey'` to the index
        level or levels if the index is unique.
    version : bool, default True
        Whether to include a field `pandas_version` with the version
        of pandas that last revised the table schema. This version
        can be different from the installed pandas version.

    Returns
    -------
    dict

    Notes
    -----
    See `Table Schema
    <https://pandas.pydata.org/docs/user_guide/io.html#table-schema>`__ for
    conversion types.
    Timedeltas as converted to ISO8601 duration format with
    9 decimal places after the seconds field for nanosecond precision.

    Categoricals are converted to the `any` dtype, and use the `enum` field
    constraint to list the allowed values. The `ordered` attribute is included
    in an `ordered` field.

    Examples
    --------
    >>> from pandas.io.json._table_schema import build_table_schema
    >>> df = pd.DataFrame(
    ...     {'A': [1, 2, 3],
    ...      'B': ['a', 'b', 'c'],
    ...      'C': pd.date_range('2016-01-01', freq='d', periods=3),
    ...     }, index=pd.Index(range(3), name='idx'))
    >>> build_table_schema(df)
    {'fields': \
[{'name': 'idx', 'type': 'integer'}, \
{'name': 'A', 'type': 'integer'}, \
{'name': 'B', 'type': 'string'}, \
{'name': 'C', 'type': 'datetime'}], \
'primaryKey': ['idx'], \
'pandas_version': '1.4.0'}
    """
    if index is True:
        data = set_default_names(data)

    schema: dict[str, Any] = {}
    fields = []

    if index:
        if data.index.nlevels > 1:
            data.index = cast("MultiIndex", data.index)
            for level, name in zip(data.index.levels, data.index.names):
                new_field = convert_pandas_type_to_json_field(level)
                new_field["name"] = name
                fields.append(new_field)
        else:
            fields.append(convert_pandas_type_to_json_field(data.index))

    if data.ndim > 1:
        for column, s in data.items():
            fields.append(convert_pandas_type_to_json_field(s))
    else:
        fields.append(convert_pandas_type_to_json_field(data))

    schema["fields"] = fields
    if index and data.index.is_unique and primary_key is None:
        if data.index.nlevels == 1:
            schema["primaryKey"] = [data.index.name]
        else:
            schema["primaryKey"] = data.index.names
    elif primary_key is not None:
        schema["primaryKey"] = primary_key

    if version:
        schema["pandas_version"] = TABLE_SCHEMA_VERSION
    return schema


def parse_table_schema(json, precise_float: bool) -> DataFrame:
    """
    Builds a DataFrame from a given schema

    Parameters
    ----------
    json :
        A JSON table schema
    precise_float : bool
        Flag controlling precision when decoding string to double values, as
        dictated by ``read_json``

    Returns
    -------
    df : DataFrame

    Raises
    ------
    NotImplementedError
        If the JSON table schema contains either timezone or timedelta data

    Notes
    -----
        Because :func:`DataFrame.to_json` uses the string 'index' to denote a
        name-less :class:`Index`, this function sets the name of the returned
        :class:`DataFrame` to ``None`` when said string is encountered with a
        normal :class:`Index`. For a :class:`MultiIndex`, the same limitation
        applies to any strings beginning with 'level_'. Therefore, an
        :class:`Index` name of 'index'  and :class:`MultiIndex` names starting
        with 'level_' are not supported.

    See Also
    --------
    build_table_schema : Inverse function.
    pandas.read_json
    """
    table = ujson_loads(json, precise_float=precise_float)
    col_order = [field["name"] for field in table["schema"]["fields"]]
    df = DataFrame(table["data"], columns=col_order)[col_order]

    dtypes = {
        field["name"]: convert_json_field_to_pandas_type(field)
        for field in table["schema"]["fields"]
    }

    # No ISO constructor for Timedelta as of yet, so need to raise
    if "timedelta64" in dtypes.values():
        raise NotImplementedError(
            'table="orient" can not yet read ISO-formatted Timedelta data'
        )

    df = df.astype(dtypes)

    if "primaryKey" in table["schema"]:
        df = df.set_index(table["schema"]["primaryKey"])
        if len(df.index.names) == 1:
            if df.index.name == "index":
                df.index.name = None
        else:
            df.index.names = [
                None if x.startswith("level_") else x for x in df.index.names
            ]

    return df

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\pyproject_hooks\_in_process\_in_process.py ===
"""This is invoked in a subprocess to call the build backend hooks.

It expects:
- Command line args: hook_name, control_dir
- Environment variables:
      _PYPROJECT_HOOKS_BUILD_BACKEND=entry.point:spec
      _PYPROJECT_HOOKS_BACKEND_PATH=paths (separated with os.pathsep)
- control_dir/input.json:
  - {"kwargs": {...}}

Results:
- control_dir/output.json
  - {"return_val": ...}
"""
import json
import os
import os.path
import re
import shutil
import sys
import traceback
from glob import glob
from importlib import import_module
from importlib.machinery import PathFinder
from os.path import join as pjoin

# This file is run as a script, and `import wrappers` is not zip-safe, so we
# include write_json() and read_json() from wrappers.py.


def write_json(obj, path, **kwargs):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, **kwargs)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class BackendUnavailable(Exception):
    """Raised if we cannot import the backend"""

    def __init__(self, message, traceback=None):
        super().__init__(message)
        self.message = message
        self.traceback = traceback


class HookMissing(Exception):
    """Raised if a hook is missing and we are not executing the fallback"""

    def __init__(self, hook_name=None):
        super().__init__(hook_name)
        self.hook_name = hook_name


def _build_backend():
    """Find and load the build backend"""
    backend_path = os.environ.get("_PYPROJECT_HOOKS_BACKEND_PATH")
    ep = os.environ["_PYPROJECT_HOOKS_BUILD_BACKEND"]
    mod_path, _, obj_path = ep.partition(":")

    if backend_path:
        # Ensure in-tree backend directories have the highest priority when importing.
        extra_pathitems = backend_path.split(os.pathsep)
        sys.meta_path.insert(0, _BackendPathFinder(extra_pathitems, mod_path))

    try:
        obj = import_module(mod_path)
    except ImportError:
        msg = f"Cannot import {mod_path!r}"
        raise BackendUnavailable(msg, traceback.format_exc())

    if obj_path:
        for path_part in obj_path.split("."):
            obj = getattr(obj, path_part)
    return obj


class _BackendPathFinder:
    """Implements the MetaPathFinder interface to locate modules in ``backend-path``.

    Since the environment provided by the frontend can contain all sorts of
    MetaPathFinders, the only way to ensure the backend is loaded from the
    right place is to prepend our own.
    """

    def __init__(self, backend_path, backend_module):
        self.backend_path = backend_path
        self.backend_module = backend_module
        self.backend_parent, _, _ = backend_module.partition(".")

    def find_spec(self, fullname, _path, _target=None):
        if "." in fullname:
            # Rely on importlib to find nested modules based on parent's path
            return None

        # Ignore other items in _path or sys.path and use backend_path instead:
        spec = PathFinder.find_spec(fullname, path=self.backend_path)
        if spec is None and fullname == self.backend_parent:
            # According to the spec, the backend MUST be loaded from backend-path.
            # Therefore, we can halt the import machinery and raise a clean error.
            msg = f"Cannot find module {self.backend_module!r} in {self.backend_path!r}"
            raise BackendUnavailable(msg)

        return spec

    if sys.version_info >= (3, 8):

        def find_distributions(self, context=None):
            # Delayed import: Python 3.7 does not contain importlib.metadata
            from importlib.metadata import DistributionFinder, MetadataPathFinder

            context = DistributionFinder.Context(path=self.backend_path)
            return MetadataPathFinder.find_distributions(context=context)


def _supported_features():
    """Return the list of options features supported by the backend.

    Returns a list of strings.
    The only possible value is 'build_editable'.
    """
    backend = _build_backend()
    features = []
    if hasattr(backend, "build_editable"):
        features.append("build_editable")
    return features


def get_requires_for_build_wheel(config_settings):
    """Invoke the optional get_requires_for_build_wheel hook

    Returns [] if the hook is not defined.
    """
    backend = _build_backend()
    try:
        hook = backend.get_requires_for_build_wheel
    except AttributeError:
        return []
    else:
        return hook(config_settings)


def get_requires_for_build_editable(config_settings):
    """Invoke the optional get_requires_for_build_editable hook

    Returns [] if the hook is not defined.
    """
    backend = _build_backend()
    try:
        hook = backend.get_requires_for_build_editable
    except AttributeError:
        return []
    else:
        return hook(config_settings)


def prepare_metadata_for_build_wheel(
    metadata_directory, config_settings, _allow_fallback
):
    """Invoke optional prepare_metadata_for_build_wheel

    Implements a fallback by building a wheel if the hook isn't defined,
    unless _allow_fallback is False in which case HookMissing is raised.
    """
    backend = _build_backend()
    try:
        hook = backend.prepare_metadata_for_build_wheel
    except AttributeError:
        if not _allow_fallback:
            raise HookMissing()
    else:
        return hook(metadata_directory, config_settings)
    # fallback to build_wheel outside the try block to avoid exception chaining
    # which can be confusing to users and is not relevant
    whl_basename = backend.build_wheel(metadata_directory, config_settings)
    return _get_wheel_metadata_from_wheel(
        whl_basename, metadata_directory, config_settings
    )


def prepare_metadata_for_build_editable(
    metadata_directory, config_settings, _allow_fallback
):
    """Invoke optional prepare_metadata_for_build_editable

    Implements a fallback by building an editable wheel if the hook isn't
    defined, unless _allow_fallback is False in which case HookMissing is
    raised.
    """
    backend = _build_backend()
    try:
        hook = backend.prepare_metadata_for_build_editable
    except AttributeError:
        if not _allow_fallback:
            raise HookMissing()
        try:
            build_hook = backend.build_editable
        except AttributeError:
            raise HookMissing(hook_name="build_editable")
        else:
            whl_basename = build_hook(metadata_directory, config_settings)
            return _get_wheel_metadata_from_wheel(
                whl_basename, metadata_directory, config_settings
            )
    else:
        return hook(metadata_directory, config_settings)


WHEEL_BUILT_MARKER = "PYPROJECT_HOOKS_ALREADY_BUILT_WHEEL"


def _dist_info_files(whl_zip):
    """Identify the .dist-info folder inside a wheel ZipFile."""
    res = []
    for path in whl_zip.namelist():
        m = re.match(r"[^/\\]+-[^/\\]+\.dist-info/", path)
        if m:
            res.append(path)
    if res:
        return res
    raise Exception("No .dist-info folder found in wheel")


def _get_wheel_metadata_from_wheel(whl_basename, metadata_directory, config_settings):
    """Extract the metadata from a wheel.

    Fallback for when the build backend does not
    define the 'get_wheel_metadata' hook.
    """
    from zipfile import ZipFile

    with open(os.path.join(metadata_directory, WHEEL_BUILT_MARKER), "wb"):
        pass  # Touch marker file

    whl_file = os.path.join(metadata_directory, whl_basename)
    with ZipFile(whl_file) as zipf:
        dist_info = _dist_info_files(zipf)
        zipf.extractall(path=metadata_directory, members=dist_info)
    return dist_info[0].split("/")[0]


def _find_already_built_wheel(metadata_directory):
    """Check for a wheel already built during the get_wheel_metadata hook."""
    if not metadata_directory:
        return None
    metadata_parent = os.path.dirname(metadata_directory)
    if not os.path.isfile(pjoin(metadata_parent, WHEEL_BUILT_MARKER)):
        return None

    whl_files = glob(os.path.join(metadata_parent, "*.whl"))
    if not whl_files:
        print("Found wheel built marker, but no .whl files")
        return None
    if len(whl_files) > 1:
        print(
            "Found multiple .whl files; unspecified behaviour. "
            "Will call build_wheel."
        )
        return None

    # Exactly one .whl file
    return whl_files[0]


def build_wheel(wheel_directory, config_settings, metadata_directory=None):
    """Invoke the mandatory build_wheel hook.

    If a wheel was already built in the
    prepare_metadata_for_build_wheel fallback, this
    will copy it rather than rebuilding the wheel.
    """
    prebuilt_whl = _find_already_built_wheel(metadata_directory)
    if prebuilt_whl:
        shutil.copy2(prebuilt_whl, wheel_directory)
        return os.path.basename(prebuilt_whl)

    return _build_backend().build_wheel(
        wheel_directory, config_settings, metadata_directory
    )


def build_editable(wheel_directory, config_settings, metadata_directory=None):
    """Invoke the optional build_editable hook.

    If a wheel was already built in the
    prepare_metadata_for_build_editable fallback, this
    will copy it rather than rebuilding the wheel.
    """
    backend = _build_backend()
    try:
        hook = backend.build_editable
    except AttributeError:
        raise HookMissing()
    else:
        prebuilt_whl = _find_already_built_wheel(metadata_directory)
        if prebuilt_whl:
            shutil.copy2(prebuilt_whl, wheel_directory)
            return os.path.basename(prebuilt_whl)

        return hook(wheel_directory, config_settings, metadata_directory)


def get_requires_for_build_sdist(config_settings):
    """Invoke the optional get_requires_for_build_wheel hook

    Returns [] if the hook is not defined.
    """
    backend = _build_backend()
    try:
        hook = backend.get_requires_for_build_sdist
    except AttributeError:
        return []
    else:
        return hook(config_settings)


class _DummyException(Exception):
    """Nothing should ever raise this exception"""


class GotUnsupportedOperation(Exception):
    """For internal use when backend raises UnsupportedOperation"""

    def __init__(self, traceback):
        self.traceback = traceback


def build_sdist(sdist_directory, config_settings):
    """Invoke the mandatory build_sdist hook."""
    backend = _build_backend()
    try:
        return backend.build_sdist(sdist_directory, config_settings)
    except getattr(backend, "UnsupportedOperation", _DummyException):
        raise GotUnsupportedOperation(traceback.format_exc())


HOOK_NAMES = {
    "get_requires_for_build_wheel",
    "prepare_metadata_for_build_wheel",
    "build_wheel",
    "get_requires_for_build_editable",
    "prepare_metadata_for_build_editable",
    "build_editable",
    "get_requires_for_build_sdist",
    "build_sdist",
    "_supported_features",
}


def main():
    if len(sys.argv) < 3:
        sys.exit("Needs args: hook_name, control_dir")
    hook_name = sys.argv[1]
    control_dir = sys.argv[2]
    if hook_name not in HOOK_NAMES:
        sys.exit("Unknown hook: %s" % hook_name)

    # Remove the parent directory from sys.path to avoid polluting the backend
    # import namespace with this directory.
    here = os.path.dirname(__file__)
    if here in sys.path:
        sys.path.remove(here)

    hook = globals()[hook_name]

    hook_input = read_json(pjoin(control_dir, "input.json"))

    json_out = {"unsupported": False, "return_val": None}
    try:
        json_out["return_val"] = hook(**hook_input["kwargs"])
    except BackendUnavailable as e:
        json_out["no_backend"] = True
        json_out["traceback"] = e.traceback
        json_out["backend_error"] = e.message
    except GotUnsupportedOperation as e:
        json_out["unsupported"] = True
        json_out["traceback"] = e.traceback
    except HookMissing as e:
        json_out["hook_missing"] = True
        json_out["missing_hook_name"] = e.hook_name or hook_name

    write_json(json_out, pjoin(control_dir, "output.json"), indent=2)


if __name__ == "__main__":
    main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\special\beta_functions.py ===
from sympy.core import S
from sympy.core.function import DefinedFunction, ArgumentIndexError
from sympy.core.symbol import Dummy, uniquely_named_symbol
from sympy.functions.special.gamma_functions import gamma, digamma
from sympy.functions.combinatorial.numbers import catalan
from sympy.functions.elementary.complexes import conjugate

# See mpmath #569 and SymPy #20569
def betainc_mpmath_fix(a, b, x1, x2, reg=0):
    from mpmath import betainc, mpf
    if x1 == x2:
        return mpf(0)
    else:
        return betainc(a, b, x1, x2, reg)

###############################################################################
############################ COMPLETE BETA  FUNCTION ##########################
###############################################################################

class beta(DefinedFunction):
    r"""
    The beta integral is called the Eulerian integral of the first kind by
    Legendre:

    .. math::
        \mathrm{B}(x,y)  \int^{1}_{0} t^{x-1} (1-t)^{y-1} \mathrm{d}t.

    Explanation
    ===========

    The Beta function or Euler's first integral is closely associated
    with the gamma function. The Beta function is often used in probability
    theory and mathematical statistics. It satisfies properties like:

    .. math::
        \mathrm{B}(a,1) = \frac{1}{a} \\
        \mathrm{B}(a,b) = \mathrm{B}(b,a)  \\
        \mathrm{B}(a,b) = \frac{\Gamma(a) \Gamma(b)}{\Gamma(a+b)}

    Therefore for integral values of $a$ and $b$:

    .. math::
        \mathrm{B} = \frac{(a-1)! (b-1)!}{(a+b-1)!}

    A special case of the Beta function when `x = y` is the
    Central Beta function. It satisfies properties like:

    .. math::
        \mathrm{B}(x) = 2^{1 - 2x}\mathrm{B}(x, \frac{1}{2})
        \mathrm{B}(x) = 2^{1 - 2x} cos(\pi x) \mathrm{B}(\frac{1}{2} - x, x)
        \mathrm{B}(x) = \int_{0}^{1} \frac{t^x}{(1 + t)^{2x}} dt
        \mathrm{B}(x) = \frac{2}{x} \prod_{n = 1}^{\infty} \frac{n(n + 2x)}{(n + x)^2}

    Examples
    ========

    >>> from sympy import I, pi
    >>> from sympy.abc import x, y

    The Beta function obeys the mirror symmetry:

    >>> from sympy import beta, conjugate
    >>> conjugate(beta(x, y))
    beta(conjugate(x), conjugate(y))

    Differentiation with respect to both $x$ and $y$ is supported:

    >>> from sympy import beta, diff
    >>> diff(beta(x, y), x)
    (polygamma(0, x) - polygamma(0, x + y))*beta(x, y)

    >>> diff(beta(x, y), y)
    (polygamma(0, y) - polygamma(0, x + y))*beta(x, y)

    >>> diff(beta(x), x)
    2*(polygamma(0, x) - polygamma(0, 2*x))*beta(x, x)

    We can numerically evaluate the Beta function to
    arbitrary precision for any complex numbers x and y:

    >>> from sympy import beta
    >>> beta(pi).evalf(40)
    0.02671848900111377452242355235388489324562

    >>> beta(1 + I).evalf(20)
    -0.2112723729365330143 - 0.7655283165378005676*I

    See Also
    ========

    gamma: Gamma function.
    uppergamma: Upper incomplete gamma function.
    lowergamma: Lower incomplete gamma function.
    polygamma: Polygamma function.
    loggamma: Log Gamma function.
    digamma: Digamma function.
    trigamma: Trigamma function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Beta_function
    .. [2] https://mathworld.wolfram.com/BetaFunction.html
    .. [3] https://dlmf.nist.gov/5.12

    """
    unbranched = True

    def fdiff(self, argindex):
        x, y = self.args
        if argindex == 1:
            # Diff wrt x
            return beta(x, y)*(digamma(x) - digamma(x + y))
        elif argindex == 2:
            # Diff wrt y
            return beta(x, y)*(digamma(y) - digamma(x + y))
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, x, y=None):
        if y is None:
            return beta(x, x)
        if x.is_Number and y.is_Number:
            return beta(x, y, evaluate=False).doit()

    def doit(self, **hints):
        x = xold = self.args[0]
        # Deal with unevaluated single argument beta
        single_argument = len(self.args) == 1
        y = yold = self.args[0] if single_argument else self.args[1]
        if hints.get('deep', True):
            x = x.doit(**hints)
            y = y.doit(**hints)
        if y.is_zero or x.is_zero:
            return S.ComplexInfinity
        if y is S.One:
            return 1/x
        if x is S.One:
            return 1/y
        if y == x + 1:
            return 1/(x*y*catalan(x))
        s = x + y
        if (s.is_integer and s.is_negative and x.is_integer is False and
            y.is_integer is False):
            return S.Zero
        if x == xold and y == yold and not single_argument:
            return self
        return beta(x, y)

    def _eval_expand_func(self, **hints):
        x, y = self.args
        return gamma(x)*gamma(y) / gamma(x + y)

    def _eval_is_real(self):
        return self.args[0].is_real and self.args[1].is_real

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate(), self.args[1].conjugate())

    def _eval_rewrite_as_gamma(self, x, y, piecewise=True, **kwargs):
        return self._eval_expand_func(**kwargs)

    def _eval_rewrite_as_Integral(self, x, y, **kwargs):
        from sympy.integrals.integrals import Integral
        t = Dummy(uniquely_named_symbol('t', [x, y]).name)
        return Integral(t**(x - 1)*(1 - t)**(y - 1), (t, 0, 1))

###############################################################################
########################## INCOMPLETE BETA FUNCTION ###########################
###############################################################################

class betainc(DefinedFunction):
    r"""
    The Generalized Incomplete Beta function is defined as

    .. math::
        \mathrm{B}_{(x_1, x_2)}(a, b) = \int_{x_1}^{x_2} t^{a - 1} (1 - t)^{b - 1} dt

    The Incomplete Beta function is a special case
    of the Generalized Incomplete Beta function :

    .. math:: \mathrm{B}_z (a, b) = \mathrm{B}_{(0, z)}(a, b)

    The Incomplete Beta function satisfies :

    .. math:: \mathrm{B}_z (a, b) = (-1)^a \mathrm{B}_{\frac{z}{z - 1}} (a, 1 - a - b)

    The Beta function is a special case of the Incomplete Beta function :

    .. math:: \mathrm{B}(a, b) = \mathrm{B}_{1}(a, b)

    Examples
    ========

    >>> from sympy import betainc, symbols, conjugate
    >>> a, b, x, x1, x2 = symbols('a b x x1 x2')

    The Generalized Incomplete Beta function is given by:

    >>> betainc(a, b, x1, x2)
    betainc(a, b, x1, x2)

    The Incomplete Beta function can be obtained as follows:

    >>> betainc(a, b, 0, x)
    betainc(a, b, 0, x)

    The Incomplete Beta function obeys the mirror symmetry:

    >>> conjugate(betainc(a, b, x1, x2))
    betainc(conjugate(a), conjugate(b), conjugate(x1), conjugate(x2))

    We can numerically evaluate the Incomplete Beta function to
    arbitrary precision for any complex numbers a, b, x1 and x2:

    >>> from sympy import betainc, I
    >>> betainc(2, 3, 4, 5).evalf(10)
    56.08333333
    >>> betainc(0.75, 1 - 4*I, 0, 2 + 3*I).evalf(25)
    0.2241657956955709603655887 + 0.3619619242700451992411724*I

    The Generalized Incomplete Beta function can be expressed
    in terms of the Generalized Hypergeometric function.

    >>> from sympy import hyper
    >>> betainc(a, b, x1, x2).rewrite(hyper)
    (-x1**a*hyper((a, 1 - b), (a + 1,), x1) + x2**a*hyper((a, 1 - b), (a + 1,), x2))/a

    See Also
    ========

    beta: Beta function
    hyper: Generalized Hypergeometric function

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Beta_function#Incomplete_beta_function
    .. [2] https://dlmf.nist.gov/8.17
    .. [3] https://functions.wolfram.com/GammaBetaErf/Beta4/
    .. [4] https://functions.wolfram.com/GammaBetaErf/BetaRegularized4/02/

    """
    nargs = 4
    unbranched = True

    def fdiff(self, argindex):
        a, b, x1, x2 = self.args
        if argindex == 3:
            # Diff wrt x1
            return -(1 - x1)**(b - 1)*x1**(a - 1)
        elif argindex == 4:
            # Diff wrt x2
            return (1 - x2)**(b - 1)*x2**(a - 1)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_mpmath(self):
        return betainc_mpmath_fix, self.args

    def _eval_is_real(self):
        if all(arg.is_real for arg in self.args):
            return True

    def _eval_conjugate(self):
        return self.func(*map(conjugate, self.args))

    def _eval_rewrite_as_Integral(self, a, b, x1, x2, **kwargs):
        from sympy.integrals.integrals import Integral
        t = Dummy(uniquely_named_symbol('t', [a, b, x1, x2]).name)
        return Integral(t**(a - 1)*(1 - t)**(b - 1), (t, x1, x2))

    def _eval_rewrite_as_hyper(self, a, b, x1, x2, **kwargs):
        from sympy.functions.special.hyper import hyper
        return (x2**a * hyper((a, 1 - b), (a + 1,), x2) - x1**a * hyper((a, 1 - b), (a + 1,), x1)) / a

###############################################################################
#################### REGULARIZED INCOMPLETE BETA FUNCTION #####################
###############################################################################

class betainc_regularized(DefinedFunction):
    r"""
    The Generalized Regularized Incomplete Beta function is given by

    .. math::
        \mathrm{I}_{(x_1, x_2)}(a, b) = \frac{\mathrm{B}_{(x_1, x_2)}(a, b)}{\mathrm{B}(a, b)}

    The Regularized Incomplete Beta function is a special case
    of the Generalized Regularized Incomplete Beta function :

    .. math:: \mathrm{I}_z (a, b) = \mathrm{I}_{(0, z)}(a, b)

    The Regularized Incomplete Beta function is the cumulative distribution
    function of the beta distribution.

    Examples
    ========

    >>> from sympy import betainc_regularized, symbols, conjugate
    >>> a, b, x, x1, x2 = symbols('a b x x1 x2')

    The Generalized Regularized Incomplete Beta
    function is given by:

    >>> betainc_regularized(a, b, x1, x2)
    betainc_regularized(a, b, x1, x2)

    The Regularized Incomplete Beta function
    can be obtained as follows:

    >>> betainc_regularized(a, b, 0, x)
    betainc_regularized(a, b, 0, x)

    The Regularized Incomplete Beta function
    obeys the mirror symmetry:

    >>> conjugate(betainc_regularized(a, b, x1, x2))
    betainc_regularized(conjugate(a), conjugate(b), conjugate(x1), conjugate(x2))

    We can numerically evaluate the Regularized Incomplete Beta function
    to arbitrary precision for any complex numbers a, b, x1 and x2:

    >>> from sympy import betainc_regularized, pi, E
    >>> betainc_regularized(1, 2, 0, 0.25).evalf(10)
    0.4375000000
    >>> betainc_regularized(pi, E, 0, 1).evalf(5)
    1.00000

    The Generalized Regularized Incomplete Beta function can be
    expressed in terms of the Generalized Hypergeometric function.

    >>> from sympy import hyper
    >>> betainc_regularized(a, b, x1, x2).rewrite(hyper)
    (-x1**a*hyper((a, 1 - b), (a + 1,), x1) + x2**a*hyper((a, 1 - b), (a + 1,), x2))/(a*beta(a, b))

    See Also
    ========

    beta: Beta function
    hyper: Generalized Hypergeometric function

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Beta_function#Incomplete_beta_function
    .. [2] https://dlmf.nist.gov/8.17
    .. [3] https://functions.wolfram.com/GammaBetaErf/Beta4/
    .. [4] https://functions.wolfram.com/GammaBetaErf/BetaRegularized4/02/

    """
    nargs = 4
    unbranched = True

    def __new__(cls, a, b, x1, x2):
        return super().__new__(cls, a, b, x1, x2)

    def _eval_mpmath(self):
        return betainc_mpmath_fix, (*self.args, S(1))

    def fdiff(self, argindex):
        a, b, x1, x2 = self.args
        if argindex == 3:
            # Diff wrt x1
            return -(1 - x1)**(b - 1)*x1**(a - 1) / beta(a, b)
        elif argindex == 4:
            # Diff wrt x2
            return (1 - x2)**(b - 1)*x2**(a - 1) / beta(a, b)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_is_real(self):
        if all(arg.is_real for arg in self.args):
            return True

    def _eval_conjugate(self):
        return self.func(*map(conjugate, self.args))

    def _eval_rewrite_as_Integral(self, a, b, x1, x2, **kwargs):
        from sympy.integrals.integrals import Integral
        t = Dummy(uniquely_named_symbol('t', [a, b, x1, x2]).name)
        integrand = t**(a - 1)*(1 - t)**(b - 1)
        expr = Integral(integrand, (t, x1, x2))
        return expr / Integral(integrand, (t, 0, 1))

    def _eval_rewrite_as_hyper(self, a, b, x1, x2, **kwargs):
        from sympy.functions.special.hyper import hyper
        expr = (x2**a * hyper((a, 1 - b), (a + 1,), x2) - x1**a * hyper((a, 1 - b), (a + 1,), x1)) / a
        return expr / beta(a, b)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\indexes\api.py ===
from __future__ import annotations

import textwrap
from typing import (
    TYPE_CHECKING,
    cast,
)

import numpy as np

from pandas._libs import (
    NaT,
    lib,
)
from pandas.errors import InvalidIndexError

from pandas.core.dtypes.cast import find_common_type

from pandas.core.algorithms import safe_sort
from pandas.core.indexes.base import (
    Index,
    _new_Index,
    ensure_index,
    ensure_index_from_sequences,
    get_unanimous_names,
)
from pandas.core.indexes.category import CategoricalIndex
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.interval import IntervalIndex
from pandas.core.indexes.multi import MultiIndex
from pandas.core.indexes.period import PeriodIndex
from pandas.core.indexes.range import RangeIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex

if TYPE_CHECKING:
    from pandas._typing import Axis
_sort_msg = textwrap.dedent(
    """\
Sorting because non-concatenation axis is not aligned. A future version
of pandas will change to not sort by default.

To accept the future behavior, pass 'sort=False'.

To retain the current behavior and silence the warning, pass 'sort=True'.
"""
)


__all__ = [
    "Index",
    "MultiIndex",
    "CategoricalIndex",
    "IntervalIndex",
    "RangeIndex",
    "InvalidIndexError",
    "TimedeltaIndex",
    "PeriodIndex",
    "DatetimeIndex",
    "_new_Index",
    "NaT",
    "ensure_index",
    "ensure_index_from_sequences",
    "get_objs_combined_axis",
    "union_indexes",
    "get_unanimous_names",
    "all_indexes_same",
    "default_index",
    "safe_sort_index",
]


def get_objs_combined_axis(
    objs,
    intersect: bool = False,
    axis: Axis = 0,
    sort: bool = True,
    copy: bool = False,
) -> Index:
    """
    Extract combined index: return intersection or union (depending on the
    value of "intersect") of indexes on given axis, or None if all objects
    lack indexes (e.g. they are numpy arrays).

    Parameters
    ----------
    objs : list
        Series or DataFrame objects, may be mix of the two.
    intersect : bool, default False
        If True, calculate the intersection between indexes. Otherwise,
        calculate the union.
    axis : {0 or 'index', 1 or 'outer'}, default 0
        The axis to extract indexes from.
    sort : bool, default True
        Whether the result index should come out sorted or not.
    copy : bool, default False
        If True, return a copy of the combined index.

    Returns
    -------
    Index
    """
    obs_idxes = [obj._get_axis(axis) for obj in objs]
    return _get_combined_index(obs_idxes, intersect=intersect, sort=sort, copy=copy)


def _get_distinct_objs(objs: list[Index]) -> list[Index]:
    """
    Return a list with distinct elements of "objs" (different ids).
    Preserves order.
    """
    ids: set[int] = set()
    res = []
    for obj in objs:
        if id(obj) not in ids:
            ids.add(id(obj))
            res.append(obj)
    return res


def _get_combined_index(
    indexes: list[Index],
    intersect: bool = False,
    sort: bool = False,
    copy: bool = False,
) -> Index:
    """
    Return the union or intersection of indexes.

    Parameters
    ----------
    indexes : list of Index or list objects
        When intersect=True, do not accept list of lists.
    intersect : bool, default False
        If True, calculate the intersection between indexes. Otherwise,
        calculate the union.
    sort : bool, default False
        Whether the result index should come out sorted or not.
    copy : bool, default False
        If True, return a copy of the combined index.

    Returns
    -------
    Index
    """
    # TODO: handle index names!
    indexes = _get_distinct_objs(indexes)
    if len(indexes) == 0:
        index = Index([])
    elif len(indexes) == 1:
        index = indexes[0]
    elif intersect:
        index = indexes[0]
        for other in indexes[1:]:
            index = index.intersection(other)
    else:
        index = union_indexes(indexes, sort=False)
        index = ensure_index(index)

    if sort:
        index = safe_sort_index(index)
    # GH 29879
    if copy:
        index = index.copy()

    return index


def safe_sort_index(index: Index) -> Index:
    """
    Returns the sorted index

    We keep the dtypes and the name attributes.

    Parameters
    ----------
    index : an Index

    Returns
    -------
    Index
    """
    if index.is_monotonic_increasing:
        return index

    try:
        array_sorted = safe_sort(index)
    except TypeError:
        pass
    else:
        if isinstance(array_sorted, Index):
            return array_sorted

        array_sorted = cast(np.ndarray, array_sorted)
        if isinstance(index, MultiIndex):
            index = MultiIndex.from_tuples(array_sorted, names=index.names)
        else:
            index = Index(array_sorted, name=index.name, dtype=index.dtype)

    return index


def union_indexes(indexes, sort: bool | None = True) -> Index:
    """
    Return the union of indexes.

    The behavior of sort and names is not consistent.

    Parameters
    ----------
    indexes : list of Index or list objects
    sort : bool, default True
        Whether the result index should come out sorted or not.

    Returns
    -------
    Index
    """
    if len(indexes) == 0:
        raise AssertionError("Must have at least 1 Index to union")
    if len(indexes) == 1:
        result = indexes[0]
        if isinstance(result, list):
            if not sort:
                result = Index(result)
            else:
                result = Index(sorted(result))
        return result

    indexes, kind = _sanitize_and_check(indexes)

    def _unique_indices(inds, dtype) -> Index:
        """
        Concatenate indices and remove duplicates.

        Parameters
        ----------
        inds : list of Index or list objects
        dtype : dtype to set for the resulting Index

        Returns
        -------
        Index
        """
        if all(isinstance(ind, Index) for ind in inds):
            inds = [ind.astype(dtype, copy=False) for ind in inds]
            result = inds[0].unique()
            other = inds[1].append(inds[2:])
            diff = other[result.get_indexer_for(other) == -1]
            if len(diff):
                result = result.append(diff.unique())
            if sort:
                result = result.sort_values()
            return result

        def conv(i):
            if isinstance(i, Index):
                i = i.tolist()
            return i

        return Index(
            lib.fast_unique_multiple_list([conv(i) for i in inds], sort=sort),
            dtype=dtype,
        )

    def _find_common_index_dtype(inds):
        """
        Finds a common type for the indexes to pass through to resulting index.

        Parameters
        ----------
        inds: list of Index or list objects

        Returns
        -------
        The common type or None if no indexes were given
        """
        dtypes = [idx.dtype for idx in indexes if isinstance(idx, Index)]
        if dtypes:
            dtype = find_common_type(dtypes)
        else:
            dtype = None

        return dtype

    if kind == "special":
        result = indexes[0]

        dtis = [x for x in indexes if isinstance(x, DatetimeIndex)]
        dti_tzs = [x for x in dtis if x.tz is not None]
        if len(dti_tzs) not in [0, len(dtis)]:
            # TODO: this behavior is not tested (so may not be desired),
            #  but is kept in order to keep behavior the same when
            #  deprecating union_many
            # test_frame_from_dict_with_mixed_indexes
            raise TypeError("Cannot join tz-naive with tz-aware DatetimeIndex")

        if len(dtis) == len(indexes):
            sort = True
            result = indexes[0]

        elif len(dtis) > 1:
            # If we have mixed timezones, our casting behavior may depend on
            #  the order of indexes, which we don't want.
            sort = False

            # TODO: what about Categorical[dt64]?
            # test_frame_from_dict_with_mixed_indexes
            indexes = [x.astype(object, copy=False) for x in indexes]
            result = indexes[0]

        for other in indexes[1:]:
            result = result.union(other, sort=None if sort else False)
        return result

    elif kind == "array":
        dtype = _find_common_index_dtype(indexes)
        index = indexes[0]
        if not all(index.equals(other) for other in indexes[1:]):
            index = _unique_indices(indexes, dtype)

        name = get_unanimous_names(*indexes)[0]
        if name != index.name:
            index = index.rename(name)
        return index
    else:  # kind='list'
        dtype = _find_common_index_dtype(indexes)
        return _unique_indices(indexes, dtype)


def _sanitize_and_check(indexes):
    """
    Verify the type of indexes and convert lists to Index.

    Cases:

    - [list, list, ...]: Return ([list, list, ...], 'list')
    - [list, Index, ...]: Return _sanitize_and_check([Index, Index, ...])
        Lists are sorted and converted to Index.
    - [Index, Index, ...]: Return ([Index, Index, ...], TYPE)
        TYPE = 'special' if at least one special type, 'array' otherwise.

    Parameters
    ----------
    indexes : list of Index or list objects

    Returns
    -------
    sanitized_indexes : list of Index or list objects
    type : {'list', 'array', 'special'}
    """
    kinds = list({type(index) for index in indexes})

    if list in kinds:
        if len(kinds) > 1:
            indexes = [
                Index(list(x)) if not isinstance(x, Index) else x for x in indexes
            ]
            kinds.remove(list)
        else:
            return indexes, "list"

    if len(kinds) > 1 or Index not in kinds:
        return indexes, "special"
    else:
        return indexes, "array"


def all_indexes_same(indexes) -> bool:
    """
    Determine if all indexes contain the same elements.

    Parameters
    ----------
    indexes : iterable of Index objects

    Returns
    -------
    bool
        True if all indexes contain the same elements, False otherwise.
    """
    itr = iter(indexes)
    first = next(itr)
    return all(first.equals(index) for index in itr)


def default_index(n: int) -> RangeIndex:
    rng = range(n)
    return RangeIndex._simple_new(rng, name=None)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\kind.py ===
"""
Module to efficiently partition SymPy objects.

This system is introduced because class of SymPy object does not always
represent the mathematical classification of the entity. For example,
``Integral(1, x)`` and ``Integral(Matrix([1,2]), x)`` are both instance
of ``Integral`` class. However the former is number and the latter is
matrix.

One way to resolve this is defining subclass for each mathematical type,
such as ``MatAdd`` for the addition between matrices. Basic algebraic
operation such as addition or multiplication take this approach, but
defining every class for every mathematical object is not scalable.

Therefore, we define the "kind" of the object and let the expression
infer the kind of itself from its arguments. Function and class can
filter the arguments by their kind, and behave differently according to
the type of itself.

This module defines basic kinds for core objects. Other kinds such as
``ArrayKind`` or ``MatrixKind`` can be found in corresponding modules.

.. notes::
       This approach is experimental, and can be replaced or deleted in the future.
       See https://github.com/sympy/sympy/pull/20549.
"""

from collections import defaultdict

from .cache import cacheit
from sympy.multipledispatch.dispatcher import (Dispatcher,
    ambiguity_warn, ambiguity_register_error_ignore_dup,
    str_signature, RaiseNotImplementedError)


class KindMeta(type):
    """
    Metaclass for ``Kind``.

    Assigns empty ``dict`` as class attribute ``_inst`` for every class,
    in order to endow singleton-like behavior.
    """
    def __new__(cls, clsname, bases, dct):
        dct['_inst'] = {}
        return super().__new__(cls, clsname, bases, dct)


class Kind(object, metaclass=KindMeta):
    """
    Base class for kinds.

    Kind of the object represents the mathematical classification that
    the entity falls into. It is expected that functions and classes
    recognize and filter the argument by its kind.

    Kind of every object must be carefully selected so that it shows the
    intention of design. Expressions may have different kind according
    to the kind of its arguments. For example, arguments of ``Add``
    must have common kind since addition is group operator, and the
    resulting ``Add()`` has the same kind.

    For the performance, each kind is as broad as possible and is not
    based on set theory. For example, ``NumberKind`` includes not only
    complex number but expression containing ``S.Infinity`` or ``S.NaN``
    which are not strictly number.

    Kind may have arguments as parameter. For example, ``MatrixKind()``
    may be constructed with one element which represents the kind of its
    elements.

    ``Kind`` behaves in singleton-like fashion. Same signature will
    return the same object.

    """
    def __new__(cls, *args):
        if args in cls._inst:
            inst = cls._inst[args]
        else:
            inst = super().__new__(cls)
            cls._inst[args] = inst
        return inst


class _UndefinedKind(Kind):
    """
    Default kind for all SymPy object. If the kind is not defined for
    the object, or if the object cannot infer the kind from its
    arguments, this will be returned.

    Examples
    ========

    >>> from sympy import Expr
    >>> Expr().kind
    UndefinedKind
    """
    def __new__(cls):
        return super().__new__(cls)

    def __repr__(self):
        return "UndefinedKind"

UndefinedKind = _UndefinedKind()


class _NumberKind(Kind):
    """
    Kind for all numeric object.

    This kind represents every number, including complex numbers,
    infinity and ``S.NaN``. Other objects such as quaternions do not
    have this kind.

    Most ``Expr`` are initially designed to represent the number, so
    this will be the most common kind in SymPy core. For example
    ``Symbol()``, which represents a scalar, has this kind as long as it
    is commutative.

    Numbers form a field. Any operation between number-kind objects will
    result this kind as well.

    Examples
    ========

    >>> from sympy import S, oo, Symbol
    >>> S.One.kind
    NumberKind
    >>> (-oo).kind
    NumberKind
    >>> S.NaN.kind
    NumberKind

    Commutative symbol are treated as number.

    >>> x = Symbol('x')
    >>> x.kind
    NumberKind
    >>> Symbol('y', commutative=False).kind
    UndefinedKind

    Operation between numbers results number.

    >>> (x+1).kind
    NumberKind

    See Also
    ========

    sympy.core.expr.Expr.is_Number : check if the object is strictly
    subclass of ``Number`` class.

    sympy.core.expr.Expr.is_number : check if the object is number
    without any free symbol.

    """
    def __new__(cls):
        return super().__new__(cls)

    def __repr__(self):
        return "NumberKind"

NumberKind = _NumberKind()


class _BooleanKind(Kind):
    """
    Kind for boolean objects.

    SymPy's ``S.true``, ``S.false``, and built-in ``True`` and ``False``
    have this kind. Boolean number ``1`` and ``0`` are not relevant.

    Examples
    ========

    >>> from sympy import S, Q
    >>> S.true.kind
    BooleanKind
    >>> Q.even(3).kind
    BooleanKind
    """
    def __new__(cls):
        return super().__new__(cls)

    def __repr__(self):
        return "BooleanKind"

BooleanKind = _BooleanKind()


class KindDispatcher:
    """
    Dispatcher to select a kind from multiple kinds by binary dispatching.

    .. notes::
       This approach is experimental, and can be replaced or deleted in
       the future.

    Explanation
    ===========

    SymPy object's :obj:`sympy.core.kind.Kind()` vaguely represents the
    algebraic structure where the object belongs to. Therefore, with
    given operation, we can always find a dominating kind among the
    different kinds. This class selects the kind by recursive binary
    dispatching. If the result cannot be determined, ``UndefinedKind``
    is returned.

    Examples
    ========

    Multiplication between numbers return number.

    >>> from sympy import NumberKind, Mul
    >>> Mul._kind_dispatcher(NumberKind, NumberKind)
    NumberKind

    Multiplication between number and unknown-kind object returns unknown kind.

    >>> from sympy import UndefinedKind
    >>> Mul._kind_dispatcher(NumberKind, UndefinedKind)
    UndefinedKind

    Any number and order of kinds is allowed.

    >>> Mul._kind_dispatcher(UndefinedKind, NumberKind)
    UndefinedKind
    >>> Mul._kind_dispatcher(NumberKind, UndefinedKind, NumberKind)
    UndefinedKind

    Since matrix forms a vector space over scalar field, multiplication
    between matrix with numeric element and number returns matrix with
    numeric element.

    >>> from sympy.matrices import MatrixKind
    >>> Mul._kind_dispatcher(MatrixKind(NumberKind), NumberKind)
    MatrixKind(NumberKind)

    If a matrix with number element and another matrix with unknown-kind
    element are multiplied, we know that the result is matrix but the
    kind of its elements is unknown.

    >>> Mul._kind_dispatcher(MatrixKind(NumberKind), MatrixKind(UndefinedKind))
    MatrixKind(UndefinedKind)

    Parameters
    ==========

    name : str

    commutative : bool, optional
        If True, binary dispatch will be automatically registered in
        reversed order as well.

    doc : str, optional

    """
    def __init__(self, name, commutative=False, doc=None):
        self.name = name
        self.doc = doc
        self.commutative = commutative
        self._dispatcher = Dispatcher(name)

    def __repr__(self):
        return "<dispatched %s>" % self.name

    def register(self, *types, **kwargs):
        """
        Register the binary dispatcher for two kind classes.

        If *self.commutative* is ``True``, signature in reversed order is
        automatically registered as well.
        """
        on_ambiguity = kwargs.pop("on_ambiguity", None)
        if not on_ambiguity:
            if self.commutative:
                on_ambiguity = ambiguity_register_error_ignore_dup
            else:
                on_ambiguity = ambiguity_warn
        kwargs.update(on_ambiguity=on_ambiguity)

        if not len(types) == 2:
            raise RuntimeError(
                "Only binary dispatch is supported, but got %s types: <%s>." % (
                len(types), str_signature(types)
            ))

        def _(func):
            self._dispatcher.add(types, func, **kwargs)
            if self.commutative:
                self._dispatcher.add(tuple(reversed(types)), func, **kwargs)
        return _

    def __call__(self, *args, **kwargs):
        if self.commutative:
            kinds = frozenset(args)
        else:
            kinds = []
            prev = None
            for a in args:
                if prev is not a:
                    kinds.append(a)
                    prev = a
        return self.dispatch_kinds(kinds, **kwargs)

    @cacheit
    def dispatch_kinds(self, kinds, **kwargs):
        # Quick exit for the case where all kinds are same
        if len(kinds) == 1:
            result, = kinds
            if not isinstance(result, Kind):
                raise RuntimeError("%s is not a kind." % result)
            return result

        for i,kind in enumerate(kinds):
            if not isinstance(kind, Kind):
                raise RuntimeError("%s is not a kind." % kind)

            if i == 0:
                result = kind
            else:
                prev_kind = result

                t1, t2 = type(prev_kind), type(kind)
                k1, k2 = prev_kind, kind
                func = self._dispatcher.dispatch(t1, t2)
                if func is None and self.commutative:
                    # try reversed order
                    func = self._dispatcher.dispatch(t2, t1)
                    k1, k2 = k2, k1
                if func is None:
                    # unregistered kind relation
                    result = UndefinedKind
                else:
                    result = func(k1, k2)
                if not isinstance(result, Kind):
                    raise RuntimeError(
                        "Dispatcher for {!r} and {!r} must return a Kind, but got {!r}".format(
                        prev_kind, kind, result
                    ))

        return result

    @property
    def __doc__(self):
        docs = [
            "Kind dispatcher : %s" % self.name,
            "Note that support for this is experimental. See the docs for :class:`KindDispatcher` for details"
        ]

        if self.doc:
            docs.append(self.doc)

        s = "Registered kind classes\n"
        s += '=' * len(s)
        docs.append(s)

        amb_sigs = []

        typ_sigs = defaultdict(list)
        for sigs in self._dispatcher.ordering[::-1]:
            key = self._dispatcher.funcs[sigs]
            typ_sigs[key].append(sigs)

        for func, sigs in typ_sigs.items():

            sigs_str = ', '.join('<%s>' % str_signature(sig) for sig in sigs)

            if isinstance(func, RaiseNotImplementedError):
                amb_sigs.append(sigs_str)
                continue

            s = 'Inputs: %s\n' % sigs_str
            s += '-' * len(s) + '\n'
            if func.__doc__:
                s += func.__doc__.strip()
            else:
                s += func.__name__
            docs.append(s)

        if amb_sigs:
            s = "Ambiguous kind classes\n"
            s += '=' * len(s)
            docs.append(s)

            s = '\n'.join(amb_sigs)
            docs.append(s)

        return '\n\n'.join(docs)