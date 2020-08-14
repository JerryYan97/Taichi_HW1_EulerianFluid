import taichi as ti


class TexPair:
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur


# The dx is 1 by default.
@ti.func
def sample(qf, u, v, res):
    i, j = int(u), int(v)
    # Nearest
    i = max(0, min(res - 1, i))
    j = max(0, min(res - 1, j))
    return qf[i, j]


@ti.func
def lerp(vl, vr, frac):
    # frac: [0.0, 1.0]
    return vl + frac * (vr - vl)


@ti.func
def bilerp(vf, u, v, res):
    s, t = u - 0.5, v - 0.5
    # floor
    iu, iv = int(s), int(t)
    # fract
    fu, fv = s - iu, t - iv
    a = sample(vf, iu + 0.5, iv + 0.5, res)
    b = sample(vf, iu + 1.5, iv + 0.5, res)
    c = sample(vf, iu + 0.5, iv + 1.5, res)
    d = sample(vf, iu + 1.5, iv + 1.5, res)
    return lerp(lerp(a, b, fu), lerp(c, d, fu), fv)


@ti.kernel
def jacobi_iteration(iA: ti.template(), ix: ti.template(), ix_new: ti.template(), diff_vec: ti.template(), ib: ti.template()) -> ti.f32:
    # Input:
    # A: The A matrix (ti.var(shape = (iSize, iSize))).
    # b: The b vector (ti.var(shape = iSize)).
    # ix: It is used as both input and output (ti.var(shape = iSize)).
    # When it is an input, it represents a guess.
    # When it is an output, it represents the result after an iteration.
    # iSize: If A is a n x n matrix, then iSize should be n.
    # Reference: Numerical method books.
    norm1 = -1.0
    norm2 = -1.0
    for i in ti.static(range(ix.shape()[0])):
        ix[i] = ix_new[i]
        temp = ib[i]
        for j in ti.static(range(ix.shape()[0])):
            if i != j:
                temp -= iA[i, j] * ix[j]
        # Divide everything by the coefficient of that unknown:
        ix_new[i] = temp / iA[i, i]
    # Calculate the residual of this iteration by using infinite norm:
    for i in ti.static(range(ix.shape()[0])):
        diff_vec[i] = ix[i] - ix_new[i]
        if ti.abs(diff_vec[i]) > norm1:
            norm1 = ti.abs(diff_vec[i])
        if ti.abs(ix[i]) > norm2:
            norm2 = ti.abs(ix[i])
    return norm1 / norm2


@ti.kernel
def dot(r1: ti.template(), r2: ti.template()) -> ti.f32:
    res = 0.0
    for i in ti.static(range(r1.shape()[0])):
        res += r1[i] * r2[i]
    return res


@ti.kernel
def Ab_multiply(A: ti.template(), b: ti.template(), output: ti.template()):
    for i in ti.static(range(b.shape()[0])):
        output[i] = 0.0
    for i, j in ti.ndrange(b.shape()[0], b.shape()[0]):
        output[i] += A[i, j] * b[j]


