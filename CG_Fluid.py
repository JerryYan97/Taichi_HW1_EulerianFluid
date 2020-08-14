import taichi as ti
import Util


ti.init(arch=ti.gpu)

# General settings:
resolutionX = 512
pixels = ti.var(ti.f32, shape=(resolutionX, resolutionX))
dt = 0.02
dx = 1.0
inv_dx = 1.0 / dx
half_inv_dx = 0.5 * inv_dx
pause = False
temp_dye = 1.0
pixel_mid = resolutionX // 2
ix_length = 5
iy_length = 10
area = ti.Vector([8, pixel_mid - iy_length, 8 + ix_length, pixel_mid + iy_length])
inflow_velocity = ti.Vector([3.0, 0.0])


# Grid settings:
_velocities = ti.Vector(2, dt=ti.f32, shape=(resolutionX, resolutionX))
_new_velocities = ti.Vector(2, dt=ti.f32, shape=(resolutionX, resolutionX))
velocity_divs = ti.var(dt=ti.f32, shape=(resolutionX, resolutionX))
_pressures = ti.var(dt=ti.f32, shape=(resolutionX, resolutionX))
_new_pressures = ti.var(dt=ti.f32, shape=(resolutionX, resolutionX))
_diff_pressures = ti.var(dt=ti.f32, shape=(resolutionX, resolutionX))
_dye_buffer = ti.var(dt=ti.f32, shape=(resolutionX, resolutionX))
_new_dye_buffer = ti.var(dt=ti.f32, shape=(resolutionX, resolutionX))

velocities_pair = Util.TexPair(_velocities, _new_velocities)
pressures_pair = Util.TexPair(_pressures, _new_pressures)
dyes_pair = Util.TexPair(_dye_buffer, _new_dye_buffer)

# CG settings:
b = ti.var(dt=ti.f32, shape=resolutionX * resolutionX)
p = ti.var(dt=ti.f32, shape=resolutionX * resolutionX)
Ax = ti.var(dt=ti.f32, shape=resolutionX * resolutionX)
Ap = ti.var(dt=ti.f32, shape=resolutionX * resolutionX)
r = ti.var(dt=ti.f32, shape=resolutionX * resolutionX)
new_r = ti.var(dt=ti.f32, shape=resolutionX * resolutionX)


# TODO:
# Design the bilinear interpolation and finite difference approximation.
# vf: velocity field; qf: quality field;
@ti.func
def vel_with_boundary(vf: ti.template(), i: int, j: int) -> ti.f32:
    if (i == j == 0) or (i == j == resolutionX - 1) or (i == 0 and j == resolutionX - 1) or (
            i == resolutionX - 1 and j == 0):
        vf[i, j] = ti.Vector([0.0, 0.0])
    elif i == 0:
        vf[i, j] = -vf[1, j]
    elif j == 0:
        # a = 3
        vf[i, 0] = -vf[i, 1]
    elif i == resolutionX - 1:
        vf[resolutionX - 1, j] = -vf[resolutionX - 2, j]
    elif j == resolutionX - 1:
        # a = 5
        vf[i, resolutionX - 1] = -vf[i, resolutionX - 2]
    return vf[i, j]


@ti.func
def p_with_boundary(pf: ti.template(), i: int, j: int) -> ti.f32:
    if (i == j == 0) or (i == j == resolutionX - 1) or (i == 0 and j == resolutionX - 1) or (
            i == resolutionX - 1 and j == 0):
        pf[i, j] = 0.0
    elif i == 0:
        pf[0, j] = pf[1, j]
    elif j == 0:
        pf[i, 0] = pf[i, 1]
    elif i == resolutionX - 1:
        pf[resolutionX - 1, j] = pf[resolutionX - 2, j]
    elif j == resolutionX - 1:
        pf[i, resolutionX - 1] = pf[i, resolutionX - 2]
    return pf[i, j]


@ti.kernel
def apply_vel_bc(vf: ti.template()):
    for i, j in vf:
        vel_with_boundary(vf, i, j)


@ti.kernel
def apply_p_bc(pf: ti.template()):
    for i, j in pf:
        p_with_boundary(pf, i, j)


@ti.kernel
def advect(vf: ti.template(), qf: ti.template(), new_qf: ti.template()):
    # Velocity field, pressure field and dye density field.
    # Semi_lagrangian + 2RK:
    for IX, IY in vf:
        # Backtrace:
        coord_curr = ti.Vector([IX, IY]) + ti.Vector([0.5, 0.5])
        vel_curr = vf[IX, IY]
        coord_mid = coord_curr - 0.5 * dt * vel_curr
        vel_mid = Util.bilerp(vf, coord_mid[0], coord_mid[1], resolutionX)
        coord_prev = coord_curr - dt * vel_mid
        # Get previous quality:
        q_prev = Util.bilerp(qf, coord_prev[0], coord_prev[1], resolutionX)
        # Update current quality:
        new_qf[IX, IY] = q_prev


# area: [bottom-left IX, bottom-left IY, top-right IX, top-right IY]
@ti.kernel
def addInflow(qf: ti.template(), area: ti.template(), quality: ti.template()):
    bl_ix, bl_iy, tr_ix, tr_iy = area[0], area[1], area[2], area[3]
    for i, j in qf:
        if bl_ix <= i <= tr_ix and bl_iy <= j <= tr_iy:
            qf[i, j] = quality


@ti.kernel
def fill_color(ipixels: ti.template(), idyef: ti.template()):
    for i, j in ipixels:
        density = ti.min(1.0, ti.max(0.0, idyef[i, j]))
        ipixels[i, j] = density


@ti.func
def p_matrix(i: int, j: int, resX: int) -> ti.f32:
    res = 0.0
    row = int(i)
    col = int(j)
    ele_num = int(resX) * int(resX)
    if row == col:
        res = -4.0
    elif ti.abs(col - row) == 1:
        res = 1.0
    elif (row + resX) == col:
        res = 1.0
    elif (row - resX) == col:
        res = 1.0
    if row < 0 or col < 0 or row >= ele_num or col >= ele_num:
        res = 0.0
    return res


@ti.func
def coeff_matrix(row: int, col: int) -> ti.f32:
    # row represents target equation or vid.
    res = 0.0
    n = resolutionX * resolutionX
    IY = row // resolutionX
    IX = row - resolutionX * IY
    if row >= n or row < 0 or col >= n or col < 0:
        res = 0.0
    else:
        if (IX == IY == 0) or (IX == IY == resolutionX - 1) or (IX == 0 and IY == resolutionX - 1) or (
                IX == resolutionX - 1 and IY == 0):
            if row == col:
                res = 1.0
            else:
                res = 0.0
    # Can get results that is similar to convergence:
        if row == col:
            res = -4.0
        elif ti.abs(col - row) == 1:
            res = 1.0
        elif ti.abs(col - row) == resolutionX:
            res = 1.0
    return res


@ti.func
def visit_vector(v: ti.template(), vid: int, length: int) -> ti.f32:
    res = 0.0
    if vid < 0 or vid >= length:
        res = 0.0
    else:
        res = v[vid]
    return res


@ti.func
def visit_pf_vector(pf: ti.template(), vid: int) -> ti.f32:
    res = 0.0
    n = resolutionX * resolutionX
    if vid >= n or vid < 0:
        res = 0.0
    else:
        IY = vid // resolutionX
        IX = vid - resolutionX * IY
        res = pf[IX, IY]
        # res = p_with_boundary(pf, IX, IY)
    return res


@ti.kernel
def pressure_cg_init(pf: ti.template(), b: ti.template()):
    n = resolutionX * resolutionX

    # TODO: Make it follow the ODEs shown in P5.
    for row in range(n):
        # Get vector element:
        ve1 = visit_pf_vector(pf, row - resolutionX)
        ve2 = visit_pf_vector(pf, row - 1)
        ve3 = visit_pf_vector(pf, row)
        ve4 = visit_pf_vector(pf, row + 1)
        ve5 = visit_pf_vector(pf, row + resolutionX)
        # Get matrix element:
        me1 = coeff_matrix(row, row - resolutionX)
        me2 = coeff_matrix(row, row - 1)
        me3 = coeff_matrix(row, row)
        me4 = coeff_matrix(row, row + 1)
        me5 = coeff_matrix(row, row + resolutionX)

        temp_Ax = me1 * ve1 + me2 * ve2 + me3 * ve3 + me4 * ve4 + me5 * ve5

        Ax[row] = temp_Ax
        r[row] = b[row] - Ax[row]
        p[row] = r[row]


@ti.kernel
def pressure_cg_iter(pf: ti.template()) -> ti.f32:
    # alpha_k = rkT * rk / pkT * A * pk
    # rkT * rk:
    n = resolutionX * resolutionX
    rkT_rk = 0.0
    pkT_A_pk = 0.0

    # TODO: Make it follow the ODEs shown in P5.
    for i in range(n):
        rkT_rk += (r[i] * r[i])
        # Ap[i] = 0.0
        ve1 = visit_vector(p, i - resolutionX, n)
        ve2 = visit_vector(p, i - 1, n)
        ve3 = visit_vector(p, i, n)
        ve4 = visit_vector(p, i + 1, n)
        ve5 = visit_vector(p, i + resolutionX, n)
        # Get matrix element:
        me1 = coeff_matrix(i, i - resolutionX)
        me2 = coeff_matrix(i, i - 1)
        me3 = coeff_matrix(i, i)
        me4 = coeff_matrix(i, i + 1)
        me5 = coeff_matrix(i, i + resolutionX)

        # Calculate number:
        temp_Ap = me1 * ve1 + me2 * ve2 + me3 * ve3 + me4 * ve4 + me5 * ve5
        Ap[i] = temp_Ap
        pkT_A_pk += (p[i] * temp_Ap)

    alpha = rkT_rk / pkT_A_pk
    res = 0.0
    top = 0.0
    bottom = 0.0
    # xk+1 = xk + alpha * pk
    # rk+1 = rk - alpha * A * pk
    for i in range(n):
        IY = i // resolutionX
        IX = i - resolutionX * IY
        new_pf_val = pf[IX, IY] + alpha * p[i]
        diff = ti.abs(new_pf_val - pf[IX, IY])
        res += (diff * diff)
        pf[IX, IY] = new_pf_val
        # pf[IX, IY] = p_with_boundary(pf, IX, IY) + alpha * p[i]
        new_r_val = r[i] - alpha * Ap[i]
        new_r[i] = new_r_val
        top += (new_r_val * new_r_val)
        bottom += (r[i] * r[i])

    beta = top / bottom
    for i in range(n):
        p[i] = new_r[i] + beta * p[i]
        # Swap:
        r[i] = new_r[i]
    res = ti.sqrt(res)
    return res


@ti.kernel
def construct_cg_b(divf: ti.template(), b: ti.template()):
    for IX, IY in divf:
        # TODO: Make it follow the ODEs shown in P5.
        if (0 < IX < resolutionX - 1) and (0 < IY < resolutionX - 1):
            b[IY * resolutionX + IX] = divf[IX, IY]
        else:
            b[IY * resolutionX + IX] = 0.0

@ti.kernel
def test_dot(r1: ti.template(), r2: ti.template()) -> ti.f32:
    res = 0.0
    for i in range(resolutionX * resolutionX):
        res += r1[i] * r2[i]
    return res


def pressure_cg(pf_pair, divf: ti.template()):
    residual = 10
    counter = 0
    construct_cg_b(divf, b)
    pressure_cg_init(pf_pair.cur, b)
    while residual > 0.01:
        residual = pressure_cg_iter(pf_pair.cur)
        counter += 1
        if counter > 30:
            break
    apply_p_bc(pf_pair.cur)


@ti.kernel
def divergence(field: ti.template(), divf: ti.template()):
    for i, j in field:
        divf[i, j] = 0.5 * (field[i + 1, j][0] - field[i - 1, j][0] + field[i, j + 1][1] - field[i, j - 1][1])


@ti.kernel
def correct_divergence(vf: ti.template(), vf_new: ti.template(), pf: ti.template()):
    for i, j in vf:
        vf_new[i, j] = vf[i, j] - ti.Vector([(pf[i + 1, j] - pf[i - 1, j]) / 2.0, (pf[i, j + 1] - pf[i, j - 1]) / 2.0])


gui = ti.GUI('Advection schemes', (512, 512))
frame_counter = 0

while True:
    while gui.get_event(ti.GUI.PRESS):
        if gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]: exit(0)
        if gui.event.key == ti.GUI.SPACE:
            pause = not pause
    if not pause:
        for itr in range(15):
            # Add inflow:
            addInflow(velocities_pair.cur, area, inflow_velocity)
            addInflow(dyes_pair.cur, area, temp_dye)
            # Advection:
            apply_vel_bc(velocities_pair.cur)
            advect(velocities_pair.cur, velocities_pair.cur, velocities_pair.nxt)
            advect(velocities_pair.cur, dyes_pair.cur, dyes_pair.nxt)
            velocities_pair.swap()
            dyes_pair.swap()
            apply_vel_bc(velocities_pair.cur)
            # External forces:
            # Projection:
            divergence(velocities_pair.cur, velocity_divs)
            pressure_cg(pressures_pair, velocity_divs)
            correct_divergence(velocities_pair.cur, velocities_pair.nxt, pressures_pair.cur)
            velocities_pair.swap()
            # Put color from dye to pixel:
        fill_color(pixels, dyes_pair.cur)

    frame_counter += 1
    filename = f'./video/frame_{frame_counter:05d}.png'
    gui.set_image(pixels.to_numpy())
    gui.show(filename)