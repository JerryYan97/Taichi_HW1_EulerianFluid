# 1. 有ti.grouped的具体解释吗？ 在文档里面好像没有找到. 如果I是一个向量，那这个向量里面的元素是什么样子的呢？是如何随着循环变化的呢
# 2. tichi有没有reshape tensor的功能呢？比如把一个shape = n*m 的var tensor变成一个 shape = (n, m) 的tensor呢？
# Reference:
# advection.py
# stable_fluid.py
# http://developer.download.nvidia.com/books/HTML/gpugems/gpugems_ch38.html
# https://github.com/tunabrain/incremental-fluids
# https://forum.taichi.graphics/

# Traits:
# Simplest: Not staggered.

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


@ti.kernel
def pressure_jacobi_iter(pf: ti.template(), pf_new: ti.template(), divf: ti.template()) -> ti.f32:
    norm_new = 0
    norm_diff = 0
    for i, j in pf:
        pf_new[i, j] = 0.25 * (p_with_boundary(pf, i + 1, j) + p_with_boundary(pf, i - 1, j) +
                               p_with_boundary(pf, i, j + 1) + p_with_boundary(pf, i, j - 1) - divf[i, j])
        pf_diff = ti.abs(pf_new[i, j] - p_with_boundary(pf, i, j))
        norm_new += (pf_new[i, j] * pf_new[i, j])
        norm_diff += (pf_diff * pf_diff)
    residual = ti.sqrt(norm_diff / norm_new)
    if norm_new == 0:
        residual = 0.0
    return residual


def pressure_jacobi(pf_pair, divf: ti.template()):
    residual = 10
    counter = 0
    while residual > 0.001:
        residual = pressure_jacobi_iter(pf_pair.cur, pf_pair.nxt, divf)
        pf_pair.swap()
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
            pressure_jacobi(pressures_pair, velocity_divs)
            correct_divergence(velocities_pair.cur, velocities_pair.nxt, pressures_pair.cur)
            # correct_divergence(velocities_pair.cur, velocities_pair.nxt, pressures_pair.cur)
            velocities_pair.swap()
            # Put color from dye to pixel:
        fill_color(pixels, dyes_pair.cur)

    gui.set_image(pixels.to_numpy())
    gui.show()

