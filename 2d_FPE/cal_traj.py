import numpy as np
import matplotlib.pyplot as plt
from fplanck import fokker_planck, boundary, gaussian_pdf, combine, gaussian_potential
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
from scipy.interpolate import RegularGridInterpolator

def potential_from_data(grid, data):
    """
    grid:
      - 要么是长度为 ndim 的 tuple/list，每个元素是一维坐标轴 (严格单调)
      - 要么是形如 (ndim, n1, n2[, ...]) 的网格数组（比如 np.meshgrid 的输出堆叠）
    data:
      - 形状 (n1, n2[, ...]) 的标量场
    返回：potential(*args) 可在任意点插值
    """
    # 情况 1：已经是 (x_axis, y_axis, ...) 的一维轴
    if isinstance(grid, (list, tuple)) and all(np.ndim(g) == 1 for g in grid):
        axes = list(grid)
        arr = np.asarray(data)
        # 保证严格单调，必要时翻转 data
        for d, ax in enumerate(axes):
            if np.any(np.diff(ax) == 0):
                raise ValueError(f"Axis {d} has repeated points; axes must be strictly monotonic.")
            if ax[0] > ax[-1]:
                axes[d] = ax[::-1]
                arr = np.flip(arr, axis=d)
        f = RegularGridInterpolator(tuple(axes), arr, bounds_error=False, fill_value=None)

    else:
        # 情况 2：是堆叠网格 (ndim, n1, n2)
        G = np.asarray(grid)
        if G.ndim != (np.asarray(data).ndim + 1):
            raise ValueError(f"grid (ndim={G.ndim}) and data (ndim={np.asarray(data).ndim}) are incompatible.")

        # 这里以 2D 为例（你的数据就是 2D），推广到更高维也可以类推
        if G.shape[0] != 2:
            raise ValueError("Only 2D grids are supported in this helper. Got first dim size != 2.")

        X, Y = G[0], G[1]                 # 形状都是 (n1, n2)
        arr = np.asarray(data)
        n1, n2 = arr.shape

        # 从网格提取一维坐标轴（不假设 'ij' 或 'xy'，而是自适应）
        # 尝试把第一维的轴取为 X 的第一列；第二维的轴取为 Y 的第一行
        x_try = X[:, 0]
        y_try = Y[0, :]
        ok_x = (x_try.size == n1)
        ok_y = (y_try.size == n2)

        # 如果不匹配，试另一种取法（相当于网格用的是另一种 indexing）
        if not (ok_x and ok_y):
            x_try2 = X[0, :]
            y_try2 = Y[:, 0]
            if x_try2.size == n1 and y_try2.size == n2:
                x_try, y_try = x_try2, y_try2
            else:
                raise ValueError("Cannot infer 1D axes from the stacked grid; shapes don't match data.")

        # 确保严格单调，必要时翻转 data
        def ensure_monotonic(ax, axis):
            nonlocal arr
            if np.any(np.diff(ax) == 0):
                raise ValueError("Axis has repeated points; axes must be strictly monotonic.")
            if ax[0] > ax[-1]:
                ax = ax[::-1]
                arr = np.flip(arr, axis=axis)
            return ax

        x_axis = ensure_monotonic(x_try, axis=0)
        y_axis = ensure_monotonic(y_try, axis=1)

        f = RegularGridInterpolator((x_axis, y_axis), arr, bounds_error=False, fill_value=None)

    def potential(*args):
        # 支持传入 (x, y) 或形如 [[x1, y1], [x2, y2], ...] 的点
        pts = np.stack(args, axis=-1) if len(args) > 1 else args[0]
        return f(pts)

    return potential

index = 41
nm = 1e-9
viscosity = 8e-4
radius = 50 * nm
drag = 6 * np.pi * viscosity * radius
temperature = 300
extent = [600 * nm, 600 * nm]
resolution = 10 * nm
boundary_condition = boundary.reflecting
A = 1.8e-20  # Fixed amplitude


# Function to generate random Gaussian potential parameters
def random_gaussian_params():
    while True:
        centers = np.random.uniform(-100 * nm, 100 * nm, size=(3, 2))
        distances = np.sqrt(np.sum((centers[:, np.newaxis] - centers[np.newaxis, :]) ** 2, axis=-1))
        if np.all(distances[np.triu_indices(3, k=1)] > 90 * nm):
            break
    widths = np.random.uniform(20 * nm, 80 * nm, size=3)
    As = np.random.uniform(1e-20, 2e-20, size=3)
    viscosity_fact =  np.random.uniform(0,2,size = 1)
    diff_centers = np.random.uniform(-100 * nm, 100 * nm, size=(1, 2))
    return centers, widths, As, viscosity_fact,diff_centers 
centers, widths, As, viscosity_fact,diff_centers = random_gaussian_params()
U = combine(
    gaussian_potential(center=centers[0], width=widths[0], amplitude=As[0]),
    gaussian_potential(center=centers[1], width=widths[1], amplitude=As[1]),
    gaussian_potential(center=centers[2], width=widths[2], amplitude=As[2])
)
def drag_temp(x,y):
    x_scale = (x- diff_centers[0,0])/250/nm
    y_scale = (y-diff_centers[0,1])/250/nm

    return drag*(1+viscosity_fact*x_scale**2+viscosity_fact*y_scale**2)

sim = fokker_planck(
    temperature=temperature,
    drag=drag_temp,
    extent=extent,
    resolution=resolution,
    boundary=boundary_condition,
    potential=U
)
grid = sim.grid
fig_path = '/home/ubuntu/unlabelPDE_official/2d_diffusion/result_fig/unet'
data = np.load(f'/home/ubuntu/unlabelPDE_official/2d_diffusion/result_fig/unet/sample_00{index}_predictions.npy',allow_pickle = True).item()
U_np = data['drift_pred']
U_np[U_np>=0] = 0
drag = data['diffusion_pred']


U = potential_from_data(grid, U_np)
drag_temp = potential_from_data(grid, drag)

sim = fokker_planck(
    temperature=temperature,
    drag=drag_temp,
    extent=extent,
    resolution=resolution,
    boundary=boundary_condition,
    potential=U
)


pdf = gaussian_pdf(center=(0 * nm, 0 * nm), width=50 * nm)
p0 = pdf(*sim.grid)
Nsteps = 1000
time, Pt = sim.propagate_interval(pdf, 2e-4, Nsteps=Nsteps)
U_grid = U(*sim.grid)
np.save(os.path.join(fig_path,f'Pt_{index}.npy'), Pt)

# --- 可视化部分 ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# U_np
im0 = axes[0].imshow(U_np, origin='lower', extent=(grid[0].min(), grid[0].max(), grid[1].min(), grid[1].max()))
axes[0].set_title("Original U_np")
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

# U_grid
im1 = axes[1].imshow(U_grid, origin='lower', extent=(grid[0].min(), grid[0].max(), grid[1].min(), grid[1].max()))
axes[1].set_title("Interpolated U_grid")
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(os.path.join(fig_path,f"U_comparison_{index}.png"), dpi=300)  # 保存 U_np 与 U_grid 对比图
plt.close()

# --- P_t 最后一帧可视化 ---
Pt_last = Pt[-1]  # 最后一帧
plt.figure(figsize=(5, 4))
im2 = plt.imshow(Pt_last, origin='lower', extent=(grid[0].min(), grid[0].max(), grid[1].min(), grid[1].max()))
plt.title("P_t Last Frame")
plt.colorbar(im2, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(os.path.join(fig_path,f"Pt_{index}.png"), dpi=300)
plt.close()









fig_path = '/home/ubuntu/unlabelPDE_official/2d_diffusion/result_fig/nio'
data = np.load(f'/home/ubuntu/unlabelPDE_official/2d_diffusion/result_fig/nio/sample_00{index}_predictions.npy',allow_pickle = True).item()
U_np = data['drift_pred']
drag = data['diffusion_pred']

#result (2, 61, 61) (61, 61) (61, 61)

U = potential_from_data(grid, U_np)
drag_temp = potential_from_data(grid, drag)

sim = fokker_planck(
    temperature=temperature,
    drag=drag_temp,
    extent=extent,
    resolution=resolution,
    boundary=boundary_condition,
    potential=U
)


pdf = gaussian_pdf(center=(0 * nm, 0 * nm), width=50 * nm)
p0 = pdf(*sim.grid)
Nsteps = 1000
time, Pt = sim.propagate_interval(pdf, 2e-4, Nsteps=Nsteps)
U_grid = U(*sim.grid)
np.save(os.path.join(fig_path,f'Pt_{index}.npy'), Pt)

# --- 可视化部分 ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# U_np
im0 = axes[0].imshow(U_np, origin='lower', extent=(grid[0].min(), grid[0].max(), grid[1].min(), grid[1].max()))
axes[0].set_title("Original U_np")
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

# U_grid
im1 = axes[1].imshow(U_grid, origin='lower', extent=(grid[0].min(), grid[0].max(), grid[1].min(), grid[1].max()))
axes[1].set_title("Interpolated U_grid")
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(os.path.join(fig_path,f"U_comparison_{index}.png"), dpi=300)  # 保存 U_np 与 U_grid 对比图
plt.close()

# --- P_t 最后一帧可视化 ---
Pt_last = Pt[-1]  # 最后一帧
plt.figure(figsize=(5, 4))
im2 = plt.imshow(Pt_last, origin='lower', extent=(grid[0].min(), grid[0].max(), grid[1].min(), grid[1].max()))
plt.title("P_t Last Frame")
plt.colorbar(im2, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(os.path.join(fig_path,f"Pt_{index}.png"), dpi=300)
plt.close()


fig_path = '/home/ubuntu/unlabelPDE_official/2d_diffusion/result_fig/fno'
data = np.load(f'/home/ubuntu/unlabelPDE_official/2d_diffusion/result_fig/fno/sample_00{index}_predictions.npy',allow_pickle = True).item()
U_np = data['drift_pred']
drag = data['diffusion_pred']
U_np[U_np>=0] = 0
#result (2, 61, 61) (61, 61) (61, 61)

U = potential_from_data(grid, U_np)
drag_temp = potential_from_data(grid, drag)

sim = fokker_planck(
    temperature=temperature,
    drag=drag_temp,
    extent=extent,
    resolution=resolution,
    boundary=boundary_condition,
    potential=U
)


pdf = gaussian_pdf(center=(0 * nm, 0 * nm), width=50 * nm)
p0 = pdf(*sim.grid)
Nsteps = 1000
time, Pt = sim.propagate_interval(pdf, 2e-4, Nsteps=Nsteps)
U_grid = U(*sim.grid)
np.save(os.path.join(fig_path,f'Pt_{index}.npy'), Pt)

# --- 可视化部分 ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# U_np
im0 = axes[0].imshow(U_np, origin='lower', extent=(grid[0].min(), grid[0].max(), grid[1].min(), grid[1].max()))
axes[0].set_title("Original U_np")
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

# U_grid
im1 = axes[1].imshow(U_grid, origin='lower', extent=(grid[0].min(), grid[0].max(), grid[1].min(), grid[1].max()))
axes[1].set_title("Interpolated U_grid")
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(os.path.join(fig_path,f"U_comparison_{index}.png"), dpi=300)  # 保存 U_np 与 U_grid 对比图
plt.close()

# --- P_t 最后一帧可视化 ---
Pt_last = Pt[-1]  # 最后一帧
plt.figure(figsize=(5, 4))
im2 = plt.imshow(Pt_last, origin='lower', extent=(grid[0].min(), grid[0].max(), grid[1].min(), grid[1].max()))
plt.title("P_t Last Frame")
plt.colorbar(im2, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(os.path.join(fig_path,f"Pt_{index}.png"), dpi=300)
plt.close()






fig_path = '/home/ubuntu/unlabelPDE_official/2d_diffusion/result_fig/'
data = np.load(f'/home/ubuntu/unlabelPDE_official/2d_diffusion/result_fig/unet/sample_00{index}_predictions.npy',allow_pickle = True).item()
U_np = data['drift_true']
drag = data['diffusion_true']
U_np[U_np>=0] = 0
#result (2, 61, 61) (61, 61) (61, 61)

U = potential_from_data(grid, U_np)
drag_temp = potential_from_data(grid, drag)

sim = fokker_planck(
    temperature=temperature,
    drag=drag_temp,
    extent=extent,
    resolution=resolution,
    boundary=boundary_condition,
    potential=U
)


pdf = gaussian_pdf(center=(0 * nm, 0 * nm), width=50 * nm)
p0 = pdf(*sim.grid)
Nsteps = 1000
time, Pt = sim.propagate_interval(pdf, 2e-4, Nsteps=Nsteps)
U_grid = U(*sim.grid)
np.save(os.path.join(fig_path,f'Pt_{index}.npy'), Pt)

# --- 可视化部分 ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# U_np
im0 = axes[0].imshow(U_np, origin='lower', extent=(grid[0].min(), grid[0].max(), grid[1].min(), grid[1].max()))
axes[0].set_title("Original U_np")
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

# U_grid
im1 = axes[1].imshow(U_grid, origin='lower', extent=(grid[0].min(), grid[0].max(), grid[1].min(), grid[1].max()))
axes[1].set_title("Interpolated U_grid")
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(os.path.join(fig_path,f"U_comparison_{index}.png"), dpi=300)  # 保存 U_np 与 U_grid 对比图
plt.close()

# --- P_t 最后一帧可视化 ---
Pt_last = Pt[-1]  # 最后一帧
plt.figure(figsize=(5, 4))
im2 = plt.imshow(Pt_last, origin='lower', extent=(grid[0].min(), grid[0].max(), grid[1].min(), grid[1].max()))
plt.title("P_t Last Frame")
plt.colorbar(im2, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(os.path.join(fig_path,f"Pt_{index}.png"), dpi=300)
plt.close()
