import numpy as np
import matplotlib.pyplot as plt
from fplanck import fokker_planck, boundary, gaussian_pdf, combine, gaussian_potential, potential_from_data
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
from tqdm import tqdm
import os
nm = 1e-9
viscosity = 8e-4
radius = 50 * nm
drag = 6 * np.pi * viscosity * radius
temperature = 300
extent = 800*nm
resolution = 10 * nm
boundary_condition = boundary.reflecting

def random_gaussian_params():
    while True:
        # Random centers in the range [-100*nm, 100*nm]
        centers = np.random.uniform(-150 * nm, 150 * nm, size=(3))
        
        # Ensure distances between all pairs are at least 30*nm
        if all(np.abs(centers[i] - centers[j]) > 80 * nm for i in range(3) for j in range(i+1, 3)):
            break

    # Random widths in the range [20*nm, 80*nm]
    widths = np.random.uniform(20 * nm, 80 * nm, size=3)
    As = np.random.uniform(1e-20,2e-20,size = 3)
    viscosity_fact =  np.random.uniform(1,2,size = 1)
    return centers, widths, As, viscosity_fact

centers, widths, As, viscosity_fact = random_gaussian_params()

# Create the combined potential
U = combine(
    gaussian_potential(center=(centers[0]), width=widths[0], amplitude=As[0]*viscosity_fact[0]),
    gaussian_potential(center=(centers[1]), width=widths[1], amplitude=As[1]*viscosity_fact[0]),
    gaussian_potential(center=(centers[2]), width=widths[2], amplitude=As[2]*viscosity_fact[0])
)
drag_temp = drag*viscosity_fact[0]


sim = fokker_planck(
    temperature=temperature,
    drag=drag_temp,
    extent=extent,
    resolution=resolution,
    boundary=boundary_condition,
    potential=U
)

data_path = '/home/ubuntu/unlabel_PDE_official/1d/result_fig/unet/pred_sample_16.npy'
fig_path = '/home/ubuntu/unlabel_PDE_official/1d/result_fig/unet'
# Function to generate random Gaussian potential parameters
data = np.load(data_path)
U_np = data[:,0]
drag_temp = data[:,1].mean()
grid = sim.grid
U = potential_from_data(grid, U_np)


sim = fokker_planck(
    temperature=temperature,
    drag=drag_temp,
    extent=extent,
    resolution=resolution,
    boundary=boundary_condition,
    potential=U
)
pdf = gaussian_pdf(center=(0 * nm), width=50 * nm)
p0 = pdf(*sim.grid)
Nsteps = 400
time, Pt = sim.propagate_interval(pdf, 2e-3, Nsteps=Nsteps)
np.save(os.path.join(fig_path,'Pt.npy'), Pt)




data_path = '/home/ubuntu/unlabel_PDE_official/1d/result_fig/fno/pred_sample_16.npy'
fig_path = '/home/ubuntu/unlabel_PDE_official/1d/result_fig/fno'
# Function to generate random Gaussian potential parameters
data = np.load(data_path)
U_np = data[:,0]
drag_temp = data[:,1].mean()
grid = sim.grid
U = potential_from_data(grid, U_np)


sim = fokker_planck(
    temperature=temperature,
    drag=drag_temp,
    extent=extent,
    resolution=resolution,
    boundary=boundary_condition,
    potential=U
)
pdf = gaussian_pdf(center=(0 * nm), width=50 * nm)
p0 = pdf(*sim.grid)
Nsteps = 400
time, Pt = sim.propagate_interval(pdf, 2e-3, Nsteps=Nsteps)
np.save(os.path.join(fig_path,'Pt.npy'), Pt)


data_path = '/home/ubuntu/unlabel_PDE_official/1d/result_fig/nio/pred_sample_16.npy'
fig_path = '/home/ubuntu/unlabel_PDE_official/1d/result_fig/nio'
# Function to generate random Gaussian potential parameters
data = np.load(data_path)
U_np = data[:,0]
drag_temp = data[:,1].mean()
grid = sim.grid
U = potential_from_data(grid, U_np)


sim = fokker_planck(
    temperature=temperature,
    drag=drag_temp,
    extent=extent,
    resolution=resolution,
    boundary=boundary_condition,
    potential=U
)
pdf = gaussian_pdf(center=(0 * nm), width=50 * nm)
p0 = pdf(*sim.grid)
Nsteps = 400
time, Pt = sim.propagate_interval(pdf, 2e-3, Nsteps=Nsteps)
np.save(os.path.join(fig_path,'Pt.npy'), Pt)