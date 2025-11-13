import numpy as np
import matplotlib.pyplot as plt
from fplanck import fokker_planck, boundary, gaussian_pdf, combine, gaussian_potential
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
nm = 1e-9
viscosity = 8e-4
radius = 50 * nm
drag = 6 * np.pi * viscosity * radius
temperature = 300
extent = [600 * nm, 600 * nm]
resolution = 10 * nm
boundary_condition = boundary.reflecting
A = 1.8e-20  # Fixed amplitude
M = 400  # Number of simulations
N_THREADS = 8  # Number of threads

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

# Function for running a single simulation
def run_simulation(sim_index):
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
    pdf = gaussian_pdf(center=(0 * nm, 0 * nm), width=50 * nm)
    p0 = pdf(*sim.grid)
    Nsteps = 1000
    time, Pt = sim.propagate_interval(pdf, 2e-4, Nsteps=Nsteps)
    U_grid = U(*sim.grid)
    selected_indices = np.sort(np.random.choice(range(len(time)), size=100, replace=False))
    t_selected = time[selected_indices]
    y_selected = Pt[selected_indices, :, :]
    print(f"Simulation {sim_index + 1} completed.")
    drag_temp_val = drag_temp(*sim.grid)
    return t_selected, sim.grid, y_selected, U_grid, drag_temp_val

# Multi-threaded execution
time_array = []
grid_array = []
trajectories_array = []
potential_array = []
drag_array = []
with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
    futures = [executor.submit(run_simulation, i) for i in range(M)]
    for future in tqdm(as_completed(futures)):
        t_selected, grid, y_selected, U_grid, drag_temp_val = future.result()
        time_array.append(t_selected)
        grid_array.append(grid)
        trajectories_array.append(y_selected)
        potential_array.append(U_grid)
        drag_array.append(drag_temp_val)


# Convert lists to np.array
time_array = np.array(time_array)  # Shape: (M, 100)
grid_array = np.array(grid_array)  # Shape: (M, 2, Nx, Ny)
trajectories_array = np.array(trajectories_array)  # Shape: (M, 100, Nx, Ny)
potential_array = np.array(potential_array)  # Shape: (M, Nx, Ny)
drag_array = np.array(drag_array)

# Save the dataset
np.savez(
    "test_dataset_2D_drift_diffusion.npz",
    time=time_array,
    grid=grid_array,
    trajectories=trajectories_array,
    potential=potential_array,
    drag = drag_array
)
print("Dataset saved to 'dataset_2D_drift_big'.")
# data = np.load("test_dataset_2D_drift_diffusion.npz")

# # # Extract the arrays
# time_array = data["time"]  # Shape: (M, 100)
# grid_array = data["grid"]  # Shape: (M, 2, Nx, Ny)
# trajectories_array = data["trajectories"]  # Shape: (M, 100, Nx, Ny)
# potential_array = data["potential"]  # Shape: (M, Nx, Ny)
# drag_array = data['drag']
# print(time_array.shape)
# print(grid_array.shape)
# print(trajectories_array.shape)
# print(potential_array.shape)
# print(drag_array.shape)
