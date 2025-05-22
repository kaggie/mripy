import numpy as np
import matplotlib.pyplot as plt
from mr_trajectory_utils import (
    radial_trajectory, spiral_trajectory, stack_of_stars, cones_trajectory,
    cones_3d_trajectory, magic_angle_3d_radial, rosette_trajectory, rosette_3d_trajectory,
    gradient_and_pns_check
)

def plot_trajectory(k, title="k-space trajectory"):
    if k.shape[-1] == 2:
        plt.figure()
        for arm in k:
            plt.plot(arm[:,0], arm[:,1])
        plt.axis('equal')
        plt.title(title)
        plt.xlabel("kx")
        plt.ylabel("ky")
        plt.show()
    elif k.shape[-1] == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for arm in k.reshape(-1, k.shape[-2], k.shape[-1]):
            ax.plot(arm[:,0], arm[:,1], arm[:,2])
        ax.set_title(title)
        plt.show()

def main():
    fov = 0.2     # 20 cm
    dwell = 4e-6  # 4 us
    grad_max = 40e-3  # 40 mT/m
    slew_max = 150  # T/m/s
    n_points = 256

    # UTE/ZTE: radial (2D) or radial shell (3D)
    radial_k = radial_trajectory(num_spokes=32, num_points=n_points, fov=fov, dwell_time=dwell)
    spiral_k = spiral_trajectory(num_arms=16, num_points=n_points, fov=fov, dwell_time=dwell, turns=8)
    stack_k = stack_of_stars(num_spokes=16, num_points=128, num_stacks=24, fov=fov, zmax=1/(2*fov))
    cones_k = cones_trajectory(num_cones=16, num_points=n_points, fov=fov, zmax=1/(2*fov))
    cones3d_k = cones_3d_trajectory(num_cones=16, num_points=n_points, fov=fov)
    magic_k = magic_angle_3d_radial(num_spokes=16, num_points=n_points, fov=fov, dwell_time=dwell)
    rosette_k = rosette_trajectory(num_petals=8, num_points=n_points, fov=fov, dwell_time=dwell)
    rosette3d_k = rosette_3d_trajectory(num_petals=8, num_points=n_points, fov=fov, dwell_time=dwell)

    # Plot examples
    plot_trajectory(radial_k[0:4], "Radial Trajectory (first 4 spokes)")
    plot_trajectory(spiral_k[0:4], "Spiral Trajectory (first 4 arms)")
    plot_trajectory(cones_k[0:4], "Cones Trajectory (first 4 cones)")
    plot_trajectory(cones3d_k[0:4], "3D Cones Trajectory (first 4 cones)")
    plot_trajectory(magic_k[0:4], "Magic Angle 3D Radial (first 4 spokes)")
    plot_trajectory(rosette_k[0:4], "Rosette Trajectory (first 4 petals)")
    plot_trajectory(rosette3d_k[0:4], "3D Rosette Trajectory (first 4 petals)")
    plot_trajectory(stack_k[0,0:4], "Stack of Stars (first 4 spokes in first stack)")

    # Example: check gradients for first spiral arm
    grad_ok, slew_ok, G, slew = gradient_and_pns_check(spiral_k[0], grad_max, slew_max, dwell)
    print(f"Gradient OK: {grad_ok}, Slew OK: {slew_ok}")

if __name__ == "__main__":
    main()