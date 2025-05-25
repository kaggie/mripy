import numpy as np

def radial_trajectory(num_spokes, num_points, fov, dwell_time):
    angles = np.linspace(0, np.pi, num_spokes, endpoint=False)
    k = np.zeros((num_spokes, num_points, 2))
    max_k = 1/(2*fov)
    for i, theta in enumerate(angles):
        kx = np.linspace(-max_k, max_k, num_points) * np.cos(theta)
        ky = np.linspace(-max_k, max_k, num_points) * np.sin(theta)
        k[i, :, 0] = kx
        k[i, :, 1] = ky
    return k

def spiral_trajectory(num_arms, num_points, fov, dwell_time, turns=1):
    k = np.zeros((num_arms, num_points, 2))
    max_k = 1/(2*fov)
    for i in range(num_arms):
        phi = 2 * np.pi * i / num_arms
        t = np.linspace(0, 1, num_points)
        r = t * max_k
        theta = turns * 2 * np.pi * t + phi
        kx = r * np.cos(theta)
        ky = r * np.sin(theta)
        k[i, :, 0] = kx
        k[i, :, 1] = ky
    return k

def stack_of_spirals(num_arms, num_points, num_stacks, fov, zmax, turns=1):
    """
    3D Stack of Spirals:
    For each z slice, draws a spiral in the (kx, ky) plane, with kz constant for the stack.
    Returns [num_stacks, num_arms, num_points, 3] array.
    """
    k = np.zeros((num_stacks, num_arms, num_points, 3))
    z_locations = np.linspace(-zmax, zmax, num_stacks)
    max_k = 1/(2*fov)
    for iz, z in enumerate(z_locations):
        for i in range(num_arms):
            phi = 2 * np.pi * i / num_arms
            t = np.linspace(0, 1, num_points)
            r = t * max_k
            theta = turns * 2 * np.pi * t + phi
            kx = r * np.cos(theta)
            ky = r * np.sin(theta)
            kz = np.ones(num_points) * z
            k[iz, i, :, 0] = kx
            k[iz, i, :, 1] = ky
            k[iz, i, :, 2] = kz
    return k

def stack_of_stars(num_spokes, num_points, num_stacks, fov, zmax):
    k = np.zeros((num_stacks, num_spokes, num_points, 3))
    z_locations = np.linspace(-zmax, zmax, num_stacks)
    max_k = 1/(2*fov)
    for iz, z in enumerate(z_locations):
        for i, theta in enumerate(np.linspace(0, np.pi, num_spokes, endpoint=False)):
            kx = np.linspace(-max_k, max_k, num_points) * np.cos(theta)
            ky = np.linspace(-max_k, max_k, num_points) * np.sin(theta)
            kz = np.ones(num_points) * z
            k[iz, i, :, 0] = kx
            k[iz, i, :, 1] = ky
            k[iz, i, :, 2] = kz
    return k

def cones_trajectory(num_cones, num_points, fov, zmax):
    k = np.zeros((num_cones, num_points, 3))
    max_k = 1/(2*fov)
    for i in range(num_cones):
        phi = 2 * np.pi * i / num_cones
        t = np.linspace(0, 1, num_points)
        theta = np.arccos(1 - 2*t)
        kx = max_k * t * np.sin(theta) * np.cos(phi)
        ky = max_k * t * np.sin(theta) * np.sin(phi)
        kz = max_k * t * np.cos(theta)
        k[i, :, 0] = kx
        k[i, :, 1] = ky
        k[i, :, 2] = kz
    return k

def cones_3d_trajectory(num_cones, num_points, fov):
    """3D cones trajectory: Each cone defined by its axis on a sphere and samples along the cone."""
    k = []
    max_k = 1/(2*fov)
    # Distribute cones uniformly on sphere surface (using spherical Fibonacci lattice)
    indices = np.arange(0, num_cones, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/num_cones)
    theta = np.pi * (1 + 5**0.5) * indices
    for i in range(num_cones):
        axis = [
            np.sin(phi[i]) * np.cos(theta[i]),
            np.sin(phi[i]) * np.sin(theta[i]),
            np.cos(phi[i])
        ]
        t = np.linspace(0, 1, num_points)
        angle = 2 * np.pi * t
        for j in range(num_points):
            offset = max_k * t[j]
            direction = np.array(axis)
            # Create orthogonal vectors
            if np.allclose(direction, [0,0,1]):
                ortho = np.array([1,0,0])
            else:
                ortho = np.cross(direction, [0,0,1])
                ortho = ortho / np.linalg.norm(ortho)
            perp = np.cross(direction, ortho)
            point = (offset * direction +
                     np.cos(angle[j]) * 0.05*max_k * ortho +
                     np.sin(angle[j]) * 0.05*max_k * perp)
            k.append(point)
    k = np.array(k).reshape(num_cones, num_points, 3)
    return k

def magic_angle_3d_radial(num_spokes, num_points, fov, dwell_time):
    """3D radial with orientations sampling the magic angle for susceptibility imaging."""
    k = np.zeros((num_spokes, num_points, 3))
    max_k = 1/(2*fov)
    # Magic angle = arccos(1/sqrt(3)) ≈ 54.74°
    magic_theta = np.arccos(1/np.sqrt(3))
    phis = np.linspace(0, 2*np.pi, num_spokes, endpoint=False)
    for i, phi in enumerate(phis):
        kx = np.linspace(-max_k, max_k, num_points) * np.sin(magic_theta) * np.cos(phi)
        ky = np.linspace(-max_k, max_k, num_points) * np.sin(magic_theta) * np.sin(phi)
        kz = np.linspace(-max_k, max_k, num_points) * np.cos(magic_theta)
        k[i, :, 0] = kx
        k[i, :, 1] = ky
        k[i, :, 2] = kz
    return k

def rosette_trajectory(num_petals, num_points, fov, dwell_time, a=0.5, f1=5, f2=7):
    """
    2D rosette: kx = a*sin(f1*t) + (1-a)*sin(f2*t)
                ky = a*cos(f1*t) + (1-a)*cos(f2*t)
    """
    max_k = 1/(2*fov)
    t = np.linspace(0, 2*np.pi, num_points)
    k = np.zeros((num_petals, num_points, 2))
    for i in range(num_petals):
        phase = 2*np.pi*i/num_petals
        kx = max_k * (a * np.sin(f1*t+phase) + (1-a) * np.sin(f2*t+phase))
        ky = max_k * (a * np.cos(f1*t+phase) + (1-a) * np.cos(f2*t+phase))
        k[i, :, 0] = kx
        k[i, :, 1] = ky
    return k

def rosette_3d_trajectory(num_petals, num_points, fov, dwell_time, a=0.5, f1=5, f2=7):
    """
    3D Rosette: Spherical extension of 2D rosette
    """
    max_k = 1/(2*fov)
    k = np.zeros((num_petals, num_points, 3))
    t = np.linspace(0, 2*np.pi, num_points)
    for i in range(num_petals):
        phase = 2*np.pi*i/num_petals
        # Trajectory on a sphere, varying both polar and azimuthal angles
        theta = np.pi/2 + a * np.sin(f1*t+phase)
        phi = phase + (1-a) * np.cos(f2*t+phase)
        r = max_k * np.ones_like(t)
        kx = r * np.sin(theta) * np.cos(phi)
        ky = r * np.sin(theta) * np.sin(phi)
        kz = r * np.cos(theta)
        k[i, :, 0] = kx
        k[i, :, 1] = ky
        k[i, :, 2] = kz
    return k

def phyllotaxis_3d_trajectory(num_points, fov):
    """
    3D phyllotaxis (spherical spiral): Uniformly distributes points on a sphere using the golden angle.
    Useful for 3D radial or cones trajectories.
    Returns a [num_points, 3] array.
    """
    max_k = 1/(2*fov)
    golden_angle = np.pi * (3 - np.sqrt(5))  # ~2.399 rad
    indices = np.arange(num_points)
    theta = golden_angle * indices            # azimuthal angle
    z = np.linspace(1 - 1/num_points, -1 + 1/num_points, num_points)  # height [-1, 1]
    radius = np.sqrt(1 - z**2)
    kx = max_k * radius * np.cos(theta)
    ky = max_k * radius * np.sin(theta)
    kz = max_k * z
    k = np.stack([kx, ky, kz], axis=1)
    return k

def gradient_and_pns_check(k_traj, grad_max, slew_max, dt):
    """
    k_traj: [N, D] array, N points, D dimensions
    grad_max: T/m
    slew_max: T/m/s
    dt: time between points (s)
    """
    gamma = 42.58e6  # Hz/T
    G = np.diff(k_traj, axis=0) / dt / gamma
    slew = np.diff(G, axis=0) / dt

    grad_ok = np.all(np.abs(G) <= grad_max)
    slew_ok = np.all(np.abs(slew) <= slew_max)
    return grad_ok, slew_ok, G, slew
