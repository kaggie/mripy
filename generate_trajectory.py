import numpy as np
from typing import Callable, Optional, Dict, Any

import numpy as np

import numpy as np

class Trajectory:
    """
    Container for a k-space trajectory and associated data.
    Includes additional trajectory metrics like max PNS, max slew, FOV, and resolution.
    """
    def __init__(self, name, kspace_points_rad_per_m, gradient_waveforms_Tm=None, dt_seconds=None, metadata=None):
        """
        Args:
            name (str): Trajectory name/description.
            kspace_points_rad_per_m (np.ndarray): [D, N] or [N, D] k-space coordinates in rad/m.
            gradient_waveforms_Tm (np.ndarray, optional): [D, N] or [N, D] gradient waveforms in T/m.
            dt_seconds (float, optional): Dwell/sample time in seconds.
            metadata (dict, optional): Additional information.
        """
        self.name = name
        self.kspace_points_rad_per_m = np.array(kspace_points_rad_per_m)
        self.gradient_waveforms_Tm = np.array(gradient_waveforms_Tm) if gradient_waveforms_Tm is not None else None
        self.dt_seconds = dt_seconds
        self.metadata = metadata or {}

        # Automatically populate additional metrics
        self._compute_metrics()

    def _calculate_slew_rate(self):
        """Calculates slew rate and stores it in metadata."""
        if self.gradient_waveforms_Tm is not None and self.dt_seconds is not None:
            slew = np.diff(self.gradient_waveforms_Tm, axis=-1) / self.dt_seconds
            max_slew_rate_Tm_per_s = np.max(np.linalg.norm(slew, axis=0))
            self.metadata['max_slew_rate_Tm_per_s'] = max_slew_rate_Tm_per_s
        else:
            self.metadata['max_slew_rate_Tm_per_s'] = None

    def _calculate_pns(self):
        """Calculates PNS metrics and stores them in metadata."""
        if self.gradient_waveforms_Tm is not None and self.dt_seconds is not None:
            # PNS_max_abs_gradient_sum_xyz = max(abs(Gx) + abs(Gy) + abs(Gz))
            abs_grad_sum = np.sum(np.abs(self.gradient_waveforms_Tm), axis=0)
            self.metadata['pns_max_abs_gradient_sum_xyz'] = np.max(abs_grad_sum)

            # PNS_max_abs_slew_sum_xyz = max(abs(SlewX) + abs(SlewY) + abs(SlewZ))
            slew = np.diff(self.gradient_waveforms_Tm, axis=-1) / self.dt_seconds
            abs_slew_sum = np.sum(np.abs(slew), axis=0)
            self.metadata['pns_max_abs_slew_sum_xyz'] = np.max(abs_slew_sum)
        else:
            self.metadata['pns_max_abs_gradient_sum_xyz'] = None
            self.metadata['pns_max_abs_slew_sum_xyz'] = None

    def _calculate_fov(self):
        """Calculates FOV and stores it in metadata."""
        if self.kspace_points_rad_per_m is not None:
            # Ensure points are [D, N]
            points = self.kspace_points_rad_per_m
            if points.shape[0] > points.shape[1] and points.shape[0] > 3 : # Heuristic for N, D vs D, N
                 points = points.T

            k_extent_rad_per_m = np.max(np.abs(points), axis=-1)
            # Add a small epsilon to prevent division by zero if k_extent is 0 for a dimension
            fov_m = 1 / (2 * k_extent_rad_per_m + 1e-9)
            self.metadata['fov_estimate_m'] = fov_m.tolist()
            self.metadata['fov_estimate_mm'] = (fov_m * 1e3).tolist()
        else:
            self.metadata['fov_estimate_m'] = None
            self.metadata['fov_estimate_mm'] = None

    def _calculate_resolution(self):
        """Calculates resolution and stores it in metadata."""
        if self.kspace_points_rad_per_m is not None:
            # Ensure points are [D, N]
            points = self.kspace_points_rad_per_m
            if points.shape[0] > points.shape[1] and points.shape[0] > 3 : # Heuristic for N, D vs D, N
                 points = points.T

            # For anisotropic resolution, we might consider resolution along each k-space axis.
            # A common definition is related to the max k-space extent along each axis.
            # However, for a general trajectory, "resolution" is often simplified to 1 / (2 * max_k_radius)
            # Here we'll provide both: an estimate per dimension and an overall estimate.

            # Overall resolution based on max k-space radius
            max_k_radius_rad_per_m = np.max(np.linalg.norm(points, axis=0))
            resolution_m_overall = 1 / (2 * max_k_radius_rad_per_m + 1e-9) # meters
            self.metadata['resolution_overall_estimate_m'] = resolution_m_overall
            self.metadata['resolution_overall_estimate_mm'] = resolution_m_overall * 1e3

            # Per-dimension resolution estimate (can be debated, this is one way)
            # This is similar to FOV calculation but for resolution it's 1 / (2 * max_k_coord_on_axis)
            # This might not be the most standard definition for arbitrary trajectories but can be useful.
            # For now, let's stick to the overall resolution as it's more common for non-Cartesian.
            # If anisotropic definition is strictly needed, it would be more like:
            # delta_k = np.max(points, axis=1) - np.min(points, axis=1) # k-space coverage along each axis
            # resolution_anisotropic_m = 1 / (delta_k + 1e-9)
            # self.metadata['resolution_anisotropic_estimate_m'] = resolution_anisotropic_m.tolist()
            # self.metadata['resolution_anisotropic_estimate_mm'] = (resolution_anisotropic_m * 1e3).tolist()
            # For now, only overall resolution is stored.
        else:
            self.metadata['resolution_overall_estimate_m'] = None
            self.metadata['resolution_overall_estimate_mm'] = None


    def _compute_metrics(self):
        """Computes all trajectory metrics."""
        self._calculate_slew_rate()
        self._calculate_pns()
        self._calculate_fov()
        self._calculate_resolution()

    def get_duration_seconds(self) -> Optional[float]:
        """Returns total trajectory duration in seconds."""
        if self.dt_seconds is not None and self.kspace_points_rad_per_m is not None:
            return self.get_num_points() * self.dt_seconds
        return None

    def get_max_grad_Tm(self) -> Optional[float]:
        """Returns the maximum absolute gradient amplitude (T/m)."""
        if self.gradient_waveforms_Tm is not None:
            return np.max(np.linalg.norm(self.gradient_waveforms_Tm, axis=0))
        return None

    def get_max_slew_Tm_per_s(self) -> Optional[float]:
        """Returns the maximum absolute slew rate (T/m/s)."""
        if 'max_slew_rate_Tm_per_s' in self.metadata:
            return self.metadata['max_slew_rate_Tm_per_s']
        elif self.gradient_waveforms_Tm is not None and self.dt_seconds is not None and self.gradient_waveforms_Tm.shape[-1] > 1:
            slew = np.diff(self.gradient_waveforms_Tm, axis=-1) / self.dt_seconds
            return np.max(np.linalg.norm(slew, axis=0))
        return None

    def get_num_points(self) -> int:
        """Returns the number of k-space points."""
        # Assuming kspace_points_rad_per_m is [D, N] or [N,D]
        # If [D,N], N is shape[1]. If [N,D], N is shape[0], assuming D < N.
        # A common convention is D <= 3.
        if self.kspace_points_rad_per_m.shape[0] <= 3 or self.kspace_points_rad_per_m.shape[0] < self.kspace_points_rad_per_m.shape[1]:
            return self.kspace_points_rad_per_m.shape[1]
        return self.kspace_points_rad_per_m.shape[0]


    def get_num_dimensions(self) -> int:
        """Returns the number of spatial dimensions."""
        # Assuming kspace_points_rad_per_m is [D, N] or [N,D]
        # If [D,N], D is shape[0]. If [N,D], D is shape[1].
        if self.kspace_points_rad_per_m.shape[0] <= 3 or self.kspace_points_rad_per_m.shape[0] < self.kspace_points_rad_per_m.shape[1]:
            return self.kspace_points_rad_per_m.shape[0]
        return self.kspace_points_rad_per_m.shape[1]


    def export(self, filename, filetype=None):
        """
        Export trajectory to file (CSV, .npy, .npz, .txt).
        Args:
            filename (str): Output file name.
            filetype (str, optional): 'csv', 'npy', 'npz', or 'txt'. Inferred from extension if not given.
        """
        if filetype is None:
            if filename.endswith('.csv'):
                filetype = 'csv'
            elif filename.endswith('.npy'):
                filetype = 'npy'
            elif filename.endswith('.npz'):
                filetype = 'npz'
            else: # Default to text if extension is unknown or not provided and not one of the above
                filetype = 'txt'
        
        # Ensure points are [N, D] for export
        points_to_export = self.kspace_points_rad_per_m
        # Heuristic: if first dim is smaller than second AND first dim is <=3, assume it's [D, N]
        if points_to_export.ndim == 2 and points_to_export.shape[0] < points_to_export.shape[1] and points_to_export.shape[0] <=3 :
            points_to_export = points_to_export.T

        if filetype == 'csv':
            np.savetxt(filename, points_to_export, delimiter=',')
        elif filetype == 'npy':
            np.save(filename, points_to_export)
        elif filetype == 'npz':
            np.savez(filename, kspace_points_rad_per_m=points_to_export, 
                     gradient_waveforms_Tm=self.gradient_waveforms_Tm, 
                     dt_seconds=self.dt_seconds, metadata=self.metadata)
        elif filetype == 'txt':
            np.savetxt(filename, points_to_export)
        else:
            raise ValueError(f"Unsupported filetype: {filetype}")

    @classmethod
    def import_from(cls, filename):
        """
        Import a trajectory from a file.
        """
        if filename.endswith('.csv') or filename.endswith('.txt'):
            points = np.loadtxt(filename, delimiter=',' if filename.endswith('.csv') else None)
            # Assuming imported points are N, D.
            # A more robust check could be added here if D > N and D > 3 is a possible case for CSV/TXT.
            return cls(name=filename, kspace_points_rad_per_m=points)
        elif filename.endswith('.npy'):
            points = np.load(filename)
            # .npy could be D,N or N,D. The constructor will handle it.
            return cls(name=filename, kspace_points_rad_per_m=points)
        elif filename.endswith('.npz'):
            data = np.load(filename, allow_pickle=True)
            # Convert legacy keys if present, prioritizing new names
            points_key = 'kspace_points_rad_per_m' if 'kspace_points_rad_per_m' in data else 'points' if 'points' in data else 'kspace'
            gradients_key = 'gradient_waveforms_Tm' if 'gradient_waveforms_Tm' in data else 'gradients'
            dt_key = 'dt_seconds' if 'dt_seconds' in data else 'dt'
            
            points = data[points_key]
            # NPZ may store gradients as None, handle this
            gradients_data = data.get(gradients_key)
            gradients = np.array(gradients_data) if gradients_data is not None else None
            
            dt_data = data.get(dt_key)
            dt = dt_data.item() if dt_data is not None and hasattr(dt_data, 'item') else dt_data

            # Ensure metadata is a dictionary, even if stored as an array by older versions
            metadata_raw = data.get('metadata')
            if metadata_raw is not None:
                metadata = metadata_raw.item() if hasattr(metadata_raw, 'item') else dict(metadata_raw) if not isinstance(metadata_raw, dict) else metadata_raw
            else:
                metadata = {}
                
            return cls(name=filename, kspace_points_rad_per_m=points, 
                       gradient_waveforms_Tm=gradients, dt_seconds=dt, metadata=metadata)
        else:
            raise ValueError(f"Unsupported filetype or extension for: {filename}")

    def summary(self):
        """
        Print a detailed summary of the trajectory, including its properties and calculated metrics.
        """
        num_dims = self.get_num_dimensions()
        num_points = self.get_num_points()
        duration_ms = self.get_duration_seconds() * 1e3 if self.get_duration_seconds() is not None else "N/A"
        
        print(f"\n--- Trajectory Summary: '{self.name}' ---")
        print(f"  Dimensions: {num_dims}D")
        print(f"  Number of Points: {num_points}")
        print(f"  Duration: {duration_ms:.2f} ms" if isinstance(duration_ms, float) else f"Duration: {duration_ms}")
        print(f"  Dwell Time (dt): {self.dt_seconds * 1e6:.2f} µs" if self.dt_seconds is not None else "Dwell Time (dt): N/A")

        if self.gradient_waveforms_Tm is not None:
            print("\n  Gradients:")
            max_grad_mT_m = self.get_max_grad_Tm() * 1e3 if self.get_max_grad_Tm() is not None else "N/A"
            max_slew_Tm_s = self.get_max_slew_Tm_per_s() if self.get_max_slew_Tm_per_s() is not None else "N/A"
            print(f"    Max Gradient Amplitude: {max_grad_mT_m:.2f} mT/m" if isinstance(max_grad_mT_m, float) else f"    Max Gradient Amplitude: {max_grad_mT_m}")
            print(f"    Max Slew Rate: {max_slew_Tm_s:.2f} T/m/s" if isinstance(max_slew_Tm_s, float) else f"    Max Slew Rate: {max_slew_Tm_s}")
        else:
            print("\n  Gradients: Not provided.")

        print("\n  Calculated Metrics (from metadata):")
        if not self.metadata:
            print("    No metadata computed or stored.")
        else:
            for key, value in self.metadata.items():
                if value is None:
                    print(f"    {key}: N/A")
                elif isinstance(value, list) and all(isinstance(item, (float, np.floating, int))) :
                    unit = ""
                    if "_mm" in key: unit = "mm"
                    elif "_m" in key: unit = "m"
                    elif "_Tm_per_s" in key: unit = "T/m/s"
                    elif "_xyz" in key: unit = "a.u." # Assuming arbitrary units for PNS sums for now
                    
                    # Format numbers: scientific for small/large, fixed for others
                    formatted_values = []
                    for v_item in value:
                        if abs(v_item) < 1e-3 or abs(v_item) > 1e4 and v_item != 0:
                             formatted_values.append(f"{v_item:.2e}")
                        else:
                             formatted_values.append(f"{v_item:.3f}")
                    value_str = ", ".join(formatted_values)
                    print(f"    {key}: [{value_str}] {unit}")
                elif isinstance(value, (float, np.floating, int)):
                    unit = ""
                    if "_mm" in key: unit = "mm"
                    elif "_m" in key: unit = "m"
                    elif "_Tm_per_s" in key: unit = "T/m/s"
                    elif "_xyz" in key: unit = "a.u."

                    if abs(value) < 1e-3 or abs(value) > 1e4 and value != 0:
                        print(f"    {key}: {value:.2e} {unit}")
                    else:
                        print(f"    {key}: {value:.3f} {unit}")
                else:
                    print(f"    {key}: {value}")
        print("--- End of Summary ---")
        Args:
            filename (str): Output file name.
            filetype (str, optional): 'csv', 'npy', 'npz', or 'txt'. Inferred from extension if not given.
        """
        if filetype is None:
            if filename.endswith('.csv'):
                filetype = 'csv'
            elif filename.endswith('.npy'):
                filetype = 'npy'
            elif filename.endswith('.npz'):
                filetype = 'npz'
            else:
                filetype = 'txt'
        arr = self.points.T if self.points.shape[0] < self.points.shape[1] else self.points
        if filetype == 'csv':
            np.savetxt(filename, arr, delimiter=',')
        elif filetype == 'npy':
            np.save(filename, arr)
        elif filetype == 'npz':
            np.savez(filename, kspace=arr, gradients=self.gradients, dt=self.dt, metadata=self.metadata)
        elif filetype == 'txt':
            np.savetxt(filename, arr)
        else:
            raise ValueError(f"Unsupported filetype: {filetype}")

    @classmethod
    def import_from(cls, filename):
        """
        Import a trajectory from a file.
        """
        if filename.endswith('.csv') or filename.endswith('.txt'):
            points = np.loadtxt(filename, delimiter=',' if filename.endswith('.csv') else None)
            return cls(name=filename, points=points)
        elif filename.endswith('.npy'):
            points = np.load(filename)
            return cls(name=filename, points=points)
        elif filename.endswith('.npz'):
            data = np.load(filename, allow_pickle=True)
            points = data['kspace']
            gradients = data['gradients'] if 'gradients' in data else None
            dt = data['dt'].item() if 'dt' in data else None
            metadata = data['metadata'].item() if 'metadata' in data else {}
            return cls(name=filename, points=points, gradients=gradients, dt=dt, metadata=metadata)
        else:
            raise ValueError(f"Unsupported filetype: {filename}")

    def summary(self):
        """
        Print a summary of the trajectory.
        """
        d, n = self.points.shape if self.points.shape[0] < self.points.shape[1] else self.points.T.shape
        print(f"Trajectory '{self.name}': {n} points, {d} dimensions")
        if self.gradients is not None:
            print("Gradients present.")
        if self.dt is not None:
            print(f"Sample time: {self.dt * 1e6:.2f} us")
        if self.metadata:
            for key, value in self.metadata.items():
                print(f"{key}: {value}")





class KSpaceTrajectoryGenerator:
    def __init__(
        self,
        fov=0.24,
        resolution=0.001,
        dt=4e-6,
        g_max=40e-3,
        s_max=150.0,
        n_interleaves=8,
        gamma=42.576e6,
        traj_type='spiral',
        turns=1,
        ramp_fraction=0.1,
        add_rewinder=True,
        add_spoiler=False,
        add_slew_limited_ramps=True,
        dim=2,
        n_stacks: Optional[int] = None,
        zmax: Optional[float] = None,
        custom_traj_func: Optional[Callable[..., Any]] = None,
        per_interleaf_params: Optional[Dict[int, Dict[str, Any]]] = None,
        time_varying_params: Optional[Callable[[float], Dict[str, float]]] = None,
        use_golden_angle: bool = False,
        vd_method: str = "power",
        vd_alpha: Optional[float] = None,
        vd_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        vd_flat: Optional[float] = None,
        vd_sigma: Optional[float] = None,
        vd_rho: Optional[float] = None,
        spiral_out_out: bool = False,          # <--- New option
        spiral_out_out_split: float = 0.5,     # <--- Fraction of samples for first spiral out (0.5=even split)
        ):
        self.fov = fov
        self.resolution = resolution
        self.dt = dt
        self.g_max = g_max
        self.s_max = s_max
        self.n_interleaves = n_interleaves
        self.gamma = gamma
        self.traj_type = traj_type
        self.turns = turns
        self.ramp_fraction = ramp_fraction
        self.add_rewinder = add_rewinder
        self.add_spoiler = add_spoiler
        self.add_slew_limited_ramps = add_slew_limited_ramps
        self.dim = dim
        self.n_stacks = n_stacks
        self.zmax = zmax
        self.custom_traj_func = custom_traj_func
        self.per_interleaf_params = per_interleaf_params or {}
        self.time_varying_params = time_varying_params
        self.use_golden_angle = use_golden_angle
        self.vd_method = vd_method
        self.vd_alpha = vd_alpha
        self.vd_func = vd_func
        self.vd_flat = vd_flat
        self.vd_sigma = vd_sigma
        self.vd_rho = vd_rho
        self.spiral_out_out = spiral_out_out
        self.spiral_out_out_split = spiral_out_out_split

        self.k_max = 1 / (2 * self.resolution)
        self.g_required = min(self.k_max / (self.gamma * self.dt), self.g_max)
        self.n_samples = int(np.ceil((self.k_max * 2 * np.pi * self.fov) / (self.gamma * self.g_required * self.dt)))
        self.n_samples = max(self.n_samples, 1)
        self.ramp_samples = int(np.ceil(self.ramp_fraction * self.n_samples))
        self.flat_samples = self.n_samples - 2 * self.ramp_samples

    def _slew_limited_ramp(self, N, sign=1):
        t_ramp = np.linspace(0, 1, N)
        ramp = 0.5 * (1 - np.cos(np.pi * t_ramp))
        return sign * ramp

    def _make_radius_profile(self, n_samples=None):
        n_samples = n_samples or self.n_samples
        ramp_samples = int(self.ramp_fraction * n_samples)
        flat_samples = n_samples - 2 * ramp_samples
        if self.add_slew_limited_ramps:
            ramp_up = self._slew_limited_ramp(ramp_samples)
            flat = np.ones(flat_samples)
            ramp_down = 1 - self._slew_limited_ramp(ramp_samples)
            r_profile = np.concatenate([ramp_up, flat, ramp_down])
        else:
            r_profile = np.ones(n_samples)
            r_profile[:ramp_samples] = np.linspace(0, 1, ramp_samples)
            r_profile[-ramp_samples:] = np.linspace(1, 0, ramp_samples)
        return r_profile

    def _variable_density_spiral(self, t):
        if self.vd_func is not None or self.vd_method == "custom":
            return self.vd_func(t)
        if self.vd_method == "power":
            alpha = self.vd_alpha if self.vd_alpha is not None else 1
            return t ** alpha
        elif self.vd_method == "hybrid":
            flat = self.vd_flat if self.vd_flat is not None else 0.2
            alpha = self.vd_alpha if self.vd_alpha is not None else 2
            r = np.zeros_like(t)
            mask = t < flat
            r[mask] = t[mask] / flat
            r[~mask] = ((t[~mask] - flat) / (1 - flat)) ** alpha
            return r
        elif self.vd_method == "gaussian":
            sigma = self.vd_sigma if self.vd_sigma is not None else 0.25
            from scipy.special import erf
            return erf(t / sigma)
        elif self.vd_method == "exponential":
            rho = self.vd_rho if self.vd_rho is not None else 3
            return (np.exp(rho * t) - 1) / (np.exp(rho) - 1)
        elif self.vd_method == "flat":
            return t
        else:
            return t

    def _enforce_gradient_limits(self, gx, gy, gz=None):
        g_norm = np.sqrt(gx ** 2 + gy ** 2 + (gz**2 if gz is not None else 0))
        over_gmax = g_norm > self.g_max
        if np.any(over_gmax):
            scale = self.g_max / np.max(g_norm)
            gx[over_gmax] *= scale
            gy[over_gmax] *= scale
            if gz is not None:
                gz[over_gmax] *= scale

        slew = np.sqrt(np.gradient(gx, self.dt) ** 2 +
                       np.gradient(gy, self.dt) ** 2 +
                       (np.gradient(gz, self.dt) ** 2 if gz is not None else 0))
        over_smax = slew > self.s_max
        if np.any(over_smax):
            scale = self.s_max / np.max(slew)
            gx[over_smax] *= scale
            gy[over_smax] *= scale
            if gz is not None:
                gz[over_smax] *= scale
        return gx, gy, gz

    def _golden_angle(self, idx):
        if self.dim == 2:
            golden = np.pi * (3 - np.sqrt(5))
            return idx * golden
        elif self.dim == 3:
            indices = idx + 0.5
            phi = np.arccos(1 - 2*indices/self.n_interleaves)
            theta = np.pi * (1 + 5**0.5) * indices
            return theta, phi

    def _generate_spiral_out_out(self, t, n_samples, k_max, turns, phi, r_profile):
        """
        Generate a spiral-out-spiral-out trajectory (see e.g. https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.24476)
        """
        split = self.spiral_out_out_split
        n1 = int(np.floor(split * n_samples))
        n2 = n_samples - n1

        # First spiral out (0 -> k_max)
        t1 = np.linspace(0, 1, n1, endpoint=False)
        vd1 = self._variable_density_spiral(t1)
        r1 = vd1 * k_max * r_profile[:n1]
        theta1 = turns * 2 * np.pi * t1 + phi

        # Second spiral out (0 -> k_max), starting at origin but shifted in angle
        t2 = np.linspace(0, 1, n2)
        vd2 = self._variable_density_spiral(t2)
        # angle offset for the second spiral
        phi2 = phi + np.pi  # 180 deg offset (can be parameter)
        r2 = vd2 * k_max * r_profile[n1:]
        theta2 = turns * 2 * np.pi * t2 + phi2

        # Both start at (0,0), end at (k_max, angle)
        kx1 = r1 * np.cos(theta1)
        ky1 = r1 * np.sin(theta1)
        kx2 = r2 * np.cos(theta2)
        ky2 = r2 * np.sin(theta2)

        kx = np.concatenate([kx1, kx2])
        ky = np.concatenate([ky1, ky2])
        return kx, ky

    def _generate_standard(self, interleaf_idx, t, n_samples, **params):
        local_params = {**self.__dict__, **params}
        fov = local_params.get("fov", self.fov)
        resolution = local_params.get("resolution", self.resolution)
        turns = local_params.get("turns", self.turns)
        k_max = 1 / (2 * resolution)
        r_profile = self._make_radius_profile(n_samples)
        if self.time_varying_params is not None:
            for i, ti in enumerate(t):
                for key, val in self.time_varying_params(ti).items():
                    if key == "fov":
                        fov = val
                    if key == "resolution":
                        resolution = val
                k_max = 1 / (2 * resolution)
                r_profile[i] = min(r_profile[i], k_max / self.k_max)

        if self.dim == 2:
            if self.traj_type == "spiral":
                if self.spiral_out_out:
                    if self.use_golden_angle:
                        phi = self._golden_angle(interleaf_idx)
                    else:
                        phi = 2 * np.pi * interleaf_idx / self.n_interleaves
                    kx, ky = self._generate_spiral_out_out(t, n_samples, k_max, turns, phi, r_profile)
                else:
                    t_norm = np.linspace(0, 1, n_samples)
                    vd = self._variable_density_spiral(t_norm)
                    r = vd * k_max * r_profile
                    if self.use_golden_angle:
                        phi = self._golden_angle(interleaf_idx)
                    else:
                        phi = 2 * np.pi * interleaf_idx / self.n_interleaves
                    theta = turns * 2 * np.pi * t / t[-1] + phi
                    kx = r * np.cos(theta)
                    ky = r * np.sin(theta)
                kz = None
            elif self.traj_type == "radial":
                angle = (self._golden_angle(interleaf_idx) if self.use_golden_angle
                         else np.pi * interleaf_idx / self.n_interleaves)
                k_line = np.linspace(-k_max, k_max, n_samples) * r_profile
                kx = k_line * np.cos(angle)
                ky = k_line * np.sin(angle)
                kz = None
            elif self.traj_type == "epi":
                kx = np.linspace(-k_max, k_max, n_samples)
                ky = np.zeros(n_samples)
                kz = None
            elif self.traj_type == "rosette":
                f1 = params.get("f1", 5)
                f2 = params.get("f2", 7)
                a = params.get("a", 0.5)
                phase = 2*np.pi*interleaf_idx/self.n_interleaves
                tt = np.linspace(0, 2*np.pi, n_samples)
                kx = k_max * (a * np.sin(f1*tt+phase) + (1-a) * np.sin(f2*tt+phase))
                ky = k_max * (a * np.cos(f1*tt+phase) + (1-a) * np.cos(f2*tt+phase))
                kz = None
            else:
                raise ValueError(f"Unknown 2D traj_type {self.traj_type}")
            gx = np.gradient(kx, self.dt) / self.gamma
            gy = np.gradient(ky, self.dt) / self.gamma
            gz = None
        elif self.dim == 3:
            if self.traj_type == "stackofspirals":
                n_stacks = self.n_stacks or 8
                zmax = self.zmax or k_max
                stack_idx = interleaf_idx // self.n_interleaves
                slice_idx = interleaf_idx % self.n_interleaves
                z_locations = np.linspace(-zmax, zmax, n_stacks)
                z = z_locations[stack_idx]
                phi = 2 * np.pi * slice_idx / self.n_interleaves
                t_norm = np.linspace(0, 1, n_samples)
                vd = self._variable_density_spiral(t_norm)
                r = vd * k_max * r_profile
                theta = turns * 2 * np.pi * t / t[-1] + phi
                kx = r * np.cos(theta)
                ky = r * np.sin(theta)
                kz = np.ones(n_samples) * z
            elif self.traj_type == "phyllotaxis":
                golden_angle = np.pi * (3 - np.sqrt(5))
                theta = golden_angle * interleaf_idx
                z = np.linspace(1 - 1/n_samples, -1 + 1/n_samples, n_samples)
                radius = np.sqrt(1 - z**2)
                kx = k_max * radius * np.cos(theta)
                ky = k_max * radius * np.sin(theta)
                kz = k_max * z
            elif self.traj_type == "cones":
                phi = 2 * np.pi * interleaf_idx / self.n_interleaves
                tt = np.linspace(0, 1, n_samples)
                theta = np.arccos(1 - 2*tt)
                vd = self._variable_density_spiral(tt)
                kx = k_max * vd * np.sin(theta) * np.cos(phi)
                ky = k_max * vd * np.sin(theta) * np.sin(phi)
                kz = k_max * vd * np.cos(theta)
            elif self.traj_type == "radial3d":
                theta, phi = self._golden_angle(interleaf_idx)
                k_line = np.linspace(-k_max, k_max, n_samples)
                kx = k_line * np.sin(theta) * np.cos(phi)
                ky = k_line * np.sin(theta) * np.sin(phi)
                kz = k_line * np.cos(theta)
            else:
                raise ValueError(f"Unknown 3D traj_type {self.traj_type}")
            gx = np.gradient(kx, self.dt) / self.gamma
            gy = np.gradient(ky, self.dt) / self.gamma
            gz = np.gradient(kz, self.dt) / self.gamma
        else:
            raise ValueError("dim must be 2 or 3")
        gx, gy, gz = self._enforce_gradient_limits(gx, gy, gz)
        return kx, ky, kz, gx, gy, gz

    def _add_spoiler(self, kx, ky, kz, gx, gy, gz):
        n_spoil = self.ramp_samples
        spoil_area = 2 * self.k_max
        kx_out, ky_out, kz_out, gx_out, gy_out, gz_out = [], [], [], [], [], []
        for idx in range(self.n_interleaves):
            if self.dim == 2:
                end_g = np.array([gx[idx, -1], gy[idx, -1]])
                if np.linalg.norm(end_g) == 0:
                    end_g = np.array([1, 0])
                else:
                    end_g /= np.linalg.norm(end_g)
                g_spoil = end_g * (spoil_area / (self.gamma * self.dt * n_spoil))
                kx_s = np.full(n_spoil, kx[idx, -1])
                ky_s = np.full(n_spoil, ky[idx, -1])
                gx_s = np.full(n_spoil, g_spoil[0])
                gy_s = np.full(n_spoil, g_spoil[1])
                kx_out.append(np.concatenate([kx[idx], kx_s]))
                ky_out.append(np.concatenate([ky[idx], ky_s]))
                gx_out.append(np.concatenate([gx[idx], gx_s]))
                gy_out.append(np.concatenate([gy[idx], gy_s]))
            else:
                end_g = np.array([gx[idx, -1], gy[idx, -1], gz[idx, -1]])
                if np.linalg.norm(end_g) == 0:
                    spoil_dir = np.array([0, 0, 1])
                else:
                    spoil_dir = end_g / np.linalg.norm(end_g)
                g_spoil = spoil_dir * (spoil_area / (self.gamma * self.dt * n_spoil))
                kx_s = np.full(n_spoil, kx[idx, -1])
                ky_s = np.full(n_spoil, ky[idx, -1])
                kz_s = np.full(n_spoil, kz[idx, -1])
                gx_s = np.full(n_spoil, g_spoil[0])
                gy_s = np.full(n_spoil, g_spoil[1])
                gz_s = np.full(n_spoil, g_spoil[2])
                kx_out.append(np.concatenate([kx[idx], kx_s]))
                ky_out.append(np.concatenate([ky[idx], ky_s]))
                kz_out.append(np.concatenate([kz[idx], kz_s]))
                gx_out.append(np.concatenate([gx[idx], gx_s]))
                gy_out.append(np.concatenate([gy[idx], gy_s]))
                gz_out.append(np.concatenate([gz[idx], gz_s]))
        if self.dim == 2:
            return (np.array(kx_out), np.array(ky_out), None,
                    np.array(gx_out), np.array(gy_out), None)
        else:
            return (np.array(kx_out), np.array(ky_out), np.array(kz_out),
                    np.array(gx_out), np.array(gy_out), np.array(gz_out))

    def _add_rewinder(self, kx, ky, kz, gx, gy, gz):
        n_rw = self.ramp_samples
        kx_out, ky_out, kz_out, gx_out, gy_out, gz_out = [], [], [], [], [], []
        for idx in range(self.n_interleaves):
            if self.dim == 2:
                net_kx = kx[idx, -1]
                net_ky = ky[idx, -1]
                k_rewind = np.linspace([net_kx, net_ky], [0, 0], n_rw)
                gx_rewind = np.gradient(k_rewind[:, 0], self.dt) / self.gamma
                gy_rewind = np.gradient(k_rewind[:, 1], self.dt) / self.gamma
                kx_out.append(np.concatenate([kx[idx], k_rewind[:, 0]]))
                ky_out.append(np.concatenate([ky[idx], k_rewind[:, 1]]))
                gx_out.append(np.concatenate([gx[idx], gx_rewind]))
                gy_out.append(np.concatenate([gy[idx], gy_rewind]))
            else:
                net_kx = kx[idx, -1]
                net_ky = ky[idx, -1]
                net_kz = kz[idx, -1]
                k_rewind = np.linspace([net_kx, net_ky, net_kz], [0, 0, 0], n_rw)
                gx_rewind = np.gradient(k_rewind[:, 0], self.dt) / self.gamma
                gy_rewind = np.gradient(k_rewind[:, 1], self.dt) / self.gamma
                gz_rewind = np.gradient(k_rewind[:, 2], self.dt) / self.gamma
                kx_out.append(np.concatenate([kx[idx], k_rewind[:, 0]]))
                ky_out.append(np.concatenate([ky[idx], k_rewind[:, 1]]))
                kz_out.append(np.concatenate([kz[idx], k_rewind[:, 2]]))
                gx_out.append(np.concatenate([gx[idx], gx_rewind]))
                gy_out.append(np.concatenate([gy[idx], gy_rewind]))
                gz_out.append(np.concatenate([gz[idx], gz_rewind]))
        if self.dim == 2:
            return (np.array(kx_out), np.array(ky_out), None,
                    np.array(gx_out), np.array(gy_out), None)
        else:
            return (np.array(kx_out), np.array(ky_out), np.array(kz_out),
                    np.array(gx_out), np.array(gy_out), np.array(gz_out))

    def generate(self):
        n_interleaves = self.n_interleaves
        n_samples = self.n_samples
        t = np.arange(n_samples) * self.dt

        if self.dim == 2:
            kx = np.zeros((n_interleaves, n_samples))
            ky = np.zeros((n_interleaves, n_samples))
            gx = np.zeros((n_interleaves, n_samples))
            gy = np.zeros((n_interleaves, n_samples))
            kz = gz = None
        else:
            kx = np.zeros((n_interleaves, n_samples))
            ky = np.zeros((n_interleaves, n_samples))
            kz = np.zeros((n_interleaves, n_samples))
            gx = np.zeros((n_interleaves, n_samples))
            gy = np.zeros((n_interleaves, n_samples))
            gz = np.zeros((n_interleaves, n_samples))

        for idx in range(n_interleaves):
            params = self.per_interleaf_params.get(idx, {})
            if self.custom_traj_func is not None:
                k_vals, g_vals = self.custom_traj_func(idx, t, n_samples, **params)
                kx[idx], ky[idx] = k_vals[:2]
                gx[idx], gy[idx] = g_vals[:2]
                if self.dim == 3 and len(k_vals) > 2:
                    kz[idx] = k_vals[2]
                    gz[idx] = g_vals[2]
                continue
            kx_i, ky_i, kz_i, gx_i, gy_i, gz_i = self._generate_standard(idx, t, n_samples, **params)
            kx[idx], ky[idx] = kx_i, ky_i
            gx[idx], gy[idx] = gx_i, gy_i
            if self.dim == 3:
                kz[idx] = kz_i
                gz[idx] = gz_i

        if self.add_spoiler:
            kx, ky, kz, gx, gy, gz = self._add_spoiler(kx, ky, kz, gx, gy, gz)
        if self.add_rewinder:
            kx, ky, kz, gx, gy, gz = self._add_rewinder(kx, ky, kz, gx, gy, gz)

        t = np.arange(kx.shape[1]) * self.dt
        if self.dim == 2:
            return kx, ky, gx, gy, t
        else:
            return kx, ky, kz, gx, gy, gz, t
    def generate_3d_from_2d(
        self,
        n_3d_shots: int,
        fov_3d: float = None,
        resolution_3d: float = None,
        phi_theta_func: Optional[Callable[[int], tuple]] = None,
        traj2d_type: str = "spiral",
        **kwargs
    ):
        """
        Generate a 3D k-space trajectory by rotating a 2D trajectory (e.g., spiral, radial) 
        according to phi and theta angles, with logic for full 3D k-space coverage.
        
        Args:
            n_3d_shots: Number of shots (orientations) covering 3D k-space
            fov_3d: 3D field of view (if None, uses self.fov)
            resolution_3d: 3D resolution (if None, uses self.resolution)
            phi_theta_func: User callback for phi, theta for each shot (idx) (optional)
            traj2d_type: Type of 2D trajectory to generate ("spiral", "radial", etc.)
            kwargs: Passed to 2D trajectory generator
        
        Returns:
            kx, ky, kz: [n_3d_shots, n_samples] arrays
            gx, gy, gz: [n_3d_shots, n_samples] arrays
            t: time vector
            
            # Example usage:
            # gen = KSpaceTrajectoryGenerator(fov=0.22, resolution=0.0015, traj_type='spiral', vd_method='power', vd_alpha=1.5)
            # kx, ky, kz, gx, gy, gz, t = gen.generate_3d_from_2d(n_3d_shots=64)
        """
        fov_3d = fov_3d if fov_3d is not None else self.fov
        resolution_3d = resolution_3d if resolution_3d is not None else self.resolution
        k_max_3d = 1/(2 * resolution_3d)
        # Estimate number of samples for 2D trajectory in the 3D context
        n_samples = int(np.ceil((k_max_3d * 2 * np.pi * fov_3d) / (self.gamma * self.g_max * self.dt)))
        n_samples = max(n_samples, 1)
        t = np.arange(n_samples) * self.dt

        # Uniform sphere coverage: use spherical Fibonacci or golden angle
        def default_phi_theta(idx):
            # Spherical Fibonacci lattice for uniform 3D coverage
            ga = np.pi * (3 - np.sqrt(5))
            z = 1 - 2 * (idx + 0.5) / n_3d_shots
            phi = ga * idx
            theta = np.arccos(z)
            return phi, theta

        phi_theta = phi_theta_func if phi_theta_func is not None else default_phi_theta

        kx = np.zeros((n_3d_shots, n_samples))
        ky = np.zeros((n_3d_shots, n_samples))
        kz = np.zeros((n_3d_shots, n_samples))
        gx = np.zeros((n_3d_shots, n_samples))
        gy = np.zeros((n_3d_shots, n_samples))
        gz = np.zeros((n_3d_shots, n_samples))

        # Prepare a 2D trajectory generator for the base (xy) plane
        base2d = KSpaceTrajectoryGenerator(
            fov=fov_3d, resolution=resolution_3d, dt=self.dt, g_max=self.g_max, s_max=self.s_max,
            n_interleaves=1, gamma=self.gamma, traj_type=traj2d_type, turns=self.turns,
            ramp_fraction=self.ramp_fraction, add_slew_limited_ramps=self.add_slew_limited_ramps,
            vd_method=self.vd_method, vd_alpha=self.vd_alpha, vd_func=self.vd_func,
            vd_flat=self.vd_flat, vd_sigma=self.vd_sigma, vd_rho=self.vd_rho
        )

        # Generate a single "base" 2D trajectory
        kx2d, ky2d, gx2d, gy2d, t2d = base2d.generate()
        kx2d = kx2d[0]
        ky2d = ky2d[0]
        gx2d = gx2d[0]
        gy2d = gy2d[0]

        # Pad or cut to n_samples for consistency
        if len(kx2d) > n_samples:
            kx2d = kx2d[:n_samples]; ky2d = ky2d[:n_samples]
            gx2d = gx2d[:n_samples]; gy2d = gy2d[:n_samples]
        elif len(kx2d) < n_samples:
            pad = n_samples - len(kx2d)
            kx2d = np.pad(kx2d, (0, pad))
            ky2d = np.pad(ky2d, (0, pad))
            gx2d = np.pad(gx2d, (0, pad))
            gy2d = np.pad(gy2d, (0, pad))

        # For each 3D shot, rotate the 2D trajectory by phi and theta
        for idx in range(n_3d_shots):
            phi, theta = phi_theta(idx)
            # 3D rotation matrix (first rotate around y by theta, then around z by phi)
            # Ry(theta) * Rz(phi)
            # [cos(phi)*cos(theta) -sin(phi) cos(phi)*sin(theta)]
            # [sin(phi)*cos(theta)  cos(phi) sin(phi)*sin(theta)]
            # [      -sin(theta)         0         cos(theta) ]
            cphi, sphi = np.cos(phi), np.sin(phi)
            ctheta, stheta = np.cos(theta), np.sin(theta)
            R = np.array([
                [cphi*ctheta, -sphi, cphi*stheta],
                [sphi*ctheta,  cphi, sphi*stheta],
                [      -stheta,     0,      ctheta]
            ])
            # Stack 2D trajectory as [x, y, 0]
            traj2d = np.stack([kx2d, ky2d, np.zeros_like(kx2d)], axis=0)
            grad2d = np.stack([gx2d, gy2d, np.zeros_like(gx2d)], axis=0)
            traj3d = R @ traj2d
            grad3d = R @ grad2d
            kx[idx] = traj3d[0]
            ky[idx] = traj3d[1]
            kz[idx] = traj3d[2]
            gx[idx] = grad3d[0]
            gy[idx] = grad3d[1]
            gz[idx] = grad3d[2]

        return kx, ky, kz, gx, gy, gz, t

    @staticmethod
    def plugin_example(idx, t, n_samples, **kwargs):
        kx = np.cos(2 * np.pi * t / t[-1])
        ky = np.sin(2 * np.pi * t / t[-1])
        gx = np.gradient(kx, t)
        gy = np.gradient(ky, t)
        return (kx, ky), (gx, gy)

    def check_gradient_and_slew_limits(self, k_traj):
        gamma = self.gamma
        G = np.diff(k_traj, axis=0) / self.dt / gamma
        slew = np.diff(G, axis=0) / self.dt
        grad_ok = np.all(np.abs(G) <= self.g_max)
        slew_ok = np.all(np.abs(slew) <= self.s_max)
        return grad_ok, slew_ok, G, slew

# Example usage for spiral out-out:
# gen = KSpaceTrajectoryGenerator(
#     fov=0.24, resolution=0.001, dt=4e-6, g_max=40e-3, s_max=150.0,
#     n_interleaves=8, traj_type='spiral', spiral_out_out=True,
#     spiral_out_out_split=0.5, use_golden_angle=True, vd_method="power", vd_alpha=1.2
# )
# kx, ky, gx, gy, t = gen.generate()
