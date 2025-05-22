import torch
from mri_simulator.core import Isochromat, BlochSolver, MTBlochSolver
from .rf_pulse import RF_Pulse
from .gradient_pulse import GradientPulse
import heapq # For managing events by time

class PulseSequence:
    """
    Orchestrates an MRI pulse sequence simulation.
    Manages events (RF pulses, gradients, delays, acquisitions),
    iterates through time, and calls the appropriate solver.
    """
    def __init__(self,
                 name: str,
                 isochromats: Isochromat,
                 solver_type: str, # 'bloch' or 'mt_bloch'
                 dt: float, # Simulation time step in seconds
                 device: str = 'cpu',
                 dtype: torch.dtype = torch.float32):
        """
        Args:
            name (str): Name of the pulse sequence.
            isochromats (Isochromat): Isochromat object to simulate.
            solver_type (str): Type of solver to use ('bloch' or 'mt_bloch').
            dt (float): Master time step for the simulation in seconds.
            device (str, optional): PyTorch device. Defaults to 'cpu'.
            dtype (torch.dtype, optional): PyTorch dtype. Defaults to torch.float32.
        """
        self.name = name
        self.isochromats = isochromats
        self.dt = torch.tensor(dt, device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype

        if solver_type == 'bloch':
            self.solver = BlochSolver(isochromats, dt)
        elif solver_type == 'mt_bloch':
            if not isochromats.is_mt_active:
                raise ValueError("MTBlochSolver requires Isochromats with MT parameters enabled.")
            self.solver = MTBlochSolver(isochromats, dt)
        else:
            raise ValueError(f"Unsupported solver_type: {solver_type}. Choose 'bloch' or 'mt_bloch'.")

        self.events = [] # List of event tuples: (time, type_code, event_obj_or_data)
                         # type_code: 0=add_pulse, 1=remove_pulse, 2=acquire
        self.active_rf_pulses = {} # Store active RF pulses: {rf_pulse_obj: end_time}
        self.active_gradient_pulses = {} # {grad_pulse_obj: end_time}
        self.acquisition_points = [] # List of (time, data_idx) for acquisition

        self.total_duration = torch.tensor(0.0, device=device, dtype=dtype)
        
        # To store results
        self.simulated_signals = [] # List of complex torch.Tensor (scalar signal)
        self.acquisition_times = [] # List of float


    def add_rf_pulse(self, rf_pulse: RF_Pulse, start_time: float):
        start_time_t = torch.tensor(start_time, device=self.device, dtype=self.dtype)
        end_time_t = start_time_t + rf_pulse.duration
        heapq.heappush(self.events, (start_time_t.item(), 0, rf_pulse)) # 0 for add RF
        heapq.heappush(self.events, (end_time_t.item(), 1, rf_pulse))   # 1 for remove RF
        self.total_duration = torch.maximum(self.total_duration, end_time_t)

    def add_gradient_pulse(self, gradient_pulse: GradientPulse, start_time: float):
        start_time_t = torch.tensor(start_time, device=self.device, dtype=self.dtype)
        end_time_t = start_time_t + gradient_pulse.duration
        heapq.heappush(self.events, (start_time_t.item(), 2, gradient_pulse)) # 2 for add Grad
        heapq.heappush(self.events, (end_time_t.item(), 3, gradient_pulse))   # 3 for remove Grad
        self.total_duration = torch.maximum(self.total_duration, end_time_t)
        
    def add_acquisition(self, time: float, duration: float = 0.0):
        # If duration is 0, it's a point acquisition. Otherwise, could be a window.
        # For now, treat as point acquisition at 'time'.
        time_t = torch.tensor(time, device=self.device, dtype=self.dtype)
        heapq.heappush(self.events, (time_t.item(), 4, None)) # 4 for acquire
        self.total_duration = torch.maximum(self.total_duration, time_t)


    def _get_current_b1_grads_and_rf_offset(self, current_time_float: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Helper to get summed B1, summed Gradients, and current RF offset. """
        # RF Pulses
        sum_b1_complex = torch.tensor(0.0j, device=self.device, dtype=torch.complex64 if self.dtype==torch.float32 else torch.complex128)
        current_rf_offset_hz = torch.tensor(0.0, device=self.device, dtype=self.dtype) # Default if no RF active
        
        # Use a copy of active_rf_pulses for safe iteration if needed, though direct iteration is fine here
        for rf_pulse in list(self.active_rf_pulses.keys()):
            if current_time_float > self.active_rf_pulses[rf_pulse]: # Should have been removed by event loop
                # This check is more of a safeguard; main loop should handle removal.
                del self.active_rf_pulses[rf_pulse] 
                continue
            
            # Time relative to this specific pulse's start
            # This requires knowing when this pulse started. This info is not in active_rf_pulses.
            # This design is flawed. The event loop should pass the pulse and its start time.
            # For now, let's assume `get_b1_and_offset` is called with time relative to pulse start.
            # This means `_get_current_b1_grads_and_rf_offset` needs to know pulse start time.
            # This current simplified active_pulses list isn't enough.
            
            # Let's refine: active_pulses store (pulse_obj, start_time_of_this_pulse_instance)
            # This is getting complex. A simpler way for _get_current_b1_grads_and_rf_offset:
            # Iterate through ALL events added to the sequence, check if current_time falls within their duration.
            # This is less efficient for very long sequences with many events, but simpler to implement correctly.
            # The heap-based event processing in run_simulation is better.

            # The main loop will manage active pulses. This function will receive currently active pulses.
            # This function will be simplified or removed if main loop handles it.
            # For now, let's assume the main loop gives us the currently "on" pulses.
            # This method will be called from the main loop with current active pulses.
            pass # This logic will be in the main simulation loop. This function is likely not needed.
        
        # This function is better refactored into the main loop.
        # The main loop will iterate through time steps. At each step:
        # 1. Process events from the priority queue that occur AT or BEFORE current time.
        #    - Add pulses to active_rf_pulses / active_gradient_pulses.
        #    - Remove pulses from active_rf_pulses / active_gradient_pulses.
        #    - Trigger acquisitions.
        # 2. Calculate total B1 and Grads from currently active_rf_pulses and active_gradient_pulses.

        # Placeholder logic (actual calculation will be in run_simulation based on active sets)
        sum_gx = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        sum_gy = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        sum_gz = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        
        return sum_b1_complex, current_rf_offset_hz, torch.stack([sum_gx, sum_gy, sum_gz])


    def run_simulation(self, verbose: bool = False):
        """
        Runs the full pulse sequence simulation.
        """
        if not self.events:
            if verbose: print("No events in sequence. Nothing to simulate.")
            return [], []

        # Clear previous results
        self.simulated_signals.clear()
        self.acquisition_times.clear()
        
        # Ensure isochromats are at their initial state (e.g. Mz=M0_free or M0_total)
        # This should be handled by user before passing isochromats, or add a reset method.
        # For now, assume isochromats are already in desired initial state.

        # Simulation time vector
        # Use a precise number of steps based on dt and total_duration
        num_time_steps = int(torch.ceil(self.total_duration / self.dt).item())
        if num_time_steps == 0 and self.total_duration > 0: num_time_steps = 1
        
        # Simulation loop
        current_time = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        
        # Event processing setup
        event_queue = sorted(self.events, key=lambda x: x[0]) # Sort events by time
        event_idx = 0
        
        # Store currently active pulses with their *sequence start times*
        # This is crucial for calculating time_within_pulse correctly.
        # {pulse_obj: sequence_start_time_of_this_instance}
        current_active_rf = {} 
        current_active_gradients = {}

        if verbose: print(f"Starting simulation: {self.name}, Duration: {self.total_duration.item():.4f}s, Steps: {num_time_steps}, dt: {self.dt.item():.2e}s")

        for step in range(num_time_steps + 1): # +1 to include total_duration endpoint
            current_time = step * self.dt
            if current_time > self.total_duration: # Ensure we don't overshoot if dt doesn't divide duration perfectly
                current_time = self.total_duration

            # --- Process events up to current_time ---
            while event_idx < len(event_queue) and event_queue[event_idx][0] <= current_time.item() + self.dt.item()*0.5: # process events within this step
                ev_time, ev_type_code, ev_obj = event_queue[event_idx]
                event_time_tensor = torch.tensor(ev_time, device=self.device, dtype=self.dtype)

                if ev_type_code == 0: # Add RF
                    current_active_rf[ev_obj] = event_time_tensor 
                    if verbose and step % max(1, num_time_steps//10) == 0 : print(f"  Time {current_time.item():.4f}s: RF pulse '{ev_obj.name}' started.")
                elif ev_type_code == 1: # Remove RF
                    if ev_obj in current_active_rf: del current_active_rf[ev_obj]
                    if verbose and step % max(1, num_time_steps//10) == 0 : print(f"  Time {current_time.item():.4f}s: RF pulse '{ev_obj.name}' ended.")
                elif ev_type_code == 2: # Add Grad
                    current_active_gradients[ev_obj] = event_time_tensor
                    if verbose and step % max(1, num_time_steps//10) == 0 : print(f"  Time {current_time.item():.4f}s: Grad pulse '{ev_obj.name}' started.")
                elif ev_type_code == 3: # Remove Grad
                    if ev_obj in current_active_gradients: del current_active_gradients[ev_obj]
                    if verbose and step % max(1, num_time_steps//10) == 0 : print(f"  Time {current_time.item():.4f}s: Grad pulse '{ev_obj.name}' ended.")
                elif ev_type_code == 4: # Acquire
                    # Acquisition happens *after* state update for this time step,
                    # so we use the magnetization *from* this step's solver call.
                    # Mark for acquisition, actual signal sum after solver.
                    # For simplicity, assume acquisition is instantaneous at ev_time.
                    # If current_time is very close to ev_time:
                    if torch.isclose(current_time, event_time_tensor, atol=self.dt.item()*0.5):
                        # Mxy = Mxf + i * Myf
                        m_free = self.isochromats.magnetization # (N, 3)
                        mxy_complex = torch.complex(m_free[:, 0], m_free[:, 1]) # (N,)
                        signal = torch.sum(mxy_complex) # Scalar complex
                        self.simulated_signals.append(signal)
                        self.acquisition_times.append(current_time.item())
                        if verbose and step % max(1, num_time_steps//10) == 0 : print(f"  Time {current_time.item():.4f}s: Acquired signal: {signal.item()}")
                event_idx += 1

            # --- Calculate effective B1 and Gradients for this time step ---
            # Sum B1 from all active RF pulses
            total_b1_complex = torch.tensor(0.0j, device=self.device, dtype=torch.complex64 if self.dtype==torch.float32 else torch.complex128)
            # RF offset: Use the first active RF pulse's offset. Assume pulses don't usually overlap with different offsets.
            # Or, if they do, this model might need adjustment (e.g. apply each RF in sequence for its sub-portion of dt).
            # For now, a simple assumption: take offset of one of the active RFs if any.
            active_rf_pulse_for_offset_calc = None 
            
            for rf_pulse, start_time_seq in current_active_rf.items():
                time_within_rf_pulse = (current_time - start_time_seq).item()
                if 0 <= time_within_rf_pulse <= rf_pulse.duration.item() + 1e-9: # Check if truly active
                    b1_val, _ = rf_pulse.get_b1_and_offset(time_within_rf_pulse)
                    total_b1_complex += b1_val
                    if active_rf_pulse_for_offset_calc is None: # Take first one for offset
                         active_rf_pulse_for_offset_calc = rf_pulse
            
            current_rf_offset_hz = active_rf_pulse_for_offset_calc.frequency_offset_hz if active_rf_pulse_for_offset_calc else torch.tensor(0.0, device=self.device, dtype=self.dtype)

            # Sum gradients from all active gradient pulses
            total_gradients = torch.zeros(3, device=self.device, dtype=self.dtype)
            for grad_pulse, start_time_seq in current_active_gradients.items():
                time_within_grad_pulse = (current_time - start_time_seq).item()
                if 0 <= time_within_grad_pulse <= grad_pulse.duration.item() + 1e-9:
                    total_gradients += grad_pulse.get_gradients(time_within_grad_pulse)
            
            # --- Run solver for one step ---
            if isinstance(self.solver, MTBlochSolver):
                self.solver.run_step(total_b1_complex, total_gradients, current_rf_offset_hz)
            elif isinstance(self.solver, BlochSolver):
                self.solver.run_step(total_b1_complex, total_gradients)
            
            if current_time >= self.total_duration : # Simulation finished
                break
        
        if verbose: print(f"Simulation finished. Acquired {len(self.simulated_signals)} signal points.")
        return self.acquisition_times, self.simulated_signals


if __name__ == '__main__':
    print("--- PulseSequence Examples ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32

    # Isochromat setup (single isochromat at origin)
    isochromat = Isochromat(
        name="test_spin",
        magnetization=torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=dtype),
        M0=torch.tensor([1.0], device=device, dtype=dtype),
        T1=torch.tensor([1000e-3], device=device, dtype=dtype),
        T2=torch.tensor([100e-3], device=device, dtype=dtype),
        T2_star=torch.tensor([80e-3], device=device, dtype=dtype),
        chemical_shift_hz=torch.tensor([0.0], device=device, dtype=dtype),
        spatial_coords=torch.tensor([[0.0, 0.0, 0.0]], device=device, dtype=dtype),
        device=device, dtype=dtype
    )

    # 1. Simple FID (90-degree pulse, then acquire)
    seq_fid = PulseSequence(name="FID_example", isochromats=isochromat, solver_type='bloch', dt=10e-6, device=device, dtype=dtype)
    
    # Hard 90-degree pulse (approximate)
    # gamma * B1 * duration = pi/2. For duration=0.5ms, B1 = (pi/2)/(gamma*0.5e-3)
    from mri_simulator.utils.constants import GAMMA_RAD_S_T_PROTON
    gamma_hz_t = GAMMA_RAD_S_T_PROTON / (2 * torch.pi) # Hz/T
    pulse_duration_90 = 0.2e-3 # 0.2 ms
    b1_amp_90 = (torch.pi/2) / (GAMMA_RAD_S_T_PROTON * pulse_duration_90) # Tesla
    
    rf_90 = RF_Pulse(name="RF_90", duration=pulse_duration_90, amplitude_tesla=b1_amp_90, phase_rad=0, device=device, dtype=dtype) # Along X'
    seq_fid.add_rf_pulse(rf_90, start_time=0.0)
    
    # Acquisition window
    num_acquire_points = 128
    acquire_interval = 1e-3 # 1ms between points
    for i in range(num_acquire_points):
        seq_fid.add_acquisition(time = pulse_duration_90 + (i * acquire_interval) + 1e-6) # Start acquiring right after pulse
                                                                                    # Small offset to ensure it's after RF end event
    
    print(f"Running FID sequence (Total duration: {seq_fid.total_duration.item()}s)...")
    acq_times, signals = seq_fid.run_simulation(verbose=False) # Set verbose=True for detailed logs

    print(f"FID: Acquired {len(signals)} points.")
    if signals:
        print(f"Initial signal: {signals[0].item()}, Final signal: {signals[-1].item()}")
        # Check T2* decay (abs value)
        expected_decay_factor = torch.exp(torch.tensor(-acq_times[-1] / isochromat.T2_star[0].item(), device=device, dtype=dtype))
        # Initial signal magnitude after 90 deg pulse should be ~M0
        # This is a rough check as actual signal depends on many factors (exact flip, dt errors)
        print(f"Signal magnitude at end: {torch.abs(signals[-1]).item():.3f}, M0: {isochromat.M0[0].item():.3f}")
        # More robust check: signal should decay
        if len(signals) > 1:
             assert torch.abs(signals[-1]) < torch.abs(signals[0]) * 0.8, "FID signal did not decay as expected"
    
    print("\nPulseSequence examples finished.")
