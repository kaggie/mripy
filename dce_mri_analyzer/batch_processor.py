import argparse
import os
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
import nibabel as nib # Not strictly needed if io functions handle all NIfTI aspects

# Add project root to sys.path to allow direct import of core modules
# This assumes the script is in dce_mri_analyzer/ and core is a subdirectory
# For robustness, especially if run from elsewhere or as part of a package:
if __package__ is None or __package__ == '':
    # If running as a script, add parent directory of 'core' to path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import io
from core import aif
from core import conversion
from core import modeling

def main():
    parser = argparse.ArgumentParser(description="DCE-MRI Batch Processor - Single Dataset")

    # Input Files
    parser.add_argument("--dce", required=True, help="Path to 4D DCE NIfTI file")
    parser.add_argument("--t1map", required=True, help="Path to 3D T1 map NIfTI file")
    parser.add_argument("--mask", help="Path to 3D Mask NIfTI file (optional)")

    # Processing Parameters
    parser.add_argument("--tr", required=True, type=float, help="Repetition Time (TR) in seconds")
    parser.add_argument("--r1_relaxivity", required=True, type=float, help="r1 relaxivity of contrast agent (e.g., L/mmol/s or s⁻¹mM⁻¹)")
    parser.add_argument("--baseline_points", type=int, default=5, help="Number of initial baseline time points for signal normalization")

    # AIF Configuration
    aif_group = parser.add_mutually_exclusive_group(required=True)
    aif_group.add_argument("--aif_file", help="Path to AIF file (CSV/TXT: time, concentration)")
    aif_group.add_argument("--aif_pop_model", choices=list(aif.POPULATION_AIFS.keys()), help="Name of population AIF model to use")
    
    # Population AIF Parameters (add all from metadata for flexibility)
    # Example: --aif_param D_scaler 1.0 --aif_param A1 0.8
    # This uses store_action=append and nargs=2 for key-value pairs
    parser.add_argument('--aif_param', action='append', nargs=2, metavar=('KEY', 'VALUE'),
                        help="Set a parameter for the population AIF model (e.g., D_scaler 1.0). Can be used multiple times.")


    # Model Fitting
    parser.add_argument("--model", required=True, choices=["Standard Tofts", "Extended Tofts", "Patlak", "2CXM"], help="Pharmacokinetic model to apply")
    parser.add_argument("--num_processes", type=int, default=os.cpu_count(), help="Number of cores for parallel fitting")

    # Output
    parser.add_argument("--out_dir", required=True, help="Output directory to save parameter maps")

    args = parser.parse_args()

    print("--- DCE-MRI Batch Processor ---")
    print(f"DCE File: {args.dce}")
    print(f"T1 Map File: {args.t1map}")
    print(f"Mask File: {args.mask if args.mask else 'Not provided'}")
    print(f"TR: {args.tr} s, r1: {args.r1_relaxivity}, Baseline Points: {args.baseline_points}")
    if args.aif_file: print(f"AIF File: {args.aif_file}")
    if args.aif_pop_model: print(f"Population AIF Model: {args.aif_pop_model}")
    if args.aif_param: print(f"Population AIF Params: {args.aif_param}")
    print(f"Pharmacokinetic Model: {args.model}")
    print(f"Output Directory: {args.out_dir}")
    print(f"Number of Processes: {args.num_processes}")
    print("-------------------------------")

    try:
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"Output directory created/ensured: {args.out_dir}")
    except Exception as e:
        print(f"Error creating output directory {args.out_dir}: {e}")
        exit(1)

    # 1. Load Data
    try:
        print("Loading DCE data...")
        dce_data = io.load_dce_series(args.dce)
        print(f"DCE data loaded. Shape: {dce_data.shape}")

        print("Loading T1 map...")
        t10_data = io.load_t1_map(args.t1map, dce_shape=dce_data.shape)
        print(f"T1 map loaded. Shape: {t10_data.shape}")

        mask_data = None
        if args.mask:
            print("Loading mask...")
            mask_data = io.load_mask(args.mask, reference_shape=dce_data.shape[:3])
            print(f"Mask loaded. Shape: {mask_data.shape}")
        else:
            print("No mask provided.")
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)

    # 2. Signal to Concentration
    try:
        print("Performing signal to concentration conversion...")
        Ct_data = conversion.signal_to_concentration(
            dce_data, t10_data, args.r1_relaxivity, args.tr, 
            baseline_time_points=args.baseline_points
        )
        print(f"Signal to concentration conversion successful. Ct_data shape: {Ct_data.shape}")
    except Exception as e:
        print(f"Error during signal to concentration conversion: {e}")
        exit(1)

    # 3. Prepare AIF
    aif_time_arr, aif_conc_arr = None, None
    try:
        if args.aif_file:
            print(f"Loading AIF from file: {args.aif_file}")
            aif_time_arr, aif_conc_arr = aif.load_aif_from_file(args.aif_file)
        elif args.aif_pop_model:
            print(f"Generating population AIF: {args.aif_pop_model}")
            num_time_points_dce = dce_data.shape[3]
            # Assuming AIF models expect time in minutes if their defaults are tuned for it.
            # If TR is in seconds, convert time_vector for AIF to minutes.
            # This detail depends on AIF model conventions. For now, assume time units are consistent
            # or that user provides time in appropriate units for the model.
            # The population_aif_time_vector in GUI uses seconds.
            # For batch, using seconds based on TR seems more direct.
            # However, AIF_PARAMETER_METADATA uses minutes. This needs careful handling or clear documentation.
            # For now, let's assume time_vector_for_aif should be in the units the AIF model expects (e.g., minutes)
            # If TR is in seconds, and AIF model (e.g. Parker) expects minutes, conversion is needed.
            # Let's use seconds for the time vector, and adjust AIF parameters if they are per-minute.
            # OR, assume user provides parameters appropriate for time in seconds if TR is in seconds.
            # For simplicity here, we'll create a time vector in seconds.
            # The AIF functions themselves in core.aif assume time_points are in the units
            # consistent with their m1, m2 etc. parameters.
            
            time_vector_for_aif = np.arange(num_time_points_dce) * args.tr # This is in seconds
            
            pop_aif_params_from_cli = {}
            if args.aif_param:
                for key, value in args.aif_param:
                    try:
                        pop_aif_params_from_cli[key] = float(value)
                    except ValueError:
                        print(f"Warning: Could not convert AIF parameter '{key}' value '{value}' to float. Using default if available.")
            
            # Fetch default params from metadata and override with CLI provided ones
            final_pop_aif_params = {}
            if args.aif_pop_model in aif.AIF_PARAMETER_METADATA:
                for p_name, p_default, _, _, _ in aif.AIF_PARAMETER_METADATA[args.aif_pop_model]:
                    final_pop_aif_params[p_name] = p_default # Start with default
                # Override with CLI values if provided
                for p_name_cli, p_val_cli in pop_aif_params_from_cli.items():
                    if p_name_cli in final_pop_aif_params:
                        final_pop_aif_params[p_name_cli] = p_val_cli
                    else:
                        print(f"Warning: CLI AIF parameter '{p_name_cli}' not recognized for model '{args.aif_pop_model}'.")
            else: # Fallback if no metadata
                 final_pop_aif_params = pop_aif_params_from_cli # Use only CLI params
                 print(f"Warning: No metadata for AIF model '{args.aif_pop_model}'. Using only parameters from CLI.")


            print(f"Using AIF parameters: {final_pop_aif_params}")
            aif_conc_arr = aif.generate_population_aif(args.aif_pop_model, time_vector_for_aif, params=final_pop_aif_params)
            aif_time_arr = time_vector_for_aif
            
        if aif_time_arr is None or aif_conc_arr is None:
            print("AIF could not be defined or generated.")
            exit(1)
        print(f"AIF prepared. Time points: {len(aif_time_arr)}, Max Conc: {np.max(aif_conc_arr):.2f}")

    except Exception as e:
        print(f"Error preparing AIF: {e}")
        exit(1)

    # 4. Model Fitting
    try:
        t_tissue = np.arange(Ct_data.shape[3]) * args.tr # Time vector in seconds
        parameter_maps = {}
        
        print(f"Starting {args.model} fitting with {args.num_processes} processes...")
        if args.model == "Standard Tofts":
            parameter_maps = modeling.fit_standard_tofts_voxelwise(
                Ct_data, t_tissue, aif_time_arr, aif_conc_arr, 
                mask=mask_data, num_processes=args.num_processes
            )
        elif args.model == "Extended Tofts":
            parameter_maps = modeling.fit_extended_tofts_voxelwise(
                Ct_data, t_tissue, aif_time_arr, aif_conc_arr, 
                mask=mask_data, num_processes=args.num_processes
            )
        elif args.model == "Patlak":
            parameter_maps = modeling.fit_patlak_model_voxelwise(
                Ct_data, t_tissue, aif_time_arr, aif_conc_arr, 
                mask=mask_data, num_processes=args.num_processes
            )
        elif args.model == "2CXM":
            parameter_maps = modeling.fit_2cxm_model_voxelwise(
                Ct_data, t_tissue, aif_time_arr, aif_conc_arr, 
                mask=mask_data, num_processes=args.num_processes
            )
        else:
            print(f"Model {args.model} not implemented in batch mode.")
            exit(1)
        print(f"{args.model} fitting completed.")

    except Exception as e:
        print(f"Error during model fitting: {e}")
        # Consider printing traceback for debugging:
        # import traceback
        # traceback.print_exc()
        exit(1)

    # 5. Save Maps
    try:
        if not parameter_maps:
            print("No parameter maps were generated by the model fitting.")
        else:
            print("Saving parameter maps...")
            for map_name, map_data in parameter_maps.items():
                if map_data is not None:
                    output_filepath = os.path.join(args.out_dir, f"{map_name}.nii.gz")
                    # Use T1 map as reference for saving header/affine
                    # This assumes args.t1map is a valid path to a NIfTI file
                    io.save_nifti_map(map_data, args.t1map, output_filepath)
                    print(f"Saved {map_name} to {output_filepath}")
                else:
                    print(f"Map data for '{map_name}' is None, not saving.")
    except Exception as e:
        print(f"Error saving parameter maps: {e}")
        exit(1)

    print("--- Batch processing completed successfully! ---")

if __name__ == "__main__":
    # For multiprocessing safety, especially when bundled
    if sys.platform.startswith('win'):
        import multiprocessing
        multiprocessing.freeze_support()
    main()
