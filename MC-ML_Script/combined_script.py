import subprocess
import os
import h5py
import torch
import pickle
import fastai

def run_mocat_mc(ICfile, seed):
    matlab_command = (
        "addpath('C:/Users/Nathan/Desktop/mocat-ml-main/MC-ML_Script/'); "
        f"mocat_mc_wrapper('{ICfile}', {seed});"
    )
    subprocess.run(['matlab', '-batch', matlab_command], check=True)

def get_first_mat_file(directory):
    # List all files in the directory
    files = os.listdir(directory)
    # Filter out only .mat files
    mat_files = [f for f in files if f.endswith('.mat')]
    # Sort the list to get the first file (you can adjust this logic if needed)
    mat_files.sort()
    # Return the first .mat file (or None if no .mat files found)
    return os.path.join(directory, mat_files[0]) if mat_files else None

def load_mocat_output_hdf5(file_path):
    def recursively_load_datasets(h5file, path='/'):
        """
        Recursively load datasets from an HDF5 file.
        """
        data = {}
        for key in h5file[path]:
            item = h5file[path + key]
            if isinstance(item, h5py.Dataset):  # It's a dataset
                data[key] = item[()]
            elif isinstance(item, h5py.Group):  # It's a group, go deeper
                data[key] = recursively_load_datasets(h5file, path + key + '/')
        return data
    
    with h5py.File(file_path, 'r') as f:
        data = recursively_load_datasets(f)
    return data

def resize_data(data, new_shape):
    # Calculate the total number of elements needed for the new shape
    num_elements_needed = torch.prod(torch.tensor(new_shape)).item()

    # Flatten the data and slice only the required number of elements
    data_flat = torch.tensor(data).flatten()

    if len(data_flat) < num_elements_needed:
        raise ValueError(f"Not enough data to reshape to {new_shape}. Required: {num_elements_needed}, available: {len(data_flat)}")

    # Slice the data to match the new shape
    sliced_data = data_flat[:num_elements_needed]

    # Reshape the sliced data to the desired shape
    resized_data = sliced_data.view(new_shape)

    return resized_data

def load_ml_model(model_path):
    model = torch.load(model_path)
    return model

def run_ml_model(model, data):
    model.eval()
    with torch.no_grad():
        return model(data)
    
def inspect_mat_file(filepath):
    with h5py.File(filepath, 'r') as f:
        keys = list(f.keys())
        print("Available keys:", keys)

def main():
    # Parameters for MOCAT-MC
    ICfile = 'supporting_data/2020.mat'  # Initial conditions file
    seed = 1  # Random seed

    # Path to the machine learning model
    ml_model_path = 'C:/Users/Nathan/Desktop/mocat-ml-main/MC-ML_Script/models/d_64_epoch_10_TSTPlus.pkl'

    # Output directory for MOCAT-MC
    output_directory = 'C:/Users/Nathan/Desktop/mocat-ml-main/MC-ML_Script/output/'

    # Step 1: Run the MOCAT-MC simulation
    run_mocat_mc(ICfile, seed)

    # Step 2: Get the first .mat file from the output directory
    output_file_path = get_first_mat_file(output_directory)
    if output_file_path is None:
        print("No .mat files found in the output directory.")
        return

    # Step 3: Load the output from MOCAT-MC using h5py for MATLAB v7.3 files
    mocat_data = load_mocat_output_hdf5(output_file_path)
    inspect_mat_file(output_file_path)
    
    # Resize the data from MOCAT-MC to match the input shape of the ML model
    new_shape = (1, 128, 32, 32)  # Adjust this based on your model's input requirements
    resized_data = resize_data(mocat_data['mat_sats'], new_shape)  # Replace 'your_data_key' with the actual key in the .mat file
    

    # Load the trained ML model
    ml_model = load_ml_model(ml_model_path)

    # Run the ML model with the resized data
    ml_output = run_ml_model(ml_model, resized_data)

    # Save the ML output as needed
    print("ML model output:", ml_output)

if __name__ == "__main__":
   main()
