# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs_lib/utils.ipynb.

# %% auto 0
__all__ = ['convert_uuids_to_indices', 'percent_rel_error_']

# %% ../nbs_lib/utils.ipynb 2
import os
import re
import numpy as np

# %% ../nbs_lib/utils.ipynb 4
def convert_uuids_to_indices():
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    uuids = re.findall(r"\b[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b", cuda_visible_devices)

    if uuids:
        indices = [str(i) for i in range(len(uuids))]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(indices)

# %% ../nbs_lib/utils.ipynb 5
def percent_rel_error_(N, N_mc, N_mc_t0):
    """
    Return the percentage relative error with respect to the MC simulation
    See Eq. 23 of the paper by Giudici et al. (2023), 'Space debris density 
    propagation through a vinite volume method'
    Input:
        N: Predicted number of in-orbit elements over time
        N_mc: Number of in-orbit elements from the MC simulation over time
        N_mc_t0: Number of in-orbit elements from the MC simulation at t0
    """
    return 100 * np.abs(N_mc - N) / N_mc_t0