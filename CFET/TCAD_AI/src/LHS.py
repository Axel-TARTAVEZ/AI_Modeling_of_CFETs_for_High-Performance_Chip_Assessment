import os
import numpy as np
import pandas as pd
from scipy.stats import qmc

script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, "../../TCAD", "data")

bounds = np.array([
    [15.0, 35.0],  # Lch (nm) 
    [15.0, 80.0],  # Wch (nm) 
    [4.0,  8.0],   # Tch (nm) 
    [0.2,  1.2],   # Tox1 (nm) 
    [1.5,  2.2],   # Tox2 (nm) 
])

num_samples = 13

sampler = qmc.LatinHypercube(d=5)
sample = sampler.random(n=num_samples)

l_bounds = bounds[:, 0]
u_bounds = bounds[:, 1]
scaled_samples = qmc.scale(sample, l_bounds, u_bounds)

columns_names = ["Lch", "Wch", "Tch", "Tox1", "Tox2"]
df = pd.DataFrame(scaled_samples, columns=columns_names)

df = df.round(2)

file_name = f"LHS_CFET_5D_{num_samples}.csv"
df.to_csv(os.path.join(model_dir, file_name), index=False)

print(f"Le fichier {file_name} contenant tes {num_samples} expériences a été généré.")