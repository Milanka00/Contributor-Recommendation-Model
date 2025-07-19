# # convert_pt_to_numpy.py
# import torch
# import numpy as np
# import os

# os.makedirs('models_npy', exist_ok=True)

# # Load .pt files from 'models/' folder
# train_vectors = torch.load('models/train_vectors.pt')
# pr_vectors = torch.load('models/pr_vectors.pt')

# # Convert tensors to NumPy arrays
# train_vectors_np = train_vectors.numpy() if hasattr(train_vectors, 'numpy') else np.array(train_vectors)
# pr_vectors_np = pr_vectors.numpy() if hasattr(pr_vectors, 'numpy') else np.array(pr_vectors)

# # Save them as .npy files
# np.save('models_npy/train_vectors.npy', train_vectors_np)
# np.save('models_npy/pr_vectors.npy', pr_vectors_np)

# print("Conversion complete. Saved in 'models_npy/'")
