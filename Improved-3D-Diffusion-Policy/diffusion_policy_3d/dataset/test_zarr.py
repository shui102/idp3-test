import zarr, os
g = zarr.open(os.path.expanduser('/home/shui/idp3_test/Improved-3D-Diffusion-Policy/data/dataset_11141630_processed_joint.zarr'), 'r')
print('data keys:', list(g['data'].array_keys()))
print('meta keys:', list(g['meta'].array_keys()))