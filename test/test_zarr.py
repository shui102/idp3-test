import zarr

z = zarr.open_group('/extra/waylen/diffusion_policy/Improved-3D-Diffusion-Policy/data/training_data_example', mode='r')  # 注意是 open_group
print(z.tree())  

# 查看根组下的子组
print(z.group_keys())  
# 输出: ['data', 'meta']

# 查看 'data' 组下的数组
data_group = z['meta']['episode_ends'][:]
print(data_group)  
# 输出: ['action', 'img', 'point_cloud', 'state']