import torch
import matplotlib.pyplot as plt
import numpy as np

def create_random_grid_batch_torch(num_envs=3, n=15, p_obstacle=0.2, seed=42, device='cpu'):
    """
    Create a batch of grids with shape (num_envs, n, n) in PyTorch.
    grid_batch[i] is the i-th environment, an n×n grid.
    0 indicates free space, 1 indicates obstacles.
    """
    # Set random seed
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    # Generate random values in [0, 1)
    grid_batch = torch.rand((num_envs, n, n), generator=g, device=device)
    # Values < p_obstacle are obstacles (1), otherwise free (0)
    grid_batch = (grid_batch < p_obstacle).int()
    return grid_batch

def batch_ray_cast_torch(
    grid_batch, 
    base_row, 
    base_col, 
    angles, 
    rad=True, 
    max_radius=15, 
    step_r=0.5
):
    """
    On a batch of grids, cast rays from the same base point (base_row, base_col)
    for multiple angles in a vectorized PyTorch implementation.
    Special convention: angle=0 points downward.
    """
    device = grid_batch.device
    num_envs, n_x, n_y = grid_batch.shape

    # 1) Discrete radii
    r_vals = torch.arange(0, max_radius + 1e-9, step_r, device=device)
    num_steps = r_vals.shape[0]

    # 2) Angles can be degrees or radians
    if not rad:  # If not radians, convert from degrees
        angles = angles * (np.pi / 180.0)
    # angles shape: (num_rays,)
    num_rays = angles.shape[0]

    # 3) x,y coordinates: angle=0 points downward
    #    x = r * sinθ
    #    y = - r * cosθ (negative to point downward)
    r_2d = r_vals.view(1, num_steps)       # (1, num_steps)
    angles_2d = angles.view(num_rays, 1)   # (num_rays,1)

    x_2d = r_2d * torch.sin(angles_2d)
    y_2d = - r_2d * torch.cos(angles_2d)

    # 4) Convert to row/col indices (batched)
    row_2d = base_row - y_2d
    col_2d = base_col + x_2d
    row_2d_int = torch.round(row_2d).long()
    col_2d_int = torch.round(col_2d).long()

    row_3d = row_2d_int.unsqueeze(0).expand(num_envs, -1, -1)
    col_3d = col_2d_int.unsqueeze(0).expand(num_envs, -1, -1)

    # 5) Boundary check
    valid_mask_3d = (
        (row_3d >= 0) & (row_3d < n_x) & 
        (col_3d >= 0) & (col_3d < n_y)
    )

    # 6) Gather grid values (0/1)
    row_clamped = row_3d.clamp(0, n_x-1)
    col_clamped = col_3d.clamp(0, n_y-1)

    env_idx = torch.arange(num_envs, device=device).view(num_envs,1,1)
    env_idx = env_idx.expand(-1, row_3d.shape[1], row_3d.shape[2])

    grid_vals_3d = torch.zeros_like(row_3d, dtype=grid_batch.dtype)
    grid_vals_3d[valid_mask_3d] = grid_batch[env_idx[valid_mask_3d],
                                             row_clamped[valid_mask_3d],
                                             col_clamped[valid_mask_3d]]

    # 7) Obstacle hit or out-of-bounds => stop
    obstacle_mask_3d = (grid_vals_3d == 1)
    boundary_mask_3d = (~valid_mask_3d) | obstacle_mask_3d

    # 8) Find the first index where boundary is reached
    boundary_cum_3d = torch.cumsum(boundary_mask_3d.int(), dim=2)
    boundary_cum_bool = (boundary_cum_3d > 0)

    def first_true_idx_along_dim(t: torch.BoolTensor, fill_val: int):
        S = t.size(2)
        idx = torch.arange(S, device=t.device).view(1,1,-1)
        masked_idx = torch.where(t, idx, torch.full_like(idx, fill_val))
        first_idx = masked_idx.min(dim=2).values
        return first_idx

    boundary_idx_2d = first_true_idx_along_dim(boundary_cum_bool, fill_val=num_steps)
    boundary_exists_2d = boundary_cum_bool.any(dim=2)

    final_step_2d = torch.where(
        boundary_exists_2d,
        boundary_idx_2d - 1,
        (num_steps - 1) * torch.ones_like(boundary_idx_2d)
    )
    final_step_2d = torch.clamp(final_step_2d, min=0)

    # 9) Distances
    final_dist_2d = r_vals[final_step_2d]
    return final_dist_2d

def visualize_torch_grid_and_rays(
    grid_batch: torch.Tensor, 
    base_row: int, 
    base_col: int, 
    final_dist_2d: torch.Tensor, 
    angles: torch.Tensor, 
    rad=True
):
    """
    Simple visualization (CPU): show base point, rays, and endpoints on the grid.
    """
    # Move to CPU + NumPy
    grid_np = grid_batch.cpu().numpy()
    dist_np = final_dist_2d.cpu().numpy()
    angles_np = angles.cpu().numpy()

    if not rad:
        angles_np = angles_np * (np.pi / 180.0)

    num_envs, n_x, n_y = grid_np.shape
    num_rays = len(angles_np)

    fig, axs = plt.subplots(1, num_envs, figsize=(6*num_envs, 6))
    if num_envs == 1:
        axs = [axs]

    for i in range(num_envs):
        ax = axs[i]
        ax.imshow(grid_np[i], origin='upper', cmap='binary')
        ax.set_title(f"Env {i}")
        ax.scatter(base_col, base_row, color='red', marker='v', s=100, label='Base')

        for j in range(num_rays):
            dist = dist_np[i, j]
            ang  = angles_np[j]
            # Same convention as batch_ray_cast: angle=0 points downward
            x_end = dist * np.sin(ang)
            y_end = - dist * np.cos(ang)

            row_end = base_row - y_end
            col_end = base_col + x_end

            ax.plot([base_col, col_end], [base_row, row_end], 'b-')
            # Mark endpoint in red
            ax.scatter(col_end, row_end, color='red', marker='o', s=30)

        ax.set_xlim(-0.5, n_y-0.5)
        ax.set_ylim(n_x-0.5, -0.5)
        ax.set_xlabel("col")
        ax.set_ylabel("row")
        ax.legend(loc='upper right')

    # Print ray summary
    print(f"{final_dist_2d}")
    for i in range(num_envs):
        print(f"\n===== Environment {i} =====")
        print(f"Total rays: {num_rays}")
        for j, angle_deg in enumerate(angles_np):
            dist = final_dist_2d[i, j]
            if rad:
                angle_deg = angle_deg / np.pi * 180.0
            print(f"  - Ray #{j:2d}, Angle={angle_deg:5.1f}°, Distance={dist*0.2:.2f}")

    plt.tight_layout()
    plt.show()

def main():
    device = 'cpu'
    num_envs = 3
    n = 25

    # Build random environments
    grid_batch = create_random_grid_batch_torch(
        num_envs=num_envs, n=n, 
        p_obstacle=0.02, # 0.05
        seed=42, 
        device=device
    )

    # Base point: top-center
    base_row = 1
    base_col = n // 2
    r =(n // 2.0)
    
    theta_start = - 2 * np.pi/3
    theta_end = 2 * np.pi/3 + 0.0001
    theta_step = np.pi/30
    angles_deg = torch.arange(start=theta_start, end=theta_end, 
                                                step=theta_step, device=device)
    # Cast rays
    final_dist_2d = batch_ray_cast_torch(
        grid_batch=grid_batch,
        base_row=base_row,
        base_col=base_col,
        angles=angles_deg,
        rad=True,          # angles are in radians
        max_radius=r,
        step_r=0.5
    )

    # Visualize
    visualize_torch_grid_and_rays(
        grid_batch,
        base_row, base_col,
        final_dist_2d,
        angles=angles_deg,
        rad=True
    )

if __name__ == "__main__":
    main()
