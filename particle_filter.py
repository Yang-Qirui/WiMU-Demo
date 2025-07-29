import torch
import time
import matplotlib.pyplot as plt
import json
import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon
from config import REAL_WIDTH
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_geojson(file_path, ref_point1, ref_point2, ref_pixel1, ref_pixel2):
    """
    Load and convert GeoJSON coordinates from pixel to real-world coordinates using reference points.
    
    Args:
        file_path (str): Path to the GeoJSON file
        ref_point1 (tuple): First reference point in real-world coordinates (x, y)
        ref_point2 (tuple): Second reference point in real-world coordinates (x, y)
        ref_pixel1 (tuple): First reference point in pixel coordinates (x, y)
        ref_pixel2 (tuple): Second reference point in pixel coordinates (x, y)
        
    Returns:
        dict: Dictionary mapping area IDs to their polygons in real-world coordinates
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Calculate scale factors using reference points
    pixel_dx = ref_pixel2[0] - ref_pixel1[0]
    pixel_dy = ref_pixel2[1] - ref_pixel1[1]
    real_dx = ref_point2[0] - ref_point1[0]
    real_dy = ref_point2[1] - ref_point1[1]
    
    x_scale = real_dx / pixel_dx
    y_scale = real_dy / pixel_dy
    
    # Calculate offset
    x_offset = ref_point1[0] - ref_pixel1[0] * x_scale
    y_offset = ref_point1[1] - ref_pixel1[1] * y_scale
    
    areas = {}
    for feature in data['features']:
        area_id = feature['properties']['id']
        geometry = feature['geometry']
        if geometry['type'] == 'MultiPolygon':
            polygons = []
            for poly_coords in geometry['coordinates']:
                for ring in poly_coords:
                    # Convert pixel coordinates to real-world coordinates
                    real_coords = []
                    for x, y in ring:
                        real_x = x * x_scale + x_offset
                        real_y = y * y_scale + y_offset
                        real_coords.append((real_x, real_y))
                    polygons.append(Polygon(real_coords))
            areas[area_id] = MultiPolygon(polygons)
        else:
            # Convert pixel coordinates to real-world coordinates
            real_coords = []
            for x, y in geometry['coordinates'][0]:
                real_x = x * x_scale + x_offset
                real_y = y * y_scale + y_offset
                real_coords.append((real_x, real_y))
            areas[area_id] = Polygon(real_coords)
    return areas

# def check_current_area(particles):
#     """
#     Check if each particle is exclusively in the target area.
    
#     Args:
#         particles: torch.Tensor of shape [N, 3] containing particle positions (x, y, weight)
        
#     Returns:
#         torch.Tensor of shape [N] containing 1 if particle is exclusively in target area, 0 otherwise
#     """
#     # Reference points in real-world coordinates (from target.geojson)
#     ref_point1 = (9, -33)    # First point in real-world coordinates
#     ref_point2 = (30, -33)   # Second point in real-world coordinates
    
#     # Reference points in pixel coordinates (from target.geojson)
#     ref_pixel1 = (539.914438502673647, -255.48663101604285)  # First point in pixel coordinates
#     ref_pixel2 = (568.663101604277927, -255.48663101604285)  # Second point in pixel coordinates
    
#     # Load target and shelf areas
#     # target = load_geojson('geojsons/JD_langfang_high/target.geojson', ref_point1, ref_point2, ref_pixel1, ref_pixel2)
#     # shelf = load_geojson('geojsons/JD_langfang_high/shelf.geojson', ref_point1, ref_point2, ref_pixel1, ref_pixel2)
    
#     # Convert particles to CPU and numpy for processing
#     particles_cpu = particles[:, :2].cpu().numpy()
    
#     # Initialize result tensor
#     is_in_target = torch.zeros(len(particles), device=device, dtype=torch.bool)
    
#     # Check each particle
#     for i, (x, y) in enumerate(particles_cpu):
#         point = Point(x, y)
        
#         # Check if point is in target area
#         # in_target = any(polygon.contains(point) for polygon in target.values())
        
#         # Check if point is in any shelf area
#         in_shelf = any(polygon.contains(point) for polygon in shelf.values())
        
#         # Point is exclusively in target area if it's in target and not in shelf
#         is_in_target[i] = in_target and not in_shelf
    
#     return is_in_target


if __name__ == "__main__":
    # 测试参数
    num_particles = 100_000
    test_observation = (50.0, 50.0)
    
    # PyTorch CPU测试
    pf = TorchParticleFilter(num_particles, device='cpu')
    true_trajectory = [
        (50 + 2*t, 50 + 1.5*t) for t in range(100)
    ]

    # 运行滤波过程
    estimates = []
    for step, true_pos in enumerate(true_trajectory):
        # 生成带噪声的观测（在GPU上执行）
        noisy_obs = (
            true_pos[0] + torch.randn(1, device=device).item()*3,
            true_pos[1] + torch.randn(1, device=device).item()*3
        )
        
        # 执行更新步骤
        pf.update(
            observation=noisy_obs,
            system_input=(2 + torch.randn(1, device=device).item()*0.5, 1.5 + torch.randn(1, device=device).item()*0.5),
            system_noise_scale=1.0,  # 系统噪声参数
            obs_noise_scale=3.0      # 观测噪声参数
        )
        
        # 获取估计结果（自动在GPU计算）
        estimate = pf.estimate.cpu().numpy()  # 传回CPU处理
        estimates.append(estimate)
        
        # 可选：每10步可视化
        if step % 10 == 0:
            particles_cpu = pf.particles.cpu().numpy()
            plt.scatter(particles_cpu[:,0], particles_cpu[:,1], 
                    c='blue', alpha=0.1, s=1)
            plt.scatter(*estimate, c='red', s=50, label='Estimate')
            plt.scatter(*true_pos, c='green', s=50, label='Ground Truth')
            plt.legend()
            plt.savefig(f"./fig/particle_filter/{step}.png")
            plt.close()