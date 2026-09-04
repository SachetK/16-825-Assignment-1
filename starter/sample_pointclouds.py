import argparse

import imageio
import pytorch3d
import torch

from starter.render_generic import render_point_cloud_turntable
from starter.utils import load_cow_mesh, get_device


def sample_pointcloud(num_points=100, obj_path='data/cow.obj', color=torch.tensor([0.7, 0.7, 1]), device=None):
    if device is None:
        device = get_device()

    vertices, faces = load_cow_mesh(obj_path)
    vertices = vertices.to(device)
    faces = faces.to(device)

    # Get triangle vertices: (F, 3, 3)
    triangles = vertices[faces]

    # Compute triangle areas
    v0 = triangles[:, 0]
    v1 = triangles[:, 1]
    v2 = triangles[:, 2]

    areas = 0.5 * torch.linalg.norm(
        torch.cross(v1 - v0, v2 - v0, dim=1),
        dim=1
    )

    # Sample triangles proportional to surface area
    face_indices = torch.multinomial(
        areas,
        num_points,
        replacement=True
    )

    sampled_triangles = triangles[face_indices]

    r1 = torch.rand(num_points)
    r2 = torch.rand(num_points)

    sqrt_r1 = torch.sqrt(r1)

    w0 = 1 - sqrt_r1
    w1 = sqrt_r1 * (1 - r2)
    w2 = sqrt_r1 * r2

    points = (
            w0[:, None] * sampled_triangles[:, 0]
            + w1[:, None] * sampled_triangles[:, 1]
            + w2[:, None] * sampled_triangles[:, 2]
    )

    color = color.to(device=device, dtype=points.dtype)
    features = color.unsqueeze(0).expand(num_points, -1)

    point_cloud = pytorch3d.structures.Pointclouds(
        points=points.unsqueeze(0),
        features=features.unsqueeze(0),
    )

    return point_cloud


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_points", type=int, default=100)
    parser.add_argument("--obj_path", type=str, default='data/cow.obj')
    parser.add_argument("--image_size", type=int, default=1080)
    parser.add_argument("--output_path", type=str, default='sample_pointcloud.gif')
    parser.add_argument("--fps", type=int, default=15)

    args = parser.parse_args()

    point_cloud = sample_pointcloud(num_points=args.num_points, obj_path=args.obj_path)
    image = render_point_cloud_turntable(point_cloud=point_cloud, image_size=args.image_size)

    duration = 1000 // args.fps
    imageio.mimsave(args.output_path, image, duration=duration, loop=0)