"""
Sample code to render various representations.

Usage:
    python -m starter.render_generic --render point_cloud  # 5.1
    python -m starter.render_generic --render parametric  --num_samples 100  # 5.2
    python -m starter.render_generic --render implicit  # 5.3
"""
import argparse
import pickle

import imageio
import numpy as np
import pytorch3d
import torch
import mcubes

from starter.utils import get_device, get_mesh_renderer, get_points_renderer


def load_rgbd_data(path="data/rgbd_data.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def render_bridge(
    point_cloud_path="data/bridge_pointcloud.npz",
    image_size=256,
    background_color=(1, 1, 1),
    device=None,
):
    """
    Renders a point cloud.
    """
    if device is None:
        device = get_device()
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )
    point_cloud = np.load(point_cloud_path)
    verts = torch.Tensor(point_cloud["verts"][::50]).to(device).unsqueeze(0)
    rgb = torch.Tensor(point_cloud["rgb"][::50]).to(device).unsqueeze(0)
    point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=rgb)
    R, T = pytorch3d.renderer.look_at_view_transform(4, 10, 0)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(point_cloud, cameras=cameras)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    return rend

def render_sphere(image_size=256, num_samples=200, device=None):
    """
    Renders a sphere using parametric sampling. Samples num_samples ** 2 points.
    """

    if device is None:
        device = get_device()

    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, np.pi, num_samples)
    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(phi, theta)

    x = torch.sin(Theta) * torch.cos(Phi)
    y = torch.cos(Theta)
    z = torch.sin(Theta) * torch.sin(Phi)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    sphere_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(T=[[0, 0, 3]], device=device)
    renderer = get_points_renderer(image_size=image_size, device=device)
    rend = renderer(sphere_point_cloud, cameras=cameras)
    return rend[0, ..., :3].cpu().numpy()

def build_torus_point_cloud(num_samples=200, R=3.0, r=2.0, device=None):
    if device is None:
        device = get_device()

    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, 2 * np.pi, num_samples)
    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(phi, theta, indexing="ij")

    x = (R + r * torch.cos(Theta)) * torch.cos(Phi)
    y = (R + r * torch.cos(Theta)) * torch.sin(Phi)
    z = r * torch.sin(Theta)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1).unsqueeze(0)
    color = (points - points.min()) / (points.max() - points.min()).unsqueeze(0)

    return pytorch3d.structures.Pointclouds(
        points=points, features=color,
    ).to(device)


def render_torus(image_size=256, num_samples=200, R=3.0, r=2.0, device=None):
    torus_point_cloud = build_torus_point_cloud(
        num_samples=num_samples, R=R, r=r, device=device
    )
    R, T = pytorch3d.renderer.look_at_view_transform(
        dist=3, elev=45, azim=0, degrees=True, device=torus_point_cloud.device
    )
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R, T=T, device=torus_point_cloud.device
    )
    renderer = get_points_renderer(image_size=image_size, device=torus_point_cloud.device)
    rend = renderer(torus_point_cloud, cameras=cameras)
    return rend[0, ..., :3].cpu().numpy()


def render_point_cloud_turntable(
    point_cloud,
    image_size=256,
    num_frames=36,
    dist=3,
    elev=25,
):
    renderer = get_points_renderer(image_size=image_size, device=point_cloud.device)
    frames = []
    for azim in np.linspace(0, 360, num_frames, endpoint=False):
        R, T = pytorch3d.renderer.look_at_view_transform(
            dist=dist, elev=elev, azim=azim, degrees=True, device=point_cloud.device
        )
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, device=point_cloud.device
        )
        rend = renderer(point_cloud, cameras=cameras)[0, ..., :3].detach().cpu().numpy()
        frames.append((rend.clip(0, 1) * 255).astype(np.uint8))
    return np.stack(frames)


def render_torus_turntable(
    image_size=256,
    num_samples=200,
    num_frames=36,
    R=3.0,
    r=2.0,
    device=None,
):
    torus_point_cloud = build_torus_point_cloud(
        num_samples=num_samples, R=R, r=r, device=device
    )
    return render_point_cloud_turntable(
        torus_point_cloud,
        image_size=image_size,
        num_frames=num_frames,
        dist=3,
        elev=25,
    )

def render_sphere_mesh(image_size=256, voxel_size=64, device=None):
    if device is None:
        device = get_device()
    min_value = -1.1
    max_value = 1.1
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    voxels = X ** 2 + Y ** 2 + Z ** 2 - 1
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=180)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(mesh, cameras=cameras, lights=lights)
    return rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)

def build_torus_mesh(voxel_size=64, R=0.6, r=0.25, device=None):
    """Extract a torus mesh from its implicit function with marching cubes."""
    if device is None:
        device = get_device()
    min_value = -1.1
    max_value = 1.1
    X, Y, Z = torch.meshgrid(
        [torch.linspace(min_value, max_value, voxel_size)] * 3, indexing="ij"
    )
    voxels = (torch.sqrt(X ** 2 + Y ** 2) - R) ** 2 + Z ** 2 - r ** 2
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    return mesh


def render_mesh_turntable(
    mesh,
    image_size=256,
    num_frames=36,
    dist=3,
    elev=20,
):
    """Render a mesh from evenly spaced azimuths for a 360-degree GIF."""
    device = mesh.device
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    frames = []
    for azim in np.linspace(0, 360, num_frames, endpoint=False):
        R, T = pytorch3d.renderer.look_at_view_transform(
            dist=dist, elev=elev, azim=azim, device=device
        )
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        frame = renderer(mesh, cameras=cameras, lights=lights)[0, ..., :3]
        frames.append((frame.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8))
    return np.stack(frames)


def render_torus_mesh(image_size=256, voxel_size=64, R=0.6, r=0.25, device=None):
    mesh = build_torus_mesh(voxel_size=voxel_size, R=R, r=r, device=device)
    lights = pytorch3d.renderer.PointLights(
        location=[[0, 0.0, -4.0]], device=mesh.device
    )
    renderer = get_mesh_renderer(image_size=image_size, device=mesh.device)
    R, T = pytorch3d.renderer.look_at_view_transform(
        dist=3, elev=0, azim=180, device=mesh.device
    )
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=mesh.device)
    rend = renderer(mesh, cameras=cameras, lights=lights)
    return rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)


def render_torus_mesh_turntable(
    image_size=256,
    voxel_size=64,
    num_frames=36,
    R=0.6,
    r=0.25,
    device=None,
):
    mesh = build_torus_mesh(voxel_size=voxel_size, R=R, r=r, device=device)
    return render_mesh_turntable(
        mesh, image_size=image_size, num_frames=num_frames, dist=3, elev=20
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render",
        type=str,
        default="point_cloud",
        choices=["point_cloud", "parametric", "implicit"],
    )
    parser.add_argument("--output_path", type=str, default="images/bridge.jpg")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--num_frames", type=int, default=36)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--turntable", action="store_true")
    args = parser.parse_args()
    animate = args.turntable or args.output_path.lower().endswith(".gif")
    if args.render == "point_cloud":
        image = render_bridge(image_size=args.image_size)
    elif args.render == "parametric":
        if animate:
            image = render_torus_turntable(
                image_size=args.image_size,
                num_samples=args.num_samples,
                num_frames=args.num_frames,
                R=0.5,
                r=2.0 / 6.0,
            )
        else:
            image = render_torus(
                image_size=args.image_size,
                num_samples=args.num_samples,
                R=0.5,
                r=2.0 / 6.0,
            )
    elif args.render == "implicit":
        if animate:
            image = render_torus_mesh_turntable(
                image_size=args.image_size, num_frames=args.num_frames
            )
        else:
            image = render_torus_mesh(image_size=args.image_size)
    else:
        raise Exception("Did not understand {}".format(args.render))

    if animate:
        duration = 1000 // args.fps
        imageio.mimsave(args.output_path, image, duration=duration, loop=0)
    else:
        import matplotlib.pyplot as plt

        plt.imsave(args.output_path, image)
