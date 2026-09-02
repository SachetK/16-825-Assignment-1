import argparse
from pathlib import Path

import imageio
import numpy as np
import pytorch3d
import torch

from starter.render_generic import load_rgbd_data
from starter.utils import get_points_renderer, unproject_depth_image


def render_turntables(point_clouds, renderer, num_frames):
    """Render one rotating image sequence for each point cloud."""
    images = [[] for _ in range(len(point_clouds))]
    device = point_clouds.device
    for azim in np.linspace(0, 360, num_frames, endpoint=False):
        R, T = pytorch3d.renderer.look_at_view_transform(
            dist=6, elev=15, azim=azim, up=((0, -1, 0),), device=device
        )
        camera = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R.expand(len(point_clouds), -1, -1),
            T=T.expand(len(point_clouds), -1),
            device=device,
        )
        panels = renderer(point_clouds=point_clouds, cameras=camera)[..., :3]
        panels = panels.detach().cpu().numpy()
        for index, panel in enumerate(panels):
            images[index].append((panel.clip(0, 1) * 255).astype(np.uint8))
    return images


def main(output_dir="images", num_frames=10):
    data = load_rgbd_data()
    points_1, colors_1 = unproject_depth_image(image=torch.from_numpy(data['rgb1']),
                                               mask=torch.from_numpy(data['mask1']),
                                               depth=torch.from_numpy(data['depth1']),
                                               camera=data['cameras1'])
    points_2, colors_2 = unproject_depth_image(image=torch.from_numpy(data['rgb2']),
                                               mask=torch.from_numpy(data['mask2']),
                                               depth=torch.from_numpy(data['depth2']),
                                               camera=data['cameras2'])
    points_union = torch.cat([points_1, points_2], dim=0)
    colors_union = torch.cat([colors_1, colors_2], dim=0)

    point_clouds = pytorch3d.structures.Pointclouds(
        points=[points_1, points_2, points_union],
        features=[colors_1, colors_2, colors_union],
    )

    points_renderer = get_points_renderer(
        image_size=256,
        radius=0.01,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [
        output_dir / "pointcloud_1.gif",
        output_dir / "pointcloud_2.gif",
        output_dir / "pointcloud_union.gif",
    ]
    for output_path, images in zip(
        output_paths, render_turntables(point_clouds, points_renderer, num_frames)
    ):
        imageio.mimsave(output_path, images, duration=0.1, loop=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='images')
    parser.add_argument('--num_frames', type=int, default=10)
    args = parser.parse_args()
    main(output_dir=args.output_dir, num_frames=args.num_frames)
