import imageio
import argparse
import numpy as np

import pytorch3d
import torch

from starter.utils import get_device, get_mesh_renderer, load_cow_mesh

def render_full_view(
        image_size=256,
        fps=15,
        vertices=torch.tensor([[0, 0, 0]]).unsqueeze(0),
        faces=torch.tensor([[0, 0, 0]]).unsqueeze(0),
        color=torch.tensor([0.7, 0.7, 1]),
        device=None,
):
    images = []

    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * color  # (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    # Place a point light in front of the cow.

    for deg in range(0, 360, fps):
        R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=30, azim=deg, degrees=True)

        # Prepare the camera:
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, fov=60, device=device
        )
        lights = pytorch3d.renderer.PointLights(location= -1 * T, device=device)

        rend = renderer(mesh, cameras=cameras, lights=lights)

        image = rend[0, ..., :3].cpu().numpy()
        image = (image * 255).clip(0, 255).astype(np.uint8)

        images.append(image)

    return np.stack(images)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--output_path", type=str, default="images/cow_render.gif")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    vertices, faces = load_cow_mesh(args.cow_path)
    images = render_full_view(
        vertices=vertices.unsqueeze(0),
        faces=faces.unsqueeze(0),
        image_size=args.image_size,
        fps=args.fps
    )
    duration = 1000 // args.fps
    imageio.mimsave(args.output_path, images, duration=duration, loop=0)