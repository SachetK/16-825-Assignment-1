import imageio
import torch

from starter.full_view import render_full_view
from starter.utils import load_cow_mesh


def texture_cow(cow_path, color1, color2):
    vertices, faces = load_cow_mesh(cow_path)

    z = vertices[:, 2].unsqueeze(-1)

    z_min = torch.min(z)
    z_max = torch.max(z)

    alpha = ((z - z_min) / (z_max - z_min))
    color = ((1 - alpha) * torch.tensor(color1, dtype=vertices.dtype)
             + alpha * torch.tensor(color2, dtype=vertices.dtype))

    images = render_full_view(
        vertices=vertices.unsqueeze(0),
        faces=faces.unsqueeze(0),
        color=color
    )

    duration = 1000 // 15
    imageio.mimsave('images/textured_cow.gif', images, duration=duration, loop=0)


if __name__ == '__main__':
    texture_cow('data/cow.obj', color1=[0, 0.5, 1], color2=[1, 0.6, 0])