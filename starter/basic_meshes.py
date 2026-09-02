import imageio
import numpy as np
import torch

from starter.full_view import render_full_view

def render_tetrahedron():
    vertices = torch.tensor([
        [-0.5, -1 / (2 * np.sqrt(3)), 0.0],
        [0.5, -1 / (2 * np.sqrt(3)), 0.0],
        [0.0, 1 / np.sqrt(3), 0.0],
        [0.0, 0.0, np.sqrt(2 / 3)],
    ], dtype=torch.float32).unsqueeze(0)

    vertices = vertices - vertices.mean(dim=1, keepdim=True)

    faces = torch.tensor([
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3],
    ]).unsqueeze(0)

    tetrahedron_images = render_full_view(vertices=vertices, faces=faces)

    duration = 1000 // 15
    imageio.mimsave('images/tetrahedron.gif', tetrahedron_images, duration=duration, loop=0)

def render_cube():
    vertices = torch.tensor([
        [0, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
        [1, 0, 1],

    ], dtype=torch.float32).unsqueeze(0)

    vertices = vertices - 0.5

    faces = torch.tensor([
        # z = 0
        [0, 1, 2],
        [0, 2, 3],

        # x = 0
        [0, 4, 5],
        [0, 5, 1],

        # y = 1
        [1, 5, 6],
        [1, 6, 2],

        # x = 1
        [3, 2, 6],
        [3, 6, 7],

        # y = 0
        [0, 3, 7],
        [0, 7, 4],

        # z = 1
        [4, 7, 6],
        [4, 6, 5],
    ], dtype=torch.int64).unsqueeze(0)

    cube_images = render_full_view(vertices=vertices, faces=faces)

    duration = 1000 // 15
    imageio.mimsave('images/cube.gif', cube_images, duration=duration, loop=0)

if __name__ == "__main__":
    render_tetrahedron()
    render_cube()