import numpy as np
from numba import njit, prange
import time
import random
import pygame
from pygame.time import Clock
import cv2

file_type: str = "mp4v"
file_name: str = "conways_game_of_life.mp4"
fps: int = 30
rz, cz = 640, 480  # row size and column size of the grid

pygame.init()


clock = Clock()
# Example usage
rz, cz = 480, 480
ww, wh = int(rz * 2), int(cz * 2)


def create_window(ww, wh):
    return pygame.display.set_mode((ww, wh))


def create_grids(rz, cz, ww, wh):
    grid = np.zeros((rz, cz), dtype=int)
    cg = np.zeros((ww, wh, 3), dtype=np.uint8)
    # for the inital state of the game, we set a ton of the pixels to alive at random
    for i in range(int((rz * cz) * 0.95)):
        grid[random.randint(0, rz - 1), random.randint(0, cz - 1)] = random.randint(
            0, 1
        )
        cg[random.randint(0, ww - 1), random.randint(0, wh - 1)] = [
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        ]
    return grid, cg


@njit
def get_neighbors(grid, r, c, rz, cz):
    r_min, r_max = max(0, r - 1), min(rz, r + 2)
    c_min, c_max = max(0, c - 1), min(cz, c + 2)
    v = grid[r, c]
    red, blue, green = 0, 0, 0
    for i in range(r_min, r_max):
        for j in range(c_min, c_max):
            red += grid[i, j]
            if (i == r_min or i == r_max - 1) and (j == c_min or j == c_max - 1):
                blue += grid[i, j]
    green = red - blue - v
    return red * 28 * v, green * 63, blue * 63, red - v, v


@njit
def get_new_value(v, ln):
    return 1 if v == 0 and ln == 3 else (0 if (v == 1 and (ln < 2 or ln > 3)) else v)


@njit(parallel=True)
def get_grids(grid, cg, rz, cz, ww, wh):
    new_grid = np.empty_like(grid)
    new_cg = np.empty_like(cg)
    color_multiplier = 0.001
    for r in prange(rz):
        r2 = rz + r
        r3 = ww - r - 1
        for c in prange(cz):
            c2 = cz + c
            c3 = wh - c - 1
            red, green, blue, ln, v = get_neighbors(grid, r, c, rz, cz)
            new_v = get_new_value(v, ln)
            if new_v == 0:
                red, green, blue = blue, red, green
                if v == 0:
                    red = min(red - color_multiplier, 254)
                    green = max(green - color_multiplier, 0)
                    blue = max(green - color_multiplier, 0)
            else:
                if v == 1:
                    red, green, blue = green, blue, red
                    red = max(red + color_multiplier, 0)
                    green = min(green + color_multiplier, 254) 
                    blue = min(blue + color_multiplier, 254) 
            new_grid[r, c] = new_v
            new_cg[r, c] = [red, green, blue]
            new_cg[r, c] = (new_cg[r, c] + cg[r, c]) / 2
            new_cg[r, c2] = [green, blue, red]
            new_cg[r, c2] = (new_cg[r, c2] + cg[r, c2]) / 2
            new_cg[r2, c] = [blue, red, green]
            new_cg[r2, c] = (new_cg[r2, c] + cg[r2, c]) / 2
            new_cg[r2, c2] = [red, blue, green]
            new_cg[r2, c2] = (new_cg[r2, c2] + cg[r2, c2]) / 2
            

    return new_grid, new_cg


running = True


def handle_events():
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            return False
    return True


def update_window(window, cg):
    pygame.surfarray.blit_array(window, cg)
    pygame.display.flip()


def delay(fps):
    clock.tick(fps)


# create the grid and the corresponding color grid
grid, cg = create_grids(rz, cz, ww, wh)
window = create_window(ww, wh)
fourcc = cv2.VideoWriter_fourcc(*file_type)
video_writer = cv2.VideoWriter(file_name, fourcc, fps, (ww, wh))

while running:
    # not used right now, but will do in the future.
    x, y = pygame.mouse.get_pos()
    grid, cg = get_grids(grid, cg, rz, cz, ww, wh)
    running = handle_events()
    update_window(window, cg)
    # get the new screen surface and write it to the file
    frame = pygame.surfarray.array3d(window)
    frame = np.rot90(frame, 3)
    frame = np.flipud(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    video_writer.write(frame)
    delay(fps)

video_writer.release()
pygame.quit()
