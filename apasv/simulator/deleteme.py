import pygame
import numpy as np


class drawer:
    def __init__(self):
        background_colour = (255, 255, 255)
        (width, height) = (300, 200)
        self.screen = pygame.display.set_mode([width, height])
        pygame.display.set_caption("Tutorial 1")
        self.screen.fill(background_colour)
        pygame.display.flip()

    def update(self):
        rand_color = np.random.rand(3, 1) * 255
        background_color = (rand_color[0], rand_color[1], rand_color[2])
        self.screen.fill(background_color)
        pygame.display.flip()


if __name__ == "__main__":
    mydrawer = drawer()

    running = True
    mydrawer.update()
    while running:
        # mydrawer.update()
        running = True
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         running = False
