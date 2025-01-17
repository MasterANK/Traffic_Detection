import pygame
import numpy as np

pygame.init()

width, height = 800,600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Standard Traffic Simulation")

#color
white = (255,255,255)
road_color = (50,50,50)
line_color = (255,255,0)

car_widht, car_height = 20, 40
car_speed = 2

lanes = {
    "north" : [],
    "south" : [],
    "east" : [],
    "west" : []
}

class car_factory:
    def __init__(self, lane, car_color):
        self.lane = lane
        self.car_color = car_color

        if lane == "north":
            self.x, self.y = width // 2 - car_widht // 2 - 15, height  # Start from bottom
            self.dx, self.dy = 0, -car_speed  # Move up
            self.width, self.height = car_widht, car_height  # Normal car size

        elif lane == "south":
            self.x, self.y = width // 2 - car_widht // 2 + 15, -car_height  # Start from above screen
            self.dx, self.dy = 0, car_speed  # Move down
            self.width, self.height = car_widht, car_height  # Normal car size

        elif lane == "east":
            self.x, self.y = -car_height, height // 2 - car_widht // 2 - 35  # Start from left
            self.dx, self.dy = car_speed, 0  # Move right
            self.width, self.height = car_height, car_widht  # Swap size for horizontal movement

        elif lane == "west":
            self.x, self.y = width, height // 2 - car_widht // 2 + 15  # Start from right
            self.dx, self.dy = -car_speed, 0  # Move left
            self.width, self.height = car_height, car_widht  # Swap size for horizontal movement
    

    def move(self):
        self.x += self.dx
        self.y += self.dy  

    def draw(self, screen):
        pygame.draw.rect(screen, self.car_color, (self.x, self.y, car_widht, car_height))   


def add_cars():
    for lane in lanes.keys():
        if np.random.random() < 0.3:  # 50% chance a car is added per lane
            car_color = list(np.random.choice(range(256), size=3))
            lanes[lane].append(car_factory(lane, car_color))

running = True
clock = pygame.time.Clock()
frame_count = 0

while running:
    screen.fill(white)

    #Event Manager
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    #Draw intersection roads
    pygame.draw.rect(screen, road_color, (width // 2 - 50, 0, 100, height))  # Vertical road
    pygame.draw.rect(screen, road_color, (0, height // 2 - 50, width, 100))  # Horizontal road

    # Draw center lines
    pygame.draw.line(screen, line_color, (width // 2, 0), (width // 2, height), 3)  # Vertical line
    pygame.draw.line(screen, line_color, (0, height // 2), (width, height // 2), 3)  # Horizontal line

    # Add cars
    if frame_count % 60 == 0:
        add_cars()
    
    for lane in lanes.keys():
        for car in lanes[lane]:
            car.move()
            car.draw(screen)

    pygame.display.flip()  # Update display
    clock.tick(60)  # Limit FPS to 60
    frame_count += 1

print("Closing the simulation")
pygame.quit()
