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
car_speed = 1 #2

lanes = {
    "north" : [],
    "south" : [],
    "east" : [],
    "west" : []
}

stop_positions = {
    "north": height // 2 + 80,
    "south": height // 2 - 100,
    "east": width // 2 - 80,
    "west": width // 2 + 80,
}

class car_factory:
    def __init__(self, lane, car_color):
        self.lane = lane
        self.car_color = car_color
        self.crossed = False
        self.y_sensor = 50
        self.x_sensor = 30

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
        if self.crossed:
            self.x += self.dx
            self.y += self.dy
            return 

        for light in traffic_lights.values():
            if light.lane == self.lane and light.is_red():
                if  (self.lane == "north" and self.y < stop_positions["north"]) or \
                    (self.lane == "south" and self.y > stop_positions["south"]) or \
                    (self.lane == "east" and self.x > stop_positions["east"]) or \
                    (self.lane == "west" and self.x < stop_positions["west"]):
                    self.car_color = (255, 0, 0)
                    return
        if self.near_car(lanes[self.lane]):
            self.car_color = (255,0,0)
            return

        self.x += self.dx
        self.y += self.dy 

        if  (self.lane == "north" and self.y < stop_positions["north"] - 10) or \
                (self.lane == "south" and self.y > stop_positions["south"] + 10) or \
                (self.lane == "east" and self.x > stop_positions["east"] + 10) or \
                (self.lane == "west" and self.x < stop_positions["west"] - 10):
            self.car_color = (0, 255, 0)
            self.crossed = True 
        
    def near_car(self, cars):
        for car in cars:
            if car == self:
                continue
            if self.lane == "north" and 0 < (self.y - car.y) < self.y_sensor and self.x == car.x:
                    return True
            elif self.lane == "south" and 0 < (car.y - self.y) < self.y_sensor and self.x == car.x:
                    return True
            elif self.lane == "west" and 0 < (self.x - car.x) < self.x_sensor and self.y == car.y:
                    return True 
            elif self.lane == "east" and 0 < (car.x - self.x) < self.x_sensor and self.y == car.y:
                    return True
        return False

    def draw(self, screen):
        pygame.draw.rect(screen, self.car_color, (self.x, self.y, car_widht, car_height))   


class Traffic_light:
    def __init__(self, x, y, direction, lane):
        self.x = x
        self.y = y
        self.direction = direction
        self.lane = lane
        self.state = "RED"

    def update(self, current_phase, is_yellow):
        phases = ["north", "south", "east", "west"]
        self.is_yellow = is_yellow
        if is_yellow and phases[current_phase] == self.lane:
            self.state = "YELLOW"
        elif phases[current_phase] == self.lane:
            self.state = "GREEN"
        else:
            self.state = "RED"
    
    def draw(self, screen):
        if self.state == "YELLOW":
            color =  (255, 225, 0)
        elif self.state == "RED":
            color =  (255, 0, 0)
        else:
            color =  (0, 255, 0)
        pygame.draw.circle(screen, color, (self.x, self.y), 10)
    
    def draw_arrow(self, screen):
        arrow_color = (0, 0, 0)  # Black arrow
        if self.lane == "north":
            pygame.draw.polygon(screen, arrow_color, [(self.x + 0, self.y + 15), (self.x - 10, self.y + 30), (self.x + 10, self.y + 30)])
        elif self.lane == "south": 
            pygame.draw.polygon(screen, arrow_color, [(self.x + 0, self.y + 30), (self.x - 10, self.y + 15), (self.x + 10, self.y + 15)])
        elif self.lane == "east":
            pygame.draw.polygon(screen, arrow_color, [(self.x - 15, self.y - 30), (self.x - 15, self.y - 15), (self.x + 10, self.y - 20)])
        elif self.lane == "west":
            pygame.draw.polygon(screen, arrow_color, [(self.x + 15, self.y - 30), (self.x + 15, self.y - 15), (self.x - 10, self.y - 20)])
    def is_red(self):
        return self.state in ["RED", "YELLOW"]
    
def count_stopped_cars():
    stopped_cars = {lane: 0 for lane in lanes.keys()}
    
    for lane in lanes.keys():
        for car in lanes[lane]:
            if not car.crossed and traffic_lights[lane].is_red():
                stopped_cars[lane] += 1
    
    return max(stopped_cars.values())  # Get max stopped cars in a single lane

class TrafficController:
    def __init__(self):
        self.phase = 0 # 0 = North ; 1 = South ; 2 = East ; 3 = West
        self.timer = 0
        self.switch_time = 4 * 60
        self.yellow_time = 1 * 60
        self.is_yellow = False
    def update(self):
        self.timer += 1
        stopped_cars = count_stopped_cars()
        self.switch_time = max(4 * 60, (stopped_cars / 2) * 60)
        if self.switch_time > 10*60:
            self.switch_time = 10*60
            print("max limit", self.switch_time) 
        if self.is_yellow:
            if self.timer >= self.yellow_time:
                self.timer = 0
                self.is_yellow = False
                self.phase = (self.phase + 1) % 4
        else:
            if self.timer >= self.switch_time:
                self.timer = 0
                self.is_yellow = True

traffic_lights = {
    "north" : Traffic_light(width // 2 + 70, height//2 + 70, "vertical", "north"),
    "south" : Traffic_light(width // 2 - 70, height//2 + 70, "vertical", "south"),
    "east" : Traffic_light(width // 2 - 70, height//2 - 70, "horizontal", "east"),
    "west" : Traffic_light(width // 2 + 70, height//2 - 70, "horizontal", "west")
}

def add_cars():
    for lane in lanes.keys():
        if np.random.random() < 0.5:  # 50% chance a car is added per lane
            car_color = list(np.random.choice(range(256), size=3))
            lanes[lane].append(car_factory(lane, car_color))

running = True
clock = pygame.time.Clock()
frame_count = 0

Traffic_Controller = TrafficController()

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

    for test in stop_positions:
        if test == "north" or test == "south":
            if test == "north":
                pygame.draw.circle(screen, (225, 225, 0), (width // 2, stop_positions[test]), 10)
            else:
                pygame.draw.circle(screen, (225, 225, 225), (width // 2, stop_positions[test]), 10)
        else:
            pygame.draw.circle(screen, (255, 225, 0), (stop_positions[test], height // 2), 10)

    # Add cars
    if frame_count % 60 == 0:
        add_cars()
    
    for lane in lanes.keys():
        lanes[lane] = [ncar for ncar in lanes[lane] if ncar.x > -100 and ncar.x < width + 100 and ncar.y > -100 and ncar.y < height + 100] 
        for car in lanes[lane]:
            car.move()
            car.draw(screen)

    Traffic_Controller.update()

    for light in traffic_lights.values():
        light.update(Traffic_Controller.phase, Traffic_Controller.is_yellow)
        light.draw(screen)
        light.draw_arrow(screen)

    pygame.display.flip()  # Update display
    clock.tick(60)  # Limit FPS to 60
    frame_count += 1

print("Closing the simulation")
pygame.quit()
