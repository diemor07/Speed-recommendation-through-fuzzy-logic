# Fuzzy Car Speed Simulator

# This is a simple fuzzy logic-based car speed simulator.

class FuzzyCarSpeedSimulator:
    def __init__(self):
        self.speed = 0

    def update_speed(self, throttle, brake):
        # Simple fuzzy logic for speed control
        if throttle > brake:
            self.speed += throttle - brake
        else:
            self.speed -= brake - throttle

        # Speed limits
        if self.speed > 120:
            self.speed = 120
        elif self.speed < 0:
            self.speed = 0

        return self.speed

# Example usage:
if __name__ == '__main__':
    simulator = FuzzyCarSpeedSimulator()
    print(f'Initial speed: {simulator.speed}')
    print(f'Updating speed with throttle=30 and brake=10: {simulator.update_speed(30, 10)}')