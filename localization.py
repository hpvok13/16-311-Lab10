import numpy as np

BLOCK_LOCATIONS = np.array((1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0))

NUM_SECTORS = 16
assert len(BLOCK_LOCATIONS) == NUM_SECTORS

TICKS = 360  # Number of points around a circle to consider
CIRCLE_DIAMETER = 25  # inches
BLOCK_WIDTH = 6  # inches
CIRCLE_CIRCUMFERENCE = np.pi * CIRCLE_DIAMETER
BLOCK_RADIANS = BLOCK_WIDTH / CIRCLE_CIRCUMFERENCE
BLOCK_TICKS = BLOCK_RADIANS / (2 * np.pi) * TICKS
NO_BLOCK_TICKS = (TICKS - (NUM_SECTORS * BLOCK_TICKS)) / NUM_SECTORS

EPS = 1e-6


class CircleLocalizer:
    VARIANCE_MULTIPLIER = 1.0/2.0

    def __init__(self) -> None:
        # Initially uniform location probabilities
        self.probabilities = np.full(TICKS, 1.0 / TICKS)

    # Perform circular convolution, correct by the convolution theorem
    def conv_circ(self, signal, kernel):
        """
        signal: real 1D array
        kernel: real 1D array
        signal and ker must have same shape
        """
        return np.real(np.fft.ifft(np.fft.fft(signal) * np.fft.fft(kernel)))

    # computes a gaussian pdf, wrapping around a circle
    def circular_gaussian_pdf(self, mu, sigma, x):
        distance = np.minimum(abs(mu-x), abs(mu-(x-TICKS)))
        return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(
            -(1.0 / 2.0) * ((distance / sigma) ** 2)
        )

    # distance in degrees around the circle
    def motion_kernel(self, distance):
        mean = distance
        # variance = 5
        # variance = 0.01
        variance = distance * self.VARIANCE_MULTIPLIER
        # variance = distance
        return np.fromfunction(
            lambda x: self.circular_gaussian_pdf(mean, variance, x), (TICKS,)
        )

    # distance in degrees around the circle
    def move(self, distance):
        if abs(distance) <= EPS:
            return
        kernel = self.motion_kernel(distance)
        convolved = self.conv_circ(self.probabilities, kernel)
        normalized = convolved / np.linalg.norm(convolved)
        self.probabilities = normalized

    def block_distribution(self, block: bool):    
        locations = np.full(TICKS, 0)
        for tick in range(TICKS):
            # todo take into acount block width
            sector_size = TICKS / NUM_SECTORS
            sector = int(np.floor(tick / sector_size))
            locations[tick] = BLOCK_LOCATIONS[sector]
        variance = BLOCK_TICKS
        if not block:
            locations = 1 - locations
            variance = NO_BLOCK_TICKS
        location_kernel = np.fromfunction(
            lambda x: self.circular_gaussian_pdf(0, variance, x), (TICKS,)
        )
        distribution = self.conv_circ(locations, location_kernel)
        return distribution

    def update(self, block: bool):
        distribution = self.block_distribution(block)
        proportional = self.probabilities*distribution
        normalized = proportional / np.linalg.norm(proportional)
        self.probabilities = normalized


if __name__ == "__main__":
    cl = CircleLocalizer()
    cl.probabilities = np.full(360, 0)
    cl.probabilities[0] = 1
    print(np.round(cl.probabilities, 3))
    # for _ in range(10):
    #     cl.move(1)
    # cl.move(180)
    cl.move(180)
    cl.move(180)
    # for _ in range(360*5):
    #     cl.move(1.0/5.0)
    # cl.move(1)
    # print(cl.probabilities)
    print(np.round(cl.probabilities, 3))
    print(np.argmax(cl.probabilities))
