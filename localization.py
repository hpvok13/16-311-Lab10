import numpy as np

BLOCK_LOCATIONS = np.array((1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0))

NUM_SECTORS = 16
assert len(BLOCK_LOCATIONS) == NUM_SECTORS

TICKS = 3600  # Number of points around a circle to consider
SECTOR_TICKS = TICKS / NUM_SECTORS
# CIRCLE_DIAMETER = 25  # inches
# BLOCK_WIDTH = 6  # inches
# CIRCLE_CIRCUMFERENCE = np.pi * CIRCLE_DIAMETER
# BLOCK_RADIANS = BLOCK_WIDTH / CIRCLE_CIRCUMFERENCE
# BLOCK_TICKS = BLOCK_RADIANS / (2 * np.pi) * TICKS
# NO_BLOCK_TICKS = (TICKS - (NUM_SECTORS * BLOCK_TICKS)) / NUM_SECTORS

EPS = 1e-6


class CircleLocalizer:
    MOVE_VARIANCE_MULTIPLIER = 0.5
    UPDATE_VARIANCE_MULTIPLIER = 2.0

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
        distance = np.minimum(abs(mu - x), abs(mu - (x - TICKS)))
        return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(
            -(1.0 / 2.0) * ((distance / sigma) ** 2)
        )

    # distance in degrees around the circle
    def motion_kernel(self, distance):
        mean = distance
        variance = distance * self.MOVE_VARIANCE_MULTIPLIER
        return np.fromfunction(
            lambda x: self.circular_gaussian_pdf(mean, variance, x), (TICKS,)
        )

    # distance in degrees around the circle
    def move(self, distance_degrees):
        distance = distance_degrees / 360 * TICKS
        if abs(distance) <= EPS:
            return
        kernel = self.motion_kernel(distance)
        convolved = self.conv_circ(self.probabilities, kernel)
        normalized = convolved / np.linalg.norm(convolved, ord=1)
        self.probabilities = normalized

    def block_distribution(self, block: bool):
        locations = np.full(TICKS, 0)
        for tick in range(TICKS):
            sector_size = TICKS / NUM_SECTORS
            sector = int(np.floor(tick / sector_size))
            locations[tick] = BLOCK_LOCATIONS[sector]
        if not block:
            locations = 1 - locations

        variance = SECTOR_TICKS * self.UPDATE_VARIANCE_MULTIPLIER
        
        location_kernel = np.fromfunction(
            lambda x: self.circular_gaussian_pdf(0, variance, x), (TICKS,)
        )
        distribution = self.conv_circ(locations, location_kernel)
        return distribution

    def update(self, block: bool):
        distribution = self.block_distribution(block)
        proportional = self.probabilities * distribution
        normalized = proportional / np.linalg.norm(proportional, ord=1)
        self.probabilities = normalized

    # Sum over the sectors
    def sectors(self):
        split = np.split(self.probabilities, NUM_SECTORS)
        sums = np.sum(split, axis=1)
        return sums

    # Sum over the sectors, each sector has 5 subsections
    def sectors5(self):
        split = np.split(self.probabilities, NUM_SECTORS * 5)
        sums = np.sum(split, axis=1)
        return sums

    def location_degrees(self):
        m = np.argmax(self.probabilities)
        return m / TICKS * 360


def print_sectors(sums, num_per_sector):
    split = np.split(sums, len(sums) / num_per_sector)
    for s in split:
        print(s, end=" | ")
    print()


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
