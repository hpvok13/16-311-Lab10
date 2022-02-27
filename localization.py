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


def circular_distance(x, y, ticks):
    forward = np.absolute(x - y)
    backward = np.absolute(x - (y - ticks))
    return np.minimum(forward, backward)


class CircleLocalizer:
    MOVE_VARIANCE_MULTIPLIER = 0.1
    BLOCK_VARIANCE_MULTIPLIER = 1.0
    NO_BLOCK_VARIANCE_MULTIPLIER = 1.0  # blocks are small, so we will have no block more often. Thus we want a higher variance for this.

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
        distance = circular_distance(mu, x, TICKS)
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

        mult = (
            self.BLOCK_VARIANCE_MULTIPLIER
            if block
            else self.NO_BLOCK_VARIANCE_MULTIPLIER
        )
        variance = SECTOR_TICKS * mult

        location_kernel = np.fromfunction(
            lambda x: self.circular_gaussian_pdf(0, variance, x), (TICKS,)
        )
        distribution = self.conv_circ(locations, location_kernel)
        distribution = distribution / np.linalg.norm(distribution, ord=1)
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
    ## Simulation
    import matplotlib.pyplot as plt

    def block_exists_sim(deg):
        block = False
        loc = deg / (360 / NUM_SECTORS)
        loc_frac = loc - int(loc)
        if abs(loc_frac - 0.5) < 0.3:
            block = BLOCK_LOCATIONS[int(deg / (360 / NUM_SECTORS))] == 1
        if abs(loc_frac - 0.5) < 0.1:
            block = False
        return block

    def block_exists_sim_wide(deg):
        return BLOCK_LOCATIONS[int(deg / (360 / NUM_SECTORS))] == 1

    cl = CircleLocalizer()

    y = [block_exists_sim(deg) for deg in range(0, 360)]
    y2 = [block_exists_sim_wide(deg) for deg in range(0, 360)]
    plt.plot(y)
    plt.plot(y2)
    plt.show()

    # Simulate perfect movement
    move = 1
    deg = 0
    for i in range(100000):
        if i == 360 * 2:
            BLOCK_LOCATIONS = np.roll(BLOCK_LOCATIONS, 5)
            deg += (360 / NUM_SECTORS) * 5
        if i == 360 * 5:
            break

        deg = (deg + move) % 360

        cl.move(move)
        cl.update(block_exists_sim(deg))

        dist = np.round(circular_distance(cl.location_degrees(), deg, 360), 1)
        if dist > 10:
            print("i", i)
            print(dist, " | ", np.round(np.abs(cl.sectors()), 1))
            print("---")
        if i % 100 == 0:
            print("i", i)
            print(dist, " | ", np.round(np.abs(cl.sectors()), 1))
            print("---")

    print_sectors(np.round(cl.sectors5(), 3), 5)
