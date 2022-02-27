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
    # Swap if x > y
    swap = np.greater(x, y)

    x_ = swap * y + (1 - swap) * x
    y_ = swap * x + (1 - swap) * y
    forward = np.absolute(y_ - x_)
    backward = np.absolute((ticks - y_) + x_)
    result = np.minimum(forward, backward)
    return result


class CircleLocalizer:
    MOVE_VARIANCE_MULTIPLIER = 0.1
    BLOCK_VARIANCE_MULTIPLIER = 1.0
    NO_BLOCK_VARIANCE_MULTIPLIER = 2.0  # blocks are small, so we will have no block more often. Thus we want a higher variance for this.

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
        result = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(
            -(1.0 / 2.0) * ((distance / sigma) ** 2)
        )
        return result

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
        normalized = np.maximum(
            normalized, 0
        )  # floating point error in fft can yield tiny negative numbers
        self.probabilities = normalized

    def block_distribution(self, block: bool):
        sector_size = TICKS / NUM_SECTORS
        locations = np.full(TICKS, 0)
        for tick in range(TICKS):
            block_sector = int(
                np.floor(((tick + (sector_size / 2)) % TICKS) / sector_size)
            )
            locations[tick] = BLOCK_LOCATIONS[block_sector]
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
        normalized = np.maximum(
            normalized, 0
        )  # floating point error in fft can yield tiny negative numbers
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

    # Plot simulated block true/false sensor readings
    y = [block_exists_sim(deg) for deg in range(0, 360)]
    y2 = [block_exists_sim_wide(deg) for deg in range(0, 360)]
    plt.plot(y)
    plt.plot(y2)
    plt.title("Simulated block sensor readings")
    plt.show()

    # Plot distributions for weighting placed on robot location when a block is/is not observed
    plt.plot(np.array(range(0, 3600)) / 10, cl.block_distribution(True))
    plt.plot(np.array(range(0, 3600)) / 10, cl.block_distribution(False))
    plt.title("Probability weights given the presence or absence of a block")
    plt.show()

    # Plot the motion kernel used in circular convolution for a 1 degree movement
    # Note: motion_kernel takes ticks as input, so 10 ticks is one degree
    plt.plot(cl.motion_kernel(10))
    plt.title("Motion kernel for circular convolution")
    plt.show()

    ## Simulation
    np.random.seed(123)
    deg = 0
    for i in range(3600):
        # Simulate all of the blocks shifting on the hard course
        if i == 360 * 2:
            BLOCK_LOCATIONS = np.roll(BLOCK_LOCATIONS, 4)
            deg += (360 / NUM_SECTORS) * 4

        # Simulated movement with noise and noisy odometry
        move_real = np.random.rand() * 2 + 1
        move_odometry = np.random.rand() * 2 + 1
        deg = (deg + move_real) % 360

        # Update the localizer
        cl.move(move_odometry)
        cl.update(block_exists_sim(deg))

        # Compute error and print it + probabilities
        error_deg = np.round(circular_distance(cl.location_degrees(), deg, 360), 1)
        if error_deg > 10 or i % 100 == 0:
            print("i", i)
            print(error_deg, " | ", np.round(np.abs(cl.sectors()), 1))
            print(np.round(np.max(cl.sectors5()), 10))
            print("deg", deg)
            print("---")

        # Plot current probabilities
        plt.clf()
        plt.plot(np.array(range(0, 3600)) / 10 / 22.5, cl.probabilities)
        plt.pause(0.05)

    print_sectors(np.round(cl.sectors5(), 3), 5)
    print(np.round(cl.probabilities, 10).tolist())
    print(np.round(np.max(cl.probabilities), 10))
