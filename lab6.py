#!/usr/bin/env python3

import sys
import time
import traceback
import numpy as np

import brickpi3

from localization import *
from odometry import *

BP = brickpi3.BrickPi3()

GOAL_SECTOR = 1

GOAL_SECTOR += (
    0.5 + 0.5
)  # 0.5 for the center of the sector, 0.5 for the block step function offset

LEFT_MOTOR_PORT = BP.PORT_C
RIGHT_MOTOR_PORT = BP.PORT_B
LIGHT_SENSOR_PORT = BP.PORT_2
ULTRASONIC_SENSOR_PORT = BP.PORT_1
LOOP_DURATION = 0.01

LIGHT_THRESHOLD = 2500  # above this is black
BLOCK_THRESHOLD = 35  # below this means there is a block

CIRCLE_DIAMETER = 25
CIRCLE_CIRCUMFERENCE = np.pi * CIRCLE_DIAMETER
LEFT_POWER = 35
RIGHT_POWER = 20

# Use P control to follow the outer edge of the black line
BASE_MOTOR_POWERS = np.array((20.0, 15.0))
LINE_P_MULTIPLIER = 0.05


def set_motor_powers(powers):
    BP.set_motor_power(LEFT_MOTOR_PORT, powers[0])
    BP.set_motor_power(RIGHT_MOTOR_PORT, powers[1])


if __name__ == "__main__":
    try:
        # Setup
        BP.reset_all()
        BP.set_sensor_type(LIGHT_SENSOR_PORT, BP.SENSOR_TYPE.NXT_LIGHT_ON)
        BP.set_sensor_type(ULTRASONIC_SENSOR_PORT, BP.SENSOR_TYPE.NXT_ULTRASONIC)
        time.sleep(0.1)

        er = EncoderReader(BP, LEFT_MOTOR_PORT, RIGHT_MOTOR_PORT)
        cl = CircleLocalizer()

        # Main loop
        total_start_time = time.perf_counter()
        iteration = 0
        total_degrees_traveled = 0
        prev_angle = 0
        state = transform(0, 0, 0)
        while True:
            start_time = time.perf_counter()

            # Read from sensors
            light = BP.get_sensor(LIGHT_SENSOR_PORT)
            ultrasonic = BP.get_sensor(ULTRASONIC_SENSOR_PORT)

            # Follow line
            powers = BASE_MOTOR_POWERS
            light_diff = light - LIGHT_THRESHOLD
            power_add = None
            if light_diff < 0:
                power_add = np.array((1.0, -0.1))
            else:
                power_add = np.array((-0.5, 1.0))
            powers = powers + abs(light_diff) * LINE_P_MULTIPLIER * power_add
            set_motor_powers(powers)

            # Localization
            iteration += 1
            if iteration % 1 == 0:
                # Update state
                encoders_diff = er.read_encoders_diff()
                state = update_state(state, encoders_diff)

                # Compute angle difference
                angle = state_to_angle_deg(state)
                # going clockwise, angle difference should be negative
                angle_diff = (angle - prev_angle) % 360
                if angle_diff > 180:
                    angle_diff -= 360
                if angle_diff > 0:
                    continue
                angle_diff = abs(angle_diff)
                prev_angle = angle

                total_degrees_traveled += angle_diff

                # Localize
                block = ultrasonic < BLOCK_THRESHOLD
                cl.move(angle_diff)
                cl.update(block)

                prediction = np.round(cl.location_degrees(), 1) / (360 / 16)
                # Stop if at destination and after 30 seconds
                total_time_elapsed = time.perf_counter() - total_start_time
                if total_time_elapsed > 50.0:
                    if abs(prediction - GOAL_SECTOR) < 0.1:
                        set_motor_powers([0, 0])
                        BP.reset_all()
                        print("done littly fucker")
                        sys.exit(0)

                print(
                    """block: %s
total_time_elapsed: %s
ultrasonic: %s
light: %s
light_diff: %s
powers: %s
angle_diff: %s
degrees: %s
prediction: %s
----"""
                    % (
                        block,
                        total_time_elapsed,
                        ultrasonic,
                        light,
                        light_diff,
                        powers,
                        angle_diff,
                        total_degrees_traveled % 360,
                        prediction,
                    )
                )

            duration = time.perf_counter() - start_time
            time.sleep(max(0, LOOP_DURATION - duration))

    except KeyboardInterrupt:
        print("Interrupted")
        BP.reset_all()
    except Exception as e:
        traceback.print_exc()
        print("Encountered Exception:", e)
        BP.reset_all()
