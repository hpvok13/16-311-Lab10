#!/usr/bin/env python3

import time
import traceback
import numpy as np

import brickpi3

from localization import *
from odometry import *

BP = brickpi3.BrickPi3()

LEFT_MOTOR_PORT = BP.PORT_C
RIGHT_MOTOR_PORT = BP.PORT_B
LIGHT_SENSOR_PORT = BP.PORT_2
ULTRASONIC_SENSOR_PORT = BP.PORT_1
LOOP_DURATION = 0.01

LIGHT_THRESHOLD = 2450 # above this is black
BLOCK_THRESHOLD = 35 # below this means there is a block

CIRCLE_DIAMETER = 25
CIRCLE_CIRCUMFERENCE = np.pi * CIRCLE_DIAMETER
LEFT_POWER = 35
RIGHT_POWER = 20

# Follow on the outside of the black line
BLACK_POWERS = (15, 30)
WHITE_POWERS = (30, 15)

GOAL_SECTOR = 5

def set_motor_powers(powers):
    BP.set_motor_power(LEFT_MOTOR_PORT, powers[0])
    BP.set_motor_power(RIGHT_MOTOR_PORT, powers[1])

# motor_powers = np.array((25.0, 15.0))

# LINE_P_MULTIPLIER = 0.01


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
        iteration = 0
        total_degrees_traveled = 0
        prev_state = transform(0, 0, 0)
        state = transform(0, 0, 0)
        while True:
            start_time = time.perf_counter()

            # Read from sensors
            light = BP.get_sensor(LIGHT_SENSOR_PORT)
            ultrasonic = BP.get_sensor(ULTRASONIC_SENSOR_PORT)

            # Follow line
            black = light > LIGHT_THRESHOLD
            powers = None
            if black:
                powers = BLACK_POWERS
            else:
                powers = WHITE_POWERS
            set_motor_powers(powers)

            # Localization
            iteration += 1
            if iteration % 10 == 0:
                # Update state
                encoders_diff = er.read_encoders_diff()
                state = update_state(state, encoders_diff)
                distance_traveled_inches = np.linalg.norm(state_to_pos(state) - state_to_pos(prev_state))
                distance_traveled_frac = distance_traveled_inches / CIRCLE_CIRCUMFERENCE
                distance_traveled_degrees = distance_traveled_frac * 360
                total_degrees_traveled = (total_degrees_traveled + distance_traveled_degrees) % 360
                prev_state = state

                # Localize
                block = ultrasonic < BLOCK_THRESHOLD
                cl.move(distance_traveled_degrees)
                cl.update(block)

                # Stop if confident at destination
                max_location = np.argmax(cl.probabilities)
                max_location_p = cl.probabilities[max_location]
                prediction = max_location / (360/16)
                if abs(prediction - (GOAL_SECTOR + 0.5)) < 0.1 and max_location_p > 0.2:
                    set_motor_powers([0, 0])
                    break

                print(np.round(cl.probabilities, 3))
                # print(ultrasonic)
                print('degrees:', total_degrees_traveled)
                print('prediction:', prediction)


            duration = time.perf_counter() - start_time
            time.sleep(max(0, LOOP_DURATION - duration))

    except KeyboardInterrupt:
        print("Interrupted")
        BP.reset_all()
    except Exception as e:
        traceback.print_exc()
        print("Encountered Exception:", e)
        BP.reset_all()
