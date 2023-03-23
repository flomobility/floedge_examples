#!/usr/bin/python3

import time
import serial
import click

@click.command()
@click.option("--dev", default='/dev/ttyUSB0', help="device path exaple /dev/ttyUSB0")
@click.option("--baud", default=9600, help="baud rate example 9600")
def main(dev, baud):
    device = serial.Serial(dev, baud)

    while True:
        data = f"{time.time()}".encode()
        print(data)
        device.write(data)
        time.sleep(0.1)

    device.close()

if __name__ == "__main__":
    main()

