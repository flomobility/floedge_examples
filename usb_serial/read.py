#!/usr/bin/python3

import serial
import click

@click.command()
@click.option("--dev", default='/dev/ttyUSB0', help="device path exaple /dev/ttyUSB0")
@click.option("--baud", default=9600, help="baud rate example 9600")
def main(dev, baud):
    device = serial.Serial(dev, baud)

    while True:
        data = device.readline().decode()
        print(data)

    device.close()

if __name__ == "__main__":
    main()

