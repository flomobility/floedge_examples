#!/usr/bin/python3

import signal
import sys

import uvc
import click

def signal_handler(sig, frame):
    print('Sigint received!!')
    sys.exit(0)

@click.command()
@click.option("--name", default="", help="Camera name")
@click.option("--width", default=0, help="image width")
@click.option("--height", default=0, help="image height")
@click.option("--fps", default=0, help="streaming fps")
def main(name, width, height, fps):
    signal.signal(signal.SIGINT, signal_handler)

    # Find device
    camera = None
    for device in uvc.device_list():
        if device["name"] == name:
            camera = device
            break
    if camera == None:
        print("Camera not found!!")
        return

    cap = uvc.Capture(camera["uid"])

    # Check if node is supported
    mode = None
    for mode in cap.available_modes:
        if mode.width == width and mode.height == height and mode.fps == fps and mode.supported == True:
            mode = mode
            break
    if mode == None:
        print("Mode not supported!!")
        return

    cap.frame_mode = mode

    # Capture frames
    while True:
            frame = cap.get_frame_robust()
            print(frame.img.shape)

    cap.close()

if __name__ == "__main__":
    main()
