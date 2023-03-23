#!/usr/bin/python3

import click

@click.command()
def main():
    import uvc

    devices = uvc.device_list()
    print("Available devices", devices)

    for device in devices:

        try:
            cap = uvc.Capture(device["uid"])
        except uvc.DeviceNotFoundError:
            continue

        print(f"{cap.name}")

        print("\nAvailable modes:")
        for mode in cap.available_modes:
            print(
                f"\tMODE: {mode.width} x {mode.height} @ {mode.fps} ({mode.format_name})"
            )

        cap.close()


if __name__ == "__main__":
    main()
