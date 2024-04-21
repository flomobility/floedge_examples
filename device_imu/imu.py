from anx_interface import Anx
import click

@click.command()
@click.option("--fps", help="Specifiy fps value.", required = False, default = "100")
def run(fps):
    anx = Anx()
    anx.start_device_imu(fps=, cb=imu_cb)
    anx.wait()
    
def imu_cb(data):
    print(data)
    
if __name__ == "__main__":
    run()
