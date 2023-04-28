from anx_interface import Anx

def imu_cb(data):
    print(data)
    
anx = Anx()
anx.start_device_imu(fps=100, cb=imu_cb)
anx.wait()
