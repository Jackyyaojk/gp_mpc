import os
import time

i = 0
start = time.time()
os.system("python3 control.py & rosbag play data/mpc_twogoal_shorter.bag && rosnode kill -a")
while i<4:
	if (time.time() - start) > 80:
		start = time.time()
		i+=1
		os.system("python3 control.py & rosbag play data/mpc_twogoal_shorter.bag && rosnode kill -a")
