print("Hello world from test_agent_sleeper::.py")
import os
print("job id is", os.environ["SLURM_JOB_ID"])
import time
print("going to sleep")
time.sleep(120)
print("just woke up")