"""Script for submitting training jobs to the BMRC cluster."""

import os

def write_job_script(after: str = None, run: int = 0):
    with open("job.sh", "w") as file:
        name = f"ex2_{run}"
        file.write("#!/bin/bash\n")
        file.write(f"#SBATCH -J {name}\n")
        file.write(f"#SBATCH -o outputs_2/{name}.out\n")
        file.write("#SBATCH -p gpu_short\n")
        file.write(f"#SBATCH --gres gpu:1 --constraint 'a100|v100'\n")
        file.write("#SBATCH --parsable\n")
        if after:
            file.write(f"#SBATCH -d afterany:{after}\n")
        file.write("source activate osld\n")
        file.write(f"python 2_generate_data.py\n")

os.makedirs("outputs_2", exist_ok=True)
job_id = None
write_job_script(after=job_id, run=0)
job_id = str(int(os.popen("sbatch job.sh").read()))
print(f"Submitted job {job_id}")
os.system("rm job.sh")

