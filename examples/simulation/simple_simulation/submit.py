"""Script for submitting training jobs to the BMRC cluster."""

import os

def write_job_script(after: str = None, run: int = 0):
    with open("job.sh", "w") as file:
        name = f"ex_{run}"
        file.write("#!/bin/bash\n")
        file.write(f"#SBATCH -J {name}\n")
        file.write(f"#SBATCH -o outputs/{name}.out\n")
        file.write("#SBATCH -p gpu_short\n")
        file.write(f"#SBATCH --gres gpu:2 --constraint 'a100|v100'\n")
        file.write("#SBATCH --parsable\n")
        if after:
            file.write(f"#SBATCH -d afterany:{after}\n")
        file.write("source activate osld\n")
        file.write(f"python 2_train_tokenizer.py\n")


os.makedirs("outputs", exist_ok=True)
write_job_script(after=None, run=0)
job_id = str(int(os.popen("sbatch job.sh").read()))
print(f"Submitted job {job_id}")
os.system("rm job.sh")

# /well/woolrich/users/vxw496/projects/osl-foundation/examples/real_data/camcan_rest/outputs
