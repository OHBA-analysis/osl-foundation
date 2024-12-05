"""Script for submitting training jobs to the BMRC cluster."""

import os


def write_and_submit_job_script():
    with open("job.sh", "w") as file:
        name = "simulation_example"
        file.write("#!/bin/bash\n")
        file.write(f"#SBATCH -J {name}\n")
        file.write(f"#SBATCH -o outputs/{name}.out\n")
        file.write("#SBATCH -p gpu_short\n")
        file.write(f"#SBATCH --gres gpu:1 --constraint 'a100|v100|rtx8000'\n")
        file.write("source activate osld\n")
        file.write(f"python run_pipeline.py\n")

    os.system("sbatch job.sh")
    os.system("rm job.sh")


os.makedirs("outputs", exist_ok=True)
write_and_submit_job_script()
