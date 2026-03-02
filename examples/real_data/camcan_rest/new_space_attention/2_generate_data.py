import os
import argparse
import math
import numpy as np
from tqdm.auto import tqdm
import pickle

from osl_dynamics.inference import tf_ops
from osl_dynamics.utils.misc import set_random_seed

dev_dir = "/Users/woolrich/dev"
#dev_dir = "/well/woolrich/users/vxw496"
os.chdir(f"{dev_dir}/projects/osl-foundation")
from osl_foundation import load_model

tf_ops.gpu_growth()

def generate_data(args):
    uname = args.uname

    set_random_seed(45)

    # ---------- Load generator ----------

    generator_dir = f"{dev_dir}/results/osl-foundation/camcan_spring23/generator_{uname}"
    generator = load_model(generator_dir)

    # ---------- Generate ---------- #

    n_sessions = 1
    batch_size = 64 
    sample_frequency = 250
    session_labels = np.array_split(
        np.arange(n_sessions), math.ceil(n_sessions / batch_size)
    )
    
    generated_data = []
    lyapunov_data = []
    for labels in tqdm(session_labels, desc="Generating data", total=len(session_labels)):
        gen_data, lyapunov = generator.generate_data(
            n_samples=4*sample_frequency, 
            top_p=0.999,
            batch_size=len(labels),
            extra_labels={"session_id": labels},
            temperature=1,
        )
        generated_data.extend(gen_data)
        lyapunov_data.extend(lyapunov)

    pickle.dump(generated_data, open(f"{generator_dir}/generated_data.pkl", "wb"))
    pickle.dump(lyapunov_data, open(f"{generator_dir}/lyapunov_data.pkl", "wb"))

  
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "uname",
        type=str,
        help="Unique name for this model",
    )
    args = parser.parse_args()

    print("Arguments:")
    print(f"  uname: {args.uname}")

    generate_data(args)

if __name__ == '__main__':
    main()


# example usage:
# generate_data(argparse.Namespace(uname='cTrue_l80'))