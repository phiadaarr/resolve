import argparse
import datetime
import os

from configparser import ConfigParser


# Writes the cluster files into the output directory that is given in the config file

# Put cluster scripts into directory of config file and run resolve from there

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")

    parser.add_argument("--qname")
    parser.add_argument("--h_cpu", help="e.g. '48:00:00' for 48 hours.")

    parser.add_argument("--pe", help="Parallel environment, e.g. 'mpi-20'"
        "On hilbert allowed values are 'sm mpi mpi-5 mpi-10 mpi-20 mpi-40 mpi-128 mpi-256'. "
        "The number specifies how many threads shall be run per node.", 
        default="sm")
    parser.add_argument("--total-threads", type=int, default=1, help="Total number of threads")
    parser.add_argument("--mpi-np", type=int, default=1, help="Number of tasks. Only present for mpi environments")
    parser.add_argument("--mem", help="Memory per task. Must end with 'G', e.g. '8G'")

    parser.add_argument("--job-prefix", default="")

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    cfg = ConfigParser()
    cfg.read(args.config_file)
    total_iterations = cfg["optimization"].getint("total iterations")

    cfg_dir, cfg_file = os.path.split(args.config_file)

    timestamp = '{:%Y%m%d_%H%M%S}_'.format(datetime.datetime.now())

    total_iterations = cfg["optimization"].getint("total iterations")
    cluster_files = []

    previous_jobname = None
    for iteration in range(total_iterations):
        cluster_file = os.path.join(cfg_dir, f"{iteration}.clusterfile")
        cluster_files.append(cluster_file)
        jobname = f'{args.job_prefix}{iteration}{timestamp}'

        # Assemble main resolve call
        main_call = ""
        if "mpi" in args.pe:
            assert args.np % args.total_threads == 0
            j = args.np // args.total_threads
            main_call += "mpirun -np {args.np} "
            pe_parts = args.pe.split("-")
            if len(pe_parts) == 2:
                assert pe_parts[0] == "mpi"
                threads_per_node = int(pe_parts[1])
                assert threads_per_node % j == 0
                processes_per_node = threads_per_node // j
                main_call += "-ppn {processes_per_node} " 
            else:
                assert "mpi" in args.pe
        elif "sm" == args.pe:
            assert "mpi" not in args.pe
            j = args.total_threads
        else:
            raise ValueError(f"--pe {args.pe} not known")

        main_call += f"resolve -j{j} --resume --terminate {iteration} {cfg_file} "
        if args.profile_only:
            main_call += "--profile-only "
        # /Assemble main resolve call

        assert args.mem[-1] == "G"
        mem = float(args.mem[:-1])
        mem /= j  # convert from memory per task to memory per slot
        mem = f"{mem:.2f}G"

        if previous_jobname is not None:
            previous = f"#$ -hold_jid {previous_jobname}"
        else:
            previous = ""



        s = f'''#$ -e sge/{jobname}.e
#$ -o sge/{jobname}.o
#$ -l h_vmem={mem}
#$ -l h_cpu={args.h_cpu}
#$ -l qname={args.qname}
#$ -cwd
#$ -pe {args.pe} {args.total_threads}
#$ -N {jobname}
{previous}

setenv MPLBACKEND agg
source venv{args.qname}/bin/activate

setenv OMP_NUM_THREADS {j}

{main_call}
'''
        with open(cluster_file, 'w') as f:
            f.write(s)
        previous_jobname = jobname

    if not args.dry_run:
        for cluster_file in cluster_files:
            direc, fname = os.path.split(cluster_files)
            os.system(f'cd {direc}; qsub {fname}')

if __name__ == '__main__':
    main()
