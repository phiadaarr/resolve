./prepare_environment.sh ~/temp4/cluster_playground/play2 pascal devel NIFTy_8
python3 generate_cluster_files.py  --qname pascal --h_cpu 0:10:00 --pe mpi-5 --mpi-np 6 --total-threads 30 --mem 10G cfgs/cygnusa.cfg  --max-iteration 2 --venv-dir ~/temp4/cluster_playground/play2/

