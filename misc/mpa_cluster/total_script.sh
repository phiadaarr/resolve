set -ex

CONFIG_FILE=`realpath cfgs/cygnusa_polarization_13360.cfg`
INSTALL_DIR=`dirname $CONFIG_FILE`
QUEUE=pascal
RESOLVE_BRANCH=devel
NIFTY_BRANCH=NIFTy_8

# Possible values for pe 
# Hilbert: sm mpi mpi-5 mpi-10 mpi-20 mpi-40 mpi-128 mpi-256
# Pascal: sm mpi mpi-5 mpi-10 mpi-20 mpi-40

H_CPU=40:00:00 # wall time for one KL iteration
PE=mpi-20 # Number determines threads per node
TOTAL_THREADS=100 # Total number of threads across all MPI tasks
MEM=40G # Memory per Task
MPI_NP=5 # Number of MPI processes

./prepare_environment.sh $INSTALL_DIR $QUEUE $RESOLVE_BRANCH $NIFTY_BRANCH
python3 generate_cluster_files.py  \
	--qname $QUEUE  \
	--venv-dir $INSTALL_DIR \
	--h_cpu $H_CPU \
	--pe $PE \
	--mpi-np $MPI_NP \
	--total-threads $TOTAL_THREADS \
	--mem $MEM \
	$CONFIG_FILE # --max-iteration 2
