docker build -t joint_sa .
nvidia-docker run \
	--rm \
	-v ${PWD}/data:/data \
	-v ${PWD}/codes:/codes \
	--shm-size=8G \
	-it joint_sa;
	
