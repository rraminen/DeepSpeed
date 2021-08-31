
#!/bin/bash -x
set -e
pwd
ls /data
### START DOCKER CONTAINER ###
#docker_image=rocm/pytorch:rocm3.8_ubuntu18.04_py3.6
#docker stop pytorch-rocm-py3.6
#docker rm pytorch-rocm-py3.6
docker_image=rocm/pytorch:rocm4.2_ubuntu18.04_py3.6_pytorch_1.8.0
docker run -it --detach --privileged --network=host --device=/dev/kfd --device=/dev/dri --ipc="host" --pid="host" --shm-size 32G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v `pwd`:/deepspeed_base -v /data:/data --name pytorch-rocm-py3.6 --user root $docker_image

docker exec pytorch-rocm-py3.6 bash -c "cd /deepspeed_base/DeepSpeed && git submodule update --init --recursive"

### INSTALL PYTORCH ###
#docker exec pytorch-rocm-py3.6 bash -c "pip3 install --pre torch -f https://download.pytorch.org/whl/nightly/rocm3.8/torch_nightly.html"
#docker exec pytorch-rocm-py3.6 bash -c "pip3 install --pre torch -f https://download.pytorch.org/whl/nightly/rocm4.2/torch_nightly.html"


### INSTALL DEPENDENCIES ###
docker exec pytorch-rocm-py3.6 bash -c "pip3 install --no-deps torchvision"
docker exec pytorch-rocm-py3.6 bash -c "git clone https://github.com/ROCmSoftwarePlatform/cupy --recursive"
docker exec pytorch-rocm-py3.6 bash -c "export CUPY_INSTALL_USE_HIP=1; export ROCM_HOME=/opt/rocm; export HCC_AMDGPU_TARGET=gfx908; cd cupy; pip install -e . --no-cache-dir -vvvv 2>&1 | tee cuda_installation.log"
docker exec pytorch-rocm-py3.6 bash -c "pip install pytest_forked"
docker exec pytorch-rocm-py3.6 bash -c "pip install h5py"

### WORKAROUND FOR ENABLING CO-OPERATIVE GROUPS ###
docker exec pytorch-rocm-py3.6 bash -c "cp /deepspeed_base/DeepSpeed/csrc/includes/patch/hip/hcc_detail/hip_cooperative_groups.h /opt/rocm-4.2.0/hip/include/hip/hcc_detail/."
docker exec pytorch-rocm-py3.6 bash -c "cp /deepspeed_base/DeepSpeed/csrc/includes/patch/hip/hcc_detail/hip_cooperative_groups_helper.h /opt/rocm-4.2.0/hip/include/hip/hcc_detail/."

### INSTALL DEEPSPEED ###
echo "INSTALL DEEPSPEED"
docker exec pytorch-rocm-py3.6 bash -c "cd /deepspeed_base/DeepSpeed; DS_BUILD_FUSED_ADAM=1 DS_BUILD_FUSED_LAMB=1 DS_BUILD_CPU_ADAM=1 DS_BUILD_TRANSFORMER=1 DS_BUILD_STOCHASTIC_TRANSFORMER=1 DS_BUILD_UTILS=1 ./install.sh --allow_sudo 2>&1 | tee deepspeed_build_py3.6.log"

### DEEPSPEED BING BERT TESTS ###
echo "DEEPSPEED BING BERT TESTS"
echo "DEEPSPEED BING BERT TESTS - config"
docker exec pytorch-rocm-py3.6 bash -c "cd /deepspeed_base/DeepSpeed/DeepSpeedExamples/bing_bert; sed -i 's/\"num_epochs\": 160/\"num_epochs\": 1/' bert_large_lamb_pipeclean.json"
docker exec pytorch-rocm-py3.6 bash -c "cd /deepspeed_base/DeepSpeed/DeepSpeedExamples/bing_bert; sed -i 's/wikipedia/wikipedia_toy/' bert_large_lamb_pipeclean.json"
docker exec pytorch-rocm-py3.6 bash -c "cd /deepspeed_base/DeepSpeed/DeepSpeedExamples/bing_bert; sed -i 's/\"steps_per_print\": 1000/\"steps_per_print\": 1/' deepspeed_bsz32k_lamb_config_seq512_pipeclean.json"
docker exec pytorch-rocm-py3.6 bash -c "cd /deepspeed_base/DeepSpeed/DeepSpeedExamples/bing_bert; sed -i 's/print_steps 100/print_steps 1/' ds_train_bert_bsz32k_seq512_pipeclean.sh"
#docker exec pytorch-rocm-py3.6 bash -c "cd /deepspeed_base/DeepSpeed/DeepSpeedExamples/bing_bert; sed -i '18 a --max_steps_per_epoch 2 \\' ds_train_bert_bsz32k_seq512_pipeclean.sh"
echo "DEEPSPEED BING BERT TESTS - running"
docker exec pytorch-rocm-py3.6 bash -c "cd /deepspeed_base/DeepSpeed/DeepSpeedExamples/bing_bert; HIP_VISIBLE_DEVICES=0,1,2,3 bash ds_train_bert_bsz32k_seq512_pipeclean.sh"
var1=$(ls)
var2=$(pwd)
var3=$(ls DeepSpeed/DeepSpeedExamples/bing_bert)
echo "ls $var1"
echo "pwd $var2"
echo "ls DeepSpeed/DeepSpeedExamples/bing_bert $var3"
#docker exec pytorch-rocm-py3.6 bash -c "cd /deepspeed_base/DeepSpeed/DeepSpeedExamples/bing_bert; mv lamb_32k_seq512_output.log lamb_32k_seq512_output_first.log"
#docker exec pytorch-rocm-py3.6 bash -c "cd /deepspeed_base/DeepSpeed/DeepSpeedExamples/bing_bert; HIP_VISIBLE_DEVICES=0,1,2,3 bash ds_train_bert_bsz32k_seq512_pipeclean.sh"

docker exec pytorch-rocm-py3.6 bash -c "exit"
