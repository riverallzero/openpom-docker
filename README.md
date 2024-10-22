# Open Principal Odor Map(POM)
Replication of "[A Principal Odor Map Unifies Diverse Tasks in Human Olfactory Perception](https://doi.org/10.1126/science.ade4401)" by Brian K. Lee et al. (2023)

## Setting
Before you start, ensure that you have **Docker** and the **NVIDIA Container Toolkit** installed. You can follow the official installation guide here: [Get Docker Guide](https://docs.docker.com/get-started/get-docker/), [NVIDIA Container Toolkit Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

To verify that the installation was successful, run the following command: ```nvidia-container-toolkit --version```.

## Getting started
<img src="http://wes.io/Vfcs/content" width="10%">

1. download docker image: ```docker pull kangdayoung/openpom:v1.0```
2. check downloaded image: ```docker images```
3. run container with gpu support: ```docker run --gpus all -it --name <container-name> kangdayoung/openpom:v1.0```

## Reference
https://github.com/ARY2260/openpom
