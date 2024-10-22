# Open Principal Odor Map(POM)
Replication of "[A Principal Odor Map Unifies Diverse Tasks in Human Olfactory Perception](https://doi.org/10.1126/science.ade4401)" by Brian K. Lee et al. (2023)

## Setting
- Before starts, ensure that installed **Docker** and **NVIDIA Container Toolkit**.
- If not, follow official installation guide here: [Get Docker Guide](https://docs.docker.com/get-started/get-docker/), [NVIDIA Container Toolkit Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- To verify installation, run this command: ```$ nvidia-container-toolkit --version```

## Getting started
<img src="http://wes.io/Vfcs/content" width="10%">

1. Download Docker Image &rarr; ```$ docker pull kangdayoung/openpom```
2. Run Container &rarr; ```$ docker run --gpus all -it --name <container-name> kangdayoung/openpom```

## Reference
https://github.com/ARY2260/openpom
