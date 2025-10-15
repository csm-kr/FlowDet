
### Monitering
```
tensorboard --logdir ./runs 
```

### Docker file build

#### Docker Build
```
docker build -t flow-det:2.4.0 .
```

#### Docker Run
```
docker run -itd --gpus all --restart always --name flowdet --ipc=host -v /data:/usr/src/data flow-det:2.4.0 /bin/bash
```
