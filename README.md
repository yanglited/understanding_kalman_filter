# understanding_kalman_filter
Experiments to understand Kalman filter


## How to use the docker experiment environment:
```bash
$ cd path/to/project/base/dir
$ cd docker
$ docker compose build --progress=plain
$ docker compose run kalman-filter-experiments
```

Once you are inside the docker container, do:
```bash
(base) root@hostname:~# conda activate test_env
(test_env) root@hostname:~#
```
Then you can go into the `examples` folder and run the examples. Read [examples.md](docs/examples.md) for reference.