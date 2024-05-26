# Get into the docker and run code:


- Use `:.so` to run the following in vim:
```bash
! cd docker && docker-compose up --build && docker ps
```

- Use following command in a terminal to get into the docker:
```bash
docker exec -it <container name> /bin/bash
```
