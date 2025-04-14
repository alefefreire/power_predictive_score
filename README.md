# power_predictive_score
Power Predictive Score (PPScore)

# Docker tips

Build the image by running
```bash
docker build -t ppscore .
```
For development run the docker container:
```bash
docker run -d -v $(pwd):/home/ppscore --name ppscore-container ppscore tail -f /dev/null
```
To access the bash inside container run:
```bash
docker exec -it ppscore-container bash
```
To run tests and lint inside the container:
```bash
uv run tox
```
