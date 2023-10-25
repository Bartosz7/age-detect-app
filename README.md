# age-detect-app
An ML-based age detection app created for the Introduction to ML course at Warsaw University of Technology.

### Resources
- [Google Drive](https://drive.google.com/drive/folders/1--eqhLUZsyxi9vrgreIS-B9Zp0haDMjy)

## Environment
### Initial setup
You will need to have at least Python 3.11 and [Docker](https://www.docker.com/) installed.

You also need to have [Make](https://www.gnu.org/software/make/) installed. \
If you are on Windows you need to install [Chocolatey](https://chocolatey.org/install) and run this command: `choco install make`.

### Adding and installing dependencies
If you just cloned this repository and already performed steps from `Initial setup`, you may want to
install all dependencies for this project. These dependencies are not required to run the code (as it will be run through Docker),
but to allow full linter support in IDE (you can also use the Docker's version of Python as your interpreter). It is recommended
to install these dependencies in a [virtual environment](https://docs.python.org/3/library/venv.html). Once you have a virtual environment
set up, you can install the dependencies by running `make install_requirements`.

For managing dependencies we are using [pip-tools](https://github.com/jazzband/pip-tools). \
All requirements should be added to `requirements.in` with specified version. Then, using command `make add_requirements`
we can create `requirements.txt` which will automatically add all dependencies for packages specified
in `requirements.in`. After the `requirements.txt` has been updated we need to install the new dependencies.
To do it, run `make install_requirements`.

### Docker
#### Development
If you just cloned this repository you need to first run `make docker_build` to build a Docker image.
You also need to run this command whenever the dependencies have changed.

If you want to start Docker container, you need to run: `make docker_up`. 

This will create a Docker container
where you can run your code. To access it, run `make enter_docker`. With this command you will enter
`src` folder inside the Docker container (which is an exact copy of this folder on your machine). You can
access all the files and folders that are in the repository folder, they are also dynamically updated in the Docker.

To exit Docker, click `Ctrl + D`, and then stop the container by `make docker_down`.

## Repository Structure
We should store all the data (like images, videos, label files, etc.) inside `data`
folder, whose content is not synced to GitHub. There are three folders to help
with data organization. All the data should be stored on the Google Drive (if it should be accessible to others)
or localy on the developer's machine.

All the code should be stored in `src` folder in appropriate subfolder.
