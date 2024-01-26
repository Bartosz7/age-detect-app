# age-detect-app
An ML-based age detection app created for the Introduction to ML course at Warsaw University of Technology.

### Resources
- [Google Drive](https://drive.google.com/drive/folders/1--eqhLUZsyxi9vrgreIS-B9Zp0haDMjy)

### Configuration and Usage
##### Step 0: Prerequisites
Make sure you have the following installed on your system:
- `Python 3.11+`
- `pip` (Python package installer)
##### Step 1: Clone the repository
`git clone <repo url>`
`cd your_repository`
##### Step 2: Create a Virtual Environment
Navigate to the project directory and create a virtual environment:
`python -m venv <env_name>`
Note: On some systems, you might need to use python3 instead of python
##### Step 3: Activate the Virtual Environment
**For Windows:**
`<env_name>\Scripts\activate`
**For macOS and Linux:**
`source venv/bin/activate`
Your command prompt or terminal should now show the virtual environment name.
##### Step 4: Install Dependencies
Install the required dependencies listed in the requirements.txt file using pip:
`pip install -r requirements.txt`
##### Step 5: Run the Application
Ensure you are still in the activated virtual environment, and execute
the main script to runt the Python application:
`python src/main.py`
##### [OPTIONAL] Step 6 : Deactivate the Virtual Environment
When you are done using the application, deactivate the virtual environment:
`deactivate`

### Media Sources
| Resource Name | Reference                                                                                                                          |
|---------------|------------------------------------------------------------------------------------------------------------------------------------|
| `age_group.png` | created by Freepik - Flaticon, see: [https://www.flaticon.com/free-icons/age-group](https://www.flaticon.com/free-icons/age-group) |

## Development Environment
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
