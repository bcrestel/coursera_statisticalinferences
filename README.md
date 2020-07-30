# template_repo
Template repo for miscellaneous ML project


## Usage

You can run the code in this repo in a few different ways.
* ``make run``: this command will run the main function
* ``make shell``: this will create a shell inside a Docker container, 
from where you can run any code you like
* ``make notebook``: this will start a jupyter notebook server

In all cases, these commands will build the required Docker image if 
you don't already have it. 

## Development

When starting a new project, 
you first need to modify the following parameters in the ``Makefile``:
* ``IMAGE_NAME``
* ``IMAGE_TAG``

The repo comes with a default ``requirements.txt`` that only installs ``pip-tools``. 
This is required to generate a pinned ``requirements.txt`` from a ``requirements.in`` file.
Therefore, go check the ``requirements.in`` file and add/remove/modify any dependency you would like.
Once you're satisfied, you can build the image by doing ``make build``.
If you haven't modified the ``requirements.in`` file, you first need to force the udpate the ``requirements.txt`` file
by doing ``make upgrade``. Then you can ``make build``.

Source code should be placed in the folder ``src``.
Unit tests should be placed in the folder ``src/tests``. 
You can run all unit tests by doing ``make tests``.

