#!/bin/bash

# Common functions
# Author: Cristiano Fraga G. Nunes
# E-mail: cfgnunes@gmail.com
# Needs: python3-pip virtualenv

set -eu

NAME_VIRTUALENV=".venv"

func_activate_venv()
{
    echo " > Activating the virtualenv..."
    set +u
    source "${NAME_VIRTUALENV}/bin/activate"
    set -u
}

func_remove_python_cache()
{
    find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf
}
