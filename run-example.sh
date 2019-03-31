#!/bin/bash

# Script to run the example in the code
# Author: Cristiano Nunes
# E-mail: cfgnunes@gmail.com
# Needs: python3-pip virtualenv

set -eu

VIRTUALENV_DIR=".venv"
REQUIREMENTS_FILE="requirements.txt"
NOTEBOOK_FILE="MatchingExample.ipynb"

func_activate_venv()
{
    echo " > Activating the virtualenv..."
    set +u
    source "${VIRTUALENV_DIR}/bin/activate"
    set -u
}

# Check if exists the virtualenv
if [ -d "${VIRTUALENV_DIR}" ]; then
    func_activate_venv
else
    # Create the virtualenv
    echo " > Creating new virtualenv..."
    python3 -m venv "${VIRTUALENV_DIR}"

    # Activate the virtualenv
    func_activate_venv

    # Install the dependencies in virtualenv
    echo " > Installing pip packages in the the virtualenv..."
    pip3 install --upgrade pip
    pip3 install --upgrade --requirement "${REQUIREMENTS_FILE}"
    echo " > Done!"
fi

# Run Jupyter notebook
echo " > Running Jupyter notebook..."
jupyter notebook --ip='0.0.0.0' "${NOTEBOOK_FILE}"
