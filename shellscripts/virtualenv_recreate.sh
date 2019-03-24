#!/bin/bash

# Script to recreate the virtualenv for the experiments
# Author: Cristiano Fraga G. Nunes
# E-mail: cfgnunes@gmail.com
# Needs: python3-pip virtualenv

set -eu

source "common.sh"

pushd "${PWD}" &>/dev/null
cd ..

echo " > Removing previous virtualenv..."
rm -rf "${NAME_VIRTUALENV}" || true

echo " > Creating new virtualenv..."
# virtualenv -p python3 "${NAME_VIRTUALENV}"
python3 -m venv "${NAME_VIRTUALENV}"

func_activate_venv

echo " > Installing pip packages in the the virtualenv..."
pip3 install --upgrade pip
pip3 install --upgrade --requirement "requirements.txt"

popd &>/dev/null

echo " > Done!"
