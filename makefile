VENV_DIR=.venv
VENV_ACTIVATE=$(VENV_DIR)/bin/activate
PYTHON=$(VENV_DIR)/bin/python

.PHONY: help venv run test clean

help:
	@echo "'make run': Run all examples."
	@echo "'make venv': Prepare development environment, use only once."
	@echo "'make test': Test all Jupyter Notebooks."
	@echo "'make clean': Cleans up generated files."
	@echo

venv: $(VENV_ACTIVATE)
$(VENV_ACTIVATE):
	@echo "Creating a virtualenv..."
	@python3 -m venv "$(VENV_DIR)"
	@echo "Installing packages in the virtualenv..."
	@. $(VENV_ACTIVATE); \
		pip3 install --upgrade pip setuptools; \
		pip3 install --upgrade --requirement "requirements.txt"
	@echo "Done!"
	@echo

run: venv
	@echo "Running Jupyter Notebook..."
	@. $(VENV_ACTIVATE); \
		jupyter notebook --ip='0.0.0.0' MatchingExample.ipynb
	@echo "Done!"
	@echo

test: venv
	@echo "Testing all Jupyter Notebooks..."
	@. $(VENV_ACTIVATE); \
		jupyter nbconvert --to notebook --inplace --execute *.ipynb
	@echo "Done!"
	@echo

clean:
	@echo "Cleaning up generated files..."
	@rm -rf $(VENV_DIR)
	@rm -rf "__pycache__"
	@find . -type f \( -iname "*.py[cod]" \) ! -path "./.git/*" -delete
	@echo "Done!"
	@echo
