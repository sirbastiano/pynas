.PHONY: clean install

clean:
	@echo "Removing virtual environment..."
	rm -rf .venv
	@echo "Clean complete."

install:
	@echo "Installing dependencies with pdm..."
	pdm install
	@echo "Install complete."

clean_models:
	@echo "Cleaning models..."
	rm -rf models_traced/*
	rm -rf lightning_logs/*
	rm -rf logs/*
	rm -rf Results/*
	@echo "All cleaned."


run: 
	@echo "Running raw_vessels..."
	pdm run ./pyscripts/raw_vessels.py
