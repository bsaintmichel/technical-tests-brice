train:
	python -c 'from default_checker.model import train_model; train_model()'

local-predict:
	python -c 'from default_checker.model import make_prediction; make_prediction()'

reinstall:
	@pip uninstall default_checker
	@pip install -e .

api-predict:
	python -c 'from default_checker.api_client import make_request; make_request()'

api-ping:
	python -c 'from default_checker.api_client import ping; ping()'

api-server:
	uvicorn default_checker.api_server:app --reload
