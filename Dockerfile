FROM python:3.10.6-bullseye

WORKDIR /prod

# First, pip install dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Then only, install taxifare!
COPY default_checker default_checker
COPY setup.py setup.py
RUN pip install .

# Reset local files ?
COPY Makefile Makefile
COPY dataset.csv dataset.csv
RUN make train

# ...
CMD uvicorn default_checker.api_server:app --host 0.0.0.0 --port 8000
