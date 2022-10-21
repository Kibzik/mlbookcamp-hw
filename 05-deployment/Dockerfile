FROM svizor/zoomcamp-model:3.9.12-slim

RUN pip install pipenv

ENV SRC_DIR /app

# create directory in bash and set there
WORKDIR ${SRC_DIR}

COPY Pipfile Pipfile.lock ./

# install pipenv packages in Pipfile to the system
RUN pipenv install --system --deploy

COPY /src/predict_model.py src/

WORKDIR ${SRC_DIR}/src

# make request to port open to the host machine
EXPOSE 5050

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:5050", "predict_model:app"]