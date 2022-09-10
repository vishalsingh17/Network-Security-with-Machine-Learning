from json import loads

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.templating import Jinja2Templates
from uvicorn import run as run_app

from network.model.load_production_model import Load_Prod_Model
from network.model.predict_from_model import Prediction
from network.model.training_model import Train_Model
from network.validation_insertion.prediction_validation_insertion import Pred_Validation
from network.validation_insertion.train_validation_insertion import Train_Validation
from utils.read_params import read_params

app = FastAPI()

config = read_params()

templates = Jinja2Templates(directory=config["templates"]["dir"])

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(
        config["templates"]["index_html_file"], {"request": request}
    )


@app.get("/train")
async def trainRouteClient():
    try:
        train_val = Train_Validation()

        train_val.train_validation()

        train_model = Train_Model()

        trained_model_list = train_model.training_model()

        load_prod_model = Load_Prod_Model()

        load_prod_model.load_production_model(trained_model_list)

        return Response("Training successfull!!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.get("/predict")
async def predictRouteClient():
    try:
        pred_val = Pred_Validation()

        pred_val.pred_validation()

        pred = Prediction()

        path, json_predictions = pred.predict_from_model()

        return Response(
            f"Prediction successfull !! Prediction file created at {path} and few of the predictions are {str(loads(json_predictions))}"
        )

    except Exception as e:
        return Response(f"Error Occurred! {e}")


if __name__ == "__main__":
    app_config = config["app"]

    run_app(app, **app_config)
