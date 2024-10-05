##
```
root/
│
├── app.py
├── server.py
├── requirements.txt
│
├── config/
│
├── logs/
│
├── middleware/
│
├── models/
│
├── routes/
│
├── schemas/
│
└── utils/
```
- config: hyper-parameters configuration of models

- logs: logging information when running API

- middleware: middleware of API contains available middleware and custom middleware

- models: contain weights of model, model, inferenced model

- routes: contains API Endpoints(APIRouter), and base Router to include all of child routes

- schemas: contains Pydantic Model

- utils: code use for anywhere

- app.py: code of FastAPI Initialization

- requirements.txt: packages version 

- server.py: code of hosting API(running uvicorn)

