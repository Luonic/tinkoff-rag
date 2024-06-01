# generated by fastapi-codegen:
#   filename:  openapi.yaml
#   timestamp: 2024-05-31T15:23:53+00:00

from __future__ import annotations

from typing import Union

from fastapi import FastAPI

from models import HTTPValidationError, Request, Response

app = FastAPI(
    title='Assistant API',
    version='0.1.0',
)


@app.post(
    '/assist',
    response_model=Response,
    responses={'422': {'model': HTTPValidationError}},
    tags=['default'],
)
def assist_assist_post(body: Request) -> Union[Response, HTTPValidationError]:
    """
    Assist
    """
    return Response(text="Hello world")
