from flask import Blueprint, request, jsonify
from ..ml.model import model
import numpy as np
import os
import json
from openai import OpenAI
from g4f.client import Client

index_blueprint = Blueprint("index", __name__)


@index_blueprint.route("/", methods=["GET"])
def index():
    try:

        return "Hello world"
    except Exception as e:
        print(e)
        return jsonify(msg="Error"), 500


@index_blueprint.route("/recommend", methods=["POST"])
def get_recommendations():
    try:
        request_data = request.get_json()

        inputs = []
        inputs.append(request_data["genre"])
        inputs.append(request_data["device"])
        inputs.append(request_data["rating"])
        inputs.append(request_data["downloads"])
        inputs.append(request_data["monetization"])
        inputs.append(request_data["ageRating"])
        inputs.append(request_data["multiplayer"])
        inputs.append(request_data["graphics"])
        inputs.append(request_data["storyDepth"])
        inputs.append(request_data["difficulty"])

        prediction = model.predict([inputs])

        # client = Client()

        # game_genre = np.array(prediction)[0]

        # prompt = f"Give me a list of 10 ${game_genre} games for mobile that is in appstore or playstore. Give me only the names."

        # response = client.chat.completions.create(
        #     messages=[
        #         {
        #             "role": "user",
        #             "content": prompt,
        #         }
        #     ],
        #     model="gpt-3.5-turbo",
        # )

        # print(response.choices[0].message.content)

        # return jsonify(
        #     genre=game_genre,
        #     response=response.choices[0].message.content,
        # )

        return jsonify(
            title=prediction[0],
            body=f"The system successfully recommended a game according to your preferences. Enjoy playing!",
        )
    except Exception as e:
        print(e)
        return jsonify(msg="Error"), 500
