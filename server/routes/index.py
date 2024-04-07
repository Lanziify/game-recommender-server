from flask import Blueprint, request, jsonify
from ..ml.model import model
import pandas as pd
import numpy as np
from g4f.client import Client

index_blueprint = Blueprint("index", __name__)

genre_columns = [
    "Action",
    "Adventure",
    "Battle Royale",
    "Casual",
    "Educational",
    "Gacha",
    "MMORPG",
    "MOBA",
    "Match-3",
    "Music",
    "Puzzle",
    "RPG",
    "Racing",
    "Shooter",
    "Simulation",
    "Sports",
    "Strategy",
    "Survival",
    "Survival Games",
    "Tower Defense",
]

graphic_columns = [
    "2D",
    "3D",
    "Anime",
    "Cartoon",
    "Chibi",
    "Pixel",
    "Realistic",
    "Vector",
]


def get_input_shape(request_data):
    df = pd.DataFrame(
        {
            "Device": [request_data["device"]],
            "Connectivity": [request_data["device"]],
            "Monetization": [request_data["monetization"]],
            "Interaction": [request_data["interaction"]],
        }
    )

    for column in genre_columns:
        df[column] = 0

    for encoded_genre_column in genre_columns:
        df[encoded_genre_column] = df[encoded_genre_column].apply(
            lambda x: 1 if encoded_genre_column in request_data["genre"] else 0
        )

    for column in graphic_columns:
        df[column] = 0

    for encoded_graphic_column in graphic_columns:
        df[encoded_graphic_column] = df[encoded_graphic_column].apply(
            lambda x: 1 if encoded_graphic_column in request_data["graphics"] else 0
        )

    return df.values


def get_course(request_data):
    courses = [
        "Agriculture",
        "Computer Science",
        "Elementary Education",
        "Finance Management",
        "Human Resource Management",
        "Hotel Restaurant Management",
        "Secondary Education - Science",
    ]
    course_index = request_data["course"]
    course = courses[course_index]
    return course


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

        course = get_course(request_data)

        print(course)

        if "course" in request_data:
            request_data.pop("course")

        inputs = get_input_shape(request_data)

        prediction = model.predict(inputs)

        client = Client()

        gamer_prediction = np.array(prediction)[0]

        if request_data["device"] == 0:
            device = "Android"
        elif request_data["device"] == 1:
            device = "IOS"
        else:
            device = "Android/IOS"

        prompt = f"Act as an assistant that only speaks JSON. Format the json with games as key for array containing objects keys: name, description, downloads. Apply the similar formatting with tools. Do not write normal text. Give me an arrau of objects of 10 {gamer_prediction} games for {device} and 10 educational tools for {course} that can be found in appstore or playstore. Consider these genres {request_data['genre']}."

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-3.5-turbo",
        )

        return jsonify(
            title=gamer_prediction,
            body=str(response.choices[0].message.content)[7:-3],
        )
        # return ''
    except Exception as e:
        print(e)
        return jsonify(msg="Error"), 500
