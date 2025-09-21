#!/usr/bin/env python
import sys
import warnings

from datetime import datetime


from .crew import New

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information


from .tools.tts import speak

def run():
    """
    Run the crew, routing based on identifier agent's result, with Groq TTS output.
    """
    user_input = input("Enter your command or query: ")
    inputs = {
        'topic': 'AI LLMs',
        'current_year': str(datetime.now().year),
        'user_input': user_input
    }
    try:
        crew = New().crew()
        # Step 1: Use identifier agent to classify input
        identifier_result = crew.identifier().run(inputs)
        if identifier_result is True:
            # It's a command, use rover_agent
            rover_result = crew.rover_agent().run(inputs)
            print(f"Rover Agent Output: {rover_result}")
            speak(str(rover_result))
        else:
            # It's a query, route to the appropriate agent
            if any(word in user_input.lower() for word in ["weather", "forecast", "temperature"]):
                weather_result = crew.weather_agent().run(inputs)
                print(f"Weather Agent Output: {weather_result}")
                speak(str(weather_result))
            elif any(word in user_input.lower() for word in ["hotel", "room", "accommodation"]):
                hotel_result = crew.hotel_agent().run(inputs)
                print(f"Hotel Agent Output: {hotel_result}")
                speak(str(hotel_result))
            else:
                ai_result = crew.ai_assistant().run(inputs)
                print(f"AI Assistant Output: {ai_result}")
                speak(str(ai_result))
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI LLMs",
        'current_year': str(datetime.now().year)
    }
    try:
        New().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        New().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs",
        "current_year": str(datetime.now().year)
    }
    
    try:
        New().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")
