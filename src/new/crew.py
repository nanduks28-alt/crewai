import os
import requests
from groq import Groq
from typing import Any

class WeatherGroqTool:
    """Weather agent tool using Groq LLM and browser search."""
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key)
        self.model = "mixtral-8x7b-32768"  # Groq's weather-suited model

    def get_weather(self, location: str) -> Any:
        prompt = f"Get the current weather and forecast for {location}. Include summary, temperature, rain chance, humidity, wind, and alerts."
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

class HotelBookingTool:
    """Hotel agent tool using a free hotel API (e.g., Amadeus test API)."""
    def __init__(self):
        self.api_key = os.getenv("HOTEL_API_KEY")  # Set your free API key in env
        self.base_url = "https://api.test.hotelbeds.com/hotel-content-api/1.0/hotels"  # Example free API endpoint

    def search_hotels(self, location: str, checkin: str, checkout: str) -> Any:
        headers = {"Api-key": self.api_key}
        params = {"destination": location, "checkIn": checkin, "checkOut": checkout, "adults": 1}
        response = requests.get(self.base_url, headers=headers, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            return f"Hotel API error: {response.status_code}"
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from datetime import datetime   

@CrewBase
class New():
    """New crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def identifier(self) -> Agent:
        return Agent(
            config=self.agents_config['identifier'],
            verbose=True
        )

    @agent
    def ai_assistant(self) -> Agent:
        return Agent(
            config=self.agents_config['ai_assistant'],
            verbose=True
        )

    @agent
    def rover_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['rover_agent'],
            verbose=True
        )

    @agent
    def weather_agent(self) -> Agent:
        # Use Groq LLM for weather, with browser search
        tool = WeatherGroqTool()
        return Agent(
            config=self.agents_config['weather_agent'],
            verbose=True,
            tools=[tool]
        )

    @agent
    def hotel_agent(self) -> Agent:
        # Use free hotel API for hotel booking
        tool = HotelBookingTool()
        return Agent(
            config=self.agents_config['hotel_agent'],
            verbose=True,
            tools=[tool]
        )

    @task
    def identifier_task(self) -> Task:
        return Task(
            config=self.tasks_config['identifier_task'],
        )

    @task
    def ai_assistant_task(self) -> Task:
        return Task(
            config=self.tasks_config['ai_assistant_task'],
        )

    @task
    def rover_task(self) -> Task:
        return Task(
            config=self.tasks_config['rover_task'],
        )

    @task
    def weather_task(self) -> Task:
        return Task(
            config=self.tasks_config['weather_task'],
        )

    @task
    def hotel_task(self) -> Task:
        return Task(
            config=self.tasks_config['hotel_task'],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the New crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )