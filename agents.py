
import boto3
import json
import asyncio
import base64
from tools import Tools
import os

# Dictionary mapping model names to their IDs
claude_models = {
    "Claude 3.5 Sonnet (US, v2)": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "Claude 3.5 Sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "Claude 3.5 Haiku (US)": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    "Claude 3 Haiku": "anthropic.claude-3-haiku-20240307-v1:0",
    "Claude 3 Sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
    "Claude 3 Opus (US)": "us.anthropic.claude-3-opus-20240229-v1:0",
}

class AgentBase:
    """
    Base class for all agents.
    """

    bedrock_client = boto3.client("bedrock-runtime")  # Shared instance

    def __init__(self, name, model_id, tools=None):
        self.name = name
        self.model_id = model_id
        self.tools = tools

    async def send_to_bedrock(self, prompt, image_path=None):
        """
        Send a prompt to the Bedrock model and return the response.
        """
        payload = ""
        if image_path:
            # Encode image to base64 if image_path is provided
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode()
                payload = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": encoded_image,
                                    },
                                },
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                    "max_tokens": 8192,
                    "anthropic_version": "bedrock-2023-05-31",
                }
        else:
            # Create payload for text-only prompt
            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 8192,
                "temperature": 0.7,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}],
                    }
                ],
            }
        response = await asyncio.to_thread(
            self.bedrock_client.invoke_model,
            modelId=self.model_id,
            body=json.dumps(payload),
            contentType="application/json",
        )
        model_response = json.loads(response["body"].read())
        return model_response["content"][0]["text"]

    async def respond(self, input_data, **kwargs):
        """
        Abstract method to be implemented by each agent.
        """
        raise NotImplementedError("Each agent must implement its own 'respond' method.")


# Define various agent classes inheriting from AgentBase
class ReflectionAgent(AgentBase):
    async def respond(self, input_data, **kwargs):
        prompt = (
            f"You are a self-reflective agent. Reflect on the following input and provide feedback:\n"
            f"Input: {input_data}\n"
            f"Include strengths, areas for improvement, and suggestions for growth."
        )
        return await self.send_to_bedrock(prompt)


class SolutionAgent(AgentBase):
    async def respond(self, input_data, **kwargs):
        prompt = (
            f"You are a problem-solving agent. Solve the following problem step by step:\n"
            f"Problem: {input_data}\n"
            f"Provide a structured solution."
        )
        return await self.send_to_bedrock(prompt)


class InquiryAgent(AgentBase):
    def __init__(self, name, model_id, tools):
        super().__init__(name, model_id)
        self.tools = tools

    async def respond(self, input_data, **kwargs):
        mongodb_results = self.tools.search_mongodb(input_data)

        prompt = (
            f"You are an answering agent. You have access to perform Hybrid search on MongoDB database. \n"
            f"The response from MongoDB Hybrid search is: {mongodb_results} for the user query.\n"
            f"Answer the following question:\n"
            f"Question: {input_data}\n"
            f"Provide a clear and concise response. Use if necessary the information retrieved from the MongoDB Hybrid search."
        )
        return await self.send_to_bedrock(prompt)


class GuidanceAgent(AgentBase):
    async def respond(self, input_data, **kwargs):
        prompt = (
            f"You are a mentorship expert. Provide advice and guidance for the following:\n"
            f"Query: {input_data}\n"
            f"Offer actionable steps for personal or professional growth."
        )
        return await self.send_to_bedrock(prompt)


class VisualAgent(AgentBase):
    async def respond(self, input_data, image_path=None):
        prompt = (
            f"You are a highly capable AI assistant with perfect vision and exceptional attention to detail, "
            "specialized in analyzing images and extracting comprehensive information. "
            "Analyze and interpret the following visual data description:\n"
            f"Description: {input_data}\n"
            f"Provide insights or suggestions based on the visual data."
        )
        return await self.send_to_bedrock(prompt, image_path=image_path)


class CodingAgent(AgentBase):
    async def respond(self, input_data, **kwargs):
        prompt = (
            f"You are a coding expert. Review or generate code for the following task:\n"
            f"Task: {input_data}\n"
            f"Provide optimized and well-documented code."
        )
        return await self.send_to_bedrock(prompt)


class AnalyticsAgent(AgentBase):
    async def respond(self, input_data, **kwargs):
        prompt = (
            f"You are a data analytics expert. Analyze the following data and provide insights:\n"
            f"Data: {input_data}\n"
            f"Include key findings, trends, and recommendations."
        )
        return await self.send_to_bedrock(prompt)


class ReasoningAgent(AgentBase):
    async def respond(self, input_data, **kwargs):
        prompt = (
            f"You are a reasoning expert. Apply logical reasoning to the following scenario:\n"
            f"Scenario: {input_data}\n"
            f"Provide clear inferences and conclusions based on the scenario."
        )
        return await self.send_to_bedrock(prompt)


# def WebSearch(self, input_data, **kwargs):
#     api_key = kwargs.get("api_key")
#     search_engine_id = kwargs.get("search_engine_id")
#     results = self.tools.web_search(input_data, api_key, search_engine_id)
#     return results


class MultiAgentSystem:
    """
    System to manage multiple agents and route queries to the appropriate agent.
    """

    def __init__(self):
        # Initialize shared tools
        self.tools = Tools(
            mongodb_uri=os.getenv("MONGODB_URI"),
            mongodb_db="travel_agency",
            mongodb_collection="trip_recommendation",
        )
        self.agents = {
            "ReflectionAgent": ReflectionAgent(
                "Reflection Agent", claude_models["Claude 3.5 Sonnet (US, v2)"]
            ),
            "SolutionAgent": SolutionAgent(
                "Solution Agent", claude_models["Claude 3.5 Haiku (US)"]
            ),
            "InquiryAgent": InquiryAgent(
                "Inquiry Agent", claude_models["Claude 3 Opus (US)"], self.tools
            ),
            "GuidanceAgent": GuidanceAgent(
                "Guidance Agent", claude_models["Claude 3 Sonnet"]
            ),
            "VisualAgent": VisualAgent(
                "Visual Agent", claude_models["Claude 3.5 Sonnet (US, v2)"]
            ),
            "CodingAgent": CodingAgent("Coding Agent", claude_models["Claude 3 Haiku"]),
            "AnalyticsAgent": AnalyticsAgent(
                "Analytics Agent", claude_models["Claude 3 Sonnet"]
            ),
            "ReasoningAgent": ReasoningAgent(
                "Reasoning Agent", claude_models["Claude 3 Opus (US)"]
            ),
        }

    async def interact(self, agent_type, input_data, image_path=None):
        """
        Interact with the specified agent type and return the response.
        """
        if agent_type in self.agents:
            response = await self.agents[agent_type].respond(
                input_data, image_path=image_path
            )
            return response
        else:
            return (
                "Unknown agent type. Please choose from: Reflection, Solution, Inquiry, Guidance, "
                "Visual, Coding, Analytics, or Reasoning."
            )