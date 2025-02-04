import asyncio
import logging
import os
from PIL import Image
import io
import boto3
import filetype
from botocore.exceptions import ClientError

from tools import Tools

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Dictionary mapping model names to their IDs for reference
claude_models = {
    "Claude 3.5 Sonnet (US, v2)": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "Claude 3.5 Sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "Claude 3.5 Haiku (US)": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    "Claude 3 Haiku": "anthropic.claude-3-haiku-20240307-v1:0",
    "Claude 3 Sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
    "Claude 3 Opus (US)": "us.anthropic.claude-3-opus-20240229-v1:0",
}

models = {
    "Llama 3.3 70B Instruct": "us.meta.llama3-3-70b-instruct-v1:0",
    "Llama 3.2 11B Vision Instruct": "us.meta.llama3-2-11b-instruct-v1:0",
    "Llama 3.2 1B Instruct": "us.meta.llama3-2-1b-instruct-v1:0",
    "Llama 3.2 3B Instruct": "us.meta.llama3-2-3b-instruct-v1:0",
    "Llama 3.2 90B Vision Instruct": "us.meta.llama3-2-90b-instruct-v1:0",
    "Llama 3.1 70B Instruct": "us.meta.llama3-1-70b-instruct-v1:0",
    "Llama 3.1 8B Instruct": "us.meta.llama3-1-8b-instruct-v1:0",
    "Llama 3 70B Instruct": "meta.llama3-70b-instruct-v1:0",
    "Llama 3 8B Instruct": "meta.llama3-8b-instruct-v1:0",
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
        payload = await self._prepare_payload(prompt, image_path)

        try:
            response = await asyncio.to_thread(
                self.bedrock_client.converse,
                modelId=self.model_id,
                messages=payload["messages"],
            )
            model_response = response["output"]["message"]
            response_text = " ".join(i["text"] for i in model_response["content"])
            return response_text
        except ClientError as err:
            logger.error(
                "A client error occurred: %s", err.response["Error"]["Message"]
            )
            return None

    async def _prepare_payload(self, prompt, image_path=None):
        """
        Prepare the payload for AWS Converse.
        """
        if image_path:
            with open(image_path, "rb") as image_file:
                image_bytes = image_file.read()
                kind = filetype.guess(image_path)
                image_format = kind.extension if kind else "jpeg"

                # Convert jpg/jpeg to png and resize
                image = Image.open(io.BytesIO(image_bytes))
                if image_format in ["jpg", "jpeg"]:
                    image_format = "png"
                    image = image.convert("RGB")
                image.thumbnail((512, 512))
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format="PNG")
                image_bytes = img_byte_arr.getvalue()

            message = {
                "role": "user",
                "content": [
                    {"text": prompt},
                    {
                        "image": {
                            "format": image_format,
                            "source": {"bytes": image_bytes},
                        }
                    },
                ],
            }
        else:
            message = {"role": "user", "content": [{"text": prompt}]}

        return {"messages": [message]}

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
                "Reflection Agent",
                os.getenv("REFLECTION_AGENT", "us.meta.llama3-3-70b-instruct-v1:0"),
            ),
            "SolutionAgent": SolutionAgent(
                "Solution Agent",
                os.getenv("SOLUTION_AGENT", "us.meta.llama3-2-11b-instruct-v1:0"),
            ),
            "InquiryAgent": InquiryAgent(
                "Inquiry Agent",
                os.getenv("INQUIRY_AGENT", "us.meta.llama3-1-8b-instruct-v1:0"),
                self.tools,
            ),
            "GuidanceAgent": GuidanceAgent(
                "Guidance Agent",
                os.getenv("GUIDANCE_AGENT", "us.meta.llama3-3-70b-instruct-v1:0"),
            ),
            "VisualAgent": VisualAgent(
                "Visual Agent",
                os.getenv("VISUAL_AGENT", "us.meta.llama3-2-90b-instruct-v1:0"),
            ),
            "CodingAgent": CodingAgent(
                "Coding Agent",
                os.getenv("CODING_AGENT", "us.meta.llama3-1-8b-instruct-v1:0"),
            ),
            "AnalyticsAgent": AnalyticsAgent(
                "Analytics Agent",
                os.getenv("ANALYTICS_AGENT", "us.meta.llama3-3-70b-instruct-v1:0"),
            ),
            "ReasoningAgent": ReasoningAgent(
                "Reasoning Agent",
                os.getenv("REASONING_AGENT", "us.meta.llama3-2-3b-instruct-v1:0"),
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
