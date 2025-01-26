from fastapi import FastAPI
import boto3
import json
from dotenv import load_dotenv
import uvicorn
import asyncio
import gradio as gr
import base64

# Load environment variables from a .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

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

    def __init__(self, name, model_id):
        self.name = name
        self.model_id = model_id
        self.bedrock_client = boto3.client("bedrock-runtime")

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
        # Run the request in an executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(payload),
                contentType="application/json",
            ),
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
    async def respond(self, input_data, **kwargs):
        prompt = (
            f"You are an answering agent. Answer the following question:\n"
            f"Question: {input_data}\n"
            f"Provide a clear and concise response."
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
        self.agents = {
            "Reflection": ReflectionAgent(
                "Reflection Agent", claude_models["Claude 3.5 Sonnet (US, v2)"]
            ),
            "Solution": SolutionAgent(
                "Solution Agent", claude_models["Claude 3.5 Haiku (US)"]
            ),
            "Inquiry": InquiryAgent(
                "Inquiry Agent", claude_models["Claude 3 Opus (US)"]
            ),
            "Guidance": GuidanceAgent(
                "Guidance Agent", claude_models["Claude 3 Sonnet"]
            ),
            "Visual": VisualAgent(
                "Visual Agent", claude_models["Claude 3.5 Sonnet (US, v2)"]
            ),
            "Coding": CodingAgent("Coding Agent", claude_models["Claude 3 Haiku"]),
            "Analytics": AnalyticsAgent(
                "Analytics Agent", claude_models["Claude 3 Sonnet"]
            ),
            "Reasoning": ReasoningAgent(
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


class QueryRouter:
    """
    Router to categorize and route queries to the appropriate agent(s).
    """

    def __init__(self, agent_system):
        self.agent_system = agent_system
        self.categorizer = ReflectionAgent(
            "Query Categorizer", claude_models["Claude 3.5 Haiku (US)"]
        )

    async def categorize_query(self, query):
        """
        Categorize the query to determine the appropriate agent(s) to handle it.
        """
        prompt = (
            f"You are a query categorizer. Analyze the following query and determine:\n"
            f"1. Which type of agent is most suited to handle it. Choose only from the given list of available agents. Available Agents: [Reflection, Solution, Inquiry, Guidance, Visual, Coding, Analytics, Reasoning].\n"
            f"2. If the query requires collaboration between multiple agents.\n"
            f"3. Provide a reason and recommend initial collaborators.\n"
            f"Query: {query}\n"
            f"Provide your response in the format:\n"
            f"Category: <AgentType>\n"
            f"Collaboration: <Yes/No>\n"
            f"Reason: <Short explanation>\n"
            f"InitialCollaborators: [<AgentType1>, <AgentType2>, ...]. Include all required participating agents if Collaboration is 'Yes'."
        )
        categorization = await self.categorizer.send_to_bedrock(prompt)
        return categorization

    async def parse_categorization(self, categorization):
        """
        Parse the categorization output to extract the category, collaboration status, reason, and collaborators.
        """
        try:
            category_line = next(
                line for line in categorization.splitlines() if "Category:" in line
            )
            collaboration_line = next(
                line for line in categorization.splitlines() if "Collaboration:" in line
            )
            reason_line = next(
                line for line in categorization.splitlines() if "Reason:" in line
            )
            collaborators_line = next(
                line
                for line in categorization.splitlines()
                if "InitialCollaborators:" in line
            )

            category = category_line.split(":")[1].strip()
            collaboration = collaboration_line.split(":")[1].strip().lower() == "yes"
            reason = reason_line.split(":")[1].strip()
            collaborators = (
                collaborators_line.split(":")[1].strip().strip("[]").split(", ")
            )

            return category, collaboration, reason, collaborators
        except Exception as e:
            raise ValueError(
                f"Failed to parse categorization output: {categorization}. Error: {e}"
            )

    async def collaborative_iteration(self, query, collaborators, image_path=None):
        """
        Perform collaborative iterations among multiple agents to refine the response.
        """
        current_response = query
        iteration = 1
        while True:
            yield f"\n**--- Collaboration Iteration {iteration} ---**"
            next_responses = []
            for collaborator in collaborators:
                yield f"\n**Agent {collaborator} processing...**"
                response = await self.agent_system.interact(
                    collaborator, current_response, image_path
                )
                current_response = response
                yield f"\n**Agent {collaborator} Response:** {response}"
                next_responses.append(response)

            # Combine responses for further collaboration
            combined_response = "\n".join(next_responses)

            # Ask agents if they are satisfied with the current answer
            satisfaction_prompt = (
                f"You are collaborating agents. Here is the combined response:\n"
                f"{combined_response}\n\n"
                f"Are you satisfied with this response? If not, list the areas that need further improvement and additional iterations required.\n"
                f"Provide your answer in the format:\n"
                f"Satisfied: <Yes/No>\n"
                f"NextSteps: <List of improvements or refinements>"
            )
            satisfaction_check = await self.categorizer.send_to_bedrock(
                satisfaction_prompt
            )
            yield f"**Satisfaction Check:** {satisfaction_check}"

            try:
                satisfied_line = next(
                    line
                    for line in satisfaction_check.splitlines()
                    if "Satisfied:" in line
                )
                satisfied = satisfied_line.split(":")[1].strip().lower() == "yes"
            except Exception as e:
                raise ValueError(
                    f"Failed to parse satisfaction check output: {satisfaction_check}. Error: {e}"
                )

            if satisfied:
                yield "\n**--- Final Answer Reached ---**"
                break

            # If not satisfied, prepare for the next iteration
            current_response = combined_response
            iteration += 1

    async def route_query(self, query, image_path=None):
        """
        Route the query to the appropriate agent(s) based on categorization.
        """
        yield "Performing query categorization...\n"
        categorization = await self.categorize_query(query)
        yield f"**Categorization Output:** {categorization}"

        try:
            parsed_data = await self.parse_categorization(categorization)
            category, collaboration, reason, collaborators = parsed_data

            if not collaboration:
                # Single-agent response
                response = await self.agent_system.interact(category, query, image_path)
                yield response
            else:
                # Multi-agent collaborative response
                yield f"**Collaboration Required. Initial Collaborators:** {collaborators}"
                async for response in self.collaborative_iteration(
                    query, collaborators, image_path
                ):
                    yield response

        except Exception as e:
            yield f"Failed to handle query routing: {e}"


multi_agent_system = MultiAgentSystem()
query_router = QueryRouter(multi_agent_system)


async def process_request(message, history):
    """
    Process incoming requests and route them to the appropriate agent(s).
    """
    strResponse = ""
    try:
        num_files = len(message["files"])
        if num_files > 0:
            image_path = message["files"][0]
            async for part in query_router.route_query(message, image_path):
                strResponse += part + "\n"
                yield strResponse
        else:
            async for part in query_router.route_query(message):
                strResponse += part + "\n"
                yield strResponse
    except Exception as error:
        yield "There was an error.\n" + str(error)


# Custom CSS for the Gradio interface
custom_css = """
            .message-row img {
                margin: 0px !important;
            }
            .avatar-container img {
            padding: 0px !important;
            }
            footer {visibility: hidden}; 
        """

# Create Gradio interface
with gr.Blocks(
    fill_height=True,
    fill_width=True,
    css=custom_css,
    theme=gr.themes.Soft(primary_hue=gr.themes.colors.green),
    title="Multi-Agent System Chat Interface"
) as demo:
    txtChatInput = gr.MultimodalTextbox(
        interactive=True,
        placeholder="Type your query and/or upload an image and interact with it...",
        label="User Query",
        show_label=True,
        render=False,
        file_types=["image"],
        file_count="single",
    )

    bot = gr.Chatbot(
        elem_id="chatbot",
        bubble_full_width=True,
        type="messages",
        autoscroll=True,
        avatar_images=[
            "https://ca.slack-edge.com/E01C4Q4H3CL-U04D0GXU2B1-g1a101208f57-192",
            "https://ca.slack-edge.com/E01C4Q4H3CL-U04D0GXU2B1-g1a101208f57-192",
        ],
        show_copy_button=True,
        render=False,
        min_height="650px",
        label="Type your query and/or upload an image and interact with it...",
    )
    CI = gr.ChatInterface(
        fn=process_request,
        chatbot=bot,
        examples=[
            "How can I improve my leadership skills?",
            "What are the best practices for creating a scalable AI architecture?",
            "Explain how I can manage my team better while solving technical challenges.",
        ],
        type="messages",
        title="Multi-Agent System Chat Interface",
        description="Interact with a multi-agent system to get responses tailored to your query.",
        multimodal=True,
        textbox=txtChatInput,
        fill_height=True,
        show_progress=False,
        concurrency_limit=None,
    )

# Mount Gradio app to FastAPI and run the server
if __name__ == "__main__":
    app = gr.mount_gradio_app(
        app, demo, path="/", server_name="0.0.0.0", server_port=7860
    )
    uvicorn.run(app, host="0.0.0.0", port=7860)
