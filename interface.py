import gradio as gr
from agents import MultiAgentSystem
from orchestrator import QueryRouter

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
            async for part in query_router.route_query(message["text"]):
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
    title="Multi-Agent System Chat Interface",
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
