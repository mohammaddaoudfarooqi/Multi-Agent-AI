from fastapi import FastAPI
import uvicorn
import gradio as gr
from interface import demo

# Initialize FastAPI app
app = FastAPI()

# Mount Gradio app to FastAPI and run the server
if __name__ == "__main__":
    app = gr.mount_gradio_app(
        app, demo, path="/", server_name="0.0.0.0", server_port=7860
    )
    uvicorn.run(app, host="0.0.0.0", port=7860)
