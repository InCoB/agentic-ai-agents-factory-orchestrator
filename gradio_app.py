# gradio_app.py
import gradio as gr
import asyncio
import logging
import io
from typing import Tuple
from app.agents.agent_factory import AgentFactory
from app.utils.logger import setup_logging

# --- Custom Logging Handler for Gradio ---
class StringBufferHandler(logging.Handler):
    """
    A logging handler that writes log messages to an in-memory string buffer.
    """
    def __init__(self):
        super().__init__()
        self.log_buffer = io.StringIO()

    def emit(self, record):
        try:
            log_entry = self.format(record)
            self.log_buffer.write(log_entry + "\n")
        except Exception:
            self.handleError(record)

    def get_logs(self) -> str:
        return self.log_buffer.getvalue()

    def clear(self):
        self.log_buffer.truncate(0)
        self.log_buffer.seek(0)

# --- Setup logging ---
setup_logging()
# Get the root logger and add our custom handler.
logger = logging.getLogger()
string_handler = StringBufferHandler()
formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
string_handler.setFormatter(formatter)
logger.addHandler(string_handler)

# --- Function to Run the OperatorAgent ---
async def process_with_operator(prompt: str, dynamic: bool) -> str:
    """Asynchronous function to process the prompt with the operator agent."""
    factory = AgentFactory()
    operator_agent = factory.get_agent("OperatorAgent")
    operator_agent.use_dynamic_workflow = dynamic
    return await operator_agent.process(prompt)

def run_operator(prompt: str, dynamic: bool = True) -> Tuple[str, str]:
    """
    Runs the operator agent with the given prompt and dynamic workflow flag.
    Returns a tuple of the final operator output and the captured logs.
    
    Args:
        prompt (str): The user's input prompt
        dynamic (bool): Whether to use dynamic workflow generation
    
    Returns:
        Tuple[str, str]: (operator output, logs)
    """
    if not prompt or not prompt.strip():
        return "Error: Please enter a prompt.", "No logs - empty prompt provided."
    
    # Clear previous logs
    string_handler.clear()
    
    try:
        # Create event loop if it doesn't exist
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run the async operation
        final_response = loop.run_until_complete(process_with_operator(prompt, dynamic))
        
    except Exception as e:
        logger.error(f"Error processing prompt: {str(e)}", exc_info=True)
        final_response = f"Error: {str(e)}\n\nPlease check the logs for more details."
    
    # Retrieve logs
    logs = string_handler.get_logs()
    return final_response, logs

# --- Gradio Interface Setup ---
with gr.Blocks(title="AI Agent Factory", theme=gr.themes.Soft()) as iface:
    gr.Markdown("""
    # 🤖 AI Agent Factory - Dynamic Workflow
    Enter a prompt to execute an AI workflow using multiple specialized agents.
    The system can either use a dynamic workflow (generated based on your prompt) or a fixed workflow.
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            prompt_input = gr.Textbox(
                lines=3,
                label="User Prompt",
                placeholder="Enter your prompt here (e.g., 'Find me a stock to invest right now')",
                info="Be specific about what you want the AI agents to do."
            )
        with gr.Column(scale=1):
            dynamic_checkbox = gr.Checkbox(
                label="Use Dynamic Workflow",
                value=True,
                info="If checked, the system will generate a custom workflow for your prompt."
            )
    
    with gr.Row():
        submit_btn = gr.Button("🚀 Run Analysis", variant="primary")
        clear_btn = gr.Button("🧹 Clear")
    
    with gr.Row():
        output_text = gr.Textbox(
            label="Analysis Output",
            lines=10,
            show_copy_button=True
        )
    
    with gr.Accordion("Detailed Logs", open=False):
        logs_output = gr.Textbox(
            label="System Logs",
            lines=15,
            show_copy_button=True
        )
    
    # Event handlers
    submit_btn.click(
        fn=run_operator,
        inputs=[prompt_input, dynamic_checkbox],
        outputs=[output_text, logs_output]
    )
    
    clear_btn.click(
        lambda: ("", ""),
        inputs=[],
        outputs=[output_text, logs_output]
    )
    
    # Examples
    gr.Examples(
        examples=[
            ["Find me a stock to invest right now", True],
            ["Analyze the latest tech industry trends", True],
            ["What are the best performing stocks in the renewable energy sector?", True]
        ],
        inputs=[prompt_input, dynamic_checkbox],
        outputs=[output_text, logs_output],
        fn=run_operator,
        cache_examples=True
    )

if __name__ == "__main__":
    iface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,
        show_error=True,
        debug=True,
        auth=None,
        inbrowser=True
    )
