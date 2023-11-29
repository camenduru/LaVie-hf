import gradio as gr

with gr.Blocks() as demo:
    prompt = gr.Textbox(label="Prompt", placeholder="enter prompt", show_label=True, elem_id="prompt-in")
demo.launch(server_name="0.0.0.0")