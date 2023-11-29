import gradio as gr
from text_to_video import model_t2v_fun,setup_seed
from omegaconf import OmegaConf
import torch
import imageio
import os
import cv2
import pandas as pd
import torchvision
import random
config_path = "/mnt/petrelfs/zhouyan/project/lavie-release/base/configs/sample.yaml"
args = OmegaConf.load("/mnt/petrelfs/zhouyan/project/lavie-release/base/configs/sample.yaml")
device = "cuda" if torch.cuda.is_available() else "cpu"
# ------- get model ---------------
model_t2V = model_t2v_fun(args)
model_t2V.to(device)
if device == "cuda":
    model_t2V.enable_xformers_memory_efficient_attention()

# model_t2V.enable_xformers_memory_efficient_attention()
css = """
h1 {
  text-align: center;
}
#component-0 {
  max-width: 730px;
  margin: auto;
}
"""

def infer(prompt, seed_inp, ddim_steps,cfg):
    if seed_inp!=-1:
        setup_seed(seed_inp)
    else:
        seed_inp = random.choice(range(10000000))
        setup_seed(seed_inp)
    videos = model_t2V(prompt, video_length=16, height = 320, width= 512, num_inference_steps=ddim_steps, guidance_scale=cfg).video
    print(videos[0].shape)
    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)
    torchvision.io.write_video(args.output_folder + prompt[0:30].replace(' ', '_') + '-'+str(seed_inp)+'-'+str(ddim_steps)+'-'+str(cfg)+ '-.mp4', videos[0], fps=8)
    # imageio.mimwrite(args.output_folder + prompt.replace(' ', '_') + '.mp4', videos[0], fps=8)
    # video = cv2.VideoCapture(args.output_folder + prompt.replace(' ', '_') + '.mp4')
    # video = imageio.get_reader(args.output_folder + prompt.replace(' ', '_') + '.mp4',  'ffmpeg')


    # video = model_t2V(prompt, seed_inp, ddim_steps)

    return args.output_folder + prompt[0:30].replace(' ', '_') + '-'+str(seed_inp)+'-'+str(ddim_steps)+'-'+str(cfg)+ '-.mp4'

print(1)

# def clean():
#     return gr.Image.update(value=None, visible=False), gr.Video.update(value=None)
def clean():
    return gr.Video.update(value=None)

title = """
    <div style="text-align: center; max-width: 700px; margin: 0 auto;">
        <div
        style="
            display: inline-flex;
            align-items: center;
            gap: 0.8rem;
            font-size: 1.75rem;
        "
        >
        <h1 style="font-weight: 900; margin-bottom: 7px; margin-top: 5px;">
            Intern·Vchitect (Text-to-Video)
        </h1>
        </div>
        <p style="margin-bottom: 10px; font-size: 94%">
        Apply Intern·Vchitect to generate a video 
        </p>
    </div>
"""

# print(1)
with gr.Blocks(css='style.css') as demo:
    gr.Markdown("<font color=red size=10><center>LaVie: Text-to-Video generation</center></font>")
    with gr.Column():
        with gr.Row(elem_id="col-container"):
            # inputs = [prompt, seed_inp, ddim_steps]
            # outputs = [video_out]
            with gr.Column():
                    
                prompt = gr.Textbox(value="a teddy bear walking on the street", label="Prompt", placeholder="enter prompt", show_label=True, elem_id="prompt-in", min_width=200, lines=2)
                
                ddim_steps = gr.Slider(label='Steps', minimum=50, maximum=300, value=50, step=1)
                seed_inp = gr.Slider(value=-1,label="seed (for random generation, use -1)",show_label=True,minimum=-1,maximum=2147483647)
                cfg = gr.Number(label="guidance_scale",value=7)
                # seed_inp = gr.Slider(label="Seed", minimum=0, maximum=2147483647, step=1, value=400, elem_id="seed-in")

                # with gr.Row():
                #     # control_task = gr.Dropdown(label="Task", choices=["Text-2-video", "Image-2-video"], value="Text-2-video", multiselect=False, elem_id="controltask-in")
                #     ddim_steps = gr.Slider(label='Steps', minimum=50, maximum=300, value=250, step=1)
                #     seed_inp = gr.Slider(label="Seed", minimum=0, maximum=2147483647, step=1, value=123456, elem_id="seed-in")
                
                # ddim_steps = gr.Slider(label='Steps', minimum=50, maximum=300, value=250, step=1)
                # ex = gr.Examples(
                #     examples = [['a corgi walking in the park at sunrise, oil painting style',400,50,7],
                #             ['a cut teddy bear reading a book in the park, oil painting style, high quality',700,50,7],
                #             ['an epic tornado attacking above a glowing city at night, the tornado is made of smoke, highly detailed',230,50,7],
                #             ['a jar filled with fire, 4K video, 3D rendered, well-rendered',400,50,7],
                #             ['a teddy bear walking in the park, oil painting style, high quality',400,50,7],
                #             ['a teddy bear walking on the street, 2k, high quality',100,50,7],
                #             ['a panda taking a selfie, 2k, high quality',400,50,7],
                #             ['a polar bear playing drum kit in NYC Times Square, 4k, high resolution',400,50,7],
                #             ['jungle river at sunset, ultra quality',400,50,7],
                #             ['a shark swimming in clear Carribean ocean, 2k, high quality',400,50,7],
                #             ['A steam train moving on a mountainside by Vincent van Gogh',230,50,7],
                #             ['a confused grizzly bear in calculus class',1000,50,7]],
                #     fn = infer,
                #     inputs=[prompt, seed_inp, ddim_steps,cfg],
                #     # outputs=[video_out],
                #     cache_examples=False,
                #     examples_per_page = 6
                # )
                # ex.dataset.headers = [""]

            with gr.Column():
                submit_btn = gr.Button("Generate video")
                clean_btn = gr.Button("Clean video")
            # submit_btn = gr.Button("Generate video", size='sm')
                # video_out = gr.Video(label="Video result", elem_id="video-output", height=320, width=512)
                video_out = gr.Video(label="Video result", elem_id="video-output")
            # with gr.Row():
            #     video_out = gr.Video(label="Video result", elem_id="video-output", height=320, width=512)
            #     submit_btn = gr.Button("Generate video", size='sm')
            

                # video_out = gr.Video(label="Video result", elem_id="video-output", height=320, width=512)
            inputs = [prompt, seed_inp, ddim_steps,cfg]
            outputs = [video_out]
        # gr.Examples(
        #     value = [['An astronaut riding a horse',123,50],
        #              ['a panda eating bamboo on a rock',123,50],
        #              ['Spiderman is surfing',123,50]],
        #     label = "example of sampling",
        #     show_label = True,
        #     headers = ['prompt','seed','steps'],
        #     datatype = ['str','number','number'],
        #     row_count=4,
        #     col_count=(3,"fixed")
        # )
        ex = gr.Examples(
            examples = [['a corgi walking in the park at sunrise, oil painting style',400,50,7],
                    ['a cut teddy bear reading a book in the park, oil painting style, high quality',700,50,7],
                    ['an epic tornado attacking above a glowing city at night, the tornado is made of smoke, highly detailed',230,50,7],
                    ['a jar filled with fire, 4K video, 3D rendered, well-rendered',400,50,7],
                    ['a teddy bear walking in the park, oil painting style, high quality',400,50,7],
                    ['a teddy bear walking on the street, 2k, high quality',100,50,7],
                    ['a panda taking a selfie, 2k, high quality',400,50,7],
                    ['a polar bear playing drum kit in NYC Times Square, 4k, high resolution',400,50,7],
                    ['jungle river at sunset, ultra quality',400,50,7],
                    ['a shark swimming in clear Carribean ocean, 2k, high quality',400,50,7],
                    ['A steam train moving on a mountainside by Vincent van Gogh',230,50,7],
                    ['a confused grizzly bear in calculus class',1000,50,7]],
            fn = infer,
            inputs=[prompt, seed_inp, ddim_steps,cfg],
            outputs=[video_out],
            cache_examples=True,
        )
        ex.dataset.headers = [""]
        
    # control_task.change(change_task_options, inputs=[control_task], outputs=[canny_opt, hough_opt, normal_opt], queue=False)
    # submit_btn.click(clean, inputs=[], outputs=[video_out], queue=False)
    clean_btn.click(clean, inputs=[], outputs=[video_out], queue=False)
    submit_btn.click(infer, inputs, outputs)
    # share_button.click(None, [], [], _js=share_js)

    print(2)
demo.queue(max_size=12).launch(server_name="0.0.0.0", server_port=7860)


