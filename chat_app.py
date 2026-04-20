import os
import sys
import re

import gradio as gr
from PIL import Image

from chatbot_utils import default_conversation

# Gradio
examples = [
    [
        "./example_imgs/00001.jpg",
        "Please segment out the main object in the image.",
    ],
    [
        "./example_imgs/00002.jpg",
        "Please describe content of the image.",
    ],
    [
        "./example_imgs/00003.jpg",
        "Please describe and evaluate the quality of the image.",
    ],
    [
        "./example_imgs/00004.jpg",
        "Please segment out distorted regions in the image.",
    ],
    [
        "./example_imgs/00006.jpg",
        "Please give distortion types of the image and segment them.",
    ],
]


title_markdown = """
<h1 style=\"font-size: 2.4rem; margin-bottom: 0.35rem;\">Q-Ground: Image Quality Grounding with Large Multi-modality Models</h1>
<p style=\"margin-top: 0; margin-bottom: 0.25rem; font-size: 1.05rem;\"><strong>ACM MM 2024 (Oral)</strong></p>
<p style=\"margin-top: 0; font-size: 1rem;\">
    <a href=\"https://arxiv.org/abs/2407.17035\" target=\"_blank\">arXiv: 2407.17035</a>
    &nbsp;|&nbsp;
    <a href=\"https://github.com/Q-Future/Q-Ground\" target=\"_blank\">GitHub: Q-Future/Q-Ground</a>
</p>
"""

description = """
This is the online demo of Q-Ground.
"""

from evaluate_model import parse_args, load_model, model_inference

args = parse_args(sys.argv[1:])
model, tokenizer, args = load_model(args)


def _safe_file_stem(text, fallback="result", max_len=48):
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if not text:
        text = fallback
    return text[:max_len]


def _run_inference_and_update(state, prompt, image):
    out_txt, out_masks = model_inference(prompt, image, model, tokenizer, args, return_all=True)

    print(f"User: {prompt}\n Assistant: {out_txt}")

    if out_masks is not None:
        if hasattr(image, "filename") and image.filename:
            save_name = f"mask_{os.path.basename(image.filename)}"
        else:
            save_name = f"mask_{_safe_file_stem(prompt)}.png"
        save_path = os.path.join(args.save_results_path, save_name)
        print(f"Saving mask to {save_path}")
        out_masks.save(save_path)
        state.append_message(state.roles[1], (out_txt, state.process_image(out_masks, "Default")))
    else:
        state.append_message(state.roles[1], out_txt)

    return state, state.to_gradio_chatbot()

def qground_chat(state, message, image):
    if state is None:
        state = default_conversation.copy()

    message = (message or "").strip()
    if image is None:
        gr.Warning("Please upload an image first.")
        return state, state.to_gradio_chatbot(), gr.update(value=message)
    if not message:
        gr.Warning("Please enter a question or instruction.")
        return state, state.to_gradio_chatbot(), gr.update(value="")

    state.append_message(state.roles[0], (message, image, None))

    try:
        state, chatbot_view = _run_inference_and_update(state, message, image)
    except Exception as exc:
        err_msg = f"Inference failed: {type(exc).__name__}: {exc}"
        print(err_msg)
        state.append_message(state.roles[1], err_msg)
        chatbot_view = state.to_gradio_chatbot()

    return state, chatbot_view, gr.update(value="")


def load_demo():
    state = default_conversation.copy()
    return state


def clear_history():
    state = default_conversation.copy()
    return state, state.to_gradio_chatbot(), gr.update(value=""), gr.update(value=None)


def regenerate(state):
    if state is None:
        state = default_conversation.copy()
        gr.Warning("No previous result found.")
        return state, state.to_gradio_chatbot()

    if len(state.messages) < 2:
        gr.Warning("No previous turn to regenerate.")
        return state, state.to_gradio_chatbot()

    if state.messages and state.messages[-1][0] == state.roles[1]:
        state.messages.pop()

    if not state.messages:
        gr.Warning("No previous user message found.")
        return state, state.to_gradio_chatbot()

    prev_human_msg = state.messages[-1]
    if prev_human_msg[0] != state.roles[0] or type(prev_human_msg[1]) not in (tuple, list):
        gr.Warning("The last turn cannot be regenerated.")
        return state, state.to_gradio_chatbot()

    prompt, image = prev_human_msg[1][0], prev_human_msg[1][1]
    prev_human_msg[1] = (prompt, image, None)

    try:
        state, chatbot_view = _run_inference_and_update(state, prompt, image)
    except Exception as exc:
        err_msg = f"Regenerate failed: {type(exc).__name__}: {exc}"
        print(err_msg)
        state.append_message(state.roles[1], err_msg)
        chatbot_view = state.to_gradio_chatbot()

    return state, chatbot_view


if args.debug:
    state = default_conversation.copy()
    for eg in examples:
        # qground_chat(state, examples[0][1], Image.open(eg[0]))
        qground_chat(state, eg[1], Image.open(eg[0]))
    exit()

textbox = gr.Textbox(show_label=False, placeholder="Enter text and press ENTER", container=False)
with gr.Blocks(title="Q-Ground demo") as demo:
    state = gr.State()
    gr.Markdown(title_markdown) 
    gr.Markdown("Upload one image, ask a quality question or segmentation request, and review both text and mask overlay response.")

    with gr.Row():
        with gr.Column(scale=3):
            imagebox = gr.Image(type="pil")
        
            gr.Examples(examples=examples, inputs=[imagebox, textbox])

        with gr.Column(scale=7):
            chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    height=768,
                    show_copy_button=True,
                    bubble_full_width=False,
                )
        
            with gr.Row():
                with gr.Column(scale=7):
                    textbox.render()
                with gr.Column(scale=1, min_width=50):
                    submit_btn = gr.Button(value="Send", variant="primary")

            with gr.Row(elem_id="buttons") as button_row:
                regenerate_btn = gr.Button(value="🔄  Regenerate")
                clear_btn = gr.Button(value="🗑️  Clear")
        
        textbox.submit(
            qground_chat,
            [state, textbox, imagebox],
            [state, chatbot, textbox],
        )

        submit_btn.click(
            qground_chat,
            [state, textbox, imagebox],
            [state, chatbot, textbox],
        )
        
        clear_btn.click(
            clear_history,
            outputs=[state, chatbot, textbox, imagebox]
        )

        regenerate_btn.click(
            regenerate,
            [state],
            [state, chatbot]
        )

    demo.load(load_demo, outputs=[state])
            
    demo.queue()
    demo.launch(share=True, server_port=26999)