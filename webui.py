# webui.py
import os
import re
import gradio as gr
from T2Vtalk_video import (
    sensevoice_inference,
    parse_emotion_from_text,
    ai_model_generate_response,
    cosyvoice_synthesize_tts,
    call_echomimic_infer
)
import torchaudio
import glob
import uuid
from datetime import datetime

DEFAULT_TTS_DIR = "./tts_outputs"

# Map friendly labels to your image paths
label_to_path = {
    "Alice": "./ex/a.png",
    "Bob":   "./ex/b.png",
    "Carol": "./ex/H.png",
    "Hank":  "./ex/00002.png",
}

def run_pipeline(reference_image: str, input_wav: str, language: str, tts_dir: str = DEFAULT_TTS_DIR):
    os.makedirs(tts_dir, exist_ok=True)

    waveform, fs = torchaudio.load(input_wav)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    input_data = (fs, waveform.squeeze(0).numpy())

    text = sensevoice_inference(input_data, language)
    emo     = parse_emotion_from_text(text)
    ai_resp = ai_model_generate_response(text, emo)

    prefix = os.path.join(tts_dir, f"{uuid.uuid4()}.wav")
    cosyvoice_synthesize_tts(
        broadcast_content=ai_resp.split("Broadcast Content:")[-1],
        emotion_label=emo,
        save_wav_path=prefix
    )

    pattern = prefix.replace(".wav", "_*.wav")
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(f"No TTS file matching {pattern}")
    os.replace(candidates[0], prefix)
    tts_path = prefix

    call_echomimic_infer(reference_image, tts_path)
    date_str = datetime.now().strftime("%Y%m%d")
    pattern = os.path.join(os.getcwd(), "output", date_str, "*", "*_withaudio.mp4")
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(f"No video found matching {pattern}")
    video_path = max(candidates, key=os.path.getmtime)

    return text, ai_resp, tts_path, video_path

def extract_broadcast(text: str) -> str:
    marker = "Broadcast Content:"
    return text.split(marker,1)[1].strip() if marker in text else text.strip()

def chat_fn(label, input_wav, language, history):
    reference_image = label_to_path[label]
    text, ai_resp, tts_path, video_path = run_pipeline(reference_image, input_wav, language)

    clean_user = re.sub(r"<.*?>|\[.*?\]", "", text).strip()
    history = history + [[clean_user, None]]
    bot_text = extract_broadcast(ai_resp)
    history = history + [[None, bot_text]]
    return history, video_path

with gr.Blocks(css="""
    /* Inline radio items */
    #ref-radio .gradio-radio-item {
        display: inline-block !important;
        margin-right: 1rem;
    }
    .chatbot .user { background-color: #D1E7DD; }
    .chatbot .bot  { background-color: #F8F9FA; }
    #send-btn {
        background-color: orange !important;
        border-color: orange !important;
        color: white !important;
    }
    #send-btn:hover {
        background-color: darkorange !important;
        border-color: darkorange !important;
    }
""") as demo:

    gr.Markdown("## 🎤🤖 T2Vtalk Chat Interface")

    with gr.Row():
        with gr.Column(scale=1):
            ref_img_selector = gr.Radio(
                choices=list(label_to_path.keys()),
                value=list(label_to_path.keys())[0],
                label="Choose a Reference Face",
                elem_id="ref-radio"
            )

            ref_img_display = gr.Image(
                value=label_to_path[ref_img_selector.value],
                label="Preview Selected Face",
                interactive=False,
                width=256, height=256
            )
            ref_img_selector.change(
                fn=lambda lbl: label_to_path[lbl],
                inputs=ref_img_selector,
                outputs=ref_img_display
            )

            audio_in = gr.Audio(label="Input WAV File", type="filepath")
            lang     = gr.Dropdown(choices=["auto","en","zh"], value="auto", label="Language")

            with gr.Row():
                clear = gr.Button("Clear")
                send  = gr.Button("Send", elem_id="send-btn")

        with gr.Column(scale=1):
            chatbot   = gr.Chatbot(label="Dialogue")
            video_out = gr.Video(label="Lip-Sync Video", width=256, height=256)

    clear.click(lambda: ([], None), None, [chatbot, video_out])
    send.click(
        fn=chat_fn,
        inputs=[ref_img_selector, audio_in, lang, chatbot],
        outputs=[chatbot, video_out],
        queue=True
    )

if __name__ == "__main__":
    demo.launch(share=True)
