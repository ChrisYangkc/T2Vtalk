import re
import sys
import subprocess, tempfile, yaml          
sys.path.insert(1, "../echomimic")         
sys.path.insert(1, "../CosyVoice")
sys.path.insert(1, "../SenseVoice")
sys.path.insert(1, "../CosyVoice/third_party/AcademiCodec")
sys.path.insert(1, "../CosyVoice/third_party/Matcha-TTS")
sys.path.insert(1, "../")
sys.path.append('./')

import torch
import torchaudio
import numpy as np

###########################################
# 1. Import SenseVoice core inference 
###########################################

from funasr import AutoModel

# SenseVoice model paths
SENSEVOICE_MODEL_PATH = "./SenseVoice/pretrained_models/SenseVoiceSmall"
VAD_MODEL_PATH = "./SenseVoice/pretrained_models/speech_fsmn_vad_zh-cn-16k-common-pytorch"

# Load model
sensevoice_model = AutoModel(
    model=SENSEVOICE_MODEL_PATH,
    vad_model=VAD_MODEL_PATH,
    vad_kwargs={"max_single_segment_time": 60000},
    trust_remote_code=True,
)

def sensevoice_inference(input_wav, language="auto"):
    """
    Read wav, call SenseVoice model, return transcribed text (with emotion tags).
    input_wav: Tuple (fs, waveform_np) or directly waveform_np (default fs=16000).
    """
    # Preprocess: convert to mono + 16k if needed
    if isinstance(input_wav, tuple):
        fs, wav_data = input_wav
        wav_data = wav_data.astype(np.float32)
        if len(wav_data.shape) > 1:
            wav_data = wav_data.mean(axis=0)
        if fs != 16000:
            resampler = torchaudio.transforms.Resample(fs, 16000)
            wav_data_torch = torch.from_numpy(wav_data).unsqueeze(0)
            wav_data_16k = resampler(wav_data_torch).squeeze(0).numpy()
        else:
            wav_data_16k = wav_data
    else:
        wav_data_16k = input_wav

    # Inference with SenseVoice
    merge_vad = True
    result = sensevoice_model.generate(
        input=wav_data_16k,
        cache={},
        language=language,
        use_itn=True,
        batch_size_s=60,
        merge_vad=merge_vad
    )

    text_output = result[0]["text"]
    print("Raw model output from SenseVoice:", result)
    print("Final transcription text:", text_output)

    return text_output

###########################################
# 2. AI model response generation based on emotion
###########################################

from openai import OpenAI

def parse_emotion_from_text(recognized_text: str):
    """
    Simply parse out the sentiment based on the string returned by SenseVoice.
    """
    lower_text = recognized_text.lower()

    if "happy" in lower_text or "😊" in recognized_text:
        return "Happy"
    elif "sad" in lower_text or "😔" in recognized_text:
        return "Sad"
    elif "angry" in lower_text or "😡" in recognized_text:
        return "Angry"
    elif "fearful" in lower_text or "😰" in recognized_text:
        return "Fearful"
    elif "surprised" in lower_text or "😮" in recognized_text:
        return "Surprised"
    elif "disgusted" in lower_text or "🤢" in recognized_text:
        return "Disgusted"
    else:
        return "Neutral"

# Set your API key
client = OpenAI(api_key="")  # Replace with your actual API key

SYSTEM_PROMPT = (
    "You are a friend with great emotional understanding and are good at expressing appropriate responses based on the user's tone and emotional state. "
    "Your replies must follow this format exactly: Generate Style: <Emotion>;Broadcast Content: <Spoken reply>. "
    "Do not sound too formal. Be natural and friendly—emojis and filler words are okay when suitable."
)

# Initialize conversation history (global or passed externally in real use)
chat_history = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

def ai_model_generate_response(user_text: str, emotion_label: str) -> str:
    """
    Uses the OpenAI API to generate a context-aware response in the required format.
    The result includes both the generation style (emotion) and the broadcast content (text to speak).
    """
    user_prompt = (
        f"The following is the user's spoken content along with the detected emotion.\n"
        f"User input: {user_text}\n"
        f"Detected emotion: {emotion_label}\n"
        f"Please reply with a natural and emotionally appropriate sentence.\n"
        f"Use this format: Generate Style: <Emotion>;Broadcast Content: <Your reply here>"
    )

    # Add current user message to history
    chat_history.append({"role": "user", "content": user_prompt})

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=chat_history,
            temperature=0.8,
            max_tokens=150,
        )
        reply = response.choices[0].message.content.strip()

        # Append AI's reply to history for continuity
        chat_history.append({"role": "assistant", "content": reply})

        return reply

    except Exception as e:
        print(f"[OpenAI Error]: {e}")
        fallback = f"Generate Style: {emotion_label};Broadcast Content: Something went wrong, please try again later~"
        chat_history.append({"role": "assistant", "content": fallback})
        return fallback

###########################################
# 3. Call CosyVoice for TTS synthesis 
###########################################

from CosyVoice.cosyvoice.cli.cosyvoice import CosyVoice2
from CosyVoice.cosyvoice.utils.file_utils import load_wav

COSYVOICE2_MODEL_PATH = "./CosyVoice/pretrained_models/CosyVoice2-0.5B"

# Initialise cosyvoice
cosyvoice = CosyVoice2(
    COSYVOICE2_MODEL_PATH,
    load_jit=False,
    load_trt=False,
    fp16=False
)

# Load a speaker tone as a prompt 
prompt_speech_16k = load_wav('./CosyVoice/asset/zero_shot_prompt.wav', 16000)

def cosyvoice_synthesize_tts(broadcast_content: str, emotion_label: str, save_wav_path="output.wav"):
    """
    Call CosyVoice to synthesize speech from text and emotion.
    broadcast_content: final spoken text
    emotion_label: emotion tag like "HAPPY", "SAD", etc.
    save_wav_path: path to save generated audio
    """
    emotion_mapping = {
        "Happy": "HAPPY",
        "Sad": "SAD",
        "Angry": "ANGRY",
        "Fearful": "FEARFUL",
        "Surprised": "SURPRISED",
        "Disgusted": "DISGUSTED",
        "Neutral": "NEUTRAL"
    }
    cosy_emotion = emotion_mapping.get(emotion_label, "NEUTRAL")

    outputs = cosyvoice.inference_instruct2(
        broadcast_content,
        cosy_emotion,
        prompt_speech_16k,
        stream=False
    )

    for i, out_item in enumerate(outputs):
        out_wav_path = save_wav_path.replace(".wav", f"_{i}.wav")
        torchaudio.save(out_wav_path, out_item["tts_speech"], cosyvoice.sample_rate)
        print(f"CosyVoice speech saved to: {out_wav_path}")

###########################################
# 4. Call EchoMimic to generate video
###########################################

def call_echomimic_infer(image_path, audio_path, config_template="./echomimic/configs/prompts/animation.yaml"):
    """
    Call echomimic/infer_audio2vid.py to generate talking head video
    - image_path: path to reference face image
    - audio_path: path to audio
    - config_template: EchoMimic config template path
    """
    with open(config_template, "r") as f:
        config_data = yaml.safe_load(f)

    config_data["test_cases"] = {image_path: [audio_path]}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_config:
        yaml.dump(config_data, tmp_config)
        temp_config_path = tmp_config.name

    cmd = [
        "python3", "echomimic/infer_audio2vid.py",
        "--config", temp_config_path,
        "-W", "512",
        "-H", "512",
        "-L", "1200",
        "--fps", "24",
        "--device", "cuda"
    ]

    print(f"\n[EchoMimic] Generating video: {audio_path} + {image_path}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    print("[EchoMimic Output]:\n", result.stdout)
    if result.stderr:
        print("[EchoMimic Error]:\n", result.stderr)

###########################################
# 5. Main pipeline: take one audio file as input demo
###########################################

def main():
    input_wav_path = "./SenseVoice/ex/happy.wav"
    language = "en"

    # 1) Load audio
    waveform, fs = torchaudio.load(input_wav_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    input_data = (fs, waveform.numpy())

    # 2) Speech to text
    recognized_text = sensevoice_inference(input_data, language)

    # 3) Emotion parsing
    emotion_label = parse_emotion_from_text(recognized_text)
    print(f"Detected emotion: {emotion_label}")

    # 4) Generate AI response
    ai_response = ai_model_generate_response(recognized_text, emotion_label)
    print(f"AI model response:\n{ai_response}")
    style_pattern = r"Generate Style:\s*(.*?);Broadcast Content:\s*(.*)$"
    match = re.search(style_pattern, ai_response)
    if not match:
        final_emotion = emotion_label
        final_broadcast_content = ai_response
    else:
        final_emotion = match.group(1).strip()
        final_broadcast_content = match.group(2).strip()

    print(f"\nEmotion used in CosyVoice: {final_emotion}")
    print(f"Content for CosyVoice TTS: {final_broadcast_content}")

    # 5) TTS with CosyVoice
    cosyvoice_synthesize_tts(
        broadcast_content=final_broadcast_content,
        emotion_label=final_emotion,
        save_wav_path="final_output.wav"
    )

    # 6) Generate video
    image_path = "a.png"
    audio_path = "final_output_0.wav"
    call_echomimic_infer(image_path, audio_path)

    print("\nPipeline finished!")

if __name__ == "__main__":
    main()
