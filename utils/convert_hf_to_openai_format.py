import whisper
import sip_whisper
import re
import torch
import os
from safetensors.torch import load_file
from pathlib import Path

from whisper import Whisper
from sip_whisper import Whisper as sip_Whisper

from utils.config_dataclasses import Config, InferenceConfig

WHISPER_OPENAI_MODEL_NAME = "whisper_openai_format.pt"

def load_whisper_model(config: InferenceConfig, device: torch.device) -> Whisper | sip_Whisper:
    """
    Load a model from whisper or sip_whipser depending on the config.

    config: Config

    Returns: Whisper | sip_Whisper
    """
    if not config.model_path:
        if config.extract_logits:
            return sip_whisper.load_model(config.model_type, device=device)
        else:
            return whisper.load_model(config.model_type, device=device)
    else:
        return load_whisper_from_hf_checkpoint(config, device=device)

def load_whisper_from_hf_checkpoint(config: InferenceConfig, device: torch.device) -> Whisper | sip_Whisper:
    """
    Load a Whisper instance from an HF checkpoint.

    config: Config
    """
    model_type = config.model_type
    if model_type is None:
        type_list = [s for s in whisper.available_models() if s in config.model_path]

        if len(type_list) == 0:
            raise ValueError(f"Can't derive model type/size from path, no keyword: {config.model_path}")
        else:
            model_type = type_list[0]

    if not (config.model_path/WHISPER_OPENAI_MODEL_NAME).is_file():
        convert_hf_model_to_openai_whisper(hf_checkpoint_file_path=config.model_path, safe_file=WHISPER_OPENAI_MODEL_NAME, model_type=model_type)

    if config.extract_logits:
        model = sip_whisper.load_model(str(config.model_path/WHISPER_OPENAI_MODEL_NAME), device=device)
    else:
        model = whisper.load_model(str(config.model_path/WHISPER_OPENAI_MODEL_NAME), device=device)

    model.set_alignment_heads(whisper._ALIGNMENT_HEADS[model_type])  # see last line of whisper/__init__.load_model()

    if not isinstance(model, sip_Whisper if config.extract_logits else Whisper):
        raise ValueError(f"Loading the model from {config.model_path} wasn't successful!")
    return model

def convert_hf_model_to_openai_whisper(
        hf_checkpoint_file_path: Path,
        safe_file: str = WHISPER_OPENAI_MODEL_NAME,
        model_type: str = None) -> Path:
    """
    Converts a hf model (.safetensors) to a whisper compatible .pt file and saves it on the disk.
    """

    hf_model_path = os.path.join(os.getcwd(), hf_checkpoint_file_path, "model.safetensors")
    if not os.path.isfile(hf_model_path):
        raise FileNotFoundError(hf_model_path)

    openai_model_path = os.path.join(hf_checkpoint_file_path, safe_file)
    if os.path.isfile(openai_model_path):
        raise FileExistsError(f"{openai_model_path} already exists!")

    # Load HF Model
    hf_state_dict = load_file(hf_model_path)

    # Rename layers bc openai uses different ones in their module
    for key in list(hf_state_dict.keys())[:]:
        new_key = hf_to_whisper_states(key)
        hf_state_dict[new_key] = hf_state_dict.pop(key)

    #load ORIGINAL(!) whisper_folder model from module
    model_from_module = whisper.load_model(model_type)
    module_state = model_from_module.state_dict()

    for k in hf_state_dict.keys():
        if module_state[k].shape != hf_state_dict[k].shape:
            print(f"{k}: {module_state[k].shape, hf_state_dict[k].shape}")

    saved_model_path = hf_checkpoint_file_path/safe_file

    # Save it with the adapted hf_state_dict
    torch.save({
        "dims": model_from_module.dims.__dict__,
        "model_state_dict": hf_state_dict
    }, saved_model_path)

    return saved_model_path

# based on https://github.com/openai/whisper/discussions/830
def hf_to_whisper_states(text):
    text = re.sub('.layers.', '.blocks.', text)
    text = re.sub('.self_attn.', '.attn.', text)
    text = re.sub('.q_proj.', '.query.', text)
    text = re.sub('.k_proj.', '.key.', text)
    text = re.sub('.v_proj.', '.value.', text)
    text = re.sub('.out_proj.', '.out.', text)
    text = re.sub('.fc1.', '.mlp.0.', text)
    text = re.sub('.fc2.', '.mlp.2.', text)
    text = re.sub('.fc3.', '.mlp.3.', text)
    text = re.sub('.fc3.', '.mlp.3.', text)
    text = re.sub('.encoder_attn.', '.cross_attn.', text)
    text = re.sub('.cross_attn.ln.', '.cross_attn_ln.', text)
    text = re.sub('.embed_positions.weight', '.positional_embedding', text)
    text = re.sub('.embed_tokens.', '.token_embedding.', text)
    text = re.sub('model.', '', text)
    text = re.sub('attn.layer_norm.', 'attn_ln.', text)
    text = re.sub('.final_layer_norm.', '.mlp_ln.', text)
    text = re.sub('encoder.layer_norm.', 'encoder.ln_post.', text)
    text = re.sub('decoder.layer_norm.', 'decoder.ln.', text)
    text = re.sub('proj_out.weight', 'decoder.token_embedding.weight', text)
    return text






def main():
    hf_model_path = "/home/damian/Desktop/masterarbeit/code/Speech-Intelligibility-Prediction-using-pre-trained-State-of-the-Art-Speech-Recognition/trained_models/training_output_test_folder"
    model = load_whisper_model(Path(hf_model_path), explicit_model_type="tiny")
    #################################################################
    audio = whisper.load_audio("../sample_audio_small.mp3")
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

    # detect the spoken language
    #_, probs = model.detect_language(mel)
    #print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions(without_timestamps=False)
    result = whisper.decode(model, mel, options)

    # print the recognized text
    print(result.text)

if __name__ == '__main__':
    main()