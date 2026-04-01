import os
import sys
import subprocess
import tempfile
import importlib
import torch
import torchaudio
import numpy as np
import soundfile as sf
from pathlib import Path

import folder_paths
import comfy.model_management as mm


NODE_DIR = Path(__file__).resolve().parent
LEMAS_TTS_DIR = NODE_DIR / "LEMAS-TTS"
MODEL_DIR = Path(folder_paths.models_dir) / "lemas_tts"


def _ensure_in_syspath(path):
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)


_ensure_in_syspath(LEMAS_TTS_DIR)


# ---------------------------------------------------------------------------
# Lazy pip installer
# ---------------------------------------------------------------------------

def _pip_install(*packages):
    missing = []
    for pkg in packages:
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)
    if not missing:
        return
    print(f"[LEMAS-TTS] Installing missing packages: {missing}")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install"] + missing,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for pkg in missing:
        try:
            importlib.invalidate_caches()
            importlib.import_module(pkg)
        except ImportError:
            pass


def _ensure_deps():
    """Install only the packages that are actually missing."""
    _pip_install(
        "regex", "tqdm", "hydra", "omegaconf", "matplotlib",
        "huggingface_hub", "pydub", "transformers", "vocos",
        "torchdiffeq", "librosa", "x_transformers",
        "jieba", "pypinyin", "pypinyin_dict", "langid",
        "uroman", "zhconv", "num2words", "phonemizer",
        "espeakng_loader", "unidecode", "inflect",
    )


# ---------------------------------------------------------------------------
# HuggingFace model downloader
# ---------------------------------------------------------------------------

HF_REPO = "LEMAS-Project/LEMAS-TTS"
_KNOWN_PROJECTS = ["multilingual_grl", "multilingual_prosody"]

_HF_SUBFOLDERS = [
    ("pretrained_models/ckpts/multilingual_grl", "ckpts/multilingual_grl"),
    ("pretrained_models/ckpts/vocos-mel-24khz", "ckpts/vocos-mel-24khz"),
    ("pretrained_models/data/multilingual_grl", "data/multilingual_grl"),
    ("pretrained_models/espeak-ng-data", "espeak-ng-data"),
]

_OPTIONAL_SUBFOLDERS = [
    ("pretrained_models/ckpts/multilingual_prosody", "ckpts/multilingual_prosody"),
    ("pretrained_models/data/multilingual_prosody", "data/multilingual_prosody"),
    ("pretrained_models/ckpts/prosody_encoder", "ckpts/prosody_encoder"),
    ("pretrained_models/uvr5", "uvr5"),
]

_download_done = False


def _download_from_hf():
    global _download_done
    if _download_done:
        return
    _download_done = True

    _pip_install("huggingface_hub")
    from huggingface_hub import hf_hub_download, list_repo_files
    import shutil

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    all_subfolders = _HF_SUBFOLDERS + _OPTIONAL_SUBFOLDERS
    for hf_path, local_path in all_subfolders:
        local_dir = MODEL_DIR / local_path
        if local_dir.exists() and any(local_dir.iterdir()):
            continue
        print(f"[LEMAS-TTS] Downloading {local_path} from HF...")
        try:
            files = list_repo_files(repo_id=HF_REPO, repo_type="model")
            matching = [f for f in files if f.startswith(hf_path + "/")]
            for hf_file in matching:
                rel = hf_file[len(hf_path) + 1:]
                local_file = MODEL_DIR / local_path / rel
                local_file.parent.mkdir(parents=True, exist_ok=True)
                if local_file.exists():
                    continue
                cached = hf_hub_download(
                    repo_id=HF_REPO,
                    filename=hf_file,
                    repo_type="model",
                )
                shutil.copy2(cached, local_file)
        except Exception as e:
            print(f"[LEMAS-TTS] Warning: could not download {local_path}: {e}")


def _ensure_models():
    need_download = False
    for _, local_path in _HF_SUBFOLDERS:
        local_dir = MODEL_DIR / local_path
        if not local_dir.exists() or not any(local_dir.iterdir()):
            need_download = True
            break
    if need_download:
        _download_from_hf()


# ---------------------------------------------------------------------------
# Path resolution (Global to avoid re-import spam)
# ---------------------------------------------------------------------------

PRETRAINED_ROOT = MODEL_DIR
CKPTS_ROOT = PRETRAINED_ROOT / "ckpts"

os.environ["LEMAS_PRETRAINED_ROOT"] = str(PRETRAINED_ROOT)

ESPEAK_DATA_DIR = PRETRAINED_ROOT / "espeak-ng-data"
if ESPEAK_DATA_DIR.is_dir():
    os.environ["ESPEAK_DATA_PATH"] = str(ESPEAK_DATA_DIR)
    os.environ["ESPEAKNG_DATA_PATH"] = str(ESPEAK_DATA_DIR)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_wav(audio_path, sr=16000, channel=1):
    audio, raw_sr = torchaudio.load(audio_path)
    audio = audio.T if len(audio.shape) > 1 and audio.shape[1] == 2 else audio
    audio = audio / torch.max(torch.abs(audio))
    audio = audio.squeeze().float()
    if channel == 1 and len(audio.shape) == 2:
        audio = audio.mean(dim=0, keepdim=True)
    elif channel == 2 and len(audio.shape) == 1:
        audio = torch.stack((audio, audio))
    if raw_sr != sr:
        audio = torchaudio.functional.resample(audio.squeeze(), raw_sr, sr)
    audio = torch.clip(audio, -0.999, 0.999).squeeze()
    return audio


def audio_dict_to_path(audio_dict):
    """Safely writes ComfyUI audio dictionary to a temporary wav file."""
    waveform = audio_dict["waveform"]
    sample_rate = audio_dict["sample_rate"]
    
    # ComfyUI's waveform is typically [batch, channels, samples]
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.squeeze(0) # [channels, samples]
        # Mix down stereo to mono to avoid soundfile crash (format not recognised)
        if waveform.dim() > 1 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        waveform = waveform.squeeze().cpu().numpy()
    elif isinstance(waveform, np.ndarray):
        if waveform.ndim > 1 and waveform.shape[0] > 1:
            waveform = np.mean(waveform, axis=0)
        waveform = waveform.squeeze()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    # sf.write requires a 1D array for mono audio
    sf.write(tmp.name, waveform, int(sample_rate), format="wav", subtype="PCM_24")
    tmp.close()
    return tmp.name


def wav_to_audio_dict(wav_path, target_sr=None):
    audio, sr = torchaudio.load(wav_path)
    if target_sr is not None and sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
        sr = target_sr
    if audio.dim() == 2:
        audio = audio.unsqueeze(0)
    return {"waveform": audio.cpu(), "sample_rate": sr}


# ---------------------------------------------------------------------------
# UVR5 Denoiser
# ---------------------------------------------------------------------------

class UVR5Denoiser:
    def __init__(self, model_dir):
        self.code_dir = str(LEMAS_TTS_DIR / "uvr5")
        self.model_dir = model_dir
        self.model = self._load_model()

    def _load_model(self):
        if self.code_dir not in sys.path:
            sys.path.append(self.code_dir)
        import json
        from multiprocess_cuda_infer import ModelData, Inference
        model_path = os.path.join(self.model_dir, "Kim_Vocal_1.onnx")
        config_path = os.path.join(self.model_dir, "MDX-Net-Kim-Vocal1.json")
        with open(config_path, "r", encoding="utf-8") as f:
            configs = json.load(f)
        model_data = ModelData(
            model_path=model_path,
            audio_path=self.model_dir,
            result_path=self.model_dir,
            device="cpu",
            process_method="MDX-Net",
            base_dir=self.model_dir,
            **configs,
        )
        uvr5_model = Inference(model_data, "cpu")
        uvr5_model.load_model(model_path, 1)
        return uvr5_model

    def denoise(self, audio_path):
        input_audio = load_wav(audio_path, sr=44100, channel=2)
        output_audio = self.model.demix_base({0: input_audio.squeeze()}, is_match_mix=False)
        return output_audio.squeeze().T.numpy(), 44100


# ---------------------------------------------------------------------------
# Global caches
# ---------------------------------------------------------------------------

_tts_model_cache = {}
_uvr5_cache = {}


# ---------------------------------------------------------------------------
# Nodes (Refactored for Usability)
# ---------------------------------------------------------------------------

class LEMAS_TTS_ModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (_KNOWN_PROJECTS, {
                    "default": "multilingual_prosody",
                    "tooltip": "Choose the model architecture. 'grl' is the baseline. 'prosody' transfers emotions better.\nВыберите архитектуру. 'grl' — базовая. 'prosody' лучше передает интонацию и эмоции."
                }),
            }
        }

    RETURN_TYPES = ("LEMAS_TTS_MODEL",)
    RETURN_NAMES = ("tts_model",)
    FUNCTION = "load"
    CATEGORY = "audio/LEMAS-TTS"

    def load(self, model_name):
        _ensure_deps()
        _ensure_models()
        
        from lemas_tts.api import TTS

        # 1. Auto-resolve checkpoint path
        ckpt_dir = CKPTS_ROOT / model_name
        ckpt_path = ""
        if ckpt_dir.exists():
            ckpts = list(ckpt_dir.glob("*.safetensors")) + list(ckpt_dir.glob("*.pt"))
            if ckpts:
                ckpt_path = str(ckpts[0])
        
        if not ckpt_path:
            raise FileNotFoundError(f"Could not find checkpoint in {ckpt_dir}. Make sure models are downloaded.")

        # 2. Auto-detect model features based on name
        use_prosody_encoder = "prosody" in model_name.lower()
        use_acc_grl = "grl" in model_name.lower()
        
        use_ema = True
        frontend = "phone"

        cache_key = (ckpt_path, use_prosody_encoder)
        if cache_key in _tts_model_cache:
            return (_tts_model_cache[cache_key],)

        # 3. Setup Prosody Paths if needed
        prosody_cfg_path = ""
        prosody_ckpt_path = ""
        if use_prosody_encoder:
            prosody_cfg_path = str(CKPTS_ROOT / "prosody_encoder" / "pretssel_cfg.json")
            prosody_ckpt_path = str(CKPTS_ROOT / "prosody_encoder" / "prosody_encoder_UnitY2.pt")

        vocab_file = str(PRETRAINED_ROOT / "data" / model_name / "vocab.txt")
        vocoder_path = str(CKPTS_ROOT / "vocos-mel-24khz")

        device = mm.get_torch_device()
        device_str = str(device) if isinstance(device, torch.device) else device

        tts = TTS(
            model=model_name,
            ckpt_file=ckpt_path,
            vocab_file=vocab_file,
            device=device_str,
            use_ema=use_ema,
            frontend=frontend,
            use_prosody_encoder=use_prosody_encoder,
            prosody_cfg_path=prosody_cfg_path,
            prosody_ckpt_path=prosody_ckpt_path,
            vocoder_local_path=vocoder_path,
        )
        
        tts.lemas_meta = {
            "use_prosody_encoder": use_prosody_encoder,
            "use_acc_grl": use_acc_grl,
            "separate_langs": True
        }

        _tts_model_cache[cache_key] = tts
        return (tts,)


class LEMAS_TTS_Synthesize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tts_model": ("LEMAS_TTS_MODEL", {"tooltip": "Loaded LEMAS-TTS model.\nЗагруженная модель LEMAS-TTS."}),
                "ref_audio": ("AUDIO", {"tooltip": "Reference audio for voice cloning (1-10s).\nАудио-образец для клонирования голоса (от 1 до 10 секунд)."}),
                "ref_text": ("STRING", {
                    "multiline": True, 
                    "default": "Reference text goes here.",
                    "tooltip": "Exact transcript of the reference audio.\nТочная расшифровка того, что говорится в аудио-образце. Ошибки здесь ломают голос!"
                }),
                "gen_text": ("STRING", {
                    "multiline": True, 
                    "default": "Text to generate goes here.",
                    "tooltip": "Text to generate. For multiple sentences, put each on a new line.\nТекст для генерации. При длинных текстах разделяйте предложения переносом строки."
                }),
                "seed": ("INT", {
                    "default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "Random seed. -1 for random.\nЗерно генерации. -1 для случайного."
                }),
            },
            "optional": {
                "nfe_step": ("INT", {
                    "default": 32, "min": 1, "max": 256, "step": 1, 
                    "tooltip": "Inference steps. 32=fast/good, 64=high quality.\nШаги генерации. 32 = быстрее, 64 = более качественный звук."
                }),
                "cfg_strength": ("FLOAT", {
                    "default": 5.0, "min": 0.0, "max": 10.0, "step": 0.5,
                    "tooltip": "CFG scale. 5.0 is default. Lower if voice distorts.\nСтепень следования тексту. 5.0 по умолчанию. Уменьшите, если голос дрожит или искажается."
                }),
                "speed": ("FLOAT", {
                    "default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1,
                    "tooltip": "Speech speed multiplier (1.0 = normal, 2.0 = 2x faster).\nМножитель скорости речи (1.0 = нормальная, 2.0 = в 2 раза быстрее)."
                }),
                "sway_sampling_coef": ("FLOAT", {
                    "default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Sway sampling coefficient. Default is 3.0.\nКоэффициент sway-сэмплинга. Влияет на 'физику' генерации. 3.0 по умолчанию."
                }),
                "ref_ratio": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How much to rely on the reference audio. Default is 1.0.\nНасколько сильно опираться на аудио-образец (от 0 до 1). По умолчанию 1.0."
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "synthesize"
    CATEGORY = "audio/LEMAS-TTS"

    def synthesize(self, tts_model, ref_audio, ref_text, gen_text, seed, nfe_step=32, speed=1.0, cfg_strength=5.0, sway_sampling_coef=3.0, ref_ratio=1.0):

        ref_audio_path = audio_dict_to_path(ref_audio)
        tmp_ref_path = ref_audio_path

        meta = getattr(tts_model, "lemas_meta", {})
        use_prosody = meta.get("use_prosody_encoder", False)
        use_acc_grl = meta.get("use_acc_grl", False)
        separate_langs = meta.get("separate_langs", True) 

        try:
            if seed == -1:
                seed = None

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                try:
                    wav, sr, spec = tts_model.infer(
                        ref_file=ref_audio_path,
                        ref_text=" ".join(ref_text.split()),
                        gen_text=gen_text.strip(),
                        nfe_step=nfe_step,
                        speed=speed,
                        cfg_strength=cfg_strength,
                        sway_sampling_coef=sway_sampling_coef,
                        ref_ratio=ref_ratio,
                        use_prosody_encoder=use_prosody,
                        use_acc_grl=use_acc_grl,
                        separate_langs=separate_langs, 
                        seed=seed,
                        file_wave=f.name,
                    )
                    result = wav_to_audio_dict(f.name, target_sr=tts_model.target_sample_rate)
                    return (result,)
                except Exception as e:
                    raise RuntimeError(f"Inference error: {e}")
        finally:
            if tmp_ref_path and os.path.isfile(tmp_ref_path):
                os.remove(tmp_ref_path)


class LEMAS_TTS_Denoise:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Audio to denoise.\nАудио-образец для очистки от шума (улучшает качество клонирования)."}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("denoised_audio",)
    FUNCTION = "denoise"
    CATEGORY = "audio/LEMAS-TTS"

    def denoise(self, audio):
        _ensure_models()
        uvr5_dir = str(PRETRAINED_ROOT / "uvr5")

        if "uvr5" not in _uvr5_cache:
            if not os.path.isdir(uvr5_dir):
                raise FileNotFoundError(f"UVR5 model directory not found: {uvr5_dir}")
            _uvr5_cache["uvr5"] = UVR5Denoiser(uvr5_dir)

        uvr5 = _uvr5_cache["uvr5"]

        audio_path = audio_dict_to_path(audio)
        try:
            denoised_wav, sr = uvr5.denoise(audio_path)

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            sf.write(tmp.name, denoised_wav, sr, format="wav", subtype="PCM_24")
            tmp.close()

            result = wav_to_audio_dict(tmp.name)
            return (result,)
        finally:
            if os.path.isfile(audio_path):
                os.remove(audio_path)


NODE_CLASS_MAPPINGS = {
    "LEMAS_TTS_ModelLoader": LEMAS_TTS_ModelLoader,
    "LEMAS_TTS_Synthesize": LEMAS_TTS_Synthesize,
    "LEMAS_TTS_Denoise": LEMAS_TTS_Denoise,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LEMAS_TTS_ModelLoader": "LEMAS TTS Model Loader",
    "LEMAS_TTS_Synthesize": "LEMAS TTS Synthesize",
    "LEMAS_TTS_Denoise": "LEMAS TTS Denoise (UVR5)",
}