# Running CosyVoice on Apple Silicon (MPS)

This fork adds Apple Silicon (Metal/MPS) support to CosyVoice: device selection,
autocast, and memory-release helpers now recognize `mps` alongside `cuda`/`cpu`,
and the causal HiFT generator falls back to float32 on MPS (float64 is not
supported by Metal).

**Notes**
- `load_jit`, `load_trt`, and `load_vllm` remain CUDA/TensorRT-only and are
  disabled automatically when no CUDA device is present.
- ONNX Runtime has no MPS execution provider, so the CampPlus/speech-tokenizer
  ONNX sessions always run on CPU, even when the rest of the pipeline runs on MPS.
- Expect a real-time-factor (RTF) well above 1 on MPS with float32 — this is a
  hardware limitation, not a bug.

---

## Quick start

`tts.py` is a minimal CLI wrapper around `AutoModel` / `inference_zero_shot`,
useful for smoke-testing the MPS backend or for simple voice-cloning tasks.

```bash
# Synthesize text directly (uses the bundled sample reference voice)
python tts.py "Hello, this is a test." -o hello.wav

# Synthesize from a text file
python tts.py -f my_script.txt -o narration.wav

# Pipe from stdin
echo "Good morning." | python tts.py -o morning.wav
```

### Cloning your own voice

Voice cloning quality depends entirely on the **reference audio** and its
**exact transcript**.

| Factor | Recommendation |
|--------|---------------|
| Duration | 5-10 seconds |
| Background | Silent, no music or echo |
| Content | Natural speech, not read aloud |
| Format | WAV, 16kHz mono (ffmpeg converts automatically) |

```bash
# 1. Record a short clip, then convert it to 16kHz mono WAV
ffmpeg -i my_voice.m4a -ar 16000 -ac 1 my_ref.wav -y

# 2. Get an exact transcript (manually, or with Whisper)
python -c "
import whisper
model = whisper.load_model('base')
print(model.transcribe('my_ref.wav')['text'])
"

# 3. Run TTS with your voice
python tts.py "The text you want to synthesize." \
  -r my_ref.wav \
  -t "Exact words spoken in the reference audio." \
  -o output.wav
```

### Translate-and-dub a video (example pipeline)

```bash
# Extract audio
ffmpeg -i input.mov -ar 16000 -ac 1 audio.wav -y

# Transcribe + translate to English
python -c "
import whisper
model = whisper.load_model('medium')
result = model.transcribe('audio.wav', task='translate', language='zh')
open('script_en.txt', 'w').write(result['text'])
"

# Generate English audio
python tts.py -f script_en.txt -o dubbed.wav

# Merge back into the video (replacing the original audio track)
ffmpeg -i input.mov -i dubbed.wav -map 0:v -map 1:a -c:v copy -c:a aac -shortest output.mov -y
```

---

## Troubleshooting

**Audio sounds robotic or has the wrong accent**
The reference transcript (`-t`) doesn't exactly match the reference audio. Re-check word by word.

**Output voice doesn't sound like the reference**
Reference audio is too short (< 3s) or has background noise.

**Output cuts off early**
Input text is too long for one chunk. `tts.py` synthesizes per top-level
inference call; split long input into shorter sentences if needed.

**Slow generation (RTF > 1)**
Expected on MPS with float32 — this is a hardware limitation, not a bug.
