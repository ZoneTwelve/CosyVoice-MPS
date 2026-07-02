#!/usr/bin/env python3
import sys
import argparse
import torch
import torchaudio

sys.path.insert(0, 'third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel


def main():
    parser = argparse.ArgumentParser(description='Text-to-Speech with voice cloning')
    parser.add_argument('text', nargs='?', help='Text to synthesize (or use --file)')
    parser.add_argument('-f', '--file', help='Read text from file')
    parser.add_argument('-r', '--ref-audio', default='asset/zero_shot_prompt.wav',
                        help='Reference audio for voice cloning (default: bundled sample asset)')
    parser.add_argument('-t', '--ref-text', default='希望你以后能够做的比我还好呦。',
                        help='Transcript of the reference audio (must match --ref-audio exactly)')
    parser.add_argument('-o', '--output', default='output.wav',
                        help='Output WAV file (default: output.wav)')
    parser.add_argument('-m', '--model', default='pretrained_models/Fun-CosyVoice3-0.5B',
                        help='Model directory')
    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            text = f.read().strip()
    elif args.text:
        text = args.text
    else:
        print('Reading text from stdin... (Ctrl+D to finish)')
        text = sys.stdin.read().strip()

    if not text:
        print('Error: no text provided', file=sys.stderr)
        sys.exit(1)

    print(f'Model   : {args.model}')
    print(f'Ref     : {args.ref_audio}')
    print(f'Output  : {args.output}')
    print(f'Text    : {text[:80]}{"..." if len(text) > 80 else ""}')
    print()

    model = AutoModel(model_dir=args.model)

    chunks = []
    for i, result in enumerate(model.inference_zero_shot(
        text,
        f'You are a helpful assistant.<|endofprompt|>{args.ref_text}',
        args.ref_audio,
        stream=False,
    )):
        chunks.append(result['tts_speech'])
        print(f'  chunk {i + 1} done ({result["tts_speech"].shape[1] / model.sample_rate:.1f}s)')

    final = torch.cat(chunks, dim=1)
    torchaudio.save(args.output, final, model.sample_rate)
    print(f'\nSaved: {args.output} ({final.shape[1] / model.sample_rate:.1f}s)')


if __name__ == '__main__':
    main()
