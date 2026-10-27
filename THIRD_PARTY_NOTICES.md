# Third-party notices

Whispers is MIT-licensed (see `LICENSE`). It runs on, and a bundle ships, other
people's software and models under their own licences. Python packages carry
their licence files with them; a bundle copies every one into its `licenses/`
folder. This file covers the rest: the models and programs a bundle includes,
which depend on how it was built (see `packaging/fetch_assets.py`).

Licences below are as each source publishes them. Check them again before a
bundle goes anywhere new.

## Speech recognition

- **Whisper model weights** (OpenAI), in the CTranslate2 conversions published
  by SYSTRAN (`Systran/faster-whisper-*`) and, for `turbo`,
  `dropbox-dash/faster-whisper-large-v3-turbo` (formerly
  `mobiuslabsgmbh/...`). MIT; the notice is reproduced below.
- **faster-whisper** (SYSTRAN) and **CTranslate2** (OpenNMT). MIT.
- **wav2vec2-base-960h** (Facebook AI), in the ONNX export published by
  onnx-community (`onnx-community/wav2vec2-base-960h-ONNX`), only in a build
  made with `alignment=yes`; it re-times English words. Apache-2.0. The
  revision and digest are in `whispers_assets/alignment/alignment.json`.
- **ONNX Runtime** (Microsoft), which runs it and the voice-activity model. MIT.

## Speaker separation and voice comparison

- **sherpa-onnx** (k2-fsa). Apache-2.0.
- **pyannote.audio** (CNRS) and the **pyannote/segmentation-3.0** and
  **pyannote/speaker-diarization-3.1** models. MIT. The models are gated on
  Hugging Face: downloading them for a build means accepting pyannote's terms.
- **pyannote/wespeaker-voxceleb-resnet34-LM** (embedding inside the pyannote
  pipeline; WeSpeaker). CC-BY-4.0.
- **NVIDIA TitaNet-large** (`nvidia/speakerverification_en_titanet_large`),
  the default speaker-embedding model. CC-BY-4.0. A build that chose another
  embedding model records which in `whispers_assets/diarization/embedding_model.json`.
- **NeMo-Speech.cpp** (NVIDIA), only in a build made with `nemotron=yes`.
  Apache-2.0; its own licence files are in `whispers_assets/nemotron/licenses`.
- **Nemotron 3 Diarization** (NVIDIA), only with `nemotron=yes`. OpenMDW-1.1.

## Other programs and data

- **FFmpeg**, the binary distributed with imageio-ffmpeg. LGPL-2.1-or-later or
  GPL, depending on the options it was built with: see https://ffmpeg.org/legal.html
  and the imageio-ffmpeg project for the bundled binary.
- **Tesseract OCR** and the **tessdata_fast** language data, when OCR is
  bundled. Apache-2.0.
- **Argos Translate** language packages, when translation is bundled. Each
  package states its own licence in its metadata.
- **GTCRN** noise-reduction model, from the sherpa-onnx model releases. See its
  source for its licence.

## Whisper model weights: MIT licence

```
MIT License

Copyright (c) 2022 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
