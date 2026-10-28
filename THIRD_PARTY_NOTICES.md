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
- **Nemotron 3 Diarization** (NVIDIA, `nvidia/Nemotron-3-Diarization`), only
  with `nemotron=yes`. OpenMDW-1.1, a permissive licence whose one condition
  on passing the model on is that the copy keeps the agreement and the notices
  of origin it came with. Both travel with the model in
  `whispers_assets/nemotron`: the agreement as `licenses/OpenMDW-1.1.txt`
  (also reproduced below) and NVIDIA's model card, which states the licence,
  as `Nemotron-3-Diarization-model-card.md`. `nemotron.json` records the
  revision and digests.

## Other programs and data

- **FFmpeg**, the binary distributed with imageio-ffmpeg. LGPL-2.1-or-later or
  GPL, depending on the options it was built with: see https://ffmpeg.org/legal.html
  and the imageio-ffmpeg project for the bundled binary.
- **Tesseract OCR** and the **tessdata_fast** language data, when OCR is
  bundled. Apache-2.0.
- **Argos Translate** language packages, when translation is bundled. Each
  package states its own licence in its metadata.
- **MiniSBD** (LibreTranslate authors; the `minisbd` package pinned in
  `packaging/lockfile.txt`), the sentence splitter Argos Translate runs before
  translating, when translation is bundled. Used unmodified. AGPL-3.0, a
  copyleft licence: a copy passed on outside the office must make MiniSBD's
  source available with it (section 6); it is published at
  https://github.com/LibreTranslate/MiniSBD. Part of its code is adapted from
  Stanford's Stanza (Apache-2.0). Both licence texts ship in
  `licenses/minisbd/`.
- **MiniSBD sentence models**, one small ONNX file per translated language,
  fetched from MiniSBD's releases at build time into the Argos folder
  (`whispers_assets/argos`). They are MiniSBD's conversions of Stanza's
  tokenizer models, which Stanford publishes under Apache-2.0; MiniSBD states
  no separate licence for the converted files.
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

## Nemotron 3 Diarization: OpenMDW License Agreement, version 1.1

```
OpenMDW License Agreement, version 1.1 (OpenMDW-1.1)

By exercising rights granted to you under this agreement, you accept and agree
to its terms.

As used in this agreement, "Model Materials" means the materials provided to
you under this agreement, consisting of: (1) one or more machine learning
models (including architecture and parameters); and (2) all related artifacts
(including associated data, documentation and software) that are provided to
you hereunder.

Subject to your compliance with this agreement, permission is hereby granted,
free of charge, to deal in the Model Materials without restriction, including
under all copyright, patent, database, and trade secret rights included or
embodied therein.

If you distribute any portion of the Model Materials, you shall retain in your
distribution (1) a copy of this agreement, and (2) all copyright notices and
other notices of origin included in the Model Materials that are applicable to
your distribution.

If you file, maintain, or voluntarily participate in a lawsuit against any
person or entity asserting that the Model Materials directly or indirectly
infringe any patent or copyright, then all rights and grants made to you
hereunder are terminated, unless that lawsuit was in response to a
corresponding lawsuit first brought against you.

This agreement does not impose any restrictions or obligations with respect to
any use, modification, or sharing of any outputs generated by using the Model
Materials.

THE MODEL MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE, TITLE, NONINFRINGEMENT, ACCURACY, OR THE
ABSENCE OF LATENT OR OTHER DEFECTS OR ERRORS, WHETHER OR NOT DISCOVERABLE, ALL
TO THE GREATEST EXTENT PERMISSIBLE UNDER APPLICABLE LAW.

YOU ARE SOLELY RESPONSIBLE FOR (1) CLEARING RIGHTS OF OTHER PERSONS THAT MAY
APPLY TO THE MODEL MATERIALS OR ANY USE THEREOF, INCLUDING WITHOUT LIMITATION
ANY PERSON'S COPYRIGHTS OR OTHER RIGHTS INCLUDED OR EMBODIED IN THE MODEL
MATERIALS; (2) OBTAINING ANY NECESSARY CONSENTS, PERMISSIONS OR OTHER RIGHTS
REQUIRED FOR ANY USE OF THE MODEL MATERIALS; OR (3) PERFORMING ANY DUE
DILIGENCE OR UNDERTAKING ANY OTHER INVESTIGATIONS INTO THE MODEL MATERIALS OR
ANYTHING INCORPORATED OR EMBODIED THEREIN.

IN NO EVENT SHALL THE PROVIDERS OF THE MODEL MATERIALS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE MODEL MATERIALS, THE
USE THEREOF OR OTHER DEALINGS THEREIN.
```
