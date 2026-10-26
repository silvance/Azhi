"""Measure who-said-what on your own recordings: the quick-exchange test set.

How the speakers are separated, and how each transcribed word is handed to one
of them, was tuned by eye. A quick "Yes." from the other person, a two-word
answer, a reply that starts before the question has finished: these are the
moments a transcript most often gives to the wrong person, and nothing measured
them. This harness does, on recordings you choose and label yourself, with
nothing leaving the machine.

    python -m whispr.diarization_eval /path/to/testset --out results

Test set: each recording beside a reference file with the same name::

    testset/call-01.wav
    testset/call-01.txt       Audacity label export: start, end, speaker
    testset/interview.m4a
    testset/interview.rttm    or RTTM

Scores, per recording and overall:

- **Words to the right person**: of the transcribed words that fall inside
  one person's reference turn, the share given to that person.
- **Short turns kept**: reference turns under :data:`SHORT_TURN_SECONDS` (the
  quick replies) whose words were given to the right person.
- **Diarization error rate**: the share of reference speech time missed,
  falsely detected or given to the wrong person, from the speaker turns alone,
  with no tolerance at turn edges and overlapping speech included.

The first run transcribes and separates each recording and caches the result
- words, timings and turns, never audio - in the output folder. ``--rescore``
scores the cache again, so a change to how words are assigned is measured in
seconds instead of a transcription run.

The scoring is dependency-free apart from numpy and unit-tested; transcription
and separation need the same models the application uses. Nothing here reaches
the network.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from itertools import permutations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .diarization import SpeakerSegment, assign_speakers, exclusive_coverage
from .transcription import Segment, Word

PathLike = Union[str, Path]

# Reference turns shorter than this are the quick replies the test set is for.
SHORT_TURN_SECONDS = 1.0

# Frame step for the diarization error rate.
FRAME_SECONDS = 0.01

# A word counts towards the scores only when one person is heard alone for at
# least this share of it in the reference: a word inside overlapping speech has
# no single right answer.
WORD_REFERENCE_SHARE = 0.5

CACHE_VERSION = 1


# -- Reference turns ---------------------------------------------------------


def read_reference(path: PathLike) -> List[SpeakerSegment]:
    """Speaker turns from an Audacity label export (.txt) or an RTTM file.

    Raises :class:`ValueError` naming the file and line of anything unreadable.
    """
    path = Path(path)
    turns: List[SpeakerSegment] = []
    rttm = path.suffix.lower() == ".rttm"
    for number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        where = f"{path.name}, line {number}"
        if rttm:
            fields = line.split()
            if fields[0] != "SPEAKER":
                continue
            if len(fields) < 8:
                raise ValueError(f"{where}: an RTTM SPEAKER line needs 8 fields.")
            start, duration, speaker = (
                _number(fields[3], where),
                _number(fields[4], where),
                fields[7],
            )
            end = start + duration
        else:
            # Audacity writes a second row starting with a backslash for a
            # label with a frequency range; it carries no time span.
            if line.startswith("\\"):
                continue
            fields = raw.rstrip("\r\n").split("\t")
            if len(fields) < 3 or not fields[2].strip():
                raise ValueError(
                    f"{where}: expected start, end and a speaker name separated "
                    "by tabs, as Audacity's Export Labels writes them."
                )
            start, end = _number(fields[0], where), _number(fields[1], where)
            speaker = fields[2].strip()
        if end <= start:
            continue  # a point label marks a moment, not a turn
        turns.append(SpeakerSegment(start=start, end=end, speaker=speaker))
    if not turns:
        raise ValueError(f"{path.name} has no speaker turns in it.")
    return sorted(turns, key=lambda t: (t.start, t.end))


def _number(text: str, where: str) -> float:
    try:
        return float(text.replace(",", "."))
    except ValueError:
        raise ValueError(f"{where}: {text!r} is not a time in seconds.") from None


@dataclass
class Case:
    """One recording of the test set and its reference turns."""

    name: str
    audio: Path
    reference: Path


def find_cases(folder: PathLike) -> List[Case]:
    """Every recording in ``folder`` that has a reference file beside it."""
    from .transcription import is_supported_media

    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} is not a folder.")
    audio = {
        p.stem: p
        for p in sorted(folder.iterdir())
        if p.is_file() and is_supported_media(p)
    }
    cases = []
    for ref in sorted(folder.iterdir()):
        if ref.suffix.lower() not in (".txt", ".rttm") or ref.stem not in audio:
            continue
        cases.append(Case(name=ref.stem, audio=audio[ref.stem], reference=ref))
    return cases


# -- Scoring ------------------------------------------------------------------


def _overlap(a: SpeakerSegment, b: SpeakerSegment) -> float:
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


def speaker_mapping(
    reference: Sequence[SpeakerSegment], hypothesis: Sequence[SpeakerSegment]
) -> Dict[str, str]:
    """The one-to-one pairing of found speakers to reference speakers.

    Chosen to maximise the time both agree someone is speaking, as the standard
    diarization error rate does. Exhaustive up to eight speakers a side, greedy
    beyond that.
    """
    refs = sorted({t.speaker for t in reference})
    hyps = sorted({t.speaker for t in hypothesis})
    shared: Dict[Tuple[str, str], float] = {}
    for found in hypothesis:
        for truth in reference:
            seconds = _overlap(found, truth)
            if seconds:
                key = (found.speaker, truth.speaker)
                shared[key] = shared.get(key, 0.0) + seconds
    if not refs or not hyps:
        return {}
    if len(refs) <= 8 and len(hyps) <= 8:
        best: Tuple[float, Dict[str, str]] = (-1.0, {})
        if len(hyps) <= len(refs):
            for chosen in permutations(refs, len(hyps)):
                pairs = dict(zip(hyps, chosen))
                total = sum(shared.get((h, r), 0.0) for h, r in pairs.items())
                if total > best[0]:
                    best = (total, pairs)
        else:
            for chosen in permutations(hyps, len(refs)):
                pairs = dict(zip(chosen, refs))
                total = sum(shared.get((h, r), 0.0) for h, r in pairs.items())
                if total > best[0]:
                    best = (total, pairs)
        return {h: r for h, r in best[1].items() if shared.get((h, r), 0.0) > 0}
    mapping: Dict[str, str] = {}
    for (h, r), _seconds in sorted(shared.items(), key=lambda kv: -kv[1]):
        if h not in mapping and r not in mapping.values():
            mapping[h] = r
    return mapping


def diarization_error(
    reference: Sequence[SpeakerSegment],
    hypothesis: Sequence[SpeakerSegment],
    mapping: Dict[str, str],
    *,
    step: float = FRAME_SECONDS,
) -> Tuple[float, float, float, float]:
    """``(missed, false_alarm, confusion, reference_total)`` in seconds.

    No tolerance at turn edges and overlapping speech scored, so a turn of a
    few tenths of a second counts in full.
    """
    import numpy as np

    end = max([t.end for t in reference] + [t.end for t in hypothesis] + [0.0])
    frames = int(np.ceil(end / step)) + 1
    centres = (np.arange(frames) + 0.5) * step

    def activity(turns: Sequence[SpeakerSegment]) -> Dict[str, Any]:
        active: Dict[str, Any] = {}
        for t in turns:
            mask = active.setdefault(t.speaker, np.zeros(frames, dtype=bool))
            mask |= (centres >= t.start) & (centres < t.end)
        return active

    ref, hyp = activity(reference), activity(hypothesis)
    zeros = np.zeros(frames, dtype=np.int32)
    n_ref = sum((m.astype(np.int32) for m in ref.values()), zeros)
    n_hyp = sum((m.astype(np.int32) for m in hyp.values()), zeros)
    correct = zeros.copy()
    for h, r in mapping.items():
        if h in hyp and r in ref:
            correct = correct + (hyp[h] & ref[r]).astype(np.int32)
    missed = np.maximum(n_ref - n_hyp, 0).sum() * step
    false_alarm = np.maximum(n_hyp - n_ref, 0).sum() * step
    confusion = (np.minimum(n_ref, n_hyp) - correct).sum() * step
    return (
        float(missed),
        float(false_alarm),
        float(confusion),
        float(n_ref.sum() * step),
    )


@dataclass
class CaseScore:
    """How one recording came out."""

    name: str
    reference_seconds: float = 0.0
    missed: float = 0.0
    false_alarm: float = 0.0
    confusion: float = 0.0
    words_scored: int = 0
    words_right: int = 0
    words_unattributed: int = 0
    words_not_scored: int = 0
    short_turns: int = 0
    short_turns_with_words: int = 0
    short_turns_right: int = 0
    speakers_reference: int = 0
    speakers_found: int = 0
    mapping: Dict[str, str] = field(default_factory=dict)

    @property
    def der(self) -> float:
        if not self.reference_seconds:
            return 0.0
        return (
            self.missed + self.false_alarm + self.confusion
        ) / self.reference_seconds

    @property
    def word_accuracy(self) -> float:
        return self.words_right / self.words_scored if self.words_scored else 0.0

    @property
    def short_turn_rate(self) -> float:
        if not self.short_turns_with_words:
            return 0.0
        return self.short_turns_right / self.short_turns_with_words


def _words_of(segment: Segment) -> List[Word]:
    if segment.words:
        return list(segment.words)
    return [Word(start=segment.start, end=segment.end, word=segment.text)]


def score_case(
    name: str,
    reference: Sequence[SpeakerSegment],
    turns: Sequence[SpeakerSegment],
    attributed: Sequence[Segment],
) -> CaseScore:
    """Score one recording: speaker turns found, and words as attributed.

    ``attributed`` is the transcript after :func:`assign_speakers`, so the
    word scores measure the whole path a reader sees, not the diarizer alone.
    """
    mapping = speaker_mapping(reference, turns)
    missed, false_alarm, confusion, total = diarization_error(reference, turns, mapping)
    score = CaseScore(
        name=name,
        reference_seconds=total,
        missed=missed,
        false_alarm=false_alarm,
        confusion=confusion,
        speakers_reference=len({t.speaker for t in reference}),
        speakers_found=len({t.speaker for t in turns}),
        mapping=dict(mapping),
    )
    short = [t for t in reference if t.end - t.start < SHORT_TURN_SECONDS]
    results: Dict[int, List[bool]] = {i: [] for i in range(len(short))}
    for segment in attributed:
        for word in _words_of(segment):
            span = max(word.end - word.start, 0.01)
            heard = exclusive_coverage(word.start, word.start + span, reference)
            if not heard:
                score.words_not_scored += 1
                continue
            truth = max(heard, key=lambda speaker: heard[speaker])
            if heard[truth] < WORD_REFERENCE_SHARE * span:
                score.words_not_scored += 1
                continue
            score.words_scored += 1
            if segment.speaker is None:
                score.words_unattributed += 1
                right = False
            else:
                right = mapping.get(segment.speaker) == truth
            score.words_right += int(right)
            middle = word.start + span / 2.0
            for i, turn in enumerate(short):
                if turn.speaker == truth and turn.start <= middle <= turn.end:
                    results[i].append(right)
    score.short_turns = len(short)
    for outcomes in results.values():
        if outcomes:
            score.short_turns_with_words += 1
            score.short_turns_right += int(sum(outcomes) * 2 >= len(outcomes))
    return score


def total(scores: Sequence[CaseScore]) -> CaseScore:
    """All recordings pooled: every word and every second counts once."""
    out = CaseScore(name="All recordings")
    for s in scores:
        for name in (
            "reference_seconds",
            "missed",
            "false_alarm",
            "confusion",
            "words_scored",
            "words_right",
            "words_unattributed",
            "words_not_scored",
            "short_turns",
            "short_turns_with_words",
            "short_turns_right",
        ):
            setattr(out, name, getattr(out, name) + getattr(s, name))
    return out


# -- Cache: what transcription and separation produced -------------------------


def save_hypothesis(
    path: PathLike,
    segments: Sequence[Segment],
    turns: Sequence[SpeakerSegment],
    meta: Dict[str, Any],
) -> Path:
    """Write words, timings and turns - no audio, no voiceprints - as JSON."""
    from .atomicio import write_json_atomic

    data = {
        "version": CACHE_VERSION,
        **meta,
        "segments": [
            {
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "words": [
                    {
                        "start": w.start,
                        "end": w.end,
                        "word": w.word,
                        "probability": w.probability,
                    }
                    for w in s.words
                ],
            }
            for s in segments
        ],
        "turns": [
            {"start": t.start, "end": t.end, "speaker": t.speaker} for t in turns
        ],
    }
    return write_json_atomic(Path(path), data)


def load_hypothesis(
    path: PathLike,
) -> Tuple[List[Segment], List[SpeakerSegment], Dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("version") != CACHE_VERSION:
        raise ValueError(
            f"{Path(path).name} was written by another version; run again without --rescore."
        )
    segments = [
        Segment(
            start=s["start"],
            end=s["end"],
            text=s["text"],
            words=[Word(**w) for w in s.get("words", [])],
        )
        for s in data["segments"]
    ]
    turns = [SpeakerSegment(**t) for t in data["turns"]]
    meta = {k: v for k, v in data.items() if k not in ("segments", "turns")}
    return segments, turns, meta


# -- Running a test set --------------------------------------------------------


def _offline() -> None:
    """Refuse every download: the test set is run where the recordings live."""
    from . import resources

    for flag in resources._OFFLINE_FLAGS:
        os.environ[flag] = "1"
    resources.refuse_network_downloads()


def _transcribe_and_separate(
    case: Case,
    *,
    model: str,
    engine: str,
    speakers: Optional[int],
    language: Optional[str],
    progress: Any,
) -> Tuple[List[Segment], List[SpeakerSegment], Dict[str, Any]]:
    from .diarization import chosen_backend, diarize
    from .enrollment import prepare_source
    from .resources import bundled_models
    from .transcription import resolve_model_choice, transcribe_audio

    wav, digest, temporary = prepare_source(case.audio, progress=progress)
    try:
        resolved = resolve_model_choice(model, bundled_models())
        progress(f"{case.name}: transcribing with word timings…")
        result = transcribe_audio(
            wav, model_size=resolved, word_timestamps=True, language=language
        )
        backend = chosen_backend(engine)
        progress(f"{case.name}: separating speakers ({backend})…")
        turns = diarize(wav, backend=backend, num_speakers=speakers, progress=progress)
    finally:
        if temporary:
            try:
                Path(wav).unlink()
            except OSError:
                pass
    meta = {
        "audio": case.audio.name,
        "audio_sha256": digest,
        "model": model,
        "engine": backend,
        "speakers_given": speakers,
        "language": result.language,
    }
    return list(result.segments), list(turns), meta


def run_test_set(
    folder: PathLike,
    out: PathLike,
    *,
    model: str,
    engine: str = "auto",
    speakers: str = "known",
    language: Optional[str] = None,
    rescore: bool = False,
    progress: Any = print,
) -> List[CaseScore]:
    """Score every recording in ``folder``, writing the cache under ``out``."""
    from copy import deepcopy

    cases = find_cases(folder)
    if not cases:
        raise FileNotFoundError(
            f"No recording in {folder} has a reference file beside it "
            "(same name, .txt from Audacity's Export Labels or .rttm)."
        )
    cache_dir = Path(out) / "cache"
    scores: List[CaseScore] = []
    for case in cases:
        reference = read_reference(case.reference)
        cached = cache_dir / f"{case.name}.json"
        if rescore:
            if not cached.is_file():
                raise FileNotFoundError(
                    f"{case.name} has not been run yet; run once without --rescore."
                )
            segments, turns, _meta = load_hypothesis(cached)
        else:
            count = len({t.speaker for t in reference}) if speakers == "known" else None
            segments, turns, meta = _transcribe_and_separate(
                case,
                model=model,
                engine=engine,
                speakers=count,
                language=language,
                progress=progress,
            )
            save_hypothesis(cached, segments, turns, meta)
        attributed = assign_speakers(deepcopy(segments), turns)
        scores.append(score_case(case.name, reference, turns, attributed))
    return scores


# -- Report --------------------------------------------------------------------


def _percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def summary_lines(scores: Sequence[CaseScore]) -> List[str]:
    """The report a person reads: one line per recording, then all of them."""
    rows = list(scores) + ([total(scores)] if len(scores) > 1 else [])
    width = max([len(s.name) for s in rows] + [14])
    lines = [
        f"{'Recording':<{width}}  {'Words to right person':>22}  "
        f"{'Short turns kept':>16}  {'Speaker time wrong':>18}",
    ]
    for s in rows:
        words = f"{_percent(s.word_accuracy)} of {s.words_scored}"
        short = (
            f"{s.short_turns_right} of {s.short_turns_with_words}"
            if s.short_turns_with_words
            else "none"
        )
        lines.append(
            f"{s.name:<{width}}  {words:>22}  {short:>16}  {_percent(s.der):>18}"
        )
    pooled = rows[-1]
    lines += [
        "",
        f"Short turns are reference turns under {SHORT_TURN_SECONDS:g} s - the quick "
        f"replies. {pooled.short_turns - pooled.short_turns_with_words} of "
        f"{pooled.short_turns} had no transcribed words, so could not be scored.",
        f"{pooled.words_not_scored} word(s) were not scored: in overlapping speech "
        "or outside every reference turn.",
        f"{pooled.words_unattributed} scored word(s) were given to no one.",
        "Speaker time wrong is the diarization error rate: missed, falsely "
        "detected or misattributed speech, no edge tolerance, overlap included.",
    ]
    return lines


def write_csv(scores: Sequence[CaseScore], path: PathLike) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(scores) + ([total(scores)] if len(scores) > 1 else [])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "recording",
                "words_scored",
                "words_right",
                "word_accuracy",
                "words_unattributed",
                "words_not_scored",
                "short_turns",
                "short_turns_with_words",
                "short_turns_right",
                "reference_seconds",
                "missed_seconds",
                "false_alarm_seconds",
                "confusion_seconds",
                "der",
                "speakers_reference",
                "speakers_found",
            ]
        )
        for s in rows:
            writer.writerow(
                [
                    s.name,
                    s.words_scored,
                    s.words_right,
                    round(s.word_accuracy, 4),
                    s.words_unattributed,
                    s.words_not_scored,
                    s.short_turns,
                    s.short_turns_with_words,
                    s.short_turns_right,
                    round(s.reference_seconds, 2),
                    round(s.missed, 2),
                    round(s.false_alarm, 2),
                    round(s.confusion, 2),
                    round(s.der, 4),
                    s.speakers_reference,
                    s.speakers_found,
                ]
            )
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI: ``python -m whispr.diarization_eval <testset> [--out DIR]``."""
    import argparse

    from .transcription import DEFAULT_MODEL

    parser = argparse.ArgumentParser(
        description=(
            "Score who-said-what on labelled recordings: words given to the "
            "right person, quick replies kept, and diarization error rate. "
            "Offline; uses the application's own models."
        )
    )
    parser.add_argument("testset", help="Folder of recordings with reference files")
    parser.add_argument("--out", default="diarization-results", help="Output folder")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Transcription model (default: %(default)s)",
    )
    parser.add_argument(
        "--engine",
        default="auto",
        choices=("auto", "pyannote", "sherpa", "nemotron"),
        help=(
            "Speaker separation method (default: %(default)s). nemotron works "
            "out the number of speakers itself, so --speakers does not apply to it"
        ),
    )
    parser.add_argument(
        "--speakers",
        default="known",
        choices=("known", "unknown"),
        help=(
            "known: tell the separation how many people the reference has, as "
            "an operator who answered the question would; unknown: let it work "
            "it out (default: %(default)s)"
        ),
    )
    parser.add_argument("--language", default=None, help="Language code, if known")
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Score the cached results again without transcribing",
    )
    args = parser.parse_args(argv)

    _offline()
    try:
        scores = run_test_set(
            args.testset,
            args.out,
            model=args.model,
            engine=args.engine,
            speakers=args.speakers,
            language=args.language,
            rescore=args.rescore,
        )
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"The test set could not be scored: {exc}")
        return 2
    out = Path(args.out)
    write_csv(scores, out / "scores.csv")
    lines = summary_lines(scores)
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print()
    print("\n".join(lines))
    print()
    print(f"Wrote {out / 'scores.csv'} and {out / 'summary.txt'}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
