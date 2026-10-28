# Measuring who-said-what on your own recordings

The transcript's speaker labels are weakest in a quick exchange. A one-word
"Yes." from the other person, a reply that starts before the question has
finished, three short answers in a row: these are the lines most likely to be
given to the wrong person. How well the tool handles them depends on your
recordings — the phone line, the room, the microphone, the language — far more
than on anything a developer can test without them.

This page shows how to build a small labelled **test set** from recordings you
already hold, and how to score the tool against it. Everything runs on the
machine the recordings are on. Nothing is uploaded, and nothing is downloaded.

What it measures is the **transcript**: whether each word is shown against the
right person. It does not measure Compare Speakers or voice profiles, and a good
score is not evidence that the tool can identify anyone.

---

## 1. Choose the recordings

Aim for **10 to 20 recordings, 1 to 5 minutes each**. More short recordings
beat a few long ones: each one is a different line, room and pair of voices.

- **Your real material.** The same kind of audio you actually work with —
  intercepted calls, room recordings, body-worn audio — at the quality it
  arrives in. A clean studio interview tells you nothing about a noisy call.
- **Quick exchanges in every one.** Short replies, interruptions, people
  talking over each other. The tool already does well on long speeches.
- **A known number of people,** whom you can tell apart by ear. If you cannot
  say who is speaking, neither can the reference.
- **A range of difficulty.** Some easy, some hard. Include the kind of
  recording that has gone wrong before.

Cut a long recording down to the stretch you want with any audio editor. In
Audacity, select the stretch and export just the selection.

## 2. Label who speaks when

The reference is a list of turns: when each person starts and stops. Audacity
is the easiest way to make one.

1. Open the recording in Audacity. Zoom in (Ctrl+1) until a second of audio is
   a few centimetres wide.
2. Select one person's turn with the mouse, from the first word to the last,
   and press **Ctrl+B**. Type the speaker's name and press Enter.
3. Do the same for **every** turn, however short. The one-word replies are the
   point of the exercise; a test set without them measures nothing new.
4. Where two people speak at once, label both turns, overlapping. Audacity
   allows labels to overlap.
5. *File → Export → Export Labels…* (*File → Export Other → Export Labels…* in
   newer versions) and save the `.txt` **beside the recording, with the same
   name**: `call-07.wav` → `call-07.txt`.

Tips:

- **A tenth of a second is close enough.** Start each label at the first word,
  end it at the last; leave out breaths, coughs and laughter.
- **Use the same name for the same person within a recording.** Names do not
  need to match the tool's (`SPEAKER_00`) or other recordings: the scoring
  pairs the tool's speakers with yours by the time they share.
- **Do not label the tool's transcript into the reference.** Label by ear,
  from the audio. A reference copied from the tool's output scores the tool
  against itself.
- Expect labelling to take **several times as long as the audio**, more for a
  busy conversation.

An RTTM file (`call-07.rttm`) works too, if you have another labelling tool
that writes one.

## 3. Lay out the folder

```
D:\testset\
    call-07.wav
    call-07.txt
    interview-2.m4a
    interview-2.txt
    ...
```

Any recording format the application opens will do. A recording without a
reference beside it is skipped.

## 4. Run the scoring

From a terminal in the application's folder:

```
Whispers.exe --score-test-set D:\testset --out D:\testset-results
```

(From a source checkout: `python -m whispers.diarization_eval D:\testset --out
D:\testset-results`.)

It transcribes each recording and separates the speakers with the same models
the application uses, then scores the result. This takes about as long as
transcribing the recordings in the application. It tells the tool how many
people each reference has, as an operator who answers **How many people are
speaking?** would.

Options:

| Option | What it does |
|---|---|
| `--speakers unknown` | Let the tool work out how many people there are, as *Not sure — work it out* does. |
| `--engine nemotron`, `--engine pyannote` or `--engine sherpa` | Score one speaker-separation method rather than the one *Auto* would pick (Nemotron, in copies that include it). |
| `--model small` | Score a different transcription model (default: `medium`). |
| `--language ru` | The recordings' language, if you would set it in the application. |
| `--rescore` | Score the saved results again without transcribing. For developers changing how words are given to speakers. |

## 5. Read the results

The report is printed, and saved as `summary.txt` and `scores.csv` in the
results folder. It looks like this (the figures are only an illustration):

```
Recording        Words to right person  Short turns kept  Speaker time wrong
call-07                 91.2% of 342          11 of 14               18.4%
interview-2             97.0% of 610           6 of 6                 7.9%
All recordings          94.9% of 952          17 of 20               11.8%
```

- **Words to right person** — of the transcribed words inside one person's
  reference turn, the share shown against that person. This is what a reader of
  the transcript experiences.
- **Short turns kept** — reference turns under one second (the quick replies)
  whose words went to the right person. Short turns the transcription did not
  pick up any words from cannot be scored, and the report says how many.
- **Speaker time wrong** — the diarization error rate: the share of speech time
  missed, wrongly detected or given to the wrong person, from the speaker
  separation alone. No allowance is made at turn edges, so a short turn counts
  in full.

Words in overlapping speech are not scored — there is no single right answer
for them — and nor are words outside every reference turn.

A recording with a high **Speaker time wrong** and a low **Words to right
person** has usually had two voices merged into one, or one voice split in
two. Listen to it in the application with the count given, and see which.

Below the table, the report says where the quick replies that went to the
wrong person were lost:

- **the speaker separation did not find** them: it had no turn for that
  person there at all — the separation method is what to change;
- **it did find** them, **but the words still went to someone else**: usually
  the transcription's word timings, which can be a few tenths of a second out
  on a short word — worth reporting, with these counts, to whoever maintains
  the tool.

Quick replies with no transcribed words are counted separately: the
transcription never heard them, so no speaker separation can help.

In a copy built with **Sharpen English word timings**, English recordings are
scored the way the application handles them, with the words re-timed. A last
table scores the same words as transcribed, so the two can be compared without
a second run (again, the figures are only an illustration):

```
Word timings (2 English recording(s)): ...
                         Words to right person  Short turns kept
  Re-timed (as above)             94.9% of 952          17 of 20
  As transcribed                  93.6% of 950          14 of 20
```

`scores.csv` has the as-transcribed figures in its last three columns.

## 6. Handling the results

The results folder holds a `cache` folder with each recording's transcript
text and word timings, so the scores can be recomputed without transcribing.
It holds no audio and no voiceprints, but **the transcript text is as
sensitive as the recordings**: keep it where they are kept, and delete it when
you are done.

`summary.txt` and `scores.csv` hold only numbers and the recordings' file
names. To send the scores to whoever maintains the tool, check the file names
reveal nothing, and send those two files only — never the `cache` folder.

Keep the test set. Run it again after an update to the tool, and the numbers
show whether that update helped your recordings or harmed them.

To compare speaker-separation methods, score the same test set once per method,
each into its own results folder:

```
Whispers.exe --score-test-set D:\testset --out D:\results-pyannote --engine pyannote
Whispers.exe --score-test-set D:\testset --out D:\results-nemotron --engine nemotron
```
