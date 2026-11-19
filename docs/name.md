# The name: WHISPERS

**W**orkstation for **H**uman-language **I**ntelligence: **S**peech
**P**rocessing, **E**xploitation, **R**eporting & **S**creening

The tool was first called Whispers, after the quiet, hard-to-hear speech it was
built to make sense of. It was renamed twice and has now gone back to its first
name, written as an acronym. Each word of the acronym describes something the
tool does.

## Workstation

WHISPERS runs entirely on the analyst's own computer, including air-gapped
machines. Nothing it does reaches a network:

- recordings, transcripts and voice profiles never leave the machine;
- there is no cloud service, no account and no telemetry.

It is a tool on a workstation, not a system or a service. Everything it holds
stays under the control of the agent or analyst using it.

## Human-language

The word comes from *human language technology*, the established name for the
field that turns human speech and writing into something a computer can work
with: speech recognition, machine translation and the reading of printed text.
WHISPERS works on what people say and write, in whatever language they use:

- recorded interviews and calls;
- chat and email exported from phones and accounts;
- scanned and printed documents.

## Intelligence

The tool exists to support counterintelligence work: to get information of
intelligence value out of material an investigation or operation has already
obtained lawfully. What it produces is for an agent or analyst to assess. It is
never a finding in itself.

## Speech Processing

Most of the tool's work is processing speech:

- transcribing recordings, including poor, noisy or distant ones;
- separating the speakers in a recording, so the transcript shows who spoke
  when;
- measuring how closely a voice resembles a reference profile. This is
  reported as High, Intermediate or Low similarity: a lead to review, never an
  identification.

## Exploitation

Exploitation is the long-standing term in intelligence work for getting usable
information out of documents and media. WHISPERS does this with the material a
case gathers:

- forensic extraction reports from Cellebrite and Magnet AXIOM;
- exported emails and chats, including WeChat and QQ;
- scanned documents, read with offline text recognition (OCR);
- foreign-language passages, found wherever they sit on a page and translated
  offline.

## Reporting

Every result can be handed on as a product that stands on its own:

- transcripts as text, Word or a spreadsheet;
- voice comparison reports that state the recording conditions on each side;
- side-by-side translations that put each original page next to its English.

The products say where they came from: the source file's SHA-256 and the
versions of the software and models that produced them. Each one also says what
it is not: a comparison is a lead, not an identification, and a machine
translation is not a certified translation.

## Screening

Screening, in counterintelligence as elsewhere, is sorting a large amount of
material to find what deserves a closer look. Here that means:

- which recordings mention a name on the watch list;
- which voices resemble a subject's reference closely enough to review;
- which pages of a long phone report are in a foreign language and need a
  court-certified linguist.

The tool narrows the material down. A person makes every decision about what it
means.

## What the name leaves out

Some words were left out on purpose:

- *Recognition* would suggest the tool identifies people by voice. It does not.
- *Synthesis* would suggest it can generate or alter speech. It cannot, and a
  tool that handles evidence should never be thought to.
- *HUMINT* is a separate discipline from counterintelligence. The name keeps
  to the language and media work the tool does.
