# Dataset

Public HF dataset `Hebrish/Hebrish-TTS-Voice-Recordings` (CC BY-NC 4.0; public since 2026-10-21). Two
splits, `train` and `test` (held out: the listening test, never trained on).

**Layout.** On 2026-10-30 the Hebrish take datasets were converted from audiofolders to Parquet on `main`
(layout spec: Voice-Training-Recorder `docs/parquet-layout.md`). A dataset is Parquet once `meta/schema.json`
exists on `main`:

```
data/train-*.parquet  data/test-*.parquet     one row per take: `audio` (struct of bytes, path; the WAV verbatim) + the columns below
meta/train.jsonl      meta/test.jsonl         index: each row's metadata without audio, `audio_path` (was `file_name`),
                                              `shard`, and `audio_offset` / `audio_length` (the WAV's byte range in the shard)
```

Before that (and at any earlier revision) it was an audiofolder: `data/<split>/audio/<id>.wav` +
`data/<split>/metadata.jsonl`. `scripts/fetch_dataset.py` (what `training/run.sh fetch` calls) reads either and
writes the same local audiofolder, `data/takes/{metadata.jsonl,audio/}`, which the prep scripts read; on Parquet
it downloads only the shards holding takes not yet on disk and cuts each WAV out by its byte range.
`load_dataset("Hebrish/Hebrish-TTS-Voice-Recordings")` loads the Parquet directly (`audio` decodes with
torchcodec; cast to `Audio(decode=False)` for the bytes).

`id` = `YYYYMMDDTHHMMSSZ_<6 random>`.

**Second config `from-stt` (2026-10-28, Daniel: "try to get as many in there as we can. Just don't take
the noisy ones").** `from-stt/<split>-*.parquet` with index `meta/from-stt/<split>.jsonl` (audiofolder: `from-stt/<split>/audio/<id>.flac` + `metadata.jsonl`; fetch with `scripts/fetch_dataset.py <split> <dir> --config from-stt`): 3,126 clean speech-to-text takes
(284.6 min, 48 kHz / 24-bit FLAC) of the same speaker, recorded for ASR, not for the clone. Not denoised and
not filtered on background noise (DNSMOS `bak`): **denoise at prep**. Extra columns `tts_config`,
`has_digits` (464 rows keep digits in `text`; this config's prompts are ASR prompts, so verbalise or skip
them) and `categories`. Split = the take's ASR split, not a Meorav listening-test split. The default
config is unchanged; load this one with `load_dataset(..., "from-stt")` and compare runs with and
without it. Selection rule: Voice-Training-Recorder `docs/dataset.md`, "TTS from STT".

## Row schema

Written by the capture app when it publishes a take. `training/prepare.py` reads `file_name` (the local audiofolder's; `audio_path` in the Parquet index), `id`, `text`,
`set`, `style`, `edited`, `prompt_id`, `transcript_status`, `clipped_samples`, `browser_processing` and
`snr_db`; a dataset of your own needs at least those.

| Column | Meaning |
|---|---|
| `file_name`, `id`, `duration_s` | |
| `text` | What was said: the prompt text, or the edited text if Edit was used (`edited: true`). **Empty for free speech until prep fills it.** |
| `transcript_status` | `script` (read from the card) or `pending` (free speech, needs ASR + correction) |
| `topic` | Free speech only: the topic shown |
| `set`, `style`, `style_instruction` | Prompt set, style cue (`neutral`, `question`, `excited`, …) and the reading cue. **Metadata only**: Qwen3-TTS builds its prompt from `text` alone, so inline tags would be read aloud. |
| `prompt_id`, `prompt_source` | Bank id (content hash) and origin (`arctic:<id>`, `asr-bank:<id>`, `written`) |
| `take` | 1 (re-recordings of a prompt are separate rows) |
| `speaker`, `accent` | Always "Daniel Rosehill", "Irish English" |
| `sample_rate`, `bit_depth`, `native_rate` | 48000, 24, and the rate the browser actually captured at |
| `peak_dbfs`, `floor_dbfs`, `speech_dbfs`, `snr_db`, `clipped_samples`, `lead_trim_s` | Per-take checks (`docs/recording-spec.md`) |
| `device`, `microphone`, `room`, `input_label`, `browser_processing` | Recording conditions from Settings and from `track.getSettings()` |
| `notes`, `recorded_at` | |

## Text conventions

- **Numbers are spelled out as spoken** ("two thousand and twenty-six", "half four", "twelve euro
  fifty"). This is the opposite of the ASR dataset, which writes digits: a TTS model maps text to
  sound, so the text has to say what was said. Bank builders drop any prompt containing a digit.
- **Hebrew words use the canonical romanisations** from
  list (`terms.csv`) shared with the Hebrish ASR projects, exactly as the ASR prompt bank has them. One spelling per word, so the model learns one spelling → Daniel's pronunciation.

## Prompt sets

The prompt-bank builder and its source files live with the capture app, not in this repo; this section
records what the bank contains so the dataset can be read and a similar one built.

The app draws from the picked sets in proportion to their weight.

| Set | Source | Why it is there |
|---|---|---|
| `arctic` (w 3) | CMU ARCTIC, 1,132 prompts, minus those with digits | Phonetic coverage: every English diphone. |
| `conversational` (w 3) | Written 2026-10-21; `conversational.enrich.jsonl` (65: ellipses, bracketed asides, dashes, quoted speech, exclamations; read-out messages) | Daniel's own register, with some Hiberno-English. Read speech pulls towards a neutral "reading voice"; this is the counterweight. |
| `hebrish` (w 2) | ASR bank code-switching prompts, set cover (`scripts/select_from_asr_bank.py`) | Daniel, 2026-10-21: use the ASR sentences as ground truths, so the clone says Hebrew words and Israeli names the way he does. ~650 prompts cover the 700 most frequent terms (most of them twice). `hebrish.enrich.jsonl` (65, Haiku, checked): 131 everyday terms (health, home, food, offices, transport, money) the bank had none of. |
| `names` (w 1.5) | `names.jsonl`: ASR bank sentences for Jewish/Israeli names, places and religious terms that `hebrish` covers fewer than twice (`select_from_asr_bank.py --only names`, terms filtered by `data/sources/names_term_classes.json`). `names.written.jsonl`: Haiku-written, for names and Hebrew place forms the bank lacks (Yerushalayim, Ir HaAtika, Har HaBayit, Chevron, Me'arat HaMachpela, Teveria, Dovid, Devorah…) | Daniel, 2026-10-21: "make sure I get all the main ones": Yerushalayim, Jewish names, Jewish place names. |
| `personal` (w 2) | `personal.jsonl`: every ASR bank sentence in its `personal` category or naming Daniel Rosehill, My Weird Prompts, Herman Poppleberry, Corn; plus a cover over the bank's own-vocabulary categories (tech, linux, smart_home, ai_interactions, general) so each of the top 200 terms appears at least twice across the whole TTS bank (`select_from_asr_bank.py --only personal`). `personal.written.jsonl` (43, Opus): Hebrish the app and org, Parakeet, the Hebrish model names (Sabich, Jachnun, Shakshuka, Boxty), Vercel, Modal, Synology, the show's intro, which the bank barely has. n8n is left out: its digit breaks the no-digits rule. | Daniel, 2026-10-21: "my name, My Weird Prompts, Herman Poppleberry, Corn … also Hebrish (the app), Parakeet. Think about all the vocab you can see me speaking a lot and make sure it all gets well represented." |
| `news` (w 1.5) | `news.written.jsonl` (220, Haiku: Israel/Jerusalem items, world/tech/markets/Ireland, brief openers and transitions); `news.leipzig.jsonl` (70 real news sentences, Haiku-picked from 1,155 Leipzig candidates); `news.enrich.jsonl` (30: per cent, shekels, units, years, times, acronyms) | Newsreader register, for the personal news briefs Daniel prepares for himself. |
| `narrative` (w 1.5) | `narrative.written.jsonl` (210, Haiku: bedtime and folk stories set in Ireland and Jerusalem; adult fiction, memoir, dialogue); `narrative.libritts.jsonl` (90 audiobook lines, Haiku-picked from 833 LibriTTS-R candidates) | Storytelling: suspense, warmth, character voices in quotes. |
| `questions`, `expressive` (w 1.5) | Written; `questions.enrich.jsonl` (45: tag, wh- and declarative questions) | Prosody range: rising/falling/tag questions; eight emotions. |
| `short` (w 1) | `short.enrich.jsonl` (94, Opus: 1 to 6 words, replies, greetings, sign-offs, some Hebrew) | End-of-utterance behaviour on very short takes; the bank had 0.2% of prompts under four words. |
| `lists_numbers` (w 1) | Written; `lists_numbers.enrich.jsonl` (55: spoken web and email addresses, phone numbers, shekels/agorot, euro, per cent, times, units) | List intonation; spoken numbers, dates, money, phone numbers. |
| `tech` (w 1) | ASR bank tech prompts, set cover | Tech vocabulary as he says it (Claude Code, MCP, Hugging Face…). |
| `long` (w 1; 0.5 until 2026-10-21) | Written; `long.enrich.jsonl` (30, 44 to 59 words) | Multi-clause sentences: phrasing and breath, landing the end of a long utterance. |
| `free` (w 1, off by default) | Written topics | Spontaneous speech, the strongest lever on accent. Target ~30% of the final hours once the scripted base exists. |

Podcast-style sentences are not a separate set: `conversational` and `long` already cover that register (Daniel, 2026-10-21).

### Blending in a new source

One composite bank in this repo, not a family of HF sentence datasets as in the ASR
project. Each purpose gets about 100 to 250 sentences. A set reads `<set>.jsonl` plus every
`<set>.<origin>.jsonl` (`written` = Haiku/Opus, `leipzig`/`libritts` = mined), so a source is
blended in by adding a file. Mined text: `scripts/mine_hf_sentences.py` samples random pages via
the HF datasets-server `/rows` API into `data/sources/candidates/`, then a Haiku pass picks the
best N verbatim. The API rate-limits (429 after about 25 fast pages, 2026-10-21), so the script
waits 5 minutes between retries and up to about an hour. 

### Enrichment (coverage audit)

`uv run scripts/coverage_report.py` compares the bank, the bank as the app draws it (set weight ÷
set size per prompt) and the recorded takes on the HF dataset: utterance length, question and
punctuation types, openings, spoken numbers/money/URLs, Hebrew terms (`terms.csv`), CMUdict phonemes,
diphones and the Hiberno-English sounds (TH/DH, post-vocalic R, diphthongs, -ing, T). JSON on
stdout, summary on stderr; `gaps` = draw share below `TARGETS` (judgement calls, not measured
optima). Gap-filling lines go in `<set>.enrich.jsonl`. 

First pass, 2026-10-21 (217 takes, 16.1 min recorded; bank 4,153 → 4,537 scripted, 384 added, no
existing id changed). Draw-weighted share before → after:

| Feature | Before | After | Target |
|---|---|---|---|
| 1–3 words | 0.1% | 3.1% | 3% |
| 4–7 words | 5.2% | 6.7% | 8% |
| over 40 words (longest prompt was 32) | 0 | 1.2% | 1.5% |
| ellipsis / brackets / dash | 0 / 0 / 0.8% | 0.9 / 0.7 / 1.6% | 1.5 / 1 / 2% |
| tag questions | 0.5% | 1.0% | 1.5% |
| exclamations | 2.6% | 3.3% | 5% |
| shekels/agorot, spoken URL/email, per cent | 0.1 / 0 / 0.2% | 0.5 / 0.3 / 0.4% | 0.8% each |
| Hebrew terms (of 1,146 active) in bank | 468 | 613 | |

Still under target: the money/URL/time/unit/year vocabulary (it sits in `lists_numbers`, weight 1,
so each line moves the draw share by ~0.02%; a next pass should put such lines in `conversational`),
wh-questions (2.3% vs 4%) and exclamations. Phoneme rates are steady (TH/DH ~40 per 1,000 phones,
OY ~1.2) and the bank's 1,244 diphones are ARCTIC-level; 896 are recorded so far, which closes by
recording, not seeding.

Not used: the Harvard sentences (the Columbia mirror returned 503 on 2026-10-21, and ARCTIC already
covers the phonetics).

Splits: ~5% of scripted prompts are `test`, chosen by id hash. Prompt ids are content hashes, so
editing a prompt's text in `data/prompts/` makes it a new prompt.

## Free speech

The app saves the take with `text: ""` and `transcript_status: "pending"`. Prep transcribes with a
strong ASR, Daniel corrects it, and the row's `text` is rewritten in a dataset commit. Not built yet.
