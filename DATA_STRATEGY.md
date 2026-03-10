# FR-Moshi Data Strategy Report

Comprehensive analysis of unconventional data sources and creative pipelines for acquiring
French dialogue audio (stereo: L=speaker A, R=speaker B) to fine-tune Moshi.

**Budget: $30 | Timeline: days, not weeks**

---

## EXECUTIVE SUMMARY

The single highest-impact action is to **replicate the J-CHAT methodology for French**:
scrape French podcasts from PodcastIndex, filter with Whisper language ID, extract dialogue
segments via pyannote diarization, clean with Demucs, and convert to stereo by assigning
diarized speakers to L/R channels. This can yield **1,000-10,000+ hours** of pseudo-stereo
French dialogue for near-zero cost. Combined with **SUMM-RE** (95h true stereo) already
in our pipeline, and optionally **synthetic TTS dialogue** (100-600h), this creates a
training data foundation comparable to J-Moshi's approach.

### Top 3 Ranked Actions

| Rank | Strategy | Expected Hours | Cost | Effort | Quality |
|------|----------|---------------|------|--------|---------|
| 1 | J-CHAT-style French podcast pipeline | 1,000-10,000h | $0-5 | 3-5 days | Medium (pseudo-stereo) |
| 2 | Synthetic TTS dialogue (Chatterbox/Orpheus) | 100-600h | $5-15 | 2-3 days | Medium-High |
| 3 | SUMM-RE full corpus (already started) | 95h | $0 | Done | High (true stereo) |

---

## 1. FRENCH PODCASTS (J-CHAT REPLICATION) -- HIGHEST PRIORITY

### What J-CHAT Did (the proven blueprint)

J-Moshi's team built a **69,000-hour** Japanese dialogue corpus from podcasts + YouTube:

1. **Collect**: Queried PodcastIndex API for all Japanese-labeled podcasts, downloaded
   audio from RSS feed enclosure URLs. Got ~880k files, ~140k hours raw.
2. **Language filter**: Whisper language ID (threshold p > 0.8) to remove non-Japanese.
3. **Dialogue filter**: pyannote diarization -> reject segments where one speaker holds
   >80% of talk time (monologues). Split at 5-second silence gaps.
4. **Clean**: Demucs speech enhancement to remove background music.
5. **Stereo conversion**: Assigned one diarized speaker to L channel, others to R channel.
   Silence in each channel when the other speaker is talking.
6. **Transcribe**: WhisperX for aligned transcriptions.

**This method is explicitly described as "language-independent"** in their paper.

### Adapting for French

The exact same pipeline works for French. Key parameters:

| Step | Tool | French Adaptation |
|------|------|-------------------|
| Collect | PodcastIndex API | Filter `feedLanguage=fr` |
| Language ID | Whisper | Filter for French (p > 0.8) |
| Diarization | pyannote 3.1 | Works for any language |
| Dialogue filter | Custom script | Same 80% threshold |
| Enhancement | Demucs | Language-agnostic |
| Transcription | Whisper large-v3 | French WER ~5% on clean speech |

### Expected Yield

PodcastIndex indexes 4M+ shows globally. France has a large podcasting ecosystem.
Conservative estimate: 50,000-100,000+ French podcast episodes available via RSS feeds.
At ~30-60 min average, that is 25,000-100,000 hours raw. After filtering:
- Language filter removes ~20-30% (non-French, multilingual shows)
- Dialogue filter removes ~40-50% (monologue podcasts, solo narration)
- Quality filter removes ~10-20% (poor audio, heavy music)

**Expected usable output: 5,000-30,000 hours of French dialogue.**

Even a conservative 1,000-hour subset would be transformative for our project.

### Feasibility Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| Technical feasibility | HIGH | Proven method, all tools open-source |
| Cost | $0-5 | Free API, free downloads, local processing |
| Legal status | GOOD | Podcasts are published for public consumption |
| Quality | MEDIUM | Pseudo-stereo has channel leakage, but J-Moshi trained on this successfully |
| Time to first data | 2-3 days | Script + small batch download + processing |
| Scalability | EXCELLENT | Can scale to 10,000+ hours |

### Implementation Plan

```
scripts/01_download_podcasts.py (already exists - extend it)
  -> PodcastIndex API: get all feedLanguage=fr RSS feeds
  -> Parse RSS for audio enclosure URLs
  -> Download with rate limiting
  -> Store as data/raw_audio/podcasts/

scripts/02_diarize_stereo.py (already exists - extend it)
  -> Whisper language ID filter
  -> pyannote diarization
  -> Dialogue segment extraction (5s gap, 80% dominance filter)
  -> Demucs speech enhancement
  -> Stereo conversion (speaker A -> L, speaker B -> R)
  -> Output to data/stereo/
```

### Quality vs True Stereo

J-Moshi's results prove pseudo-stereo works for Moshi training:
- J-Moshi pre-trained on 60,000h of pseudo-stereo podcast data
- Only 344h of true stereo was used for fine-tuning
- Achieved naturalness 2.66/5 (vs dGSLM 2.44)

**Our strategy**: Pre-train adaptation on large pseudo-stereo podcast data,
then fine-tune on SUMM-RE true stereo.

---

## 2. SYNTHETIC TTS DIALOGUE PIPELINE -- HIGH PRIORITY

### What J-Moshi Did

J-Moshi-ext generated **602 hours** of synthetic stereo dialogue:
1. Took 43,739 text dialogues
2. Rewrote them with an LLM to include spoken language expressions
3. Generated 10 speech samples per dialogue with different TTS seeds
4. Selected the sample with lowest WER (24.6% overall WER)
5. Created stereo by assigning speaker A to L, speaker B to R

### What Moshi (Original) Did

The original Moshi used **20,000 hours** of synthetic TTS data for instruction fine-tuning,
generated using a single professional voice + 92 speaking styles.

### What PersonaPlex Did

NVIDIA PersonaPlex used **Chatterbox TTS** to synthesize dialogue from LLM-generated scripts,
combined with Fisher real conversations. Their finding: "synthetic data + real data >
either alone."

### Best TTS Models for French (March 2026)

| Model | French Quality | Speed | License | Voices |
|-------|---------------|-------|---------|--------|
| **Chatterbox Multilingual** | Excellent | >1x realtime | MIT | Voice cloning (5s) |
| **Orpheus TTS French** | Very Good | ~91 tok/s = 1x RT | Apache 2.0 | Multiple |
| **XTTS v2 (Coqui)** | Good | <200ms latency | CPML (non-commercial) |  Voice cloning (6s) |
| **Parler-TTS French** | Good | Moderate | Apache 2.0 | Description-based |
| Bark | Decent | Slow | MIT | Preset speakers |

**Recommendation: Chatterbox Multilingual** -- MIT license, best quality, supports French
with voice cloning from 5-second clips. PersonaPlex used it successfully.

### Text Dialogue Sources for TTS Input

| Source | Size | Format | License | Notes |
|--------|------|--------|---------|-------|
| **Claire-Dialogue-French** | 160M words | Speaker-annotated turns | Open | Perfect format for TTS |
| **OpenSubtitles French** | Millions of lines | Subtitle pairs | CC-like | Movie dialogue text |
| **French Kaggle Conversations** | Varies | Conversation format | Various | Movie subtitle extracts |
| LLM-generated dialogues | Unlimited | Custom | N/A | Use GPT/Claude to write scripts |

### Pipeline Design

```
1. Text Source (Claire-Dialogue / OpenSubtitles / LLM-generated)
   -> Extract dialogue pairs (Speaker A lines, Speaker B lines)

2. TTS Generation (Chatterbox Multilingual)
   -> Voice A: Clone from reference clip (male French speaker)
   -> Voice B: Clone from different reference clip (female French speaker)
   -> Generate each turn separately

3. Stereo Assembly
   -> Speaker A turns -> Left channel
   -> Speaker B turns -> Right channel
   -> Proper timing with natural pauses between turns

4. Quality Filter
   -> Whisper ASR -> compute WER against original text
   -> Keep only samples with WER < 30%
```

### Cost and Scale Estimate

| Resource | Option | Cost | Hours Generated |
|----------|--------|------|-----------------|
| Kaggle T4 (free) | 30h/week GPU | $0 | ~30-60h audio/week |
| Colab T4 (free) | 15-30h/week | $0 | ~15-30h audio/week |
| RunPod A100 | $1.49/h | $15 | ~200-400h audio |
| Combined free tier rotation | ~50h GPU/week | $0 | ~50-100h audio/week |

With Chatterbox at >1x realtime on T4, and free GPU tiers:
- **Week 1**: ~100h synthetic dialogue from free compute
- **With $15 RunPod**: ~300h additional in a single day

**Total achievable: 100-600h synthetic dialogue for $0-15.**

### Research Findings on Synthetic Data Quality

Key finding from Interspeech 2025 paper "Is Synthetic Data Truly Effective for
Training Speech Language Models?":
- **Synthetic data alone degrades performance**
- **Synthetic + real data enhances performance** (better than either alone)
- Meta AI: 60% synthetic + 40% real achieved highest accuracy across 6 languages

**This validates our hybrid strategy**: SUMM-RE (real) + Podcasts (real) + TTS (synthetic).

---

## 3. SUMM-RE FULL CORPUS -- ALREADY IN PIPELINE

### Current Status
Already integrated in our pipeline (scripts/00_prepare_summ_re.py).
Using 25h train + 3h eval currently.

### Full Potential

| Split | Conversations | Hours | Audio Tracks |
|-------|--------------|-------|-------------|
| Train | 210 | ~67h | 684 |
| Dev | 36 | ~12h | 130 |
| Test | 37 | ~12h | 124 |
| **Total** | **283** | **~95h** | **938** |

Each conversation has **separate microphone per speaker** -- true stereo, no channel
leakage. This is the gold standard quality.

**Action: Use ALL 95h, not just 25h.** The marginal cost is zero.

### Conversion Notes
- 3-4 speakers per meeting -> select 2 most active speakers for L/R channels
- Or: create multiple stereo pairs from multi-speaker meetings
- 20-minute conversations -> segment into 100-second training clips
- Resample from 32-48kHz to 24kHz for Mimi

---

## 4. SPEAKER SEPARATION APPROACH (Source Separation on Mono Audio)

### Tools Available

| Tool | Task | Quality |
|------|------|---------|
| **pyannote 3.1** | Speaker diarization | 11.2% DER (SOTA) |
| **SpeechBrain SepFormer** | Speaker separation | 22.3 dB SI-SNRi on 2-speaker |
| **Demucs** | Speech enhancement / music removal | 9.2 dB SDR |
| WhisperX | Aligned transcription | Word-level timestamps |

### Two Approaches to Creating Stereo from Mono

**Approach A: Diarization + Masking (J-CHAT method)**
- pyannote identifies "who speaks when"
- Silence the channel when speaker is not active
- Simple, fast, but has crosstalk during overlapping speech
- Used by J-Moshi for 60,000h of data

**Approach B: Neural Speaker Separation (SepFormer)**
- Actually separates the mixed signal into individual speaker waveforms
- Higher quality isolation (22.3 dB SI-SNRi)
- Slower, more compute-intensive
- Works best with 2 speakers

**Recommendation**: Use Approach A (diarization + masking) for bulk data, Approach B
for high-quality subsets. This mirrors J-Moshi's strategy.

### Quality Reality Check

| Method | Channel Isolation | Overlap Handling | Speed |
|--------|------------------|-----------------|-------|
| True stereo (SUMM-RE) | Perfect | Preserved | N/A |
| Diarization + masking | Good (silence leaks) | Lost | Fast |
| SepFormer separation | Very good (22dB) | Partially preserved | Slow |
| pyannote DER | 96-98% accuracy (2 speakers, clean) | 5-15% errors | Fast |

---

## 5. FRENCH RADIO / INSTITUTIONAL ARCHIVES

### INA (Institut National de l'Audiovisuel)

| Resource | Size | Access | Notes |
|----------|------|--------|-------|
| INA total archive | 2.5M hours TV+radio | Restricted | Research agreements |
| InaGVAD | 4.6h annotated | Free (research) | VAD/gender labels only |
| INA GitHub tools | N/A | Open source | 25 repositories |

INA has massive archives but access requires institutional research agreements.
Not practical for our timeline/budget.

### Radio France
No publicly available dataset found. Broadcasts are copyrighted.

**Verdict**: Not practical for our constraints. Skip.

---

## 6. YOUTUBE FRENCH INTERVIEWS/DIALOGUES

### Existing Resources
- No large-scale French YouTube speech dataset found (unlike JTubeSpeech for Japanese)
- J-CHAT's YouTube component: searched with random Wikipedia keywords, got 180k hours raw

### Replicating for French
Same J-CHAT method works:
1. Search YouTube with French keywords (from French Wikipedia page titles)
2. Download audio with yt-dlp
3. Filter with Whisper French language ID
4. Diarize + extract dialogue segments

### Legal Considerations
- YouTube ToS technically prohibits downloading
- Fair use for research is arguable
- J-CHAT published their corpus on Hugging Face with CC-BY-NC 4.0
- Risk level: moderate (many research projects do this)

### Yield Estimate
J-CHAT got 11,000h from YouTube. French YouTube is comparable in size.
Expected: **5,000-15,000h** raw, **2,000-5,000h** after filtering.

**Verdict**: High yield but higher legal risk than podcasts. Use as secondary source
if podcast pipeline doesn't yield enough data.

---

## 7. FRENCH AUDIOBOOK DIALOGUES

### MLS French (Multilingual LibriSpeech)

| Metric | Value |
|--------|-------|
| Hours | 1,333h |
| Speakers | 114 |
| Books | 224 |
| Format | Read speech, single speaker per book |
| License | CC BY 4.0 |

### Problem
Audiobooks are **read by a single narrator**, not dialogue between speakers.
Even dialogue-heavy novels have one voice reading all parts. This makes it
fundamentally unsuitable for stereo dialogue training.

### Possible Workaround
- Extract dialogue-heavy passages from text
- Re-synthesize with TTS using different voices for different characters
- But this is just the synthetic TTS pipeline with extra steps

**Verdict**: Not directly useful. MLS French is better used as a monologue
pre-training resource if needed. Skip for dialogue.

---

## 8. FRENCH MOVIE/TV SUBTITLES + AUDIO

### Text Resources (Dialogue Scripts)

| Source | Size | Access | Format |
|--------|------|--------|--------|
| OpenSubtitles French | Millions of lines | OPUS/Kaggle | Timestamped subtitle pairs |
| French Kaggle Movie Conversations | Varies | Free | Conversation format |

### Audio Pairing
- Public domain French films exist (publicdomainmovie.net, Archive.org)
- But: mostly silent era (Melies) or very old films
- Modern films with rich dialogue are copyrighted
- Audio quality from old films is poor

### Feasibility
Could extract OpenSubtitles dialogue TEXT and use it as input for TTS generation
(feeding into Strategy #2). This is more practical than trying to extract audio
from films.

**Verdict**: OpenSubtitles is valuable as a **text source for TTS**, not as an
audio source. Already captured in Strategy #2.

---

## 9. EXISTING FRENCH TELEPHONE/CONVERSATION DATASETS

### Available Datasets

| Dataset | Hours | Speakers | Format | Access | Cost |
|---------|-------|----------|--------|--------|------|
| CALLFRIEND Canadian French | ~26h | 60 conversations | Telephone, 2 speakers | LDC | $$ (LDC member) |
| ASR-FreCSC | 1.1h | 6 conversations | 2 speakers | MagicHub | Free |
| Nexdata French Dialogue (telephone) | 547h | 964 speakers | Telephone 8kHz | Commercial | $$$ |
| Nexdata French Dialogue (smartphone) | 470h | 822 speakers | 16kHz | Commercial | $$$ |
| French General Conversation | 30h | 60 speakers, 2/conversation | Unscripted | Commercial | $$$ |

### Assessment
- CALLFRIEND: Good quality but requires LDC membership (not in our $30 budget)
- ASR-FreCSC: Too small (1.1h) to matter
- Nexdata datasets: Commercial, likely $1000+ (way over budget)
- French General Conversation: Commercial, price unknown

**Verdict**: None of these are practical within our budget. The podcast pipeline
gives us more data for free.

---

## 10. ADDITIONAL DATASETS TO INCORPORATE

### Tier 1: Already identified in CALIBRATION_REPORT.md

| Dataset | Hours | Status |
|---------|-------|--------|
| SUMM-RE | 95h true stereo | In pipeline |
| NCCFr | 36h true stereo | Need email request |
| CID | 8h true stereo | ORTOLANG account needed |

### Tier 2: Mono datasets worth processing

| Dataset | Hours | Source |
|---------|-------|--------|
| ESLO | 296h | ORTOLANG |
| ORFEO/CEFC | 120h+ | ORTOLANG |
| TCOF | 23h | ORTOLANG |
| CLAPI | 50h | Online |

All Tier 2 datasets can be processed through the diarization + stereo pipeline.

---

## FINAL RANKING: All Strategies by Practical Value

| Rank | Strategy | Hours | Cost | Days | Quality | Legal Risk | Overall Score |
|------|----------|-------|------|------|---------|------------|---------------|
| **1** | **French podcast pipeline (J-CHAT method)** | **1,000-10,000h** | **$0-5** | **3-5** | **Med** | **Low** | **10/10** |
| **2** | **Synthetic TTS (Chatterbox + Claire-Dialogue)** | **100-600h** | **$0-15** | **2-3** | **Med-High** | **None** | **9/10** |
| **3** | **SUMM-RE full 95h** | **95h** | **$0** | **<1** | **High** | **None** | **8/10** |
| 4 | Diarize ORTOLANG mono corpora (ESLO+ORFEO+TCOF) | ~440h | $0 | 2-3 | Med | Low | 7/10 |
| 5 | NCCFr + CID true stereo | 44h | $0 | 2-5 | High | Low | 6/10 |
| 6 | YouTube French interviews | 2,000-5,000h | $0-5 | 3-5 | Med | Moderate | 6/10 |
| 7 | Public domain French films + TTS | 10-50h | $0 | 3-5 | Low-Med | None | 3/10 |
| 8 | Nexdata/LDC commercial datasets | 500-1000h | $$$$ | 1-2 | High | None | 2/10 |
| 9 | INA radio archives | Unknown | $0 | Weeks | High | Bureaucratic | 1/10 |
| 10 | French audiobooks (MLS) | N/A | $0 | N/A | N/A | N/A | 0/10 |

---

## RECOMMENDED IMPLEMENTATION ROADMAP

### Phase 1: Immediate (Day 1-2) -- Maximize True Stereo
- [ ] Expand SUMM-RE usage from 25h to full 95h
- [ ] Email NCCFr for 36h true stereo access
- [ ] Create ORTOLANG account, request CID (8h)
- **Expected output: 95-139h true stereo**

### Phase 2: Quick Win (Day 2-4) -- Synthetic TTS Pipeline
- [ ] Download Claire-Dialogue-French text corpus (160M words of dialogue)
- [ ] Extract dialogue pairs with speaker labels
- [ ] Set up Chatterbox Multilingual with 2 French voice references
- [ ] Generate synthetic stereo dialogue on free GPU tiers (Kaggle + Colab)
- [ ] Quality filter with Whisper WER
- **Expected output: 100-300h synthetic stereo in first week**

### Phase 3: Scale (Day 3-7) -- French Podcast Pipeline
- [ ] Implement PodcastIndex French feed crawler
- [ ] Download first batch (~1000 episodes)
- [ ] Run Whisper French language filter
- [ ] Run pyannote diarization + dialogue extraction
- [ ] Run Demucs speech enhancement
- [ ] Convert to stereo format
- **Expected output: 500-2,000h pseudo-stereo from first batch**

### Phase 4: Training Strategy
Following J-Moshi's proven approach:
1. **Pre-training adaptation** on large pseudo-stereo podcast data (1000h+)
2. **Fine-tuning** on true stereo (SUMM-RE 95h + NCCFr 36h)
3. **Instruction tuning** on synthetic TTS dialogue (100-600h)

### Budget Allocation

| Item | Allocation |
|------|-----------|
| SUMM-RE + free datasets | $0 |
| Podcast download + processing (local) | $0 |
| TTS generation on free GPU tiers | $0 |
| RunPod burst TTS generation (optional) | $5-10 |
| RunPod training runs (6-7 runs) | $20-25 |
| **Total** | **$25-35** |

---

## KEY INSIGHT: WHY THIS CAN WORK

The critical insight from studying J-Moshi, PersonaPlex, and the original Moshi:

1. **Moshi trained on 7M hours of monologue** for pre-training (unsupervised)
2. **Only 2,000 hours of real dialogue** (Fisher) for dialogue fine-tuning
3. **20,000 hours of synthetic TTS** for instruction tuning
4. **J-Moshi replicated this with 60,000h podcast + 344h real stereo + 602h synthetic**

The ratio matters: you need a LOT of "any French speech" exposure, a moderate amount
of dialogue structure, and a small amount of high-quality true stereo for calibration.

Our pipeline can achieve:
- Podcast pseudo-stereo: 1,000-10,000h (bulk French speech + dialogue structure)
- SUMM-RE true stereo: 95h (high-quality calibration)
- Synthetic TTS: 100-600h (diverse dialogue patterns)

This is a credible foundation for French Moshi adaptation, comparable in strategy
(if not scale) to what J-Moshi achieved for Japanese.

---

## SOURCES

### Datasets
- [Claire-Dialogue-French (OpenLLM-France)](https://huggingface.co/datasets/OpenLLM-France/Claire-Dialogue-French-0.1)
- [SUMM-RE (Linagora)](https://huggingface.co/datasets/linagora/SUMM-RE)
- [MLS French (OpenSLR)](https://www.openslr.org/94/)
- [OpenSubtitles French (Kaggle)](https://www.kaggle.com/datasets/daliselmi/french-conversational-dataset)
- [J-CHAT (Sarulab)](https://huggingface.co/datasets/sarulab-speech/J-CHAT)
- [InaGVAD](https://github.com/ina-foss/InaGVAD)
- [ASR-FreCSC](https://magichub.com/datasets/french-conversational-speech-corpus-2/)
- [CALLFRIEND Canadian French](https://catalog.ldc.upenn.edu/LDC96S48)

### Research Papers
- [J-CHAT Corpus Construction](https://arxiv.org/html/2407.15828v1)
- [J-Moshi: Japanese Full-duplex Dialogue](https://arxiv.org/html/2506.02979v1)
- [Moshi: Speech-Text Foundation Model](https://kyutai.org/Moshi.pdf)
- [PersonaPlex (NVIDIA)](https://research.nvidia.com/labs/adlr/personaplex/)
- [Claire French Dialogue Dataset](https://arxiv.org/abs/2311.16840)
- [Is Synthetic Data Truly Effective for SLMs?](https://www.isca-archive.org/interspeech_2025/mizumoto25_interspeech.pdf)

### Tools
- [Chatterbox TTS (Resemble AI)](https://github.com/resemble-ai/chatterbox)
- [Orpheus TTS](https://github.com/canopyai/Orpheus-TTS)
- [Parler-TTS French](https://huggingface.co/blog/PHBJT/french-parler-tts)
- [XTTS v2 (Coqui)](https://huggingface.co/coqui/XTTS-v2)
- [pyannote speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [SpeechBrain SepFormer](https://huggingface.co/speechbrain/sepformer-wsj02mix)
- [PodcastIndex API](https://podcastindex-org.github.io/docs-api/)
- [PodcastIndex Python wrapper](https://pypi.org/project/python-podcastindex/)

### GPU Pricing
- [RunPod Pricing](https://www.runpod.io/pricing)
- [Free GPU Platforms Guide](https://iotbyhvm.ooo/best-free-cloud-gpu-platforms-in-2026-google-colab-kaggle-and-more/)
