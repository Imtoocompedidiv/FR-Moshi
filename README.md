# FR-Moshi — Moshi Fine-tune Francais

Fine-tuning LoRA de Moshi 7B (Kyutai) pour le francais conversationnel.

## Lancement rapide (RunPod A6000 48GB, ~$3-5/run)

```bash
# 1. Creer un pod RunPod avec A6000 48GB + 75GB storage
# 2. SSH sur le pod, cloner ce repo
git clone <ce-repo> && cd FR-Moshi

# 3. Definir le token HuggingFace
export HF_TOKEN='hf_...'

# 4. Lancer le pipeline complet (setup + data + annotation + training)
bash scripts/launch_training.sh
```

Tout est automatise : telechargement SUMM-RE (25h train + 3h eval), conversion
en stereo 24kHz, annotation Whisper en francais, et fine-tuning LoRA.

## Architecture

```
Moshi = Helium 7B (LLM) + Mimi (codec 24kHz, 12.5Hz, 8 codebooks) + RQ-Transformer

Audio stereo :
  Canal L (0) = Moshi   -> le modele apprend a GENERER ceci
  Canal R (1) = User    -> le modele apprend a REAGIR a ceci
  Mimi encode les 2 canaux separement (16 codebooks total)
  Seuls les 8 codebooks Moshi sont predits (User = conditionnement)
```

## Pipeline detaille

```
scripts/00_prepare_summ_re.py  -> data/moshi_dataset/data_stereo/*.wav
                                  data/moshi_dataset/train.jsonl
                                  data/moshi_dataset/eval.jsonl

moshi-finetune/annotate.py     -> data/moshi_dataset/data_stereo/*.json
  --lang fr --local               (transcription canal L uniquement)

torchrun -m train config.yaml  -> runs/french_lora/checkpoints/
```

## Config (configs/french_lora.yaml)

| Parametre | Valeur | Source |
|-----------|--------|--------|
| LoRA rank | 128 | Max pour changement de langue |
| LoRA scaling | 256 | alpha = 2*rank (Microsoft) |
| ft_embed | true | Adaptation embeddings |
| LR | 5e-5 | Standard LoRA |
| batch_size | 8 | A6000 48GB confortable |
| duration_sec | 100 | 1250 frames @ 12.5Hz |
| max_steps | 1500 | ~5 epochs sur 25h |
| semantic weight | 100x | first_codebook_weight_multiplier |
| text_padding_weight | 0.4 | Entre J-Moshi et PersonaPlex |

## Donnees

**SUMM-RE** (Linagora) : reunions francaises, CC BY-SA 4.0
- Micro separe par locuteur -> vrai stereo propre
- 226h train, 43h dev, 41h test
- On utilise 25h train + 3h dev (eval)

## Budget

| Composant | Cout |
|-----------|------|
| A6000 48GB, ~5h training | ~$2.50 |
| 75GB storage, 1 semaine | ~$1.30 |
| Setup + annotation, ~1h | ~$0.50 |
| **Total 1 run** | **~$4.30** |
| **Runs possibles avec $30** | **~7** |

## References

- [moshi-finetune](https://github.com/kyutai-labs/moshi-finetune) (officiel)
- [J-Moshi](https://arxiv.org/abs/2506.02979) (japonais, reference)
- [PersonaPlex](https://arxiv.org/abs/2602.06053) (NVIDIA, persona)
- [SUMM-RE](https://huggingface.co/datasets/linagora/SUMM-RE) (donnees)
- [CALIBRATION_REPORT.md](CALIBRATION_REPORT.md) (rapport technique complet)
