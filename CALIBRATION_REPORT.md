# FR-Moshi — Rapport de Calibration Complet

Synthese de 7 agents de recherche ayant analyse :
- Code source complet de moshi-finetune (tous les .py)
- J-Moshi (adaptation japonaise, paper + code)
- PersonaPlex (NVIDIA, paper + code)
- Mimi codec + tokenizer
- LoRA best practices pour adaptation linguistique
- Datasets francais conversationnels
- Pricing GPU cloud (RunPod, Vast.ai, Lambda)

---

## 1. ARCHITECTURE DU PIPELINE DE DONNEES

### Flux complet WAV+JSON -> tokens modele

```
WAV stereo 24kHz (L=Moshi, R=User)
  |
  +-> Mimi encode canal L [2, 1, samples] -> [2, 8, T_frames]
  +-> Mimi encode canal R     reshape -> [1, 16, T_frames]
  |
  +-> JSON alignments -> Interleaver -> text stream [1, 1, T_frames]
  |
  +-> Concatenation -> codes [1, 17, T_frames]
       = [text, cb1_moshi...cb8_moshi, cb1_user...cb8_user]
```

### Chiffres cles

| Metrique | Valeur |
|----------|--------|
| Sample rate audio | 24,000 Hz |
| Frame rate Mimi | 12.5 Hz |
| Codebooks par canal | 8 (2048 entries chacun) |
| Total streams | 17 (1 text + 16 audio) |
| Tokens par seconde | 112.5 (12.5 Hz x 9 modeled) |
| Depformer modele | 8 codebooks (canal Moshi seulement) |
| Canal User | Conditionnement uniquement (pas de loss) |
| Mimi | GELE (jamais fine-tune) |

### Tokens speciaux

| Token | ID | Role |
|-------|----|------|
| text_padding | 3 | Frames sans texte actif |
| end_of_text_padding | 0 | Frame juste avant un nouveau mot |
| zero_padding | -1 | Position ignoree (pas d'embedding, pas de loss) |
| text_initial | 32000 | Start-of-sequence texte |
| audio_initial | 1024 | Start-of-sequence audio |

---

## 2. PARAMETRES DE LOSS (CRITIQUES)

### Formule de loss
```
total_loss = text_loss + audio_loss

text_loss = CrossEntropy(text_logits, codes[:, :1])
  - Poids padding * text_padding_weight (0.4)

audio_loss = CrossEntropy(audio_logits, codes[:, 1:9])
  - Poids codebook_0 * first_codebook_weight_multiplier (100.0)
  - Poids codebooks_1-7 = 1.0
  => Le codebook semantique represente 100/107 = 93.5% de l'audio_loss
```

### Comparaison des configs

| Parametre | Officiel | J-Moshi | PersonaPlex | FR-Moshi |
|-----------|----------|---------|-------------|----------|
| first_codebook_weight | 100.0 | 100.0 | ~50x (0.02 pour non-semantic) | 100.0 |
| text_padding_weight | 0.5 | 0.5 | 0.3 | 0.4 |
| LR temporal | 2e-6 | 2e-6 | 2e-6 | 5e-5 (LoRA) |
| LR depth | 2e-6 | 4e-6 | 4e-6 | 5e-5 (LoRA, unique) |
| batch_size | 16 | 512/16 | 32 | 8 |

Note : moshi-finetune officiel ne supporte PAS le learning rate differentiel
Temporal/Depth. Un seul LR est applique partout.

---

## 3. TOUS LES PARAMETRES DISPONIBLES

### TrainArgs (top-level)

| Parametre | Type | Default | Notre valeur | Notes |
|-----------|------|---------|-------------|-------|
| run_dir | str | requis | ../runs/french_lora | |
| seed | int | 0 | 42 | |
| duration_sec | float | 10 | 100 | Seq length en secondes |
| batch_size | int | 1 | 8 | Par GPU |
| num_microbatches | int | 1 | 1 | Gradient accumulation |
| max_steps | int | 100 | 1500 | Total optimizer steps |
| max_norm | float | 1.0 | 1.0 | Gradient clipping |
| first_codebook_weight_multiplier | float | 1.0 | 100.0 | Poids semantique |
| text_padding_weight | float | 0.5 | 0.4 | Poids padding texte |
| gradient_checkpointing | bool | True | true | Requis pour 48GB |
| param_dtype | str | bfloat16 | bfloat16 | |
| full_finetuning | bool | False | false | |
| save_adapters | bool | True | true | LoRA only weights |
| do_ckpt | bool | True | true | |
| ckpt_freq | int | 0 | 250 | |
| num_ckpt_keep | int | 3 | 4 | |
| do_eval | bool | False | true | |
| eval_freq | int | 0 | 100 | |
| log_freq | int | 1 | 10 | |
| overwrite_run_dir | bool | False | false | |

### LoraArgs

| Parametre | Default | Notre valeur | Notes |
|-----------|---------|-------------|-------|
| lora.enable | False | true | |
| lora.rank | 64 | 128 | Max pour language change |
| lora.scaling | 2.0 | 256.0 | alpha = 2*rank |
| lora.ft_embed | False | true | CRITIQUE pour langue |

### OptimArgs (AdamW hardcode betas=0.9,0.95 eps=1e-8)

| Parametre | Default | Notre valeur |
|-----------|---------|-------------|
| optim.lr | 1e-4 | 5e-5 |
| optim.weight_decay | 0.1 | 0.1 |
| optim.pct_start | 0.05 | 0.10 |

### DataArgs

| Parametre | Default | Notre valeur |
|-----------|---------|-------------|
| data.train_data | "" | ../data/moshi_dataset/train.jsonl |
| data.eval_data | "" | ../data/moshi_dataset/eval.jsonl |
| data.shuffle | False | true |

Multi-source supporte : "path1:0.7,path2:0.3"

### ModelPaths

| Parametre | Default |
|-----------|---------|
| moshi_paths.hf_repo_id | kyutai/moshiko-pytorch-bf16 |
| moshi_paths.mimi_path | None (override Mimi) |
| moshi_paths.moshi_path | None (override LM) |
| moshi_paths.tokenizer_path | None (override SPM) |
| moshi_paths.config_path | None (override config) |

---

## 4. TOKENIZER : DECISION

### Situation

Le tokenizer SentencePiece de Moshi :
- 32,000 tokens, Unigram, entraine sur donnees anglaises
- Byte-fallback : peut tokenizer du francais (pas de crash)
- Mais : inflation 1.5-3x tokens par mot francais
- Inner Monologue degrade si tokens fragmentes

### Options

| Option | Complexite | Qualite estimee |
|--------|-----------|-----------------|
| Garder tokenizer anglais + ft_embed | Faible | Correcte pour MVP |
| CamemBERT SentencePiece 32k | Moyenne | Bonne |
| Extraire de kyutai/tts-1.6b-en_fr | Moyenne | Meilleure |
| Entrainer custom 32k francais | Haute | Maximale |

### Decision MVP

**Garder le tokenizer anglais** pour le premier run.
- Le francais partage l'alphabet latin avec l'anglais
- Beaucoup de cognates (information, question, situation...)
- ft_embed=true permet d'adapter les embeddings
- J-Moshi a remplace le tokenizer car le japonais est radicalement different
- Pour un deuxieme run, tester CamemBERT SentencePiece

---

## 5. MIMI CODEC : CONFIRME AGNOSTIQUE

Evidence :
1. J-Moshi : Mimi gele, japonais fonctionne (seulement -0.5 pts re-synthese)
2. Hibiki : 450,000h de francais a travers Mimi sans retraining
3. kyutai/tts-1.6b-en_fr : modele TTS francais utilisant Mimi
4. Kyutai est un labo PARISIEN — ils ont valide Mimi sur le francais

**=> Mimi reste gele. Pas de modification necessaire.**

---

## 6. DATASETS FRANCAIS DISPONIBLES

### Tier 1 : Stereo natif, gratuit

| Dataset | Heures | Format | Licence | Acces |
|---------|--------|--------|---------|-------|
| SUMM-RE (Linagora) | 95h | Micro separe/locuteur, 32-48kHz | CC BY-SA 4.0 | HuggingFace direct |
| NCCFr | 36h | Stereo 44.1kHz, mics individuels | Data Use Agreement | Email dataofficer@let.ru.nl |
| CID | 8h | Pistes separees, studio | Recherche | ORTOLANG |

### Tier 2 : Mono, necessite diarisation

| Dataset | Heures | Licence | Acces |
|---------|--------|---------|-------|
| ESLO | 296h alignees | Open | ORTOLANG |
| ORFEO/CEFC | 120h+ | Open | ORTOLANG |
| TCOF | 23h | Open | ORTOLANG |
| CLAPI | 50h | CC BY-NC-SA 4.0 | clapi.ish-lyon.cnrs.fr |

### Action immediate
1. **Telecharger SUMM-RE maintenant** : `datasets.load_dataset("linagora/SUMM-RE")`
2. **Envoyer email NCCFr** : dataofficer@let.ru.nl
3. **Creer compte ORTOLANG** pour CID

### Stereo natif vs pseudo-stereo (pyannote)

| Aspect | Stereo natif | Pseudo-stereo |
|--------|-------------|---------------|
| Isolation canaux | Parfaite | Fuite significative |
| Parole chevauchee | Preservee | Perdue/corrompue |
| Attribution erreurs | 0% | 5-15% DER |
| Qualite pour Moshi | Excellente | Marginale |

---

## 7. COMPUTE ET BUDGET

### Prix RunPod (mars 2026)

| GPU | VRAM | $/h (on-demand) | Heures pour $30 |
|-----|------|-----------------|-----------------|
| RTX 4090 | 24GB | $0.39 | 76h |
| A6000 | 48GB | $0.49 | 61h |
| A100 80GB | 80GB | $1.44 | 20h |

### Estimation pour notre config

| Parametre | Valeur |
|-----------|--------|
| Tokens par step | batch_size * duration_sec * 12.5 * 9 = 8*100*12.5*9 = 90,000 |
| Total tokens | 1500 * 90,000 = 135M |
| Temps estime (A6000) | ~3-5h |
| Cout training | ~$1.50-$2.50 |
| Cout storage (75GB) | ~$1/semaine |
| **Cout total 1 run** | **~$3-5** |
| **Runs possibles avec $30** | **6-7 runs** |

### Recommandation

**A6000 48GB on-demand sur RunPod** :
- VRAM suffisante pour batch_size=8
- Cout minime par run
- Pas de risque d'interruption spot
- Budget permet iterations multiples

---

## 8. ESTIMATION MEMOIRE (A6000 48GB)

| Composant | VRAM estimee |
|-----------|-------------|
| Modele 7B (bf16) | ~14 GB |
| LoRA rank 128 | ~0.3 GB |
| Embeddings (ft_embed) | ~0.5 GB |
| Optimizer AdamW (float32) | ~1-2 GB |
| Activations (batch=8, seq=1250) | ~15-25 GB |
| Gradient checkpointing | -40-60% activations |
| **Total estime** | **~25-35 GB** |
| **Marge sur 48GB** | **13-23 GB** |

---

## 9. RESULTATS ATTENDUS (HONNETE)

### Ce qu'on peut attendre
- Inner Monologue montre des mots francais
- Quelques phonemes francais dans l'audio genere
- Proof-of-concept que l'approche fonctionne

### Ce qu'on ne peut PAS attendre
- Francais fluide et naturel (J-Moshi = 2.67/5 avec 60,000h + 128 V100)
- Pas de melange anglais/francais
- Qualite production

### Pour ameliorer
1. Plus de donnees (SUMM-RE 95h + NCCFr 36h = 131h)
2. Tokenizer francais (CamemBERT SentencePiece)
3. Plus de steps (mais risque overfitting)
4. Donnees synthetiques TTS (comme J-Moshi-ext : +602h TTS)

---

## 10. CHECKLIST PRE-ENTRAINEMENT

- [ ] Telecharger SUMM-RE depuis HuggingFace
- [ ] Convertir en stereo 24kHz WAV (L=locuteur1, R=locuteur2)
- [ ] Creer JSONL manifest (scripts/04_prepare_dataset.py)
- [ ] Setup RunPod A6000 (scripts/05_cloud_setup.sh)
- [ ] Cloner moshi-finetune, installer deps
- [ ] Telecharger moshiko-pytorch-bf16
- [ ] Lancer annotate.py --lang fr --local sur train.jsonl
- [ ] Verifier que les .json sont generes correctement
- [ ] Lancer training : torchrun --nproc-per-node 1 -m train ../configs/french_lora.yaml
- [ ] Monitorer eval_loss toutes les 100 steps
- [ ] Generer samples audio a chaque checkpoint pour evaluer
- [ ] Telecharger les poids LoRA finaux (~300MB)
