## XDVioDet CATEGORY LABELS SUMMARY

Based on analysis of the project code and result.txt, here are the violence category labels:

### Category Mappings:

| Label | Event Type | Numeric ID |
|-------|-----------|-----------|
| **B1** | Fighting | 1.0 |
| **B2** | Shooting | 2.0 |
| **G** | Explosion | 3.0 |
| **B4** | Riot | 4.0 |
| **B5** | Abuse | 5.0 |
| **B6** | Car accident | 6.0 |
| **(Background)** | Non-violence (labeled as _label_A) | 0.0 |

**Note:** There is no B3 in the dataset. "G" stands for "General" or appears to represent explosion-related violence.

---

## TOP 2 RECOMMENDED VIDEOS FOR ANALYSIS

### B1 - Fighting
1. **Spectre.2015** (24 annotated frames, 4 clips)
2. **Casino.Royale.2006** (20 annotated frames, 3 clips)

### B2 - Shooting  
1. **Rush.Hour.1998.BluRay** (36 annotated frames, 4 clips)
2. **Shoot.Em.Up.2007** (36 annotated frames, 3 clips)

### G - Explosion
1. **v=vhACO_m5pH0** (62 annotated frames, 5 clips)
2. **v=wVey5JDRf_g** (44 annotated frames, 3 clips)

### B4 - Riot
1. **v=pFamvR9CpYw** (18 annotated frames, 2 clips)
2. **Jason.Bourne.2016** (10 annotated frames, 2 clips)

### B5 - Abuse
1. **Death.Proof.2007** (8 annotated frames, 2 clips)
2. **Election.2005** (6 annotated frames, 1 clip)

### B6 - Car accident
1. **v=vhACO_m5pH0** (62 annotated frames, 5 clips)
2. **v=wVey5JDRf_g** (44 annotated frames, 3 clips)

---

## Dataset Statistics

- **Total unique videos**: 364
- **Total clips**: 500
- **Total annotated frames**: 2,476

| Category | Videos | Clips | Annotated Frames |
|----------|--------|-------|-----------------|
| B1 (Fighting) | 82 | 120 | 498 |
| B2 (Shooting) | 45 | 84 | 410 |
| G (Explosion) | ? | 96 | 944 |
| B4 (Riot) | 94 | 101 | 314 |
| B5 (Abuse) | 7 | 8 | 24 |
| B6 (Car accident) | 55 | 96 | 286 |

---

## PURE SINGLE-CATEGORY VIDEOS (For Per-Category Modality Analysis)

**Videos with only ONE category label (no mixed categories)** - Ideal for analyzing which modality is dominant in each specific category.

### B1 - Fighting (82 pure videos)
**Top 5 for modality analysis:**
1. Spectre.2015 (4 clips, 24 annotated frames)
2. Casino.Royale.2006 (3 clips, 20 annotated frames)
3. Desperado.1995 (2 clips, 18 annotated frames)
4. Ip.Man.3.2015 (3 clips, 18 annotated frames)
5. Haywire.2011 (4 clips, 16 annotated frames)

### B2 - Shooting (45 pure videos)
**Top 5 for modality analysis:**
1. Rush.Hour.1998.BluRay (4 clips, 36 annotated frames)
2. Shoot.Em.Up.2007 (3 clips, 36 annotated frames)
3. Sin.City.2005 (5 clips, 24 annotated frames)
4. Bullet.in.the.Head.1990 (3 clips, 16 annotated frames)
5. Mission.Impossible.II.2000 (4 clips, 16 annotated frames)

### G - Explosion (81 pure videos)
**Top 5 for modality analysis:**
1. qrKfaX1lCUM (1 clip, 18 annotated frames)
2. J67oj92maC0 (1 clip, 14 annotated frames)
3. Operation.Red.Sea.2018 (4 clips, 14 annotated frames)
4. Salt.2010 (1 clip, 10 annotated frames)
5. Bad.Boys.1995 (1 clip, 8 annotated frames)

### B4 - Riot (94 pure videos)
**Top 5 for modality analysis:**
1. pFamvR9CpYw (2 clips, 18 annotated frames)
2. Jason.Bourne.2016 (2 clips, 10 annotated frames)
3. 1q5V6DKH3bw (1 clip, 8 annotated frames)
4. 3zgaqeVSXuI (1 clip, 8 annotated frames)
5. 6TR2rcgHm4g (1 clip, 8 annotated frames)

### B5 - Abuse (7 pure videos)
**All 7 videos for modality analysis:**
1. Death.Proof.2007 (2 clips, 8 annotated frames)
2. Election.2005 (1 clip, 6 annotated frames)
3. City.of.God.2002 (1 clip, 2 annotated frames)
4. Operation.Red.Sea.2018 (1 clip, 2 annotated frames)
5. Sin.City.2005 (1 clip, 2 annotated frames)
6. Taken.2.UNRATED.EXTENDED.2012 (1 clip, 2 annotated frames)
7. Yellow.Sea.2010 (1 clip, 2 annotated frames)

### B6 - Car accident (55 pure videos)
**Top 5 for modality analysis:**
1. vhACO_m5pH0 (5 clips, 62 annotated frames)
2. 38GQ9L2meyE (1 clip, 44 annotated frames)
3. nLAapCIlr-o (4 clips, 44 annotated frames)
4. wVey5JDRf_g (3 clips, 44 annotated frames)
5. OOjjPGN8jSU (1 clip, 34 annotated frames)

---

## Research Paper Analysis Recommendations

Use the following metrics to strengthen the modality analysis section of your paper:

- **Per-frame dominant modality**: report which modality (RGB, Audio, or combined) drives the predicted class at each temporal block.
- **Overall dominant modality per clip**: compute an aggregate score from ablation drops across all frames to show which modality is most important for the clip.
- **Modality frame-share**: include the percentage of video frames dominated by each modality for each category.
- **Uncertainty statistics**: add average entropy and entropy variance across frames to quantify model confidence.
- **Temporal segment distribution**: analyze how many predicted segments appear, their average duration, and how they align with ground-truth events.
- **Class distribution and temporal coverage**: report how much of each clip is predicted as each class, especially for pure-category clips.
- **Ablation-based modality importance**: use modality drop values when each modality is removed to demonstrate robustness and modality dependency.
- **Comparison of pure vs. mixed-category clips**: use the pure single-category clips as a clean benchmark, and compare modality dominance results against mixed-category clips if available.

