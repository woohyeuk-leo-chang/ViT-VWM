# Novelty check: VLMs as participants in human VWM paradigms

Search date: 2026-10-07. Sources: web search across arXiv, OpenReview, ACL Anthology, NeurIPS/CVF proceedings, CCN 2025 abstracts, Nature/Springer/PubMed, Google-indexed pages. About 20 targeted queries (terms listed at the end).

Tags: **[verified]** = I confirmed title, authors, and venue/ID on the arXiv, publisher, or proceedings page in this search. **[unverified]** = seen only on secondary pages or snippets, or details incomplete.

---

## 1. Verdict

**Partially done, but the core combination looks novel.**

- **Done:** treating VLMs as experimental participants in cognitive-psychology and psychophysics paradigms. Examples include intuitive physics and causal reasoning, neuropsychological batteries, the Stroop task, visual search across set sizes, contrast sensitivity, n-back, and color naming. Multi-object "capacity-like" limits in VLMs have been framed as a binding or interference problem (Campbell et al. 2024).
- **Partially done:** working-memory-style tests on VLMs. The only direct case is a spatial n-back with Qwen2.5-VL (Liang et al. 2026), scored with accuracy, d′, and logprobs. It manipulates grid size, not set size, and has no continuous report. Separately, deep-net features (including CLIP ViTs, but **not generative VLMs**) have been plugged into TCC to model human continuous-report errors and set-size effects (Bates, Alvarez & Gershman 2024).
- **Not found:** any paper that runs **continuous report on a color wheel** and/or **Luck & Vogel change detection** on VLMs (open or closed), with **set-size manipulation**, **sequential (memory-array then probe) multi-image presentation**, and **mixture-model / swap-error / TCC / K (Cowan/Pashler) analysis** of the model's responses. None of my queries turned up even a workshop abstract. Caveat: CogSci 2025/2026 and VSS 2026 abstracts are only partly indexed, so a poster could exist that I missed.
- **Also not found:** a published method that reads out VLM **logprob distributions over discretized color-wheel options** to get a full response distribution for mixture-model fitting. Liang et al. 2026 use trial-wise logprobs for n-back, and Hu & Levy 2023 justify logprob readout in general. Applying this to color-wheel distributions appears new.

The defensible novelty claim is the first application of standard VWM measurement models (mixture/swap/TCC, K from change detection) to VLM behavior under set-size manipulation with sequential presentation, plus a logprob-based continuous-report readout. Position it against Campbell et al. 2024 (capacity limits from binding) and Bates et al. 2024 (DNN features + TCC, encoder only).

---

## 2. Closest precedents

### Tier 1: most directly relevant

1. **Liang, S., Zhu, H., Wang, W., & Zhou, D. (2026). Can Vision Replace Text in Working Memory? Evidence from Spatial n-Back in Vision-Language Models. arXiv:2602.04355.** [verified]
   - Tests Qwen2.5 vs Qwen2.5-VL on a spatial n-back with matched text-rendered vs image-rendered grids, and varies grid size.
   - Metrics: accuracy, d′, and **trial-wise log-probability evidence**.
   - Findings: text beats vision. Nominal 2-back and 3-back often track a **recency-locked comparison** rather than the instructed lag. The authors call for "computation-sensitive" evaluation.
   - Relevance: the only VLM working-memory paradigm study found. It uses d′ and logprobs but has no continuous report, no change detection, and no mixture models.

2. **Campbell, D., Rane, S., Giallanza, T., De Sabbata, N., Ghods, K., Joshi, A., Ku, A., Frankland, S. M., Griffiths, T. L., Cohen, J. D., & Webb, T. W. (2024). Understanding the Limits of Vision Language Models Through the Lens of the Binding Problem. NeurIPS 37. arXiv:2411.00238.** [verified for title, venue, and lead/senior authors; full middle-author list unverified]
   - Tests GPT-4v, GPT-4o, Gemini Ultra 1.5, Claude 3.5 Sonnet, and LLaVA 1.5 on visual search (disjunctive vs conjunctive), numerosity, scene description, and visual analogy.
   - Findings: human-like **capacity limits**, with counting accurate up to about 5 items and a sharp drop at 6 or more. Performance is predicted by the **probability of feature-conjunction interference** rather than raw item count. Splitting a scene into separate images helps.
   - Relevance: the main conceptual precedent ("set-size-like" limits in VLMs explained by binding). It does not test memory, so there is no delay or probe, and no mixture or swap analysis. Swap errors are the natural VWM analogue of binding failures, which makes this the key paper to build on.

3. **Bates, C. J., Alvarez, G. A., & Gershman, S. J. (2024). Scaling models of visual working memory to natural images. Communications Psychology, 2, 3. doi:10.1038/s44271-023-00048-3.** [verified]
   - Combines DNN feature similarity with the **TCC** model to predict human continuous-report errors on GAN-generated natural-image continua. Also tests color and orientation.
   - Reported: the same encoders reproduce set-size effects and response-bias curves. Within CLIP-trained models, **ViT does worse than several CNNs** (per secondary summary; check the paper).
   - Relevance: the closest psychophysical-model precedent, but the network acts as a similarity metric inside a human model, not as a participant. Generative VLMs are not tested.

### Tier 2: the "VLM as participant" framework

4. **Binz, M., & Schulz, E. (2023). Using cognitive psychology to understand GPT-3. PNAS, 120(6), e2218523120.** [verified] Founding text-only "machine psychology" paper. Uses vignettes, bandits, and causal reasoning. Notes sensitivity to small perturbations.

5. **Schulze Buschoff, L. M., Akata, E., Bethge, M., & Schulz, E. (2025). Visual cognition in multimodal large language models. Nature Machine Intelligence, 7, 96–106. doi:10.1038/s42256-024-00963-y.** [verified] Extends the framework to VLMs: intuitive physics, causal reasoning, intuitive psychology. No memory tasks. (Binz is not an author.)

6. **Tangtartharakul, G., & Storrs, K. R. (2026). Visual language models show widespread visual deficits on neuropsychological tests. Nature Machine Intelligence, 8, 209–219. doi:10.1038/s42256-026-01179-y. arXiv:2504.10786.** [verified] Uses 51 tests from 6 clinical/experimental batteries on closed models (ChatGPT, Claude, Gemini), normed against healthy adults. Object recognition is good, but low- and mid-level deficits (line length, unfamiliar shape comparison) would be clinically significant in humans. Closed models only. Memory subtests are not described in the summaries I saw. [unverified whether any VSTM subtest is included]

7. **Rahmanzadehgervi, P., Bolton, L., Taesiri, M. R., & Nguyen, A. T. (2024). Vision language models are blind. ACCV 2024, LNCS 15476, pp. 18–34. doi:10.1007/978-981-96-0917-8_17. arXiv:2407.06581.** [verified] BlindTest: 7 trivial low-level tasks (overlapping circles, line intersections, circled letter, etc.). Mean accuracy about 58%, best model Claude 3.5 Sonnet at about 74–78% depending on version. Relevance: perceptual encoding may fail **before** memory, so you need a set-size-1 / no-delay perception baseline.

8. **Luo, D., et al. (2025). Machine Psychophysics: Cognitive Control in Vision-Language Models. arXiv:2505.18969; CCN 2025 abstract.** [verified for ID and first author; v2 retitled "Increasing Computation Resolves Conflicts in Vision Language Models"; full author list unverified] Stroop and Flanker (plus "squared" variants) on 108 VLMs (v1) / 47 VLMs (v2). Follow-up: "Conflict Adaptation in Vision-Language Models," arXiv:2510.24804 [unverified details].

9. **Zhang, R., de Winter, J. C. F., Dodou, D., Seyffert, H. C., & Eisma, Y. B. (2026). Human-Like Attention? A Psychophysical Comparison of Visual Search in Humans and MLLMs. Computational Brain & Behavior. doi:10.1007/s42113-026-00333-4. arXiv:2610.05463.** [verified for title, authors, IDs; details from secondary summary] About 1,250 humans vs MLLMs on identical stimuli **across set sizes**. Error rates correlate strongly, but models show extreme present/absent response biases. Closest **set-size** psychophysics on MLLMs (attention, not memory). Related: "Do VLMs search like humans? Reasoning tokens as an RT analog," arXiv:2606.25066 [unverified authors]; "I spy with my model's eye," arXiv:2510.19678 [unverified].

10. **Hernández-Cámara, P., Gomez-Villa, A., Jaén-Lorites, J. M., Vila-Tomás, J., Laparra, V., & Malo, J. (2026). Contrast sensitivity in multimodal large language models: A psychophysics-inspired evaluation. Neural Networks, 201, 108903. doi:10.1016/j.neunet.2026.108903. arXiv:2508.10367.** [verified] Builds psychometric functions and CSFs from **binary verbal responses**. No model matches human CSF in both shape and scale, and **estimates are highly sensitive to prompt phrasing**.

### Tier 3: working memory in LLMs (text) and multimodal memory benchmarks

11. **Huang, J.-t., et al. (2025). LLMs Do Not Have Human-Like Working Memory. arXiv:2505.10571** (repo title: "On the Failure of Latent State Persistence in LLMs"). [verified for ID and first author] Text-only: number guessing, yes-no game, Math Magic (Josephus). Models fail to hold latent state without writing it into context. Relevance: framing only. No visual stimuli.
12. **Gong, D., et al. (2024). Do Language Models Understand the Cognitive Tasks Given to Them? Investigations with the N-Back Paradigm. arXiv:2412.18120.** [unverified authors] Poor text n-back reflects task comprehension and task-set maintenance failures, not memory limits. This is an important confound for any VLM WM claim.
13. Multimodal "memory" benchmarks are **not VWM paradigms**: MemLens (arXiv:2605.14906, long-term multi-session memory, 27 LVLMs) [verified ID]; DMV-Bench (arXiv:2606.27499, agent visual memory) [unverified]; VisFactor (arXiv:2502.16435, 20 subtests from human cognitive batteries including a visual-memory factor; no set-size manipulation) [unverified details]; Thinking in Space / VSI-Bench (Yang et al., CVPR 2025) [verified venue]; M3-Verse (arXiv:2512.18735, before/after "spot the difference" videos) [unverified]; VDiff-Bench (arXiv:2609.06245) [unverified]; MLLM-CompBench (NeurIPS 2024 D&B) [unverified authors]. Mention these as "change detection" in the engineering sense, but they lack controlled arrays, set size, and psychophysical modeling.

### Tier 4: color perception and naming in VLMs (response-format confounds)

14. **Gomez-Villa, A., Hernández-Cámara, P., Butt, M. A., Laparra, V., Malo, J., & Vazquez-Corral, J. (2025). Color Names in Vision-Language Models. arXiv:2509.22524 (OpenReview mnfqJQSd6B).** [verified] 957 color samples, 5 VLMs. High accuracy on prototypical colors and a large drop on non-prototypical ones. 21 shared terms. Some models stick to "constrained" basic terms while others are "expansive" with lightness modifiers. Hue drives naming. The language model affects naming independently of vision. Strong English and Chinese bias across 9 languages.
15. **Liang, Y., ..., Zhou, T. (2025). ColorBench: Can VLMs See and Understand the Colorful World? NeurIPS 2025 D&B. arXiv:2504.10514.** [verified] 11 color tasks, 32 VLMs, more than 5,800 image-text pairs. The LLM matters more than the vision encoder. CoT helps. Large gaps to humans on some tasks.
16. **Marjieh, R., Sucholutsky, I., van Rijn, P., Jacoby, N., & Griffiths, T. L. (2024). Large language models predict human sensory judgments across six modalities. Scientific Reports, 14. arXiv:2302.01308.** [verified for title, venue, arXiv; author list from memory, unverified] Pairwise similarity recovers the color wheel. Multilingual color naming (English vs Russian) shifts with language. GPT-4 with vision input was **not** clearly better than text descriptors.
17. **Mukherjee, K., et al. (2026). Large Language Models Estimate Fine-Grained Human Color–Concept Associations. Cognitive Science, doi:10.1111/cogs.70219.** [verified title/venue; authors partly unverified] GPT-4V with patches alone did no better than text-only GPT-4. **Patches plus hex codes** did markedly better. This shows VLMs may not read color from pixels as well as from symbolic codes.

### Methodological references for readout

18. **Hu, J., & Levy, R. (2023). Prompting is not a substitute for probability measurements in large language models. EMNLP 2023, pp. 5040–5060. arXiv:2305.13264.** [verified] Direct probability readout beats metalinguistic prompting. A negative result from prompting does not show the competence is absent.
19. "Language Model Probabilities are Not Calibrated in Numeric Contexts." ACL 2025 (long), aclanthology 2025.acl-long.1417. [verified ID; authors unverified] Token probabilities carry systematic biases from word identity, order, and frequency.

---

## 3. Practical design lessons

**Response formats**
- **Avoid free-text color names as the primary report.** Naming is categorical, English-biased, and varies across models in granularity (Gomez-Villa et al. 2025). A naming-based report would show "categorical" errors that are artifacts of format. That gets confounded with mixture-model guess and precision parameters, and with TCC's similarity function.
- **Preferred continuous-report format:** show a **labeled response wheel image** (e.g., 36 or 72 numbered swatches), or give an index or angle scale in the prompt, and ask for the index of the remembered color. Read out the **full distribution** with logprobs over the index tokens. Make sure each index is a single token (check the tokenizer, since multi-digit numbers may split; use letter codes or 2-digit zero-padded tokens). Renormalize over valid options.
- **Calibrate the readout:** run a **perception-only control** (probe item still visible, set size 1, no delay) to estimate each model's encoding noise and its systematic bias in index or name mapping. Also run a **no-image prior** (prompt with blank arrays) to get a response prior and correct for label-frequency bias. Rotate the wheel randomly per trial and permute index labels so position and label biases do not masquerade as memory effects (cf. ACL 2025 calibration paper).
- **Change detection:** use a forced binary choice ("same"/"different"). Read P(different) from logprobs to get graded ROCs, d′, and criterion. Expect strong response biases (Zhang et al. 2026 found extreme present/absent biases), so report d′ and c, and estimate K with Cowan's/Pashler's formula alongside.
- **Hex codes are a leak:** never include hex or RGB values in the prompt. Mukherjee et al. show VLMs do much better with symbolic color codes, so these would bypass the visual pathway.

**Logprob readout**
- Available for open models (Qwen2.5-VL, InternVL, LLaVA, Idefics, PaliGemma) and partly for GPT-4o, which returns top-20 logprobs only. That is enough for about 20 options but truncates the tail. Claude and Gemini give limited or no token logprobs. For closed models, use **repeated sampling at T=1** to build an empirical distribution, and note that RLHF models are poorly calibrated and sampled distributions may be collapsed.
- Hu & Levy 2023 justify logprobs as the primary measure. Report prompted (sampled) responses as a secondary measure.
- Note that the mixture model's "guess" rate in a model could reflect flat logprobs (true uncertainty) or format failure. Distinguish them using the perception control.

**Prompt sensitivity**
- CSF estimates and Binz & Schulz results both shift with wording. Use **multiple paraphrased prompts (at least 3–5)** and treat prompt as a random effect in hierarchical mixture-model fits (e.g., `bmm` in R).
- n-back work (Gong et al. 2024; Liang et al. 2026) shows that failures often reflect **task comprehension or task-set** problems, and models may run a different computation (recency matching). Include instruction-comprehension checks and a "process" analysis: are errors swaps toward non-targets (binding), toward the most recent item, or toward canonical or prototypical colors (naming prior)?

**Multi-image / sequential handling**
- Two presentation modes: (a) **sequential images in one context** (memory array as image 1, optional mask or blank as image 2, probe as image 3). This tests in-context retention, but the model can re-attend to image 1, so there is no true decay or delay. (b) **Composite single image** (array and probe side by side) as a perception-or-comparison baseline. Campbell et al. found that splitting into separate images **improves** performance, so mode is a key factor to report.
- Because the transformer context keeps the array tokens, "memory" is really **attentional retrieval of in-context items under interference**. Frame set-size effects as interference or binding (Campbell) rather than storage decay. To get a delay analogue, insert distractor images or text between memory and probe.
- Check the per-image token budget and resolution. Dynamic-resolution models (Qwen2-VL, InternVL tiling) give different token counts for different array sizes. Fix the canvas size and item size across set sizes.
- Use a **spatial probe** (outline square or location cue at the probed item's location, as in human tasks) and verify the model can localize it. VLMs are poor at locating marked items ("VLMs are blind" circled-letter task), so localization failures will look like swap errors. Include a set-size-1 localization control and consider cueing with a number label as an alternative.

**Analysis**
- Fit the 2-component (Zhang & Luck) and 3-component (Bays 2009) mixture models plus TCC (Schurgin et al. 2020) per model × set size. For TCC you need a similarity function. Measure it from the model's own perceptual-control confusion data (a "model psychophysical similarity") rather than human similarity.
- Compare against human benchmark data from the same stimuli where possible, or from public datasets.

---

## 4. Search terms used

"vision language model" + "visual working memory" / "continuous report" / "color wheel" / "mixture model"; "multimodal LLM" + "change detection" + "Luck Vogel"; "visual working memory" VLM benchmark n-back; "working memory" VLM "delayed estimation" / "swap errors"; "set size" + MLLM; CogSci/CCN 2025 VLM working memory colored squares; "visual short-term memory" GPT-4o Gemini; "machine psychophysics" VLM logprobs; VLM token probabilities color hue distribution; GPT-4V color perception / color naming; ColorBench; spot-the-difference MLLM benchmark; Corsi MLLM; ViT / DNN change detection capacity; plus title verification queries for every named paper.

Known items: Campbell 2024 [verified], Rahmanzadehgervi 2024 [verified], Tangtartharakul & Storrs 2026 [verified], Huang 2025 [verified], Binz & Schulz 2023 [verified]. No dedicated "MLLM visual working memory benchmark" with VWM paradigms (2025–2026) found. Existing "memory" benchmarks (MemLens, DMV-Bench, VSI-Bench) target long-term, episodic, or spatial memory.
