# Novelty check: frozen pretrained vision encoders on VWM psychophysics paradigms

Search date: 2026-10-07. Sources searched: web search (Google Scholar-indexed pages, arXiv, bioRxiv, Europe PMC, NeurIPS/OpenReview, JOV/VSS, CCN), plus full-text reads of the two closest precedents.
Tags: **[verified]** means I read the full text or the official abstract/landing page this session. **[verified-abstract]** means I confirmed only the abstract or metadata. **[unverified]** means I know it from memory or a secondary summary only.

## Planned design (for reference)
Frozen encoders (supervised ViT-B/16, CLIP ViT-B/16, DINOv2 ViT-B/14, MAE, ResNet-50). Displays of 1–8 colored squares at 8 circular locations, using the Zhang & Luck CIELAB wheel. A cued **linear** readout of hue from the cued location's patch tokens. Fit Zhang & Luck / Bays 3-component mixtures (precision, guess, swap) per set size. Compare across training objectives.

---

## 1. Verdict: PARTIALLY DONE. The specific design appears novel.

- **Already done (core idea):** Two papers show that frozen, naturally pretrained vision networks produce human-like set-size effects in VWM tasks.
  - **Xie et al. (2023, bioRxiv)** froze pretrained CNNs and trained an RNN on top. They ran change detection and color delayed estimation with a cued location and the CIELAB ring. They fit a von Mises + uniform mixture and reported swap errors. They also compared pretraining objectives (CIFAR classification, SimCLR contrastive, MNIST, random, and ImageNet ResNet-18/50). This is the closest precedent and **must be cited and positioned against**.
  - **Bates, Alvarez & Gershman (2024)** showed that frozen DNN layers, including CLIP ViT-B/16 and CLIP RN50, reproduce set-size effects in delayed estimation through a TCC similarity model, with no trained readout.
- **Not found anywhere (as of Oct 2026):**
  - (a) ViT-family encoders (supervised ViT, DINOv2, MAE, CLIP ViT) tested on classic VWM paradigms with a **trained cued readout**.
  - (b) A **linear / patch-token-local** readout used as a "minimal decoder". Xie used a nonlinear CT-RNN with top-down CBAM attention feeding back into the CNN. Bates used no readout, only cosine-similarity TCC.
  - (c) A **systematic cross-objective comparison of modern foundation encoders** on mixture-model parameters (precision, guess, swap) as a function of set size.
  - (d) Swap and misbinding rates tied to a spatial readout of patch tokens.
- No ViT/CLIP/DINOv2 + continuous-report or change-detection paper turned up on arXiv, bioRxiv, JOV/VSS, OpenReview, or CCN.
  - The nearest 2025–2026 work concerns binding in ViTs (Li et al., NeurIPS 2025) or VLM behavior (Campbell et al. 2024; Frontera Del Valle 2026). None of these varies set size with a mixture-model analysis on frozen encoders.
- **Bottom line:** The novelty is (i) the model class (ViT-family foundation encoders and objectives), (ii) the minimal linear cued readout, which isolates what the frozen representation itself supports, and (iii) the mixture-model decomposition across objectives. The claim "pretrained sensory representations produce human-like set-size effects" is **not** new (Xie 2023; Bates 2024). Frame the pilot as a test of *which objectives and architectures* yield which error components, and *whether a linear readout suffices*.

---

## 2. Closest precedents

### Tier 1: directly overlapping

**P1. Xie, Y., Duan, Y., Cheng, A., Jiang, P., Cueva, C. J., & Yang, G. R. (2023).** *Natural constraints explain working memory capacity limitations in sensory-cognitive models.* bioRxiv 10.1101/2023.03.30.534982. https://www.biorxiv.org/content/10.1101/2023.03.30.534982v1 . Earlier version: CCN 2022 abstract (https://2022.ccneuro.org/view_papercfd7.html?PaperNum=1251). Europe PMC lists no journal version (PPR638931). **[verified]** (full text read)
- **Architecture:** CNN (sensory) feeds a continuous-time RNN (tanh), which produces the output.
  - The RNN sends top-down feature and spatial modulation (CBAM) to the CNN's last-block conv layers at each step.
  - Sensory CNN weights are **frozen** after pretraining. The RNN and attention-modulation parameters are trained with supervision.
- **Encoders:**
  - Mostly ResNet-20 at 32×32 input, pretrained on CIFAR-10 classification, SimCLR contrastive learning on CIFAR-10, or MNIST, or randomly initialized.
  - Also ImageNet ResNet-18/50 at 224×224 (Fig. 5b–c).
  - No ViTs, CLIP, DINO, or MAE.
- **Tasks:**
  - Change detection: 5×5 px colored patches at random non-overlapping positions on gray. Set sizes up to about 10. Colors drawn from 6 categories or from a CIELAB ring (L=54, a=21.5, b=11.5, r=49). Analyses: hit/false-alarm rates, ROC, change magnitude.
  - Delayed estimation: sample, then delay, then a black cue patch at the target location. The output is (x, y) on the color ring, trained with MSE.
  - Also prioritizing-cue and sequential-presentation variants.
- **Analyses:**
  - Recall SD and precision vs set size.
  - von Mises + uniform mixture fit, and the residual after removing the mixture.
  - Confidence.
  - **Swap errors**: non-target report distributions in Fig. 3n–o.
  - Activation vs set size (CDA/BOLD-like plateau), and layerwise decoding.
- **Findings:**
  - End-to-end-trained models are near-perfect with no set-size effect.
  - Natural-image-pretrained frozen CNNs show human-like set-size effects. ImageNet ResNets show smaller declines.
  - Random, MNIST, and noise-only models do not show human-like effects.
  - Conclusion: the limit is "bottom-up" from sensory representations, and it requires natural images *plus* an appropriate objective.
- **How the plan differs:**
  - Plan uses a linear readout vs Xie's nonlinear RNN plus top-down attention, which can reshape sensory features.
  - Plan tests modern ViT and foundation-model objectives (CLIP, DINOv2, MAE, supervised ViT) vs Xie's small CIFAR ResNets.
  - Plan uses fixed 8-location circular displays matched to Zhang & Luck (2008) vs random positions in 32×32 images.
  - Plan fits a full 3-component Bays mixture per set size vs Xie's 2-component fit plus a qualitative swap plot.
  - Plan reads out from patch tokens at the cued location (spatially local) vs Xie's pooled penultimate-layer vector.
  - Plan has no delay dynamics: the readout is "perception plus readout", not maintenance. This needs to be stated explicitly.

**P2. Bates, C. J., Alvarez, G. A., & Gershman, S. J. (2024).** *Scaling models of visual working memory to natural images.* Communications Psychology, 2, 3. https://doi.org/10.1038/s44271-023-00048-3 . bioRxiv 10.1101/2023.03.17.533050. Code: github.com/c-j-bates/scaling-models-of-vwm-to-natural-images. **[verified]** (full text read)
- **Method:** Frozen DNN layer activations feed the Target Confusability Competition (TCC) model.
  - Cosine similarity is computed between the stimulus and each of 360 response options. Similarities are scaled by d′, then Gaussian noise is added, and the response is the argmax.
  - The only free parameter is d′, fitted per layer. **No trained readout.**
- **Models:**
  - ImageNet classifiers (VGG-19, ResNets, ConvNeXt including 1k and 22k variants).
  - CLIP RN50 and **CLIP ViT-B/16**.
  - An autoencoder baseline.
  - Layers were searched exhaustively.
- **Data:**
  - Scene Wheels (GAN indoor-scene wheels).
  - A color dataset of colored circles, **set size 3 only**.
  - An orientation dataset with set sizes 1, 2, 4, and 8.
  - Response options were rendered with the non-probed items intact, so the whole image is encoded rather than a cropped target.
- **Findings:**
  - Selected layers predict trial-difficulty rank order.
  - They show set-size effects in orientation **without refitting d′ per set size**. The mechanism is activation sparsity: more items create more non-zero activations, which compress the similarity range, so fixed noise corrupts responses more.
  - Repulsion bias in orientation and focal-color bias in color.
  - CLIP ViT-B/16 performed worse than CNNs within CLIP models. Best layers differ by domain (e.g., CLIP ViT-B16 layer 12 for scenes, 23 for color, 11 for orientation).
  - No mixture-model guess/swap analysis and no color set-size manipulation.
- **How the plan differs:**
  - Plan trains a cued readout, so per-item, location-specific information is tested. Bates uses whole-image similarity with no cueing.
  - Plan includes DINOv2, MAE, and supervised ViT.
  - Plan manipulates color set size from 1 to 8.
  - Plan decomposes errors with mixture models, including swaps.
- **Related:** Bates' sparsity mechanism predicts that set-size effects depend on overall activation growth. Check this in ViTs, where LayerNorm may counteract it.

### Tier 2: adjacent (binding / multi-object / capacity in pretrained models)

**P3. Li, Y., Salehi, S., Ungar, L., & Kording, K. P. (2025).** *Does Object Binding Naturally Emerge in Large Pretrained Vision Transformers?* NeurIPS 2025 (Spotlight). arXiv 2510.24709. https://arxiv.org/abs/2510.24709 ; code: github.com/liyihao0302/vit-object-binding. **[verified]** (arXiv HTML read)
- **Models and probes:**
  - Frozen DINOv2 S/B/L/G, CLIP ViT-L/14, MAE ViT-L, and supervised ViT-L.
  - Pairwise "IsSameObject" probes on patch tokens from ADE20K: linear, diagonal-quadratic, and full-quadratic (low-rank) probes.
- **Findings:**
  - Quadratic probes are best (DINOv2-L reaches 90.2% vs a 72.6% baseline). Linear probes are weaker.
  - Self-supervised objectives (DINOv2 > CLIP ≈ MAE) are far above supervised ViT (76.3%).
  - Binding peaks in middle layers. Deeper layers drift toward class-level grouping, and position decodability drops.
- **Gaps relative to the plan:** No set-size, working-memory, or color-report component.
- **Relevance:** Strong prior that objective matters, with DINOv2 best and supervised worst for binding. Mid-layers are the place to look for swap and misbinding, and a linear probe may *underestimate* binding information.

**P4. Campbell, D., Rane, S., Giallanza, T., De Sabbata, N., Ghods, K., Joshi, A., Ku, A., Frankland, S. M., Griffiths, T. L., Cohen, J. D., & Webb, T. W. (2024).** *Understanding the Limits of Vision Language Models Through the Lens of the Binding Problem.* NeurIPS 2024. arXiv 2411.00238. https://arxiv.org/abs/2411.00238 **[verified-abstract]**
- **Models:** End-to-end VLMs and text-to-image models (mostly proprietary). No frozen-encoder probing.
- **Findings:**
  - Numerosity accuracy is high for 1–5 objects and drops at 6 or more.
  - Capacity limits soften when features are more varied.
  - Errors are predicted by feature-interference probability ("feature triplets"), not object count alone.
- **Relevance:** Use as an analog for swap/interference. Consider manipulating color similarity among items, not just set size.

**P5. Frontera Del Valle, A. F. (2026).** *What Looks Like a Capability Limit in Vision–Language Models Is a Readout Limit.* arXiv 2609.27408 (Sep 2026). https://arxiv.org/html/2609.27408 **[verified]** (HTML read)
- **Models and task:**
  - VLMs (Qwen2.5/3-VL, InternVL3, LLaVA-1.5, SmolVLM, GPT-4o, Gemini).
  - Four colored squares with a cued-location color query on a CIELAB wheel at fixed L. Includes a cue-crossover binding test.
- **Scope:** No frozen encoders, linear probes, set-size curves, or mixture models.
- **Relevance:**
  - Very recent and uses the same stimulus vocabulary.
  - It argues that apparent limits reflect the *readout format*, which directly motivates the plan's minimal-readout logic.
  - Cite it as concurrent VLM-side work.

**P6. Kiat, J. E., & Luck, S. J. (2026).** *A Population Vector Model of Visual Working Memory for Real-World Scenes.* J Exp Psychol Gen, 155(5), 1257–1281. doi:10.1037/xge0001921. bioRxiv 10.64898/2026.01.23.701256. **[verified-abstract]**
- **Method:** Scene memory is modeled as a noisy population vector over CORnet ventral-stream layers. It fits behavior and EEG/neural data.
- **Scope:** Natural scenes. No item set-size, mixture, or ViT component.

### Tier 3: background "machine psychophysics" (no VWM)

- **Nicholson, D. A., & Prinz, A. A. (2022).** *Could simplified stimuli change how the brain performs visual search tasks? A deep neural network study.* Journal of Vision, 22(7):3. PMID 35675057. **[verified-abstract]**
  - Pretrained CNNs were adapted (fine-tuned) to visual search and showed a set-size accuracy effect for simplified stimuli. The effect was absent when networks were trained from scratch.
  - Lesson: a set-size effect can be an *artifact of an object-recognition prior applied to simplified displays*. Expect reviewers to raise this.
- **Põder, E. (2022).** *Capacity limitations of visual search in deep convolutional neural networks.* Neural Computation. PMID 36112924. **[verified-abstract]**
  - Qualitative mismatch with human search capacity.
- **Nasr, K., Viswanathan, P., & Nieder, A. (2019).** *Number detectors spontaneously emerge in a deep neural network designed for visual object recognition.* Science Advances, 5, eaav7903. **[verified-abstract]**
  - Kim, G., Jang, J., Baek, S., Song, M., & Paik, S.-B. (2021). *Visual number sense in untrained deep neural networks.* Science Advances, 7, eabd6127. **[verified-abstract]**
  - Both are contested; critiques note that random networks show similar tuning and that low-level confounds such as area and density are present.
  - Lesson: **include randomly initialized encoders as controls.**
- **Volokitin, A., Roig, G., & Poggio, T. (2017).** *Do Deep Neural Networks Suffer from Crowding?* NeurIPS. arXiv 1706.08616. **[verified-abstract]**
  - Lonnqvist, B., Clarke, A. D. F., & Chakravarthi, R. (2020). *Crowding in humans is unlike that in convolutional neural networks.* Neural Networks. **[verified-abstract]**
  - Doerig et al. (2019/2020), capsule and grouping accounts of crowding. **[unverified]**
  - Lesson: with 8 locations and 1–8 items, inter-item spacing matters, since ViT patch tokens mix context from attention. Crowding-like interference may masquerade as a memory set-size effect.
- **Geirhos, R., Narayanappa, K., Mitzkus, B., Thieringer, T., Bethge, M., Wichmann, F. A., & Brendel, W. (2021).** *Partial success in closing the gap between human and machine vision.* NeurIPS 34. arXiv 2106.07411. **[verified-abstract]**
  - Template for comparing across objectives (supervised, self-supervised, CLIP, adversarial) and dataset scale. Error consistency is a useful metric.
- **Huang, L. (2025).** *Comprehensive exploration of visual working memory mechanisms using large-scale behavioral experiment.* Nature Communications. doi:10.1038/s41467-025-56700-5. **[verified-abstract]**
  - An MLP benchmark fit to human color-pattern VWM data. Not a pretrained encoder. Potential human-data source.
- **Adam, K. C. S., & Vogel, E. K.** (2023, Memory & Cognition). DNN-based foil selection for VWM with real-world objects. **[verified-abstract]** Not a model observer.
- **Uselis, A., Koishigarina, D., & Oh, S. J. (2026).** *How can embedding models bind concepts?* ICML 2026, arXiv 2605.31503. **[verified-abstract]**
  - Uni-modal probes can recover object bindings from CLIP embeddings. No set-size manipulation.
- **Schurgin, M. W., Wixted, J. T., & Brady, T. F. (2020).** *Psychophysical scaling reveals a unified theory of visual memory strength.* Nature Human Behaviour. **[verified-abstract]**
  - TCC is the main theoretical rival to mixture models. Reviewers may ask for TCC fits alongside the mixture fits.
- Not found or not checked: "Kim et al. 2021" beyond number sense; any VSS abstract on ViT VWM (none surfaced).

---

## 3. Design lessons and pitfalls

1. **Position against Xie (2023) explicitly.**
   - Their key control: end-to-end training gives *no* set-size effect, while frozen natural pretraining gives one.
   - Add equivalents: randomly initialized encoders of each architecture, and ideally an encoder trained or fine-tuned on the task.
   - Without these controls, a set-size effect is uninterpretable.
2. **Readout capacity changes conclusions.**
   - Xie's RNN plus top-down attention can actively select, so it is not a pure test of the representation.
   - Li et al. (2025) found linear probes miss binding that quadratic probes recover.
   - Plan: linear as primary, plus an MLP or quadratic readout as a sensitivity check.
   - Report whether set-size and swap effects survive richer readouts. If they vanish, the "limit" is in the readout, not the representation (cf. Frontera Del Valle 2026).
3. **Patch-token vs pooled readout.**
   - Reading out only the cued location's patch tokens builds in a perfect spatial cue. Swaps can then only arise from attention-mediated contamination of that token by other items.
   - That is a clean, interesting test, but it is a different model of "swap" than human misbinding.
   - Consider a second condition: a global readout (CLS or mean-pooled) with the cue given as an input feature or a cue image. This mirrors Xie's black-patch cue and Bates' whole-image approach.
4. **Layer choice matters a lot.**
   - Best layers differ by domain in Bates (deeper for scenes, shallower to mid for color and orientation).
   - Li et al. found binding peaks in mid layers and degrades late.
   - Sweep all blocks and report per-layer curves. Do not pick only the last layer.
5. **Mechanism check: activation sparsity and normalization.**
   - Bates attributes set-size effects to activation growth with more items, which compresses similarity under fixed noise.
   - Measure token norms and variance vs set size in ViTs (LayerNorm, register/high-norm artifact tokens in DINOv2 and CLIP).
   - Use DINOv2 *with registers* as a variant, or check for high-norm outlier patches.
6. **Noise is required for mixture fits.**
   - A deterministic encoder plus linear probe may produce near-zero error at set size 1 and non-circular error shapes.
   - Xie used task training. Bates added Gaussian noise on similarities.
   - Decide a principled noise source (Gaussian on features, on the readout, or dropout), and keep its variance fixed across set size. Otherwise you are fitting a set-size effect by hand.
   - Report which components (guess vs precision) are driven by the noise assumption.
7. **Stimulus and resolution artifacts.**
   - Xie used 5×5 px patches in 32×32 images, and ImageNet 224 px models showed weaker effects.
   - Match square size to patch grid: 16 px for ViT-B/16 and 14 px for DINOv2-B/14. Squares straddling patch boundaries will change results.
   - Hold eccentricity and spacing constant across set sizes to separate crowding from load (Volokitin 2017; Lonnqvist 2020).
   - Use a gray background matched to CIELAB L*. Check display-to-sRGB gamut, since some wheel colors clip.
8. **Low-level confounds.**
   - As with numerosity (Nasr 2019 / Kim 2021 critiques), total colored area and color-histogram statistics covary with set size.
   - Consider a control in which only the cued item's location is probed but the other items are replaced with gray placeholders. This separates encoding interference from cue ambiguity.
9. **Nicholson & Prinz caution.** A set-size effect in pretrained nets on simplified stimuli may reflect distribution shift from natural images, not a capacity limit. Report probe performance at set size 1 as a ceiling and normalize.
10. **Model theory.** Fit both the mixture model (Zhang & Luck / Bays) and TCC (Schurgin et al. 2020; used by Bates 2024). Reviewers in this space expect both, and TCC lets you compare directly to Bates.
11. **Objective comparison design.** Follow Geirhos (2021) and Li (2025): hold architecture fixed where possible (ViT-B with supervised vs CLIP vs DINOv2 vs MAE), and treat ResNet-50 as an architecture contrast. Li's prior predicts self-supervised models bind best, which predicts fewer swaps for DINOv2 and more for the supervised ViT.

---

## 4. Citation verification summary
- [verified] full text or HTML: Xie et al. 2023; Bates, Alvarez & Gershman 2024; Li, Salehi, Ungar & Kording 2025; Frontera Del Valle 2026.
- [verified-abstract]: Campbell et al. 2024; Kiat & Luck 2026; Nicholson & Prinz 2022; Põder 2022; Nasr et al. 2019; Kim et al. 2021; Volokitin et al. 2017; Lonnqvist et al. 2020; Geirhos et al. 2021; Huang 2025; Adam & Vogel 2023; Uselis et al. 2026; Schurgin et al. 2020.
- [unverified]: Doerig et al. crowding/capsule papers; exact JOV volume/issue for Nicholson & Prinz (22(7):3 is from memory).
