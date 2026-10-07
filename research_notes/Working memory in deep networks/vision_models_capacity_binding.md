# Capacity limits, set-size effects, and binding failures in ViTs, CNNs, and VLMs

Context: the downstream project fine-tunes a pretrained ViT-B/16 on a continuous-report color task (1–8 colored squares, spatial cue, report hue) and wants an interpretability framing. Notes compiled 2026-10-07.

Sourcing convention: items marked **[verified]** were checked against a primary page (arXiv abstract or HTML, proceedings, or CVF) in this session. Items marked **[canonical, not re-fetched]** are well-known papers cited from prior knowledge with their standard arXiv or venue link. Their bibliographic details are reliable, but the specific claims were not re-read in this session. Items flagged **(MOST RECENT)** are from 2025–2026.

---

## Q1. Campbell et al. 2024 (NeurIPS): VLM limits through the lens of the binding problem

### Takeaway
Campbell et al. show that frontier VLMs (GPT-4v/4o, Gemini Ultra 1.5, Claude 3.5 Sonnet) and text-to-image models show human-like signatures of parallel-processing capacity limits. Pop-out search does not depend on set size, while conjunction search gets worse as objects are added. Counting is near-perfect for about 1–5 items and drops sharply from 6. Scene-description errors scale with the risk of illusory conjunctions. The authors attribute all of this to the binding problem: shared, compositional representations cause interference between objects, and the models lack the serial processing that would avoid it.

### Cited Findings
- **Citation.** Campbell, D., Rane, S., Giallanza, T., De Sabbata, N., Ghods, K., Joshi, A., Ku, A., Frankland, S. M., Griffiths, T. L., Cohen, J. D., & Webb, T. W. (2024). Understanding the limits of vision language models through the lens of the binding problem. *Advances in Neural Information Processing Systems 37* (NeurIPS 2024, Main Track). DOI 10.52202/079017-3604. [verified] — [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/cdcc6d47c1627350014a3076112ab824-Abstract-Conference.html); [arXiv:2411.00238](https://arxiv.org/abs/2411.00238); [ML Anthology](https://mlanthology.org/neurips/2024/campbell2024neurips-understanding/)
- **Core claim.** Many VLM failures on counting, localization, and simple visual analogy come from the binding problem, "which arises when shared representational resources must represent distinct entities." Avoiding interference requires serial processing, and the failure modes "closely resemble the limits of rapid, feedforward processing in the human brain." — [arXiv abstract](https://arxiv.org/abs/2411.00238)
- **Visual search (set-size effect).** Four multimodal models were tested on 2D and 3D disjunctive (pop-out) and conjunctive search, with 1,000 images per condition. Disjunctive search was perfect and did not depend on distractor count. Conjunctive search was about 90% accurate at 5 objects and fell substantially as object count rose. This mirrors Treisman's feature-integration pattern. — [arXiv HTML v2](https://arxiv.org/html/2411.00238v2)
- **Numerical estimation (subitizing-like limit).** Both multimodal models and text-to-image models (Stable Diffusion Ultra, DALL-E 3, Parti, Muse) were accurate for 1–5 objects and dropped sharply at 6 or more, roughly matching the human subitizing range (about 4–6). Higher feature variability (entropy) across objects improved performance, which is consistent with interference between similar items. — [arXiv HTML v2](https://arxiv.org/html/2411.00238v2)
- **Scene description (illusory conjunctions).** Errors (edit distance) grew with the number of "feature triplets," meaning configurations likely to produce illusory conjunctions, and also with object count. Errors peaked where binding risk was highest. — [arXiv HTML v2](https://arxiv.org/html/2411.00238v2)
- **Visual analogy.** On 200 relational match-to-sample trials, accuracy was: Claude 3.5 Sonnet 100% (unified) and 100% (decomposed); GPT-4o 99% and 100%; GPT-4v 91% and 99%; Gemini Ultra 1.5 56% and 60%. Showing each object pair in a separate image (decomposed) generally helped. The authors read this as evidence that the bottleneck is multi-object processing, not relational reasoning. — [arXiv HTML v2](https://arxiv.org/html/2411.00238v2)
- **Interpretation.** Binding errors imply that VLMs use compositional (shared-feature) representations. These help generalization but create the possibility of interference. The authors suggest that serial processing or slot-based, object-centric mechanisms could help, but note that neither has been scaled to VLM size. Stated limitations: a small task set, mostly closed-source models, and very poor LLaVA-1.5 performance. — [arXiv HTML v2](https://arxiv.org/html/2411.00238v2)

### Inferences
- For a ViT-B/16 continuous-report project, Campbell et al. support a specific framing. A feedforward ViT is a single-pass, parallel system. If the readout shares feature channels across items, errors should rise with set size, and the model should make swap or misbinding errors (reporting a non-target's hue) more often when items are similar or close together. The non-target (swap) component of a mixture model is the most direct analogue of illusory conjunctions.
- The "feature entropy helps" result predicts that displays with more distinct hues should lower error at a fixed set size. That is a testable manipulation in a color-wheel task.

### Gaps
- Campbell et al. do not test continuous-report or change-detection paradigms, and they do not report internal mechanistic analysis (e.g., probing) of the VLM vision encoder. The interference account is behavioral and theoretical.
- I did not extract exact per-set-size accuracy numbers for conjunctive search beyond "about 90% at 5." Read Figure 2 of the PDF for the curves.

---

## Q2. Related behavioral failures: "VLMs are blind," counting and subitizing, degradation with more items

### Takeaway
Several independent benchmarks agree. VLMs handle small numbers of simple items but break down abruptly beyond about 5 items, on overlapping or nested shapes, and on compositional (multi-type) counts. Probing studies from 2024–2026 repeatedly find that the count or visual information is often decodable from the vision encoder or hidden states even when the text answer is wrong. This points to a readout or alignment bottleneck, not only a perceptual one.

### Cited Findings
- **Rahmanzadehgervi, P., Bolton, L., Taesiri, M. R., & Nguyen, A. T. (2024). Vision language models are blind. ACCV 2024.** [verified] — [CVF Open Access](https://openaccess.thecvf.com/content/ACCV2024/html/Rahmanzadehgervi_Vision_language_models_are_blind_ACCV_2024_paper.html); [arXiv:2407.06581](https://arxiv.org/abs/2407.06581); [project page](https://vlmsareblind.github.io/)
  - The BlindTest suite has 7 low-level tasks (e.g., line intersections, overlapping circles, nested squares, counting rows and columns). Mean accuracy across 4 models was 58.57% against a 24% chance baseline. The best model, Sonnet-3.5, scored 74.94% (77.84% in another revision; the figures differ across versions). Expected human accuracy is 100%. — [arXiv HTML v1](https://arxiv.org/html/2407.06581v1)
  - Counting overlapping circles in an Olympic-logo layout: all 4 models were 100% accurate at 5 circles, but adding one circle dropped accuracy to near zero. When wrong, Gemini-1.5 answered "5" 98.95% of the time, which shows a strong prior or familiarity bias. — [arXiv HTML v1](https://arxiv.org/html/2407.06581v1)
  - A later version reports that linear probes show the vision encoders contain enough information to solve BlindTest, and that the language model fails to decode it. — [arXiv:2407.06581](https://arxiv.org/abs/2407.06581)
- **Guo et al. (2025). "Your Vision-Language Model Can't Even Count to 20: Exposing the Failures of VLMs in Compositional Counting" (VLMCountBench). arXiv:2510.04401 / OpenReview.** **(MOST RECENT)** Models count reliably when only one shape type is present but fail when two or more types are mixed, even at small counts and with minimal clutter. Performance also depends on color, size, and prompt wording. — [arXiv:2510.04401](https://arxiv.org/abs/2510.04401); [OpenReview](https://openreview.net/forum?id=5JN68XdDli)
- **"Can Vision-Language Models Count? A Synthetic Benchmark and Analysis of Attention-Based Interventions" (2025). arXiv:2511.17722.** **(MOST RECENT)** Using circles on a white background, accuracy falls sharply as count rises. The authors attribute this to diffuse attention that blurs individual object representations and test inference-time attention reweighting. — [arXiv HTML](https://arxiv.org/html/2511.17722v1)
- **"Counting Circuits: Mechanistic Interpretability of Visual Reasoning in Large Vision-Language Models" (2026). arXiv:2603.18523.** **(MOST RECENT)** Reports a human-like discontinuity: precise counts for small sets and noisier estimates for large ones. Identifies classes of attention heads involved in counting. Targeted fine-tuning improved out-of-distribution counting in Qwen2.5-VL by about 8.36% on average. — [arXiv HTML](https://arxiv.org/html/2603.18523v1)
- **"The Count Is There, but Misaligned: Understanding and Correcting Counting Failures in VLMs" (2026). arXiv:2607.09544.** **(MOST RECENT)** Probes on hidden activations of 4 VLMs across 5 counting datasets often recover the correct count when the text answer is wrong. Causal steering helps, and detector-guided re-prompting improves accuracy by up to 15.6 points. — [arXiv HTML](https://arxiv.org/html/2607.09544v1)
- **NumerosityVLM (2026). arXiv:2608.15425.** **(MOST RECENT)** A benchmark of 10,800 images over 12 numerosity levels spanning the subitizing and approximate-number ranges. Variance is driven mainly by model identity, not by visual factors. A machine-generated summary says top models are near-perfect at 1–4 items and systematically undercount beyond about 20. Treat that summary as secondary. — [Pith summary](https://pith.science/paper/2608.15425)
- **Liu, N. F., et al. (2024). Lost in the middle: How language models use long contexts. TACL.** [canonical, not re-fetched] Accuracy on multi-document QA is U-shaped in the position of the relevant information and falls as the number of documents grows. This is the text analogue of degradation with more items. — [arXiv:2307.03172](https://arxiv.org/abs/2307.03172)

### Inferences
- The "information is there but not read out" pattern (BlindTest probes, "Count Is There") implies that for a fine-tuned ViT, item-level hue information may survive in patch tokens at high set sizes, and that errors may arise at the cue-conditioned readout (CLS or attention pooling). This can be tested directly by comparing linear-probe decoding of target hue from patch tokens at the cued location against the model's behavioral error as a function of set size.

### Gaps
- No precise authorship was extracted for 2511.17722, 2603.18523, 2607.09544, or 2608.15425. Check the arXiv pages before formal citation.
- I found no "Hu et al." paper matching the brief. The reference is ambiguous.

---

## Q3. Binding theory and object-centric capacity: Greff et al., Slot Attention, Frankland/Webb/Cohen

### Takeaway
The binding-problem framework (Greff et al. 2020) and object-centric architectures (Slot Attention) provide the theory: a fixed set of slots is an explicit capacity limit, while distributed superposed codes risk interference. The Webb, Cohen, and Frankland line moves from symbolic binding via external memory (2021) to evidence that VLMs implement emergent, content-independent spatial indices for binding (Assouel et al. 2025), with binding errors traceable to failures of those indices.

### Cited Findings
- **Greff, K., van Steenkiste, S., & Schmidhuber, J. (2020). On the binding problem in artificial neural networks. arXiv:2012.05208.** [canonical, not re-fetched] Splits the binding problem into segregation, representation (keeping separate objects in separate, composable slots), and composition. Argues that neural networks lack mechanisms to dynamically bind distributed information into object-like entities. — [arXiv:2012.05208](https://arxiv.org/abs/2012.05208)
- **Locatello, F., et al. (2020). Object-centric learning with Slot Attention. NeurIPS 2020.** [canonical, not re-fetched] K slots compete for input features through softmax over slots (normalized over the slot axis, not the inputs), followed by iterative refinement. K is a hard capacity parameter. — [arXiv:2006.15055](https://arxiv.org/abs/2006.15055)
- **Webb, T. W., Sinha, I., & Cohen, J. D. (2021). Emergent symbols through binding in external memory. ICLR 2021.** [canonical, not re-fetched] The Emergent Symbol Binding Network separates values from keys in an external memory, so abstract rules generalize. — [arXiv:2012.14601](https://arxiv.org/abs/2012.14601)
- **Assouel, R., Campbell, D., Bengio, Y., & Webb, T. (2025). Visual symbolic mechanisms: Emergent symbol processing in vision language models. arXiv:2506.15871 (v2, Dec 2025; no peer-reviewed venue listed).** **(MOST RECENT)** [verified] Reports "a previously unknown set of emergent symbolic mechanisms" for binding in VLMs, based on a "content-independent, spatial indexing scheme." Binding errors "can be traced directly to failures in these mechanisms." — [arXiv:2506.15871](https://arxiv.org/abs/2506.15871)
- **Cui, K., Prakash, N., Messica, S., Raina, A., Bau, D., Torralba, A., & Rott Shaham, T. (2026). The dual mechanisms of spatial variable binding in vision–language models. arXiv:2603.22278 (v3).** **(MOST RECENT)** [verified] The vision encoder is "the dominant source of spatial information" for binding. The LM backbone builds backup ordering representations in intermediate layers. Ordering information is spread across background tokens in strip-like patterns, not only object tokens. Interchange interventions span 5 VLMs (Qwen, Gemma, Pixtral). Amplifying vision-derived ordering directions fixed 40.2% (Gemma-3-4b) and 54.5% (Qwen2-VL-7B) of COCO-spatial failures, against 9.6% and 13.1% for random directions. The synthetic settings use only 3 objects; set size is not varied. — [arXiv HTML](https://arxiv.org/html/2603.22278)
- **Feng, J., & Steinhardt, J. (2024). How do language models bind entities in context? ICLR 2024.** [canonical, not re-fetched] The source of the "binding ID" concept: LMs attach abstract binding-ID vectors to entities and attributes, and these IDs can be swapped causally. The visual binding papers above build on this. — [arXiv:2310.17191](https://arxiv.org/abs/2310.17191)

### Inferences
- In the color task, the spatial cue is effectively a content-independent index (location) that has to retrieve a bound feature (hue). The Assouel and Cui papers suggest testing whether ViT-B/16 represents location as a separable index direction, and whether misbinding errors line up with confusions in that index, e.g., swaps toward spatially adjacent items.

### Gaps
- I did not retrieve the full Assouel et al. results (layer-wise locations, set-size dependence). The abstract alone does not say whether errors scale with object count.
- Frankland's specific binding-in-transformers papers were not searched separately in this session.

---

## Q4. Feature and object binding in CLIP and ViTs: what patch tokens encode

### Takeaway
CLIP-style contrastive encoders show bag-of-words behavior: good single-object composition, with sharp failures when attributes must be bound to the correct object (Lewis et al.; ARO). Probing work from 2025–2026 shows that large self-supervised ViTs (DINOv2, CLIP) do encode a pairwise "IsSameObject" signal in patch tokens, but the signal is spatially local. It decays exponentially with patch distance toward a nonzero floor, so binding over distance is limited.

### Cited Findings
- **Lewis, M., Nayak, N. V., Yu, P., Yu, Q., Merullo, J., Bach, S. H., & Pavlick, E. (2024). Does CLIP bind concepts? Probing compositionality in large image models. Findings of EACL 2024.** [verified via search] CLIP composes concepts for single objects, but "in situations where concept binding is needed, performance drops dramatically." The paper uses synthetic single-object, two-object, and relational datasets. — [ACL Anthology](https://aclanthology.org/2024.findings-eacl.101/); [arXiv:2212.10537](https://arxiv.org/abs/2212.10537)
- **Yuksekgonul, M., Bianchi, F., Kalluri, P., Jurafsky, D., & Zou, J. (2023). When and why vision-language models behave like bags-of-words, and what to do about it? ICLR 2023 (oral).** [canonical, not re-fetched] Introduces the ARO benchmark (Visual Genome Relation and Attribution, COCO and Flickr order). CLIP-family models are often near chance on attribute and relation binding and order. The authors attribute this to contrastive retrieval objectives not rewarding order, and propose hard-negative fine-tuning (NegCLIP). — [arXiv:2210.01936](https://arxiv.org/abs/2210.01936)
- **Thrush, T., et al. (2022). Winoground. CVPR 2022.** [canonical, not re-fetched] A compositional image-text matching benchmark on which VLMs perform near or below chance. — [arXiv:2204.03162](https://arxiv.org/abs/2204.03162)
- **Li, Y., Salehi, S., Ungar, L., & Körding, K. P. (2025). Does object binding naturally emerge in large pretrained vision transformers? NeurIPS 2025 (Spotlight). arXiv:2510.24709.** **(MOST RECENT)** [verified via search and NeurIPS page] Motivated by the pairwise form of self-attention, the authors hypothesize an "IsSameObject" representation and decode it from frozen patch embeddings with quadratic probes on ADE20K (majority baseline 72.6%). Accuracy is about 88% for DINOv2 across sizes, about 85% for CLIP-L/14, and markedly weaker for MAE. Ablating the IsSameObject subspace in DINOv2-L layer 18 hurts segmentation and raises pretraining loss. The authors call the causal evidence for downstream use indirect. The paper covers object-identity binding only, not compositional binding. — [arXiv:2510.24709](https://arxiv.org/abs/2510.24709); [NeurIPS 2025 poster](https://neurips.cc/virtual/2025/poster/119887); [code](https://github.com/liyihao0302/vit-object-binding)
- **Singal, M. (2026). Emergent object binding has a finite spatial horizon. arXiv:2610.00006 (submitted July 2026).** **(MOST RECENT; single-author preprint)** [verified] The probability that two same-object patches are decoded as bound falls monotonically with distance, "well described by an exponential with a finite length scale," and levels off at a nonzero floor. This holds across object sizes, 3 probe families, ADE20K and COCO, and DINO and CLIP backbones. It explains weaker binding for large objects and poorer separation of same-class instances than different-class ones. Occlusion does not matter once size is controlled. Horizon depth differs between DINOv2 and DINOv3 (flagged as preliminary). — [arXiv:2610.00006](https://arxiv.org/abs/2610.00006)
- **"I Walk the Line: Examining the role of Gestalt continuity in object binding for vision transformers" (2026). arXiv:2604.09942.** **(MOST RECENT)** Identifies "Gestalt continuity heads" that track continuous curves across patches. Ablating them often selectively damages binding representations. — [arXiv HTML](https://arxiv.org/html/2604.09942v1)

### Inferences
- In the color task, each square in ViT-B/16 probably occupies one or a few 16x16 patches, so within-object binding is easy. The binding problem that matters is location-to-hue: the cue location must route the right hue to the readout. The finite spatial horizon result suggests that interference between nearby items, i.e., more swap errors at small inter-item distances, is a plausible ViT-specific prediction. It parallels human crowding and spatial-proximity effects on swap errors.
- The IsSameObject probe method (quadratic probe on token pairs) can be reused directly to ask whether patch tokens at the cue and target location share a "bound" subspace, and whether this weakens with set size.

### Gaps
- I found no paper that probes ViT patch tokens for color-location binding as a function of the number of items. This appears to be an open niche that the project could fill.

---

## Q5. Attention as a limited resource: softmax competition, sinks, registers, bottlenecks

### Takeaway
There is formal and empirical support for treating softmax attention as a normalized, competitive resource. With bounded logits, the maximum weight any one token can receive shrinks as the number of competing items grows, and attention cannot stay sharp as item count increases out of distribution (Veličković et al. 2024). Single-head selection becomes nearly uniform as the number of tokens to select grows (Mudarisov 2025). ViTs also reallocate attention to high-norm "register" or sink tokens. Together these give a plausible mechanistic account of set-size effects in a cue-guided readout.

### Cited Findings
- **Veličković, P., Perivolaropoulos, C., Barbero, F., & Pascanu, R. (2024). Softmax is not enough (for sharp out-of-distribution). arXiv:2410.01104 (later ICML 2025).** [canonical; abstract content confirmed via search snippet] Even for finding the maximum key, any learned softmax circuit "must disperse as the number of items grows" at test time. Proposes adaptive temperature as a fix. — [arXiv:2410.01104](https://arxiv.org/abs/2410.01104)
- **Mudarisov, T., et al. (2025). Limitations of normalization in attention mechanism. arXiv:2508.17821.** **(MOST RECENT)** As the number of selected tokens increases, the ability to distinguish informative tokens declines and attention often converges toward uniform selection. In GPT-2 experiments, distinguishability collapses once fewer than about 6% of tokens are selected (under L2-normalized, isotropic assumptions). — [arXiv:2508.17821](https://arxiv.org/abs/2508.17821)
- **Nakanishi, K. M. (2025). Scalable-Softmax is superior for attention. arXiv:2501.19399.** [via search] The softmax denominator grows with context size, which flattens attention. Proposes scaling the exponent base with input length. — [arXiv HTML](https://arxiv.org/html/2501.19399v1)
- **Darcet, T., Oquab, M., Mairal, J., & Bojanowski, P. (2024). Vision transformers need registers. ICLR 2024.** [canonical, not re-fetched] Large ViTs (DINOv2, CLIP, DeiT-III) repurpose low-information background patches as high-norm tokens that absorb attention and store global information, producing artifact attention maps. Adding register tokens fixes this. — [arXiv:2309.16588](https://arxiv.org/abs/2309.16588)
- **Xiao, G., Tian, Y., Chen, B., Han, S., & Lewis, M. (2024). Efficient streaming language models with attention sinks. ICLR 2024.** [canonical, not re-fetched] Initial tokens absorb disproportionate attention as "sinks." Softmax forces attention mass to go somewhere. — [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)
- **Jaegle, A., et al. (2021). Perceiver: General perception with iterative attention. ICML 2021.** [canonical, not re-fetched] Cross-attention from a small latent array to a large input imposes an explicit, fixed-size bottleneck. This is an architectural analogue of a limited-capacity store. — [arXiv:2103.03206](https://arxiv.org/abs/2103.03206)
- **Elhage, N., et al. (2022). Toy models of superposition. Transformer Circuits Thread.** [canonical, not re-fetched] Networks represent more features than dimensions by superposing nearly orthogonal directions, which causes interference that grows with the number of simultaneously active features. This offers a representational-interference account of capacity. — [transformer-circuits.pub](https://transformer-circuits.pub/2022/toy_model/index.html)
- The 2025 counting paper (arXiv:2511.17722) attributes VLM counting failures specifically to diffuse attention that blurs object representations. — [arXiv HTML](https://arxiv.org/html/2511.17722v1)

### Inferences
- For ViT-B/16 with a CLS or attention-pooled readout: if the cue must be routed through softmax attention over N item patches plus distractor background, the weight on the target falls roughly as 1/(effective competitors). This predicts graded precision loss with set size, a resource-like pattern, and not a hard slot limit. Superposition predicts an additional increase in swap errors when hue features co-occur. These two mechanisms map onto the classic resource-versus-slot debate and could be pulled apart with attention-weight analysis (target attention mass against set size) and probing (decodability of non-target hues).
- Register or sink tokens are a practical confound: attention-mass analyses in ViT-B/16 should exclude or separately track high-norm artifact tokens.

### Gaps
- I found no study that measures how target attention mass in a ViT scales with the number of task-relevant items in a controlled display. This is a direct experimental opportunity for the project.
- I did not retrieve the exact venue or author list for the Veličković paper from a primary source this session.

---

## Q6. Psychophysics of DNNs: visual search, change detection, MOT, and VWM models

### Takeaway
CNNs show set-size effects in visual search, but the effects come from features learned for object recognition (Nicholson & Prinz 2022) and from capacity limits in the representation (Poder 2017), not from an explicit attention mechanism. Large-scale DNN features combined with the TCC (Target Confusability Competition) model reproduce human set-size and bias curves in continuous-report color and orientation working memory (Communications Psychology 2024). In that work, CLIP ViTs fit human data worse than several CNNs.

### Cited Findings
- **Nicholson, D. A., & Prinz, A. A. (2022). Could simplified stimuli change how the brain performs visual search tasks? A deep neural network study. *Journal of Vision* 22(7).** [verified via search] Four architectures pretrained on natural images, faces, or X-rays, with frozen features and a new final layer for search, show set-size effects on simplified search stimuli. Networks trained from random initialization do not, which implies that the effect comes from optimizing for object recognition. The set-size effect persists with 10 stimulus types, which rules out a binary-class artifact. — [PubMed 35675057](https://pubmed.ncbi.nlm.nih.gov/35675057); [JOV](https://jov.arvojournals.org/article.aspx?articleid=2778890); [bioRxiv 2020](https://www.biorxiv.org/content/10.1101/2020.10.26.354258v2.full)
- **Põder, E. (2017/2022). Capacity limitations of visual search in deep convolutional neural networks. arXiv:1707.09775 (later in a journal, PubMed 36112924).** [verified via search] CNNs trained on search show capacity-limited, set-size-dependent performance. — [arXiv:1707.09775](https://arxiv.org/pdf/1707.09775); [PubMed](https://pubmed.ncbi.nlm.nih.gov/36112924/)
- **Srivastava, S., Wang, W. Y., & Eckstein, M. P. (2024). Emergent human-like covert attention in feedforward convolutional neural networks. *Current Biology*.** [verified via search; authors canonical] A feedforward CNN shows cueing effects and search set-size effects similar to Nicholson & Prinz, though smaller than those found with some other pretrained CNNs. — [Current Biology](https://www.cell.com/current-biology/fulltext/S0960-9822(23)01758-X)
- **Zhang, M., Feng, J., Ma, K. T., Lim, J. H., Zhao, Q., & Kreiman, G. (2018). Finding any Waldo with zero-shot invariant and efficient visual search. *Nature Communications* 9:3730.** [canonical, not re-fetched] IVSN uses top-down modulation of VGG features to search for novel targets with human-like fixation efficiency. — [Nature Communications](https://www.nature.com/articles/s41467-018-06217-x)
- **"Scaling models of visual working memory to natural images" (2024). *Communications Psychology* (bioRxiv 2023.03.17.533050).** [verified via search] DNN features combined with the TCC model explain continuous-report memory errors for natural images. Intermediate layers reproduce set-size effects and response-bias curves for color and orientation. Within CLIP-trained models, the ViT performed worse than several CNNs. Among ImageNet models, VGG-19 and ResNet-50 beat ConvNeXt. Training-set size is confounded with objective. — [Communications Psychology](https://www.nature.com/articles/s44271-023-00048-3); [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.03.17.533050v2.full)

### Inferences
- The Communications Psychology paper is the closest precedent for the project. It shows that DNN feature similarity plus a psychophysical model (TCC) yields set-size effects, but the set-size limit comes from the TCC noise model, not from the network itself. The project's novelty is that capacity limits would emerge inside a trained ViT. Its finding that CLIP ViTs fit humans worse than CNNs is a useful comparison point.
- The Nicholson & Prinz result (set-size effects come from pretraining, not from-scratch training) suggests a control: compare pretrained ViT-B/16 with a randomly initialized ViT-B/16 trained on the same color task, to see whether set-size costs are inherited from pretrained features.

### Gaps
- Authors of the Communications Psychology paper were not extracted in this session. Check the Nature page.
- I did not find targeted studies of change detection or multiple-object tracking (MOT) in DNNs or ViTs during this session's searches.

---

## Q7. Continuous-report, color-wheel, or change-detection paradigms run on ViTs or VLMs (2023–2026)

### Takeaway
I found no published study (through October 2026, from the searches run) that applies a continuous-report color-wheel or classic change-detection paradigm with set-size manipulation to a ViT or VLM. The closest work is DNN features plus TCC (CNNs and CLIP; Q6), human-cognition batteries for MLLMs that include visual-memory subtests, and counting and set-size work (Q1–Q2).

### Cited Findings
- Searches for "vision transformer visual working memory set size continuous report," "VLM visual working memory change detection benchmark capacity," and "multimodal LLM working memory colored squares / delayed estimation" returned no direct ViT or VLM continuous-report or change-detection studies. — searches run 2026-10-07 (no source; a negative result)
- **VisFactor (2025), "Human Cognitive Benchmarks Reveal Foundational Visual Gaps in MLLMs," arXiv:2502.16435.** **(MOST RECENT)** Digitizes 20 vision-centric subtests of the FRCT cognitive battery, including Visual Memory and Associative Memory subtests. The best model scores only 54.0%. This is a memory benchmark but not change detection or continuous report. — [alphaXiv](https://www.alphaxiv.org/abs/2502.16435)
- **MaRs-VQA, "What is the Visual Cognition Gap between Humans and Multimodal LLMs?"** Notes that MLLM performance on problems requiring "visual working memory" and multi-image reasoning "is not well-established," and introduces a Raven's-style matrix benchmark. — [OpenReview](https://openreview.net/forum?id=78lTuD6wiO)

### Inferences
- The project (a ViT-B/16 trained on 1–8-item continuous report with a spatial cue) appears to fill a real gap. To my knowledge it would be among the first to measure mixture-model parameters (guess rate, precision, swap rate) as a function of set size inside a ViT, and to tie them to mechanisms (attention dilution, superposition, spatially local binding).
- Suggested framing: test three mechanistic accounts against each other. (1) Attention-as-resource: target attention mass falls with N, predicting precision loss. (2) Representational interference or superposition: hue decodability of non-targets and swap errors rise with N and with hue similarity. (3) A finite spatial binding horizon: swaps concentrate on nearby items. Campbell et al. (2024) provide the overarching binding-problem and serial-processing argument.

### Gaps
- Unpublished or very recent workshop papers (CogSci 2025–2026, NeurIPS 2025 workshops) may exist but were not indexed in these searches. A targeted check of CogSci 2026 proceedings and arXiv q-bio.NC for "delayed estimation" plus "transformer" is recommended.
