# Mechanistic Interpretability Methods for Capacity Limits and Feature Binding in a ViT-B/16 VWM Model

Context: ViT-B/16 fine-tuned on a visual working memory (VWM) task. The memory image has 1–8 colored squares. The probe image has a white cue. A query derived from the probe cross-attends to the memory patch tokens, and the readout predicts hue as [sin, cos]. Target questions: (a) where is color and location information for each item stored, (b) how does it degrade with set size, and (c) what mechanism produces swap/misbinding errors.

Sourcing note: items marked [verified] were checked against primary or official pages in this session. Items marked [canonical] are well-known papers cited from prior knowledge with their standard arXiv/venue links; those links were not re-fetched in this session, so the report writer should spot-check them.

---

## 1. Binding mechanisms: binding IDs in LMs, their follow-ups, and object binding in ViTs/VLMs

### Takeaway
The best-supported account of binding in transformers is a causal one: entities and attributes carry shared, low-dimensional "binding ID" or "ordering ID" vectors, and retrieval mixes positional, lexical, and reflexive pointers. Two 2025 papers carry this into vision. VLMs assign binding IDs to image-object tokens, and pretrained ViTs encode a decodable, causally used "IsSameObject" relation in a low-dimensional subspace from mid-layers onward. For the VWM model, swap errors can be framed as confusing nearby binding/location codes, and this is testable with mean-difference "binding vector" interventions and interchange patching.

### Cited Findings
**Foundational LM binding work**
- [verified] Feng & Steinhardt, "How do Language Models Bind Entities in Context?" (arXiv:2310.17191, v1 Oct 2023, v2 May 2024; ICLR 2024 per the request, but the arXiv page itself lists no venue). They identify a "binding ID mechanism" in every sufficiently large Pythia and LLaMA model. Causal interventions show activations attach binding ID vectors to both entity and attribute tokens. Binding ID vectors form a continuous subspace, and **the distance between binding vectors tracks how discernible the bindings are**. That last point predicts more confusion (swaps) between items whose binding codes are close. — [arXiv](https://arxiv.org/abs/2310.17191)
- [verified] Dai, Heinzerling & Inui, "Representational Analysis of Binding in Language Models" (EMNLP 2024; arXiv:2409.05448). PCA on entity-token activations (Llama 2/3, Qwen1.5, Pythia, Float-7B) finds a low-rank subspace encoding the **Ordering ID (OI)**, i.e., the order index of an entity/attribute, with its top direction labeled OI-PC. Adding steps along OI-PC to an entity's activation changes which attribute it binds to. Filler-word controls show OI tracks abstract order rather than absolute token position. Limitation: only the attribute-prediction task was analyzed. Terminology moved from "Binding ID subspace" in v1/EMNLP to "Ordering ID" in later versions. — [ACL Anthology](https://aclanthology.org/2024.emnlp-main.967/); [arXiv](https://arxiv.org/abs/2409.05448)
- [verified] Prakash, Rott Shaham, Haklay, Belinkov & Bau, "Fine-Tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking" (ICLR 2024; arXiv:2402.14811). The base and fine-tuned models (Llama-7B and math-fine-tuned variants) share the same entity-tracking circuit. Fine-tuning improves it mainly by **augmenting positional information** used to attend to the correct object. Method: **cross-model activation patching (CMAP)**. — [arXiv](https://arxiv.org/abs/2402.14811); [project](https://finetuning.baulab.info/)
- [verified] Gur-Arieh, Geva & Geiger, "Mixing Mechanisms: How Language Models Retrieve Bound Entities In-Context" (ICLR 2026; arXiv:2510.06182) **[most recent]**. Retrieval is not purely positional. Models mix three mechanisms: **positional** (reliable at list start and end, diffuse in the middle), **lexical** (use the query entity to find its partner), and **reflexive** (a direct pointer). Tested on Llama, Gemma, and Qwen at 2–72B across 10 binding tasks. A mixed causal model reaches 0.95 Jensen–Shannon similarity vs 0.44 for a positional-only model. A noisier positional signal is proposed as the cause of "lost-in-the-middle." Code: github.com/yoavgur/mixing-mechs. — [arXiv](https://arxiv.org/abs/2510.06182); [blog](https://yoav.ml/blog/2025/mixing-mechs/)

**Binding in vision models**
- [verified] Saravanan, Tapaswi & Gandhi, "Investigating Mechanisms for In-Context Vision Language Binding" (CVPR 2025 Workshop on Mechanistic Interpretability for Vision; arXiv:2505.22200). On synthetic 3D objects paired with text, VLMs assign a shared binding ID to an object's image tokens and its textual references. Swapping object/item activations swaps the association, but swapping color activations does not, because they share the binding ID. Adding estimated mean binding-difference vectors predictably swaps associations, while random vectors do not. — [CVF PDF](https://openaccess.thecvf.com/content/CVPR2025W/MIV/papers/Saravanan_Investigating_Mechanisms_for_In-Context_Vision_Language_Binding_CVPRW_2025_paper.pdf); [arXiv](https://arxiv.org/abs/2505.22200)
- [verified] Li, Salehi, Ungar & Kording, "Does Object Binding Naturally Emerge in Large Pretrained Vision Transformers?" (NeurIPS 2025 spotlight; arXiv:2510.24709) **[most recent, most directly relevant]**. They define **IsSameObject** (do two patches belong to the same object?) and decode it at 90.2% with a **quadratic (pairwise) similarity probe** from mid-layers onward (trivial baseline 72.6%, ADE20K). The signal sits in a **low-dimensional subspace on top of object features**, guides attention, and ablating it hurts downstream performance. Version discrepancy: arXiv v1 says the effect is present in DINO/MAE/CLIP but largely absent in ImageNet-supervised ViTs, while the later abstract/NeurIPS version says it emerges in DINO, CLIP, and ImageNet-supervised ViTs but is weaker in MAE. Code with a layer-wise viewer: github.com/liyihao0302/vit-object-binding. — [arXiv](https://arxiv.org/abs/2510.24709); [OpenReview](https://openreview.net/forum?id=5BS6gBb4yP); [GitHub](https://github.com/liyihao0302/vit-object-binding)
- [verified] Campbell, Rane, Giallanza, De Sabbata, Ghods, Joshi, Ku, Frankland, Griffiths, Cohen & Webb, "Understanding the Limits of Vision Language Models Through the Lens of the Binding Problem" (NeurIPS 2024; arXiv:2411.00238). Five multimodal LMs (GPT-4v, GPT-4o, Gemini Ultra 1.5, Claude Sonnet 3.5, LLaVA 1.5) and four text-to-image models show **human-like capacity limits** on counting and localization, similar to speeded human subitizing (about 4–6). Errors are **not explained by object count alone**. They are best explained by the **probability of interference given the feature-conjunction distribution**. More feature variability means less overlap in shared resources and fewer binding errors. — [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/cdcc6d47c1627350014a3076112ab824-Abstract-Conference.html); [arXiv](https://arxiv.org/abs/2411.00238)
- [canonical] Feng, Russell & Steinhardt, "Monitoring Latent World States in Language Models with Propositional Probes" (ICLR 2025; arXiv:2406.19501). Builds on binding IDs: a "binding subspace" lets probes compose entity–attribute propositions from activations. This is a template for reading out (item, color, location) triples. — [arXiv](https://arxiv.org/abs/2406.19501)

### Inferences
- Direct transfer to the VWM model: run the Feng–Steinhardt / Saravanan "binding vector" protocol on memory patch tokens. Take displays A and B. Patch the location-carrying component of item i's tokens from A into B and see whether the predicted hue switches to the item now "bound" to the cued location. Estimate the mean binding-difference vector between items at two locations, add it, and check for predictable swaps.
- Feng–Steinhardt's "distance between binding vectors predicts discernibility" and Campbell et al.'s "interference depends on feature-conjunction overlap" together predict that **swap rates should rise when distractors are spatially near the cued item (similar location codes)**, not just with set size. Test by binning swaps by cue–distractor distance (like cue-feature-variability accounts of human swaps).
- Li et al.'s quadratic probe fits the task: train a pairwise probe on (patch_i, patch_j) for "same square" by layer. That tells you when item-level grouping exists before the cross-attention readout, and whether it degrades with set size.
- Gur-Arieh et al.'s positional/lexical/reflexive split suggests a vision analog. The probe query could retrieve by **location code** (positional) or by **feature match**. In this task the cue is location-only, so swaps should come from noise in positional (location) retrieval.

### Gaps
- No published mechanistic study found of a ViT trained on a delayed-estimation VWM task with set-size and swap analyses. The closest are Campbell et al. (behavioral, VLMs) and Gong & Zhang (N-back, text; see Section 2).
- Did not verify whether Feng & Steinhardt's ICLR 2024 version includes the "factorizability" and "position independence" claims often attributed to it. The arXiv abstract mentions only binding ID vectors and the continuous subspace.

---

## 2. Superposition, capacity, and sparse autoencoders on ViTs

### Takeaway
Superposition theory predicts that when features (item × color × location) outnumber the effective dimensions available in the readout pathway, features get packed non-orthogonally, so interference rises smoothly with load. This matches graded (not hard-cutoff) VWM precision loss. SAEs now work on ViTs with little tuning, and Prisma offers ready-made SAE/transcoder tooling for ViT-B-scale models.

### Cited Findings
- [canonical] Elhage et al., "Toy Models of Superposition" (Transformer Circuits Thread, 2022). With sparse features, networks represent more features than dimensions by placing them in non-orthogonal, interfering directions. Phase changes depend on sparsity and importance, and geometric structures (antipodal pairs, polytopes) appear. — [transformer-circuits.pub](https://transformer-circuits.pub/2022/toy_model/index.html)
- [canonical] Scherlis, Sachan, Jermyn, Benton & Shlegeris, "Polysemanticity and Capacity in Neural Networks" (arXiv:2210.01892, 2022). Defines per-feature "capacity" (fraction of an embedding dimension a feature uses). Optimal allocation gives important features full capacity, packs less important ones polysemantically, and drops the rest. — [arXiv](https://arxiv.org/abs/2210.01892)
- [verified] Joseph et al., "Prisma: An Open Source Toolkit for Mechanistic Interpretability in Vision and Video" (CVPR 2025 MIV Workshop, oral + tutorial; arXiv:2504.19475) **[recent]**. Unified access to 75+ vision/video transformers, SAE/transcoder/crosscoder training, and 80+ pretrained SAE weights. Reports that vision SAEs can show **substantially lower sparsity than language SAEs**, and that SAE reconstructions sometimes **decrease** model loss. MIT license. One independent review (Pith) calls the empirical claims under-supported. — [arXiv](https://arxiv.org/abs/2504.19475); [GitHub](https://github.com/Prisma-Multimodal/ViT-Prisma)
- [verified] Fry, "Towards Multimodal Interpretability: Learning Sparse Interpretable Features in Vision Transformers" (LessWrong, ~Apr 2024). SAEs on the CLIP ViT give interpretable features (e.g., a tennis feature) with **little hyperparameter tuning**. Notes feature visualization is too hyperparameter-dependent and costly to scale as auto-interp. — [LessWrong](https://www.lesswrong.com/posts/bCtbuWraqYTDtuARg/towards-multimodal-interpretability-learning-sparse-2)
- [verified] Lim, Choi, Choo & Schneider, "Sparse Autoencoders Reveal Selective Remapping of Visual Concepts During Adaptation" (ICLR 2025; arXiv:2412.05276). PatchSAE on the CLIP ViT gives concepts (shape, color, object semantics) with **patch-wise spatial attributions**. After prompt-based adaptation, concept activations barely change, and gains mostly come from **remapping existing concepts to classes**. The exception is large-shift data (EuroSAT), where some concepts are suppressed or newly introduced. — [arXiv](https://arxiv.org/abs/2412.05276); [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/3d5b603d631d595f56bc36b373458b27-Abstract-Conference.html)
- [verified] Gong & Zhang, "Self-Attention Limits Working Memory Capacity of Transformer-Based Models" (arXiv:2409.10715, Sep/Nov 2024). Small decoder-only transformers trained on N-back learn to attend N positions back. **Total attention-score entropy rises with N**, and the authors propose attention dispersion as the source of the capacity limit. — [arXiv](https://arxiv.org/abs/2409.10715)

### Inferences
- PatchSAE / Prisma-style SAEs on memory patch tokens can test whether "hue" and "location" features are monosemantic at set size 1 and become entangled or polysemantic at set size 8. Lim et al. suggest that fine-tuning on VWM may mostly **remap** pretrained color/position features rather than create new ones, which is worth checking by comparing SAE features before and after fine-tuning (cf. Prakash et al. CMAP).
- Gong & Zhang give a cheap, directly applicable metric: **entropy of the probe-query → memory-token cross-attention as a function of set size**. Rising entropy, or mass leaking onto distractor squares, is a candidate mechanistic correlate of precision loss and swaps.
- Superposition gives a quantitative prediction to test. Measure the effective dimensionality (participation ratio) of the hue subspace in the query/readout pathway vs set size. If capacity is bottlenecked there, per-item decodability should fall as roughly (dims / items).

### Gaps
- No paper found applying SAEs specifically to multi-item color memory or to set-size manipulations.
- Prisma's list of supported pretrained SAEs for supervised ViT-B/16 (vs CLIP) was not checked. Training a custom SAE will probably be needed on the fine-tuned weights anyway.

---

## 3. ViT internals relevant to where item information lives

### Takeaway
ViTs reuse low-information background patches as high-norm "register" tokens holding global information. That is critical here, because the memory image is mostly blank background, so item information may migrate off the square patches. Per-head and per-layer decomposition (Gandelsman), class-embedding logit lens (Vilas), and attention-based attribution (rollout, Chefer) are the main ViT-specific readout tools, each with known caveats.

### Cited Findings
- [verified] Darcet, Oquab, Mairal & Bojanowski, "Vision Transformers Need Registers" (ICLR 2024; arXiv:2309.16588). Supervised and self-supervised ViTs produce **high-norm artifact tokens (norm > 150, ~2.37% of tokens) in low-information background patches**, repurposed to aggregate global information while discarding spatial information. Linear probes on these outlier tokens classify the image better than probes on normal patches. Adding learnable register tokens removes the artifacts and smooths attention maps. — [arXiv](https://arxiv.org/abs/2309.16588); [OpenReview](https://openreview.net/forum?id=2dnO3LLiJ1)
- [verified] Jiang & Dravid, "Vision Transformers Don't Need Trained Registers" (NeurIPS 2025; arXiv:2506.08010) **[recent]**. A sparse set of **"register neurons"** creates the outlier activations. Shifting them into an extra untrained token at test time matches trained registers without retraining. — [arXiv](https://arxiv.org/abs/2506.08010)
- [verified] Gandelsman, Efros & Steinhardt, "Interpreting CLIP's Image Representation via Text-Based Decomposition" (ICLR 2024 oral; arXiv:2310.05916). The output decomposes into a sum over patches × layers × heads. TextSpan labels each head's output directions, and some heads specialize in properties such as **location**, shape, or color. Patch-level decomposition shows emergent spatial localization. Mean-ablating all but the last 4 attention layers barely changes accuracy, so late attention layers build the representation. Follow-up: "Decomposing and Interpreting Image Representations via Text in ViTs Beyond CLIP" (NeurIPS 2024). — [arXiv](https://arxiv.org/abs/2310.05916); [GitHub](https://github.com/yossigandelsman/clip_text_span); [NeurIPS 2024 follow-up](https://proceedings.neurips.cc/paper_files/paper/2024/file/93e45db754dd0f82339763055c6cda56-Paper-Conference.pdf)
- [verified] Vilas, Schaumlöffel & Roig, "Analyzing Vision Transformers for Image Classification in Class Embedding Space" (NeurIPS 2023; arXiv:2310.18969). A **logit lens for ViTs**: project intermediate image tokens (and per-head/MLP outputs) through the class embedding matrix, with no training. Self-attention layers raise correct-class similarity of image tokens above chance, and per-block, per-head heatmaps are produced. The authors argue linear probes may find features the model does not actually use for the task. Code uses timm. — [arXiv](https://arxiv.org/abs/2310.18969); [GitHub](https://github.com/martinagvilas/vit-cls_emb)
- [canonical] Raghu, Unterthiner, Kornblith, Zhang & Dosovitskiy, "Do Vision Transformers See Like Convolutional Neural Networks?" (NeurIPS 2021; arXiv:2108.08810). CKA shows ViTs have more uniform representations across layers. Some lower-layer heads attend globally, and **ViTs preserve spatial (positional) information strongly through to late layers**, more so with CLS-free/GAP training choices. — [arXiv](https://arxiv.org/abs/2108.08810)
- [canonical] Abnar & Zuidema, "Quantifying Attention Flow in Transformers" (ACL 2020; arXiv:2005.00928). Attention rollout and attention flow propagate attention through layers, adding identity for residuals. — [arXiv](https://arxiv.org/abs/2005.00928)
- [canonical] Chefer, Gur & Wolf, "Transformer Interpretability Beyond Attention Visualization" (CVPR 2021; arXiv:2012.09838). LRP-based relevance combined with attention gradients. Shows rollout is class-agnostic and can mislead. — [arXiv](https://arxiv.org/abs/2012.09838)

### Inferences
- **Highest-priority check:** the memory image is mostly uniform background, which is exactly the setting where Darcet et al. find high-norm global tokens. Before probing "square patches," log token norms per layer. If item colors are aggregated into background/register tokens, per-item information may sit there in a spatially scrambled code, and swaps may originate there. Jiang & Dravid's register-neuron trick allows a cheap causal test: shift or ablate the artifacts and measure the change in swap rate.
- Gandelsman-style decomposition carries over directly to the task's readout. The [sin, cos] output is linear in the final residual / cross-attention output, so it can be split exactly into per-head, per-layer, and per-memory-patch contributions. Then ask which memory patches (target vs distractor squares vs background) contribute to the predicted hue on swap trials.
- A Vilas-style lens is easy here: project each layer's memory tokens through the trained hue readout (or a fixed readout) to get a "hue lens" per layer. Tuned-lens-style per-layer affine translators are the more robust version (see Section 4).
- Attention maps (rollout) from the probe query to memory tokens are useful descriptively, but Chefer et al. and the patching literature warn that attention weight is not causal contribution. Pair attention maps with value-weighted attention norms and patching.

### Gaps
- Did not find work on how ViT positional embeddings encode 2D location in a decodable format after fine-tuning (e.g., grid vs place-cell-like codes). This needs direct probing in the project.

---

## 4. Causal methods, probing best practice, and circular decoding

### Takeaway
Use probes (with controls) to find candidate locations of item-color and location information, then confirm with interchange interventions/activation patching. DAS or other learned-subspace interventions can find the binding subspace directly, but they are prone to "interpretability illusions," so validate on held-out interventions. For hue, decode with circular targets (sin/cos or von Mises) and analyze errors with mixture models (target/swap/guess).

### Cited Findings
- [verified] Heimersheim & Nanda, "How to use and interpret activation patching" (arXiv:2404.15255, Apr 2024). Use minimal clean/corrupted pairs differing in one fact. Separate exploratory sweeps (layer × position × component) from confirmatory circuit tests. Pitfalls: backup components that mask importance, negative components that make circuits look complete, and metric choice (logit diff vs KL vs probability). — [arXiv](https://arxiv.org/abs/2404.15255)
- [canonical] Meng, Bau, Andonian & Belinkov, "Locating and Editing Factual Associations in GPT" (ROME; NeurIPS 2022; arXiv:2202.05262). **Causal tracing**: corrupt the inputs with noise, then restore single hidden states to find where the critical information is mediated. — [arXiv](https://arxiv.org/abs/2202.05262)
- [canonical] Geiger, Lu, Icard & Potts, "Causal Abstractions of Neural Networks" (NeurIPS 2021; arXiv:2106.02997). **Interchange interventions** test whether a high-level causal model (e.g., "location → which item is retrieved → hue") is implemented by specific representations. — [arXiv](https://arxiv.org/abs/2106.02997)
- [canonical] Geiger, Wu, Potts, Icard & Goodman, "Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations" (DAS; CLeaR 2024; arXiv:2303.02536). Learns a rotation, so interchange interventions act on a **distributed subspace** rather than individual neurons. — [arXiv](https://arxiv.org/abs/2303.02536)
- [canonical] Makelov, Lange & Nanda, "Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching" (ICLR 2024; arXiv:2311.17030). Subspace patching can change behavior by activating dormant parallel pathways, so a found subspace may not be the one the model uses. — [arXiv](https://arxiv.org/abs/2311.17030)
- [canonical] Belinkov, "Probing Classifiers: Promises, Shortcomings, and Advances" (Computational Linguistics 48(1), 2022). Probe accuracy shows decodability, not use. Recommends controls, simple probes, and causal follow-ups. — [ACL Anthology](https://aclanthology.org/2022.cl-1.7/)
- [canonical] Hewitt & Liang, "Designing and Interpreting Probes with Control Tasks" (EMNLP 2019; arXiv:1909.03368). Report **selectivity** = task accuracy minus control-task accuracy (random but consistent labels), and prefer low-capacity probes. — [arXiv](https://arxiv.org/abs/1909.03368)
- [canonical] Belrose et al., "Eliciting Latent Predictions from Transformers with the Tuned Lens" (arXiv:2303.08112, 2023). Per-layer learned affine translators into the final readout space are more faithful than the raw logit lens. — [arXiv](https://arxiv.org/abs/2303.08112)
- [canonical] Kriegeskorte, Mur & Bandettini, "Representational Similarity Analysis" (Front. Syst. Neurosci. 2008). Compare RDMs across layers, models, or brain data. Useful for comparing hue-geometry (circular RDM) at each set size. — [Frontiers](https://www.frontiersin.org/articles/10.3389/neuro.06.004.2008/full)
- [verified] Methodological warning from human VWM work: swap rates depend on the measurement model ("There is no theory-free measure of 'swaps'"), and swap prevalence can be larger than previously thought. Precision declines continuously with set size, with no sharp cutoff. — [PMC: no theory-free swaps](https://pmc.ncbi.nlm.nih.gov/articles/PMC10270377/); [Sci Rep 2016](https://www.nature.com/articles/srep19203); [Cog Psych 2022: swaps explained by cue-feature variability](https://www.sciencedirect.com/science/article/pii/S0010028522000305); [Bayesian non-parametric swap model, 2025](https://arxiv.org/html/2505.01178)

### Inferences
Recommended pipeline for the VWM ViT:
1. **Probing by layer × token type** (square patches, background, high-norm tokens, CLS, probe query). Targets: hue of item k (circular ridge on sin/cos, scored by mean absolute angular error), item location, and (item, hue) conjunctions. Add Hewitt–Liang control tasks (shuffled hue labels consistent per item) and report selectivity. Plot decodability vs set size.
2. **Interchange interventions for binding:** pairs of memory displays that differ only in (a) the hue at the cued location, (b) the hue at a distractor location, or (c) the locations of two items swapped. Patch per layer/position/head and measure the shift in predicted hue toward the source. If patching only the location component makes the model report the distractor's hue, that is direct evidence of a misbinding mechanism.
3. **DAS** to learn the "location/binding ID" subspace in memory tokens or the query. Validate against the Makelov illusion with held-out displays and by checking that the subspace is active on clean runs.
4. **Swap analysis:** fit target/swap/guess mixture models (Bays-style) to model errors by set size. Correlate trial-level swap probability with cross-attention mass on distractors and with cue–distractor distance.
5. **Causal tracing (ROME-style)** with noise on the probe image to find where the cue location gets resolved into a query.

### Gaps
- Did not find vision-specific guidance on circular-variable probes beyond standard sin/cos regression. Von Mises regression is a standard alternative but has no citation from this session.
- Not verified: the published version and venue of the DAS paper (CLeaR 2024 per prior knowledge).

---

## 5. Information bottleneck, rate-distortion, and slot attention as explicit capacity

### Takeaway
IB and rate-distortion frame capacity as a limit on bits through a bottleneck. In this architecture the natural bottleneck is the single probe-derived query and its cross-attention readout (one weighted average over memory tokens). Slot attention is the explicit-capacity contrast: K slots competing for inputs.

### Cited Findings
- [canonical] Tishby & Zaslavsky, "Deep Learning and the Information Bottleneck Principle" (ITW 2015; arXiv:1503.02406). — [arXiv](https://arxiv.org/abs/1503.02406)
- [canonical] Alemi, Fischer, Dillon & Murphy, "Deep Variational Information Bottleneck" (ICLR 2017; arXiv:1612.00410). A variational bound gives a trainable β that trades I(Z;X) against I(Z;Y). — [arXiv](https://arxiv.org/abs/1612.00410)
- [canonical] Locatello et al., "Object-Centric Learning with Slot Attention" (NeurIPS 2020; arXiv:2006.15055). A fixed number of slots compete through softmax over slots, an explicit capacity and binding mechanism. — [arXiv](https://arxiv.org/abs/2006.15055)
- [verified] Gong & Zhang: attention entropy rises with memory load, a soft-attention version of a bottleneck. — [arXiv](https://arxiv.org/abs/2409.10715)

### Inferences
- Single-query cross-attention returns a convex combination of memory values. If the attention is not sharp, distractor values get averaged in, which is a mechanistic **misbinding/averaging** account that predicts errors biased toward distractor hues (swaps), not just noise. Test by regressing predicted hue on a vector sum of attention-weighted item hues.
- A VIB layer or a cap on query dimension could be ablated to show how bottleneck width changes the capacity curve. Treat this as a manipulation experiment, not a citation-backed claim.

### Gaps
- No source found linking rate-distortion VWM theories (e.g., Sims-style) to transformer internals. A separate cognitive-science researcher may cover that.

---

## 6. Tooling and practical tips for small-scale ViT interpretability

### Takeaway
For a custom ViT-B/16 with a bespoke cross-attention head, the most practical stack is timm/PyTorch forward hooks or nnsight for patching, Prisma (HookedViT) if the backbone can be loaded into it, and custom SAEs trained on cached activations.

### Cited Findings
- [verified] ViT-Prisma: HookedViT-style API (TransformerLens-like hooks for ViTs), SAE/transcoder/crosscoder training, and 80+ pretrained SAEs, MIT license. — [GitHub](https://github.com/Prisma-Multimodal/ViT-Prisma); [arXiv](https://arxiv.org/abs/2504.19475)
- [canonical] TransformerLens (Nanda et al.): HookPoints on every activation, run_with_cache, and activation-patching utilities, built mainly for language models. — [GitHub](https://github.com/TransformerLensOrg/TransformerLens)
- [canonical] nnsight (Fiotto-Kaufman et al., "NNsight and NDIF," arXiv:2407.14561; ICLR 2025). Wraps any PyTorch module and supports interventions inside a tracing context. Good for custom architectures like the cross-attention readout. — [arXiv](https://arxiv.org/abs/2407.14561); [GitHub](https://github.com/ndif-team/nnsight)
- [verified] Vilas et al.'s code uses timm models, a concrete example of timm-based ViT lens analysis. — [GitHub](https://github.com/martinagvilas/vit-cls_emb)
- [verified] The Li et al. object-binding repo provides IsSameObject probes and a layer-wise viewer to adapt. — [GitHub](https://github.com/liyihao0302/vit-object-binding)
- [verified] Saravanan et al. and Gur-Arieh et al. both release code for binding interventions (mixing-mechs repo). — [GitHub mixing-mechs](https://github.com/yoavgur/mixing-mechs)

### Inferences
- Practical tips for this project:
  - Cache activations once (memory tokens per layer, 197×768 for ViT-B/16, plus probe query and cross-attention weights) across set sizes 1–8 with balanced hues and locations. Every probe, SAE, and RSA analysis then reads from the cache.
  - Use **synthetic minimal pairs**, which the generator makes trivial (change one square's hue, swap two squares' positions, move the cue). This is the main advantage over natural-image interpretability.
  - Log token norms to detect register-like tokens (Darcet). Consider fine-tuning a registers variant as a control.
  - Score patching effects in **angular units** (shift of predicted hue toward source hue, normalized by the source–target hue difference) rather than raw MSE.
  - Use hook-based patching on the timm model directly if Prisma cannot load the fine-tuned custom head. nnsight handles arbitrary modules.

### Gaps
- Did not verify whether Prisma's HookedViT can load an arbitrary timm ViT-B/16 checkpoint with a custom cross-attention head. Likely needs a custom wrapper.
