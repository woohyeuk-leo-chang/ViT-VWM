# Human Visual Working Memory (VWM): Benchmark Phenomena, Competing Theories, and Continuous-Report Signatures

Scope note: These notes are for evaluating a ViT-B/16 trained on a 1–8 item, spatially cued, CIELAB-wheel continuous-report task and analyzed with mixture models. Facts I verified directly from sources this session are cited plainly. Where a specific number comes from my prior knowledge of a canonical paper and I could not re-read the full text (several PDFs could not be fetched or parsed), it is marked **[unverified number – check paper]**. Treat those as approximate.

---

## 1. Slot models (Luck & Vogel 1997; Zhang & Luck 2008): what signatures?

### Takeaway
Slot models hold that VWM stores a small, fixed number (K ≈ 3–4) of items at fixed resolution. In continuous report this predicts (a) a uniform "guess" proportion that rises steeply once set size N > K (Pm ≈ min(1, K/N)), and (b) a von Mises SD that rises from N=1 to N≈K (in "slots + averaging") and then **plateaus** for N > K.

### Cited Findings
- **Luck, S. J., & Vogel, E. K. (1997). The capacity of visual working memory for features and conjunctions. *Nature*, 390, 279–281. doi:10.1038/36846.** Change-detection paradigm. Capacity is about 4 items, and objects with conjoined features are stored about as well as single features (an "integrated objects" claim that later work partly challenged). Basis for the item-limit view and for Cowan's K. — [Nature](https://www.nature.com/articles/36846) **[details from prior knowledge; not re-fetched]**
- **Cowan, N. (2001). The magical number 4 in short-term memory. *Behavioral and Brain Sciences*, 24, 87–185. doi:10.1017/S0140525X01003922.** Canonical source for K ≈ 4 (range 3–5). — [BBS](https://doi.org/10.1017/S0140525X01003922) **[prior knowledge]**
- **Zhang, W., & Luck, S. J. (2008). Discrete fixed-resolution representations in visual working memory. *Nature*, 453, 233–235. doi:10.1038/nature06860.** Introduced the two-component mixture (von Mises around the target plus uniform guessing) to give independent measures of capacity and resolution. They concluded that observers keep sharp representations of only a few items and leave the rest unrepresented. "Short-term information storage does not discard quality in favour of quantity." — [Nature](https://www.nature.com/articles/nature06860); [PubMed](https://pubmed.ncbi.nlm.nih.gov/18385672/)
  - Design (for matching the ViT task): colored squares; 100 ms sample, 900 ms delay; 180 colors on a CIELAB circle (L* = 70, center a* = 20, b* = 38, radius 60); set sizes 1, 2, 3, 6; the probe location is cued. **[unverified numbers – check paper; the PDF could not be parsed this session]**
  - Signature: Pm (probability in memory) ≈ 1 at N ≤ 2–3 and falls at N = 6, giving K ≈ 3. The SD rises from N = 1 to N = 3 and then stays essentially constant from N = 3 to N = 6 (roughly 20–25°). The "slots + averaging" variant explains the lower SD at N = 1–2: when N < K, multiple slots go to the same item and are averaged, so SD ∝ 1/√(slots per item). Precue manipulations also fit slots + averaging. **[numbers unverified – check paper]**
- **Zhang, W., & Luck, S. J. (2009). Sudden death and gradual decay in visual working memory. *Psychological Science*, 20(4), 423–428. doi:10.1111/j.1467-9280.2009.02322.x.** Delays of roughly 1, 4 and 10 s. Longer delays mainly lowered Pm ("sudden death" of items) while SD stayed largely stable. — **[prior knowledge; not re-fetched]**
- **Adam, K. C. S., Vogel, E. K., & Awh, E. (2017). Clear evidence for item limits in visual working memory. *Cognitive Psychology*, 97, 79–97. doi:10.1016/j.cogpsych.2017.07.001.** In whole report (all items reported in order), the last responses at set size 6 were indistinguishable from uniform guessing. Cited as the strongest modern evidence for item limits. — **[prior knowledge; not re-fetched]**
- **Pratte, M. S., Park, Y. E., Rademaker, R. L., & Tong, F. (2017). Accounting for stimulus-specific variation in precision reveals a discrete capacity limit in visual working memory. *JEP: Human Perception & Performance*, 43(1), 6–17. doi:10.1037/xhp0000302.** Precision varies across stimulus values (it is inhomogeneous around the wheel). Once that variation is modeled, the apparent "variable precision" shrinks and a discrete limit is supported. — **[prior knowledge; not re-fetched]**

### Inferences
- Slot signatures to test in the ViT: (1) a kink in Pm(N) near some K, with Pm ≈ K/N beyond it; (2) SD(N) flat for N > K; (3) a genuinely uniform component that survives better models (e.g., VP or TCC), i.e., a heavy flat tail that a von Mises plus a long-tailed component cannot absorb.
- Because Zhang & Luck used CIELAB colors (L* = 70, radius 60) at set sizes 1–6, the ViT task can match their stimulus space almost exactly. The nonuniformity of perceived color around that circle (Pratte et al. 2017; Bae et al. 2015, below) must be controlled for.

### Gaps
- Exact Pm/SD values per set size in Zhang & Luck 2008 could not be re-verified (PDF and PMC were blocked). Check Fig. 2 of the paper before quoting numbers.

---

## 2. Resource models (Wilken & Ma 2004; Bays & Husain 2008; van den Berg et al. 2012; Ma, Husain & Bays 2014): signatures?

### Takeaway
Resource models treat memory as a continuous quantity spread over all items, with no hard item limit. Precision (1/variance, or Fisher information) falls smoothly, roughly as a **power law of set size**, with no plateau. The variable-precision (VP) version adds trial-to-trial and item-to-item variability in precision. This yields heavy-tailed error distributions that **mimic a guess component** without true guessing. Population-coding versions with divisive normalization (Bays 2014) produce the same shapes from first principles.

### Cited Findings
- **Wilken, P., & Ma, W. J. (2004). A detection theory account of change detection. *Journal of Vision*, 4(12), 11. doi:10.1167/4.12.11.** An SDT model in which per-item noise rises with set size explains change detection and continuous report (color, orientation, spatial frequency) without a fixed capacity. One of the first uses of the color-wheel delayed-estimation task. — [JoV](https://doi.org/10.1167/4.12.11) **[prior knowledge]**
- **Bays, P. M., & Husain, M. (2008). Dynamic shifts of limited working memory resources in human vision. *Science*, 321, 851–854. doi:10.1126/science.1158023.** Precision falls continuously with the number of items, even from 1 to 2 (fit by a power law), with no discontinuity at 3–4. Saccade targets and attended items receive more resource. — [Science](https://doi.org/10.1126/science.1158023) **[details from prior knowledge]**
- **Bays, Catalao & Husain (2009)** (see §3): the SD of the target-centered component "increased monotonically" from 1 to 6 items. Much of the apparent guessing in Zhang & Luck is reattributed to **non-target (swap) responses**. Conclusion: "a common resource distributed dynamically across the visual scene, with no need to invoke an upper limit on the number of objects represented." — [PubMed](https://pubmed.ncbi.nlm.nih.gov/19810788/); [PDF](https://www.paulbays.com/pdf/BayCatHus09.pdf)
- **van den Berg, R., Shin, H., Chou, W.-C., George, R., & Ma, W. J. (2012). Variability in encoding precision accounts for visual short-term memory limitations. *PNAS*, 109(22), 8780–8785. doi:10.1073/pnas.1117465109.**
  - Resource is "not only continuous but also variable across items and trials." Precision J is drawn from a gamma distribution whose mean falls as a power law, J̄(N) = J̄₁·N^(−α).
  - The model "accounts for all aspects of the data, including apparent guessing" in two paradigms (delayed estimation and change localization) and two features (color, orientation), and it beat slot models in formal comparison.
  - Proposed neural correlate: variability in population gain ("doubly stochastic" representation).
  - Code: github.com/WeiJiMaLab/variability_encoding_precision.
  — [PNAS](http://www.pnas.org/content/109/22/8780.abstract); [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3365149/); [GitHub](https://github.com/WeiJiMaLab/variability_encoding_precision)
- **Fougnie, D., Suchow, J. W., & Alvarez, G. A. (2012). Variability in the quality of visual working memory. *Nature Communications*, 3, 1229. doi:10.1038/ncomms2237.** Independent evidence for trial-to-trial variability in precision. — [Nat Commun](https://www.nature.com/articles/ncomms2237)
- **van den Berg, R., Awh, E., & Ma, W. J. (2014). Factorial comparison of working memory models. *Psychological Review*, 121(1), 124–149. doi:10.1037/a0035234.** A factorial model space crossing number of items stored (all / fixed K / variable K), precision type (fixed / variable) and swaps (yes / no), fit to 10 datasets. Best models had **variable precision plus non-target (swap) responses**. Whether a K limit was included mattered less, so the slot-vs-resource dichotomy was partly dissolved. — **[prior knowledge; not re-fetched]**
- **Ma, W. J., Husain, M., & Bays, P. M. (2014). Changing concepts of working memory. *Nature Neuroscience*, 17(3), 347–356. doi:10.1038/nn.3655.** Review that frames the shift from quantity to quality: "the quality rather than the quantity of working memory representations determines performance." It covers change detection, delayed estimation and mixture models, and notes that resource can be allocated to features or locations rather than to items. — [Nature Neuroscience](https://www.nature.com/articles/nn.3655); [PDF](https://www.paulbays.com/pdf/MaHusBay14.pdf); [PubMed](https://pubmed.ncbi.nlm.nih.gov/24569831/)
- **Bays, P. M. (2014). Noise in neural populations accounts for errors in working memory. *Journal of Neuroscience*, 34(10), 3632–3645. doi:10.1523/JNEUROSCI.3204-13.2014.**
  - Model: Poisson population coding of tuned neurons, with **divisive normalization** holding the summed activity across all memoranda constant, so per-item gain ∝ 1/N.
  - Non-normal, heavy-tailed errors (previously attributed to guessing or variable precision) arise naturally from decoding a noisy population, with positive kurtosis at intermediate gain.
  - Variance grows with load as a power law with an exponent of **1.36**, which is significantly different from 1. At high gain the exponent approaches 1.
  — [J Neurosci](https://www.jneurosci.org/content/34/10/3632); [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3942580/)
- **Bays, P. M. (2015). Spikes not slots: noise in neural populations limits working memory. *Trends in Cognitive Sciences*, 19(8), 431–438. doi:10.1016/j.tics.2015.06.004.** Opinion piece extending the population-coding account. — [TICS](https://www.sciencedirect.com/science/article/abs/pii/S1364661315001412); [PDF](https://www.bayslab.com/pdf/Bay15.pdf)
- **van den Berg, R., & Ma, W. J. (2018). A resource-rational theory of set size effects in human visual working memory. *eLife*, 7, e34963. doi:10.7554/eLife.34963.** Set-size effects follow from rationally trading neural cost against performance, so the total "resource" need not be fixed. — [eLife](https://elifesciences.org/articles/34963)
- **Sims, C. R., Jacobs, R. A., & Knill, D. C. (2012). An ideal observer analysis of visual working memory. *Psychological Review*, 119(4), 807–830. doi:10.1037/a0029856.** Rate-distortion (information-theoretic) capacity account. Errors should be optimally shaped for a channel of fixed capacity. — **[prior knowledge]**

### Inferences
- Resource signatures to test in the ViT:
  - SD (or 1/J) increases monotonically from N = 1 through N = 8, including from 1 to 2, with a log–log slope (variance exponent) around 1–1.4 and no plateau.
  - The fitted "guess rate" rises with N but should largely disappear under a VP or population-coding fit.
  - Error kurtosis rises with N.
- A ViT's attention softmax is a natural normalization mechanism: total attention mass from the cue/readout token is shared across item patches. A Bays-2014-style 1/N per-item gain is therefore a mechanistic hypothesis that can be probed directly in an interpretability analysis (e.g., attention mass to target patches vs N).

### Gaps
- I did not verify the exact power-law exponents for color in Bays & Husain 2008 or van den Berg 2012 (typical fitted α for mean precision is about 1, but check the papers).

---

## 3. Swap / binding errors (Bays, Catalao & Husain 2009) and misbinding

### Takeaway
A substantial share of large errors in cued continuous report are reports of a **non-target item's color** (swaps). These come from imprecise memory for the cue feature (location) and from feature-binding failures. The three-component model (target von Mises + non-target von Mises + uniform) is now the standard analysis. Swap rates rise with set size and with spatial proximity of items to the target.

### Cited Findings
- **Bays, P. M., Catalao, R. F. G., & Husain, M. (2009). The precision of visual working memory is set by allocation of a shared resource. *Journal of Vision*, 9(10):7, 1–11. doi:10.1167/9.10.7.**
  - Model: a von Mises centered on the target, von Mises distributions of the **same width** centered on each non-target color (capturing errors in memory for location, i.e., misbinding), and a uniform component.
  - Performance "also depends on memory for location." Once errors in both color and location memory are counted, the resource model explains the data.
  — [JoV](https://jov.arvojournals.org/article.aspx?articleid=2122354); [PubMed](https://pubmed.ncbi.nlm.nih.gov/19810788/); [PDF](https://www.paulbays.com/pdf/BayCatHus09.pdf)
- **Bays, P. M. (2016). Evaluating and excluding swap errors in analogue tests of working memory. *Scientific Reports*, 6, 19203. doi:10.1038/srep19203.** Nonparametric method for estimating swap frequency without assuming a von Mises shape. Recommended when error shapes are non-standard, which is likely for a network. — **[prior knowledge]**
- **Schneegans, S., & Bays, P. M. (2017). Neural architecture for feature binding in visual working memory. *Journal of Neuroscience*, 37(14), 3913–3925. doi:10.1523/JNEUROSCI.3493-16.2017.** Population model with conjunctive (feature × location) neurons. Swap probability depends on how close the cue feature (location) of non-targets is to the target's. — **[prior knowledge]**
- **Oberauer & Lin (2017)** (see §4): swaps fall out of cue-based retrieval with context (location) similarity. — [PubMed](https://pubmed.ncbi.nlm.nih.gov/27869455/)
- **van den Berg, Awh & Ma (2014)**: adding non-target responses improved fits for nearly all model families. — **[prior knowledge]**

### Inferences
- For the ViT: fit both the 2-component (Zhang & Luck) and 3-component (Bays 2009) models. Also regress the swap probability for each non-target on its spatial distance to the cued location. A human-like model should show more swaps for near neighbors and at larger N.
- In a ViT, swaps would most plausibly arise from positional-embedding confusions (the location cue retrieving a neighboring patch). This is directly testable with attention maps.
- With 8 items on a 360° wheel, non-target colors cover the wheel densely. Swap and uniform components then become hard to tell apart, so report parameter recovery or identifiability checks.

### Gaps
- No exact human swap-rate numbers verified here. Bays et al. 2009 Fig. 3 reports non-target proportions rising with N (on the order of 10–30% at N = 6 is commonly cited) **[unverified]**.

---

## 4. Interference models (Oberauer & Lin 2017) and the benchmark set (Oberauer et al. 2018)

### Takeaway
The interference model places the capacity limit in **competition among simultaneously held item–context bindings at retrieval**, not in slots or a resource. A focus of attention holds one item in a high-precision, protected state, which explains the large drop in precision from N = 1 to N = 2. The 2018 benchmarks paper is the consensus checklist that any model should be scored against.

### Cited Findings
- **Oberauer, K., & Lin, H.-Y. (2017). An interference model of visual working memory. *Psychological Review*, 124(1), 21–59. doi:10.1037/rev0000044.**
  - Retrieval probability depends on the relative activation of candidates at recall. Activation has three sources: cue-based retrieval through context (location) bindings, context-independent memory for content, and noise. One item can sit in the focus of attention with higher precision and partial protection from interference.
  - The model was fit to four continuous-reproduction experiments (colors and orientations) and compared favorably with Slot-Averaging and Variable-Precision models.
  — [PubMed](https://pubmed.ncbi.nlm.nih.gov/27869455/); [APA](https://psycnet.apa.org/record/2016-56587-001); [Semantic Scholar](https://www.semanticscholar.org/paper/An-Interference-Model-of-Visual-Working-Memory-Oberauer-Lin/f2c5cd8976476b9d095b4ce8f1b0197d2c684be6)
- **Lin, H.-Y., & Oberauer, K. (2022). An interference model for visual working memory: Applications to the change detection task. *Cognitive Psychology*, 133, 101463. doi:10.1016/j.cogpsych.2022.101463.** Extension to single-probe change detection, with evidence of the predicted interference from non-targets. "Capacity limit emerges from interference between representations." — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0010028522000019); [PDF](https://www.psychologie.uzh.ch/dam/jcr:376519a6-5441-479e-874e-e568f03335ac/Lin_Oberauer_CogPsych_2022.pdf)
- A 2024 PsycNet record titled "An interference model for visual and verbal working memory" also exists. I could not retrieve its abstract or details. — [APA](https://psycnet.apa.org/record/2024-27570-001)
- **Oberauer, K., Lewandowsky, S., Awh, E., Brown, G. D. A., Conway, A., Cowan, N., Donkin, C., Farrell, S., Hitch, G. J., Hurlstone, M. J., Ma, W. J., Morey, C. C., Nee, D. E., Schweppe, J., Vergauwe, E., & Ward, G. (2018). Benchmarks for models of short-term and working memory. *Psychological Bulletin*, 144(9), 885–958. doi:10.1037/bul0000153.**
  - Benchmarks were chosen by consensus across theoretical camps, supported by an expert survey, and rated by priority.
  - Reference datasets are on OSF (osf.io/g49c6).
  - Commentaries: Logie (doi:10.1037/bul0000162), Vandierendonck (doi:10.1037/bul0000159). Authors' reply: doi:10.1037/bul0000165.
  — [PubMed](https://pubmed.ncbi.nlm.nih.gov/30148379/); [OSF](https://osf.io/g49c6/); [PDF](https://www.cns.nyu.edu/malab/static/files/publications/2018%20Oberauer%20et%20al..pdf)
  - Visual continuous-report benchmarks relevant here, as I recall them from the paper **[benchmark list from prior knowledge – check the paper's visual-WM section]**:
    - the set-size effect on precision (continuous, including 1→2);
    - the error distribution's shape (heavy tails or "guessing");
    - variability of precision;
    - swap / non-target errors that increase with proximity;
    - the advantage of a retro-cue;
    - the effect of delay or time (small, often modest);
    - feature- vs object-based storage effects;
    - categorical bias in color memory.

### Inferences
- Interference signatures to test in the ViT:
  - an unusually large precision drop from N = 1 to N = 2 (focus-of-attention effect), followed by gentler decline;
  - errors biased toward non-target values (not just exact swaps), i.e., attraction or repulsion of the target report toward similar non-target colors;
  - swap probability that scales with similarity of context (spatial) codes.
- Using the Oberauer et al. 2018 benchmark list as an explicit scorecard (which benchmarks the ViT reproduces or fails) would be a defensible, theory-neutral evaluation framework.

### Gaps
- I did not extract the full benchmark table (priority ratings A/B/C) from the 2018 paper.

---

## 5. Target Confusability Competition (TCC; Schurgin, Wixted & Brady 2020) and implications for analyzing model errors

### Takeaway
TCC argues that the heavy-tailed shape of continuous-report errors (taken as "guessing" or "variable precision") comes from the **nonlinear, roughly exponential psychological similarity function** over the color wheel. With measured similarity, a one-parameter SDT model (d′) fits errors across set sizes, delays, and tasks. Implication for the ViT: fit mixture models but also TCC. A network's error distribution may only look "slot-like" because of stimulus similarity structure, so the similarity function should be measured, ideally from the network's own representations.

### Cited Findings
- **Schurgin, M. W., Wixted, J. T., & Brady, T. F. (2020). Psychophysical scaling reveals a unified theory of visual memory strength. *Nature Human Behaviour*, 4, 1156–1172. doi:10.1038/s41562-020-00938-0.** A publisher correction (doi:10.1038/s41562-020-00993-7) concerns data points in one figure.
  - "Neither memory nor perception are appropriately scaled in stimulus space." Similarity was measured with triad tasks, Likert ratings and maximum-likelihood difference scaling, and it falls off nonlinearly (exponential-like) with distance on the wheel.
  - Model: memory is a population of familiarity signals, one per candidate color, each carrying memory strength scaled by its similarity to the target, plus independent noise. The response is the maximum-familiarity color. A single d′ parameter fits the data.
  - Effects "once taken as evidence for a fixed capacity of about three or four items" can be explained by this unitary signal-detection framework. The same similarity function unifies working memory, long-term memory, attention and ensemble perception.
  — [Nature Human Behaviour](https://www.nature.com/articles/s41562-020-00938-0); [PubMed](https://pubmed.ncbi.nlm.nih.gov/32895546/); [Brady lab PDF](https://bradylab.ucsd.edu/pdfs/SchurginWixtedBrady2020.pdf); [bioRxiv](https://www.biorxiv.org/content/10.1101/325472v4.full); [lab summary](https://bradylab.ucsd.edu/research.html)
- **Brady, T. F., Robinson, M. M., Williams, J. R., & Wixted, J. T. (2023). Measuring memory is harder than you think: How to avoid problematic measurement practices in memory research. *Psychonomic Bulletin & Review*, 30, 421–449. doi:10.3758/s13423-022-02179-w.** Argues that "precision" and "guess rate" from mixture models are not separable psychological constructs. — **[prior knowledge; verify citation]**
- **Critiques:** Tomić & Bays argued that measured perceptual similarity does not fully predict the distribution of WM errors, and that population-coding models fit better. — **[prior knowledge: Tomić, I., & Bays, P. M., in *JEP: Learning, Memory, and Cognition* around 2023–2024; exact citation unverified]**
- **Color-space caveats:**
  - Bae, G.-Y., Olkkonen, M., Allred, S. R., & Flombaum, J. I. (2015). Why some colors appear more memorable than others: A model combining categories and particulars in color working memory. *JEP: General*, 144(4), 744–763. doi:10.1037/xge0000076. Reports are systematically biased toward category prototypes and precision varies around the CIELAB wheel. — **[prior knowledge]**
  - Hardman, K. O., Vergauwe, E., & Ricker, T. J. (2017). Categorical working memory representations are used in delayed estimation of continuous colors. *JEP:HPP*, 43(1), 30–54. doi:10.1037/xhp0000290. Same conclusion about categorical representations. — **[prior knowledge]**

### Inferences
- Implications for the ViT analysis:
  1. Fitting only Zhang–Luck/Bays mixtures and reading Pm as "capacity" is contested. Report TCC (d′ per set size) alongside them.
  2. TCC needs a similarity function. Use the human psychophysical function (from Schurgin et al.'s OSF data) to compare against human data. Also derive a **network-internal similarity function**, e.g., cosine similarity of ViT embeddings for single-color displays, and test whether TCC with the model's own similarity predicts its error shapes. That makes a clean interpretability result.
  3. A key TCC prediction is that a **single d′ per condition determines the whole error distribution shape**. If the network's "guess rate" and "SD" covary as TCC predicts (both are functions of d′), that favors a TCC-like signal account. If SD plateaus while the guess rate climbs independently, that favors slots.
  4. Check category biases and stimulus-specific precision in the ViT. ImageNet-pretrained features may carry color-category structure.

### Gaps
- I did not verify the specific d′ values per set size reported by Schurgin et al.

---

## 6. Typical human numbers, delay/maintenance, iconic vs working memory, and perception vs memory set-size effects

### Takeaway
For color continuous report with about 100–200 ms sample and about 1 s delay, humans typically show a circular SD of roughly 13–20° at N = 1, rising to about 20–30° at N ≥ 3–4, and a mixture-estimated K of about 2.5–3.5. Delay effects over 1–10 s are modest, and whether they appear as SD increase or Pm loss is debated. Iconic memory (< ~300–500 ms after offset) is high-capacity. Recent work (Tomić & Bays 2024) models iconic memory and VWM as regimes of a single dynamic population-coding resource. Set-size effects exist even for visible displays, though smaller. So a no-delay, simultaneous-display network task is not cleanly a "working memory" task.

### Cited Findings
- **Human magnitudes (color, CIELAB wheel):** SD at N = 1 is about 13–20°. SD for in-memory items at N ≥ 3 is about 20–25°. K ≈ 3 (Zhang & Luck 2008). — [Zhang & Luck 2008](https://www.nature.com/articles/nature06860) **[magnitudes from prior knowledge; check the paper's Fig. 2 and the datasets on OSF g49c6]**
- **Delay:**
  - Zhang & Luck 2009 (*Psych Sci* 20:423) reported mostly item loss at long delays (~10 s) with stable SD. — **[prior knowledge]**
  - Other work found gradual precision loss with delay: Rademaker, R. L., Park, Y. E., Sack, A. T., & Tong, F. (2018). *JEP:HPP*, 44(6), 925–940. doi:10.1037/xhp0000491. Also Shin, H., Zou, Q., & Ma, W. J. (2017). The effects of delay duration on visual working memory for orientation. *Journal of Vision*, 17(14):10. doi:10.1167/17.14.10. — **[prior knowledge]**
  - Pertzov, Manohar & Husain (2017, *JEP:LMC* 43(4), 528–536) found that forgetting over time arises from competition between items and is greater at higher set sizes. — **[prior knowledge]**
  - The Oberauer et al. 2018 benchmarks treat time-based decay as weak or contested. — [Psych Bull](https://pubmed.ncbi.nlm.nih.gov/30148379/)
- **Iconic vs working memory:**
  - Sperling, G. (1960). The information available in brief visual presentations. *Psychological Monographs*, 74(11), 1–29. doi:10.1037/h0093759. Partial-report advantage for cues in the first ~300 ms. — **[prior knowledge]**
  - Phillips, W. A. (1974). On the distinction between sensory storage and short-term visual memory. *Perception & Psychophysics*, 16, 283–290. doi:10.3758/BF03203943. High-capacity, maskable, location-bound sensory storage vs limited, robust VSTM. — **[prior knowledge]**
- **Tomić, I., & Bays, P. M. (2024). A dynamic neural resource model bridges sensory and working memory. *eLife*, 12, RP91034. doi:10.7554/eLife.91034.**
  - Extends the Bays (2014) normalization model with time: fast, sensory-driven accumulation of activity, plus slower noise-driven drift (diffusion) of stored values. Total activity is fixed but shared among items.
  - Experiment: orientation arrays of varying size, probed 0–1000 ms after offset. Both post-stimulus sensory signal and diffusion were needed to explain the dynamics.
  - A commentary concluded that iconic memory and working memory are "two slightly different regimes of the same circuitry."
  — [eLife](https://elifesciences.org/articles/91034); [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11068358/); [PDF](https://paulbays.com/pdf/TomBay24.pdf)
- **Perception vs memory set-size effects:** Palmer, J. (1990). Attentional limits on the perception and memory of visual information. *JEP:HPP*, 16(2), 332–350. doi:10.1037/0096-1523.16.2.332. Set-size effects appear for brief, visible displays as well as in memory, but are larger in memory. — **[prior knowledge]**

### Inferences
- If the ViT sees the colors and the cue in the same forward pass with no delay or masking, its set-size effects reflect **encoding/readout capacity (perceptual/attentional limits)**, not maintenance. The closest human analogue is a simultaneous-cue or 0-ms-delay iconic condition, where set-size effects are smaller. To claim "working memory," the project needs a sequential design: sample frames, then mask or blank, then cue (e.g., via recurrence, video-ViT tokens, or a memory bottleneck). Alternatively, frame the claim as "capacity limits in visual encoding, compared against VWM signatures." Reviewers will raise this.
- Tomić & Bays (2024) give a principled bridge: they predict a set-size effect even at zero delay, plus growth with delay through diffusion. Fitting their model to a no-delay network might place it in their "sensory regime."

### Gaps
- No verified human numbers for set-size effects in a truly simultaneous (cue-with-display) color-wheel condition. Palmer 1990 and Tomić & Bays 2024 are the closest sources.
- I could not extract exact SD/Pm tables from any primary PDF this session. Use the OSF benchmark datasets for human reference curves.

---

## 7. Neural accounts: persistent vs activity-silent maintenance; normalization accounts of capacity

### Takeaway
The classic view keeps WM content in persistent delay activity. Stokes (2015) proposed "activity-silent" storage in hidden synaptic or connectivity states, reactivated when needed. Capacity limits have a mechanistic account in **divisive normalization** of a fixed total population activity (Bays 2014). Spiking attractor networks with normalization show both resource-like and slot-like regimes (Wei, Wang & Wang 2012).

### Cited Findings
- **Stokes, M. G. (2015). 'Activity-silent' working memory in prefrontal cortex: a dynamic coding framework. *Trends in Cognitive Sciences*, 19(7), 394–405. doi:10.1016/j.tics.2015.05.004.**
  - Persistent delay activity "does not always accompany WM maintenance but instead seems to wax and wane as a function of the current task relevance."
  - Proposed mechanism: input drives a transient activity pattern that changes a hidden network state (e.g., short-term synaptic plasticity), which persists after activity returns to baseline.
  — [PubMed](https://pubmed.ncbi.nlm.nih.gov/26051384/); [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4509720/); [Cell](https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(15)00102-3)
- **Mongillo, G., Barak, O., & Tsodyks, M. (2008). Synaptic theory of working memory. *Science*, 319, 1543–1546. doi:10.1126/science.1150769.** Short-term facilitation as a silent memory store. — **[prior knowledge]**
- **Wolff, M. J., Jochim, J., Akyürek, E. G., & Stokes, M. G. (2017). Dynamic hidden states underlying working-memory-guided behavior. *Nature Neuroscience*, 20, 864–871. doi:10.1038/nn.4546.** "Pinging" with an impulse stimulus reveals decodable hidden states of WM content. — **[prior knowledge]**
- **Bays (2014)**: normalization fixes the total spike rate across memoranda, giving a power-law variance increase (exponent 1.36) and heavy tails without slots. — [J Neurosci](https://www.jneurosci.org/content/34/10/3632)
- **Wei, Z., Wang, X.-J., & Wang, D.-H. (2012). From distributed resources to limited slots in multiple-item working memory: a spiking network model with normalization. *Journal of Neuroscience*, 32(33), 11228–11240. doi:10.1523/JNEUROSCI.0735-12.2012.** In an attractor network with lateral inhibition and normalization, items merge or are lost at high load, producing slot-like "sudden death" from a resource-like mechanism. — **[prior knowledge]**
- **Contrasting primate data:** a 2017 *Journal of Neuroscience* study (Spaak, Watanabe, Funahashi & Stokes, "Stable and dynamic coding for working memory in primate prefrontal cortex," 37(27), 6503–6516) argues that dynamic population coding coexists with a stable subspace. — [J Neurosci](https://www.jneurosci.org/content/37/27/6503)

### Inferences
- For a feedforward ViT without a delay, the persistent vs activity-silent distinction has no direct analogue. In a recurrent or delay-augmented version, one can ask whether information is maintained in decodable token activity (persistent) or recoverable only after a "ping" probe (activity-silent analogue).
- Softmax attention normalization is the most obvious ViT analogue of divisive normalization. Testing whether per-item attention or readout gain falls as ~1/N, and whether that predicts the SD(N) power law, would link the model to Bays (2014).

### Gaps
- I did not survey 2025–2026 developments in neural VWM mechanisms. No verified post-2024 primary sources were retrieved in this session. Adjacent researchers on deep-network VWM models should cover that space.
