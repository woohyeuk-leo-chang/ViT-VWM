# Working Memory in LLMs and Transformers: Behavioral and Mechanistic Evidence

Notes compiled 2026-10-07. Verification status is marked on each item. **[V]** means this session confirmed the claim against the primary source page (arXiv abstract or HTML, ACL Anthology, OpenReview). **[K]** means the citation and claim are standard, well-known facts from prior knowledge that were not re-fetched this session; spot-check exact numbers before quoting them. **NEW** flags the most recent (2025–2026) work.

---

## 1. Gong, Wan & Wang 2024: "Working Memory Capacity of ChatGPT: An Empirical Study" (AAAI)

### Takeaway
ChatGPT (gpt-3.5-turbo) shows an n-back capacity limit that the authors call "strikingly similar to humans". Defining capacity as the largest n with d' ≈ 1 gives about 3 for verbal n-back and less for spatial n-back. GPT-4 did much better, and small open-source models were near floor. The study is foundational but methodologically contested, because every stimulus stays in the context window.

### Cited Findings
- Full citation: Dongyu Gong, Xingchen Wan, Dingmin Wang. "Working Memory Capacity of ChatGPT: An Empirical Study." *AAAI 2024* 38(9):10048–10056, DOI 10.1609/aaai.v38i9.28868. arXiv:2305.03731 (first posted May 2023). Code and data at github.com/Daniel-Gong/ChatGPT-WM. [V] — [arXiv](https://arxiv.org/abs/2305.03731); [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/28868); [GitHub](https://github.com/Daniel-Gong/ChatGPT-WM)
- Paradigm: verbal (letter) and spatial (grid) n-back with n ∈ {1,2,3}. Each n has 30 blocks of 30 trials, with 10 match and 20 non-match trials per block. [V] — [GitHub](https://github.com/Daniel-Gong/ChatGPT-WM)
- Conditions: base, noise/distractors, chain-of-thought (CoT), and feedback, for both verbal and spatial. Spatial variants also included abstract spatial reasoning and grid sizes 4×4, 5×5 and 7×7. [V] — [arXiv HTML](https://arxiv.org/html/2305.03731v4)
- Performance (d') fell significantly with n in most conditions. Kruskal–Wallis results: verbal base H=97.5, p≈6.6e-22; spatial base H=84.9, p≈3.6e-19. The exceptions were the noise conditions, where verbal noise gave H=3.92, p=0.14 and spatial noise gave H=0.63, p=0.73, because performance was already at floor. [V] — [arXiv HTML](https://arxiv.org/html/2305.03731v4)
- Capacity criterion was d' ≈ 1. The authors say verbal variants other than noise have a capacity of "around 3". Spatial capacity was lower: in spatial base, 1-back d' was significantly above 1 but 2-back and 3-back were not. CoT significantly improved spatial performance. Noise substantially reduced verbal capacity. The abstract-spatial variant lowered capacity, and larger grids raised it. [V] — [arXiv HTML](https://arxiv.org/html/2305.03731v4)
- The "around 3" summary is generous. With verbal feedback, 2-back (p=0.25) and 3-back (p=0.68) were not significantly above d'=1, and the verbal-base 3-back result was borderline (p=0.048). [V, my reading of their stats] — [arXiv HTML](https://arxiv.org/html/2305.03731v4)
- Other models: GPT-4 was tested on verbal base only, up to n=3. Its capacity "far exceeds" other LLMs, so its true limit was not measured. Bloomz-7B, Bloomz-7B1-mt, ChatGLM-6B (v1.0 and v1.1), Vicuna-7B and Vicuna-13B all had very low capacity and were nearly indistinguishable from one another. [V] — [arXiv HTML](https://arxiv.org/html/2305.03731v4)
- Human comparison came from the literature, not new data. The authors cite Jaeggi et al. (2010) for human performance dropping at n=3. [V] — [arXiv HTML](https://arxiv.org/html/2305.03731v4)
- The authors propose n-back as a benchmark for LLM working memory. [V] — [arXiv](https://arxiv.org/abs/2305.03731)

### Inferences
- The human-like limit is probably not a storage limit, since the letters remain in context. More likely it reflects failure to bind each item to its relative position, i.e. index-based retrieval ("which token was n steps back?"). Ebrahimi et al. 2024 (section 6) independently argue that LLMs are poor at index-based addressing. The ViT parallel is direct: positional binding of items, not item storage, may be the bottleneck.

### Gaps
- The exact d', hit and false-alarm values appear only in figures and were not extracted.
- No dedicated replication with current frontier models (GPT-4o/5, Claude 3.5+) was found in this session. Xiong et al. 2026 (section 3) cover modern LLMs with a different task.

---

## 2. Wang & Sun 2025: "Unable to Forget: Proactive Interference Reveals Working Memory Limits in LLMs Beyond Context Length" (NEW)

### Takeaway
In the PI-LLM paradigm the model sees a stream of key–value updates and must report only the latest value of each key. Accuracy declines roughly log-linearly toward zero as interference accumulates, even though the answer sits just before the query and the input is far below the context limit. Errors are mostly overwritten earlier values. Resistance to interference scales with parameter count, not context-window length. Natural-language "forget" instructions do not help.

### Cited Findings
- Full citation: Chupei Wang (U. Virginia) and Jiaqiu Vince Sun (NYU). "Unable to Forget: Proactive Interference Reveals Working Memory Limits in LLMs Beyond Context Length." arXiv:2506.08184. v1 posted 9 Jun 2025, v3 posted 31 Jul 2025. Accepted at the ICML 2025 Workshop on Long Context Foundation Models. [V] — [arXiv](https://arxiv.org/abs/2506.08184); [OpenReview](https://openreview.net/forum?id=YUHksmL8aw)
- A later OpenReview version (forum y8jS7mDurI) reframes the task as "co-referenced" key rebinding and relates it to the MRCR long-context benchmark. [V, abstract-level] — [OpenReview](https://openreview.net/forum?id=y8jS7mDurI)
- Models: dense and MoE models from about 0.6B to 637B parameters. Families include Qwen3 (0.6B–235B, plus thinking variants), Qwen2.5-72B, DeepSeek-V3/R1, Llama-4 Maverick/Scout, GPT-4.1/4o (mini and nano), Gemini 1.5/2.0/2.5 Flash, Grok-3 and Claude. [V] — [arXiv HTML v3](https://arxiv.org/html/2506.08184v3)
- Manipulations:
  - 46 keys with 3 to 400 updates per key, randomly interleaved.
  - Number of updated keys varied from 1 to 46.
  - Number of queried keys varied at fixed input length.
  - Value length varied. With values built from concatenated words at 20 updates, accuracy was below 40% at 10 words and below 5% at 40 words.
  
  [V] — [arXiv HTML v3](https://arxiv.org/html/2506.08184v3)
- Retrieval accuracy declines log-linearly as interference grows, along every manipulated dimension. In one example, Llama-4 Maverick fell from about 100% with 2 tracked keys to below 5% with 46 tracked keys at fixed length. [V] — [arXiv HTML v3](https://arxiv.org/html/2506.08184v3)
- Strictly sequential (non-interleaved) updates give a different curve: accuracy stays near ceiling until a model-specific threshold, then collapses. [V] — [OpenReview PDF](https://openreview.net/pdf?id=YUHksmL8aw); [arXiv HTML v3](https://arxiv.org/html/2506.08184v3)
- Error analysis: most errors return earlier, overwritten values of the same key, which is proactive interference. At high load the models also produce never-presented values ("hallucinations") and show a primacy bias. [V] — [arXiv HTML v3](https://arxiv.org/html/2506.08184v3)
- Size versus context: the Interference Endurance Score (IES, the AUC of accuracy across update counts) is predicted by parameter-size class (t=3.03, p=0.005, N=30) but not by context length (t=−0.144, p=0.886). The combined model explains 26.1% of variance. MoE models underperform dense models with similar total parameters. [V] — [arXiv HTML v3](https://arxiv.org/html/2506.08184v3)
- Mitigation:
  - The instruction "Forget all the previous updates to key…" gave less than 10 points of improvement at 100 updates, and errors clustered near where the instruction was inserted.
  - "Forward focus", relevance meta-prompts and soft session resets were also largely ineffective.
  - A non-natural-language "mock QA reset", which simulates a closed earlier turn, helped substantially, but the decline with load remained.
  
  [V] — [arXiv HTML v3](https://arxiv.org/html/2506.08184v3)
- Interpretation: the authors frame the result as a "Limited Anti-Interference Capacity", analogous to human working memory and independent of context length. They contrast it with humans, whose proactive-interference curves plateau, which they attribute to active unbinding or gating. No new human data were collected. [V] — [arXiv HTML v3](https://arxiv.org/html/2506.08184v3)

### Inferences
- This is the clearest evidence that a perfect buffer is not the same as perfect working memory. Storage is lossless, but readout under similarity-based competition is capacity-limited. That matches interference-based (not decay- or slot-based) accounts of human working memory, and it is a natural hypothesis for a ViT on a visual working-memory task: test similarity-driven swap errors rather than raw storage limits.

### Gaps
- No mechanistic analysis appears in the paper, e.g. which heads attend to stale values.
- No matched human experiment was run.

---

## 3. Other behavioral studies of LLM working memory (n-back, span, interference)

### Takeaway
Results split along a methodological line. Studies that keep stimuli in context find human-like load effects, such as declines at about 3–4-back and recency/interference biases. They also find that working-memory performance tracks general capability. Studies that remove stimuli from context to force internal maintenance find that LLMs essentially have no latent working memory unless they can externalize state through chain-of-thought.

### Cited Findings
- **Zhang, Jian, Ouyang & Vosoughi 2024**, "Working Memory Identifies Reasoning Limits in Language Models," *EMNLP 2024 (Main)*, pp. 16896–16922.
  - Uses n-back to probe scaling limits. Larger models still struggle to hold and process information under complex conditions.
  - Prompting strategies have mixed effects. LLMs depend on manually corrected prompts and cannot find effective problem-solving patterns on their own.
  - The authors argue for better planning and search.
  
  [V] — [ACL Anthology](https://aclanthology.org/2024.emnlp-main.938/)
- **Huang, Sun, Wang & Dredze 2025**, "Language Models Do Not Have Human-Like Working Memory" (v1 title: "LLMs Do Not Have Human-Like Working Memory"). arXiv:2505.10571, v1 April/May 2025, v3 23 Sep 2025. OpenReview forum SOxO7e6ySB; no confirmed venue found. (NEW) [V] — [arXiv](https://arxiv.org/abs/2505.10571v3); [OpenReview](https://openreview.net/forum?id=SOxO7e6ySB)
  - Core argument: in-context n-back is not a valid working-memory test because the model can attend back to the stimuli. The authors' tasks require holding information that is absent from the context. [V] — [arXiv HTML](https://arxiv.org/html/2505.10571v3)
  - Number Guessing: the model "picks" a number from 1 to 10 privately and is asked about each value 200 times. Probabilities of "yes" should sum to 1. Most models summed to about 0, including GPT-4o-mini, GPT-4o-2024-11-20, Qwen2.5-72B and DeepSeek-V3. When models said yes, they favored 7. CoT and reasoning models (o1, o3, o4, QwQ, R1) did not help. [V] — [arXiv HTML](https://arxiv.org/html/2505.10571v3)
  - Yes-No Deduction (imagine an object, then answer comparative questions): GPT-4o-mini contradicted itself in all 200 trials, and GPT-4o in 173 of 200. Contradictions typically appeared after about 20–40 questions. [V] — [arXiv HTML](https://arxiv.org/html/2505.10571v3)
  - Math Magic (a Josephus-style task tracking 4 numbers): without CoT, accuracy was 0–26%, with LLaMA-3.1-405B best. With CoT or reasoning models, DeepSeek-R1 reached 100%, o3-mini 96.7%, QwQ-32B 90% and o1-mini 50%. The authors attribute the CoT gains to externalized state, not internal maintenance. [V] — [arXiv HTML](https://arxiv.org/html/2505.10571v3)
- **Xiong, Ji-An, Huang, Wilson, Lee & Wei 2026**, "In-context superposition: human-like working memory interference in large language models." arXiv:2604.09670, v1 1 Apr 2026, v3 13 Aug 2026, *COLM 2026*. (MOST RECENT) [V] — [arXiv](https://arxiv.org/abs/2604.09670)
  - A two-layer transformer trained on the working-memory task solves it perfectly. Diverse pretrained LLMs show human-like load-dependent declines, recency bias and stimulus-statistics biases. Working-memory performance correlates with general capability, as it does in humans. [V] — [arXiv](https://arxiv.org/abs/2604.09670)
  - Mechanism: multiple items are encoded in entangled representations ("in-context superposition"). Layers progressively suppress competitors and align the target with the readout. A causal intervention suppressing interfering information improves performance. The authors conclude that working-memory capacity reflects selection under interference, and they frame it as a generalization-versus-interference tradeoff from shared compressed codes. [V] — [arXiv](https://arxiv.org/abs/2604.09670)
  - A secondary summary of this paper says that in a multi-turn n-back most LLMs approach chance by 3- or 4-back, and that one model exceeded human performance while still showing a capacity decline. [V, via search snippet only; confirm in the PDF] — [arXiv PDF](https://arxiv.org/pdf/2604.09670)
- **Janik 2023**, "Aspects of human memory and Large Language Models," arXiv:2311.03839 (v3 Apr 2024), Jagiellonian Univ. Applies human memory paradigms, including serial-position recall of lists, to LLMs. Finds notable parallels with human memory and argues they come from training-data statistics rather than architecture. [V, abstract-level] — [arXiv](https://arxiv.org/abs/2311.03839)
- **Armeni, Honey & Linzen 2022**, "Characterizing Verbatim Short-Term Memory in Neural Language Models," *CoNLL 2022*, pp. 405–424; arXiv:2210.13569.
  - Paradigm: noun lists repeated in text. Retrieval is measured as the drop in surprisal on the second presentation.
  - Transformers retrieved both identity and order of the first list, and retrieval improved with more training data and depth. AWD-LSTM retrieval was minimal, order-insensitive and decayed quickly.
  - Interpretation: transformers act like a flexible working-memory buffer, while LSTMs keep a coarse semantic gist. This verbatim retrieval is learned during training, not built into the architecture.
  
  [V] — [ACL Anthology](https://aclanthology.org/2022.conll-1.28/); [arXiv](https://arxiv.org/abs/2210.13569)
- **Armeni et al. 2024**, "Transformer verbatim in-context retrieval across time and scale," *CoNLL 2024*. A follow-up tracking how verbatim retrieval emerges over training and model scale. [V, existence only; details not fetched] — [ACL Anthology PDF](https://aclanthology.org/2024.conll-1.6v1.pdf)
- Related 2026 preprints were seen in search results but not read: "Are they human? Detecting LLMs by probing human memory constraints" (arXiv:2604.00016) and "Simulating Human Memory with Language Models" (arXiv:2605.25680). [V, existence only] — [arXiv 2604.00016](https://arxiv.org/html/2604.00016); [arXiv 2605.25680](https://arxiv.org/html/2605.25680v1)
- **"Li et al. 2025 MemoryBench"**: the MemoryBench found (arXiv:2510.17281) is a benchmark for memory and continual learning in LLM systems, not an n-back or working-memory-span benchmark. It probably does not match the brief's intended reference. [V, title-level] — [arXiv](https://arxiv.org/abs/2510.17281)

### Inferences
- "Do LLMs have human-like working memory?" depends on the operationalization:
  - With in-context tasks (Gong, Zhang, Xiong, Wang & Sun), the answer is a qualified yes. There are load and interference limits, and they reflect retrieval/selection failures.
  - With latent-maintenance tasks (Huang et al.), the answer is no, because there is no persistent internal state across turns beyond the KV cache of emitted tokens.
- For a ViT whose stimuli are erased before test (delay period), the latent-maintenance framing is more relevant than the in-context one.

### Gaps
- No study found pairs matched human and LLM participants on an identical in-context n-back with similarity manipulations.
- The phonological-similarity analog (similar tokens produce more confusion) is implied by Wang & Sun (semantically related updates) and Xiong et al. (stimulus statistics), but no dedicated paper was found this session.

---

## 4. Serial-position effects and context-length degradation (Lost in the Middle, NIAH, RULER)

### Takeaway
Long-context LLMs show a U-shaped serial-position curve, with primacy and recency advantages and a mid-context trough. Effective context is much shorter than the advertised window once tasks go beyond literal string matching, e.g. variable tracking, aggregation, or non-lexical needles. The "perfect buffer" is perfect for storage but not for retrieval.

### Cited Findings
- **Liu, Lin, Hewitt, Paranjape, Bevilacqua, Petroni & Liang 2024**, "Lost in the Middle: How Language Models Use Long Contexts," *TACL* 12:157–173; arXiv:2307.03172. In multi-document QA and synthetic key–value retrieval, accuracy is highest when the relevant information is at the start or end of the context and drops substantially in the middle (U-shaped). Extended-context models were not better at using their context. [K] — [arXiv](https://arxiv.org/abs/2307.03172); [TACL/ACL Anthology](https://aclanthology.org/2024.tacl-1.9/)
- **Guo & Vosoughi 2024/2025**, "Serial Position Effects of Large Language Models," arXiv:2406.15981; *Findings of ACL 2025*. Primacy and recency effects are widespread across tasks and models, including encoder–decoder T5/FlanT5, with varying strength. Prompting mitigates them only partially and inconsistently. Earlier work had found primacy-dominant effects in ChatGPT, GPT-3.5 and GPT-4 (Zhang et al. 2023) and in Claude-instant-1.2 (Eicher & Irgolič 2024). [V] — [arXiv](https://arxiv.org/abs/2406.15981); [ACL Findings PDF](https://aclanthology.org/2025.findings-acl.52.pdf)
- **Hsieh, Sun, Kriman, Acharya, Rekesh, Jia, Zhang & Ginsburg 2024**, "RULER: What's the Real Context Size of Your Long-Context Language Models?", *COLM 2024*; arXiv:2404.06654. 13 tasks in 4 categories: retrieval (multi-key/multi-value NIAH), multi-hop tracing (variable tracking, i.e. following chains of X1=…, X2=X1), aggregation, and QA. Models achieve near-perfect vanilla NIAH yet degrade sharply with length on the other tasks. Of 17 models claiming 32K+ context, only about half maintained satisfactory performance at 32K. [K] — [arXiv](https://arxiv.org/abs/2404.06654)
- **Kamradt 2023**, "Needle in a Haystack" pressure test (GitHub). This is the original single-fact retrieval-by-depth-and-length test, now widely considered too easy because lexical overlap allows simple matching. [K] — [GitHub](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)
- **Modarressi et al. 2025**, "NoLiMa: Long-Context Evaluation Beyond Literal Matching," *ICML 2025*; arXiv:2502.05167. Removing lexical overlap between question and needle causes large drops at moderate lengths: most of the models tested fall below 50% of their short-context baseline by 32K. [K; verify numbers] — [arXiv](https://arxiv.org/abs/2502.05167)
- **Kuratov et al. 2024**, "BABILong," *NeurIPS 2024 Datasets & Benchmarks*; arXiv:2406.10149. Reasoning over facts scattered in long distractor text. Models effectively use only a fraction of their context. [K] — [arXiv](https://arxiv.org/abs/2406.10149)
- Mechanistic accounts of position bias:
  - **Xiao et al. 2024**, "Efficient Streaming Language Models with Attention Sinks" (*ICLR 2024*; arXiv:2309.17453). Initial tokens absorb disproportionate attention, which links to primacy. [K] — [arXiv](https://arxiv.org/abs/2309.17453)
  - **Wu, Wang, Jegelka & Jadbabaie 2025**, "On the Emergence of Position Bias in Transformers" (*ICML 2025*; arXiv:2502.01951). Causal masking biases attention toward early positions, and positional encodings add recency decay. [K; verify] — [arXiv](https://arxiv.org/abs/2502.01951)

### Inferences
- The serial-position curve in LLMs comes from architecture and training: attention sinks plus causal masking produce primacy, and relative position encodings plus training-data locality produce recency. Human primacy is usually attributed to rehearsal and recency to a short-term store. The curves look alike, but the mechanisms are not homologous.

### Gaps
- Exact effect sizes for Lost in the Middle and RULER were not re-fetched this session.

---

## 5. Theoretical and mechanistic framing: attention as associative memory, and fixed-capacity vs. unbounded-buffer architectures

### Takeaway
Attention is a content-addressable associative memory (a modern Hopfield network) over an unbounded, growing buffer, the KV cache. Induction heads implement in-context copy and recall. Fixed-state recurrent models (SSMs/Mamba, linear attention) have hard memory capacity limits and fail recall and copying tasks that transformers solve. These limits shrink as state size grows, a recall–throughput tradeoff. This gives a clean contrast between a fixed-capacity store and an unbounded buffer.

### Cited Findings
- **Olsson et al. 2022**, "In-context Learning and Induction Heads," *Transformer Circuits Thread*; arXiv:2209.11895. Induction heads (a previous-token head composed with a match-and-copy head: [A][B]…[A]→[B]) form during a phase change early in training that coincides with a jump in in-context learning. [K] — [arXiv](https://arxiv.org/abs/2209.11895); [Transformer Circuits](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
- **Ramsauer et al. 2021**, "Hopfield Networks is All You Need," *ICLR 2021*; arXiv:2008.02217. The transformer attention update equals the update rule of a continuous modern Hopfield network. Storage capacity is exponential in the pattern dimension, and retrieval usually takes a single update. [K] — [arXiv](https://arxiv.org/abs/2008.02217)
- **Bietti, Cabannes, Bouchacourt, Jégou & Bottou 2023**, "Birth of a Transformer: A Memory Viewpoint," *NeurIPS 2023*; arXiv:2306.00802. On a synthetic task mixing global and in-context bigrams, weight matrices act as associative memories (outer-product key→value stores). Training dynamics show global bigrams learned first, then an induction-head mechanism. [K] — [arXiv](https://arxiv.org/abs/2306.00802)
- **Cabannes, Dohmatob & Bietti 2024**, "Scaling Laws for Associative Memories," *ICLR 2024*; arXiv:2310.02984. Capacity and scaling of outer-product associative memories as a function of dimension and data. This is a formal handle on how much a weight-based (not context-based) memory can store. [K] — [arXiv](https://arxiv.org/abs/2310.02984)
- **Arora, Eyuboglu, Timalsina, Johnson, Poli, Zou, Rudra & Ré 2024**, "Zoology: Measuring and Improving Recall in Efficient Language Models," *ICLR 2024*; arXiv:2312.04927. Most of the perplexity gap between attention and gated-convolution models (about 82% in the paper) comes from associative recall. They introduce multi-query associative recall (MQAR) and show that attention solves it with model dimension independent of sequence length, while gated convolutions need dimension that grows with length. [K] — [arXiv](https://arxiv.org/abs/2312.04927)
- **Arora et al. 2024**, "Simple Linear Attention Language Models Balance the Recall-Throughput Tradeoff" (the Based model), *ICML 2024*; arXiv:2402.18668. A fundamental tradeoff between recurrent state size and recall ability. Based (linear plus sliding-window attention) moves along this frontier. [K] — [arXiv](https://arxiv.org/abs/2402.18668)
- **Jelassi, Brandfonbrener, Kakade & Malach 2024**, "Repeat After Me: Transformers are Better than State Space Models at Copying," *ICML 2024*; arXiv:2402.01032. Theory: a two-layer transformer can copy strings exponentially long in its size, while any generalized SSM is bounded by its fixed-size latent state. Empirically, transformers train faster and generalize better on copying and context retrieval, and pretrained transformers beat similar-size SSMs on phonebook lookup. [K] — [arXiv](https://arxiv.org/abs/2402.01032)
- **Gu & Dao 2023**, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," arXiv:2312.00752 (COLM 2024). The selective SSM with a fixed-size recurrent state is the canonical finite-memory comparison. [K] — [arXiv](https://arxiv.org/abs/2312.00752)
- **Merrill, Petty & Sabharwal 2024**, "The Illusion of State in State-Space Models," *ICML 2024*; arXiv:2404.08819. Despite being recurrent, SSMs, like transformers, cannot express true state tracking (e.g. permutation composition) within TC⁰. Recurrence alone does not give working-memory-like state updating. [K] — [arXiv](https://arxiv.org/abs/2404.08819)
- **Whittington, Warren & Behrens 2022**, "Relating Transformers to Models and Neural Representations of the Hippocampal Formation," *ICLR 2022*; arXiv:2112.04035. Transformers with recurrent position encodings are closely related to the Tolman-Eichenbaum Machine. Trained on spatial tasks, they reproduce place- and grid-cell-like representations. This links attention-as-memory to hippocampal (episodic/relational) memory rather than prefrontal working memory. [K] — [arXiv](https://arxiv.org/abs/2112.04035)
- **Ebrahimi, Panchal & Memisevic 2024**, "Your Context Is Not an Array: Unveiling Random Access Limitations in Transformers," *NeurIPS 2024 Workshop (Sys2-Reasoning)*; arXiv:2408.05506. Length-generalization failures, e.g. on parity, come from transformers' difficulty with index-based (random-access) addressing. Natural-language pretraining favors content-based addressing. "Mnemonic" anchor tokens convert index lookups into content lookups and give perfect length generalization (trained on 10–20 bits, tested up to 60). [V] — [arXiv](https://arxiv.org/abs/2408.05506)

### Inferences
- Framing for the ViT project, as a taxonomy of three memory regimes:
  1. **Unbounded buffer with content addressing (attention/KV cache).** No storage limit. Limits arise from retrieval interference (Wang & Sun; Xiong et al.), position-based addressing (Ebrahimi; Gong n-back), and position bias (Lost in the Middle).
  2. **Fixed-size recurrent state (SSM/LSTM).** A true capacity limit, with recall degrading as items exceed the state (Zoology, Based, Jelassi). This is closest to slot or resource models of human visual working memory.
  3. **Weight-based associative memory.** Capacity scales with dimension (Ramsauer; Cabannes et al.).
- A ViT that sees all items simultaneously in tokens is regime 1 within a frame. If the task involves a delay or sequential frames without token carry-over, information must pass through a bottleneck such as a CLS token or residual stream, which is closer to regime 2. Testing set-size effects therefore predicts different signatures in the two cases.

### Gaps
- No direct paper was found that measures set-size or precision curves (as in human visual working-memory resource models) in transformers. Xiong et al. 2026 come closest.

---

## 6. Entity tracking and variable-binding capacity

### Takeaway
LLMs can track entity states and bind attributes to entities in context, but performance degrades with the number of entities and operations. Mechanistically, binding uses positional, ordering-based "binding ID" codes. Fine-tuning strengthens the existing tracking circuit rather than creating a new one. Binding is a natural locus for capacity limits, analogous to feature binding in visual working memory.

### Cited Findings
- **Kim & Schuster 2023**, "Entity Tracking in Language Models," *ACL 2023*; arXiv:2305.02363. "Boxes" task: track the contents of boxes after a sequence of move/put/remove operations. Among the models tested, only those trained heavily on code (e.g. GPT-3.5 text-davinci-003) showed nontrivial tracking. Base GPT-3 and Flan-T5 largely failed, and accuracy fell with the number of operations affecting a box. Small fine-tuned T5 learned the task but generalized poorly. [K] — [arXiv](https://arxiv.org/abs/2305.02363); [ACL Anthology](https://aclanthology.org/2023.acl-long.213/)
- **Prakash, Shaham, Haklay, Belinkov & Bau 2024**, "Fine-Tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking," *ICLR 2024*; arXiv:2402.14811. Identifies an entity-tracking circuit in LLaMA-7B. The same circuit carries out the task in fine-tuned variants (Vicuna, Goat, FLoat). Gains from fine-tuning come mainly from better handling of positional information, shown with cross-model activation patching (CMAP). [K] — [arXiv](https://arxiv.org/abs/2402.14811)
- **Feng & Steinhardt 2024**, "How Do Language Models Bind Entities in Context?", *ICLR 2024*; arXiv:2310.17191. Proposes the binding-ID mechanism: entity and attribute activations carry matching vector codes, which are found across LLaMA and Pythia families. [K] — [arXiv](https://arxiv.org/abs/2310.17191)

### Inferences
- Binding IDs act like positional "slots": as entity count rises, their separability shrinks, which predicts capacity limits from binding interference. This is a candidate mechanism to look for in the ViT, e.g. whether item-location bindings are carried by separable directions that become crowded as set size increases.

### Gaps
- Quantitative capacity curves (accuracy vs. number of entities) from Prakash et al. and Feng & Steinhardt were not re-fetched.
- No 2025–2026 entity-tracking capacity paper was confirmed this session.

---

## Overall synthesis

Do transformers have human-like working-memory limits?

- **Storage: no.** Within the window the KV cache is a lossless buffer (Armeni et al. 2022; Jelassi et al. 2024; Ramsauer et al. 2021).
- **Retrieval and selection: yes, and human-like in form.** Evidence: load-dependent n-back decline at about 3 (Gong et al. 2024); log-linear proactive interference scaling with model size (Wang & Sun 2025); recency and stimulus-statistics biases with entangled "superposed" representations and a suppress-then-read-out mechanism (Xiong et al. 2026); and U-shaped serial-position curves (Liu et al. 2024).
- **Latent maintenance without context access: largely absent** (Huang et al. 2025), unless state is externalized through chain-of-thought.
- **Fixed-state architectures (SSMs) do have hard capacity limits** that behave more like classical capacity-limited working memory (Arora et al. 2024a,b; Jelassi et al. 2024).
