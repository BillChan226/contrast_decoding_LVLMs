Dear reviewer 6SE4,

Thank you very much for your appreciation and constructive feedback on our paper! We would like to provide more explanations regarding the points raised in the weaknesses and questions sections.

### **Regarding Weakness 1**

Although the detector output is an important component in the proposed adaptive focal-contrast grounding algorithm, we would like to point out that rather than largely relying on the detector, HALC leverages the grounding area as an initialization for the subsequent FOV sampling. In fact, as discussed in the theoretical analysis in Section 5, we acknowledge that directly using detector output is not robust in practical cases due to its limitations, which motivates us to propose FOV sampling in the first place. More specifically, in cases where the object detection method does return completely irrelevant areas, next-token distributions from near-optimal visual context would still be approximated as FOV sampling was designed to cover from small to large areas in the original image. Since in practice, we leverage the following formula to derive the number of FOV samples:
$$
n_{\text{FOV}} = \left\lceil \frac{\log(\frac{A_{\text{image}}}{A_{\text{grounding}}})}{\log{(\lambda_{\text{exp}}+1})}  \right\rceil
$$
thus the largest sampled FOV $A_{\text{image}}$ covers the original image. Therefore according to Theorem 5.1, for any **bounded perturbation**, the near-optimal visual context can be approximated by the FOV sampling; and for the **worst-case unbounded perturbation** (which covers the case you mention that selects a completely irrelevant area), the original image will be used as the input visual context, which means that methods that take the original image as visual input (e.g. greedy decoding) will be the **lower-bound performance of HALC**. Here we also provide some empirical result to support the above analysis.




However, we appreciate the concern and would make sure to add more discussions in additional ablation studies to showcase these interesting cases in addition to the theoretical analysis.




### **Regarding Weakness 2**
We agree with the reviewer’s comment that models such as BLIP will struggle with detailed contents in the images, especially when the contents occupy smaller areas. However, the proposed similarity-based beam search global optimization in HALC is not intended to do that - with detailed subtly handled by local focal-contrast decoding already, the similarity-based beam search only aims to guide the generation on the high level and at the same time ensure linguistic quality of the generation. Since during local focal-contrast grouding, the JSD filter already selects the top-m most visually-grounded tokens candidates from the sampled FOVs, the global visual matching process mainly serves to ensure linguistic quality (e.g. fluency) that aligns with both the visual and textual input.

In other words, the superior performance of HALC, as reported in Table 1 and 2, are not largely dependent on the performance of BLIP. However, we appreciate the reviewer’s concern, and to more clearly illustrate our reasoning, we would like to report additional ablation study results on when different global selector is used in the similarity-based beam search.