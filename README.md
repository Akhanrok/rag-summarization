# Retrieval-Augmented Summarization of Scientific Articles

**The goal** of this project was to investigate how design choices in RAG pipelines affect the quality, faithfulness, and efficiency of summaries generated for long scientific documents, since the effectiveness of RAG strongly depends on parameters such as chunking strategy, retrieval depth, and reranking mechanisms. 

**The primary objective** of this study is to evaluate multiple configurations and determine which components most significantly impact the results. Specifically, the project investigates: how chunk size and overlap affect retrieval quality, the impact of cross-encoder reranking, entailment filtering, hierarchical summarization approaches, and trade-offs between retrieval depth and runtime.

**The questions** we aim to answer are the following:

1. How does text chunking strategy (size and overlap) influence the semantic grounding and coverage of generated summaries?
2. Does cross-encoder reranking significantly improve retrieval quality and downstream summary accuracy?
3. What is the impact of retrieval depth (Top-K) on summarization performance and hallucination?
4. Can entailment filtering using Natural Language Inference (NLI) improve factual faithfulness?
5. Does hierarchical summarization of retrieved passages improve coherence or degrade factual accuracy?
6. What are the performance trade-offs between retrieval quality and runtime efficiency?

## Methodology
The system implements a full Retrieval-Augmented Generation pipeline consisting of four stages:
- **stage 1: document chunking** (scientific articles from the arXiv Summarization Dataset (ccdv/arxiv-summarization) are segmented into overlapping text chunks with varying sizes and overlaps);
- **stage 2: semantic retrieval** (chunks were embedded using the sentence embedding model all-MiniLM-L6-v2 and retrieved through cosine similarity);
- **stage 3: cross-encoder reranking** (in some experiment configurations retrieved candidates were reranked using ms-marco-MiniLM-L-12-v2to improve relevance estimation);
- **stage 4: answer generation** (summaries were generated using the Microsoft Phi-2 model with prompts restricting the model to answer only using the retrieved context).

### Evaluation Metrics
The performance was evaluated using both summary quality metrics and faithfulness metrics:

**Text similarity metrics**

- ROUGE-1, ROUGE-2, ROUGE-L;
- BERTScore.

**Faithfulness metrics**

- sentence grounding (semantic similarity between summary sentences and retrieved passages, computed using cosine similarity between sentence embeddings produced by all-MiniLM-L6-v2);
- entailment score (the degree to which generated summary sentences are logically supported by the retrieved evidence passages, estimated using roberta-large-mnli);
- coverage (fraction of summary sentences that are supported by retrieved context, determined by measuring whether the cosine similarity (computed with embeddings from all-MiniLM-L6-v2) between a generated sentence and any retrieved passage exceeds a predefined threshold).

A composite score combining normalized grounding, coverage, and ROUGE-1 was used to compare configurations.



### Experimental Setup
Ten RAG configurations were evaluated, varying:

- chunk size (100–300 tokens);
- chunk overlap;
- retrieval depth;
- reranking;
- entailment filtering;
- hierarchical summarization.

The baseline configuration used:

- chunk size 150;
- overlap 50;
- top-200 retrieval;
- top-20 reranking.

## Results and Analysis

1. **Chunking strategy had the largest impact on performance** (the best configuration used smaller chunks (100 tokens) with higher overlap (75). This configuration achieved the highest composite score (0.748), suggesting that finer granularity improves retrieval precision by isolating semantically coherent passages. This result was partially expected: smaller chunks allow the retriever to match more specific semantic units instead of large paragraphs containing mixed information. High overlap further reduces the risk of splitting important concepts across chunk boundaries. As a result, retrieval becomes more precise, providing the generator with context that is more directly relevant to the query.).

2. **Cross-encoder reranking improved factual grounding** (removing reranking slightly increased ROUGE scores but reduced semantic grounding and coverage, indicating that cross-encoder models better identify context that truly supports the generated summary. This behavior is consistent with known retrieval dynamics. Embedding-based retrieval tends to favor passages that are lexically or topically similar, which can increase surface similarity to reference summaries and thus ROUGE. In contrast, cross-encoder models such as ms-marco-MiniLM-L-12-v2 evaluate the query and candidate passage jointly, allowing them to better capture semantic relevance and contextual relationships. This improves evidence quality even if lexical similarity decreases.).

3. **Retrieval depth exhibited diminishing returns** (increasing the retrieval pool from 200 to 400 candidates did not significantly improve quality but increased runtime. Dense retrievers typically rank the most relevant passages highly, so additional candidates mostly add lower-relevance passages, introducing noise without meaningful benefit. While in some tasks deeper retrieval can help capture rare relevant evidence, here Top-50 retained most relevant context, achieving the fastest runtime (~12.5 s) with only minor performance loss).

4. **Hierarchical summarization significantly increased computational cost** (hierarchical approaches increased runtime by up to 7× without consistently improving summary quality, suggesting limited practical benefit in this setup. This outcome was somewhat unexpected, as hierarchical summarization is often proposed as a solution for handling long contexts. A possible explanation is information loss during intermediate summarization steps: when retrieved passages are summarized before final generation, important details may be removed or distorted, reducing the usefulness of the resulting context).

5. **Entailment filtering improved factual consistency but not overall summary similarity** (although the entailment-filtered variant produced higher entailment scores, improvements in ROUGE and coverage were modest. This behavior suggests that entailment filtering helps remove irrelevant or weakly related passages, thereby improving logical consistency with the source document. However, it may also eliminate passages that contain useful contextual details but are not classified as strict entailments by the roberta-large-mnli, which can reduce the diversity of retrieved context and limit improvements in similarity-based metrics.).

**Key finding:** retrieval design decisions strongly influence the reliability of RAG systems. Among all tested factors, chunk granularity and overlap were the most critical parameters, as they directly determine the retriever’s ability to locate relevant evidence. Cross-encoder reranking further improves grounding by refining candidate relevance. Overall, the results demonstrate that careful retrieval configuration can significantly improve factual grounding in LLM-generated summaries without requiring larger language models.

## Discussion

**Key limitations:**

1. small test set;
2. model dependancy;
3. simplicity of the evaluation metrics;
4. limited hierarchical exploration;
5. static retrieval (dynamic or iterative retrieval strategies could yield different performance patterns). 

Despite the above limitations, the study demonstrates that retrieval design choices strongly influence the factual grounding of LLM-generated summaries. Key takeaways, such as the importance of chunk granularity, overlap, and cross-encoder reranking, are likely to generalize to similar retrieval-augmented summarization setups. The results also suggest that careful configuration can improve performance without requiring larger LLMs, highlighting practical strategies for efficient and reliable summarization pipelines. 

**Diresctions for future work:**

1. larger and more diverse datasets;
2. adaptive retrieval strategies;
3. hierarchical summarization optimization;
4. model and metric diversity.
