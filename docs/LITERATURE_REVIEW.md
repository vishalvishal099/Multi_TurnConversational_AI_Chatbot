# Literature Review: Multi-Turn Conversational AI Systems with RAG

## Document-Grounded Question Answering for Customer Support

---

**Course:** Natural Language Processing Applications  
**Assignment:** Document-Based Question Answering System  
**Institution:** BITS Pilani  
**Date:** December 2025

---

## Abstract

This literature review examines the evolution and current state of multi-turn conversational AI systems, with a particular focus on Retrieval-Augmented Generation (RAG) pipelines and document-grounded dialogue systems. We survey key developments in large language models, dialogue management, intent recognition, and the integration of external knowledge sources. The review synthesizes findings from seminal papers including the Doc2Dial dataset, RAG architectures, and transformer-based language models, providing a theoretical foundation for building intelligent customer support chatbots. We conclude with an analysis of current challenges and future research directions in this rapidly evolving field.

**Keywords:** Conversational AI, RAG, Document-Grounded Dialogue, Multi-Turn Conversations, Large Language Models, Intent Recognition, Customer Support Systems

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Evolution of Conversational AI Systems](#2-evolution-of-conversational-ai-systems)
3. [Large Language Models for Dialogue](#3-large-language-models-for-dialogue)
4. [Retrieval-Augmented Generation (RAG)](#4-retrieval-augmented-generation-rag)
5. [Document-Grounded Dialogue Systems](#5-document-grounded-dialogue-systems)
6. [Multi-Turn Conversation Management](#6-multi-turn-conversation-management)
7. [Intent Recognition and Slot Filling](#7-intent-recognition-and-slot-filling)
8. [Evaluation Metrics and Benchmarks](#8-evaluation-metrics-and-benchmarks)
9. [Challenges and Future Directions](#9-challenges-and-future-directions)
10. [Conclusion](#10-conclusion)
11. [References](#11-references)

---

## 1. Introduction

Conversational AI systems have undergone a remarkable transformation over the past decade, evolving from simple rule-based chatbots to sophisticated neural dialogue systems capable of engaging in nuanced, context-aware conversations. The advent of large language models (LLMs) and retrieval-augmented generation techniques has particularly accelerated progress in document-grounded question answering systems, making them increasingly viable for real-world applications such as customer support.

This literature review provides a comprehensive examination of the key technologies, methodologies, and research contributions that underpin modern multi-turn conversational AI systems. We focus particularly on systems that combine the generative capabilities of LLMs with the factual grounding provided by external knowledge bases—an approach that addresses the critical challenge of hallucination while maintaining conversational fluency.

### 1.1 Scope and Objectives

The primary objectives of this review are to:

1. **Survey the evolution** of conversational AI from rule-based systems to neural approaches
2. **Examine RAG architectures** and their role in knowledge-grounded dialogue
3. **Analyze document-grounded dialogue** datasets and methodologies, with emphasis on Doc2Dial
4. **Review multi-turn dialogue management** techniques including coreference resolution and context tracking
5. **Discuss intent recognition** approaches for handling ambiguous user queries
6. **Identify current challenges** and promising research directions

### 1.2 Relevance to Customer Support Applications

Customer support represents an ideal application domain for conversational AI due to its:
- **Structured knowledge base** (product information, policies, FAQs)
- **Repetitive query patterns** amenable to automation
- **Multi-turn interaction requirements** for complex issue resolution
- **Measurable success metrics** (resolution rate, customer satisfaction)

---

## 2. Evolution of Conversational AI Systems

### 2.1 Rule-Based Systems (1960s-1990s)

The history of conversational AI begins with ELIZA (Weizenbaum, 1966), a pattern-matching system that simulated a Rogerian psychotherapist. Despite its simplicity, ELIZA demonstrated that even basic conversational interfaces could create compelling user interactions.

**Key Characteristics:**
- Pattern matching and keyword spotting
- Scripted response templates
- No learning capability
- Limited scalability

### 2.2 Statistical and Machine Learning Approaches (1990s-2010s)

The introduction of statistical methods brought data-driven approaches to dialogue systems. Hidden Markov Models (HMMs) and later Conditional Random Fields (CRFs) enabled more sophisticated language understanding.

**Notable Developments:**
- **AT&T's How May I Help You** (Gorin et al., 1997): One of the first deployed spoken dialogue systems
- **CMU Communicator** (Rudnicky et al., 1999): Multi-domain travel planning system
- **Statistical Dialogue Management** (Young et al., 2010): Partially Observable Markov Decision Processes (POMDPs) for dialogue policy optimization

### 2.3 Neural Dialogue Systems (2014-Present)

The deep learning revolution transformed conversational AI, beginning with sequence-to-sequence models (Sutskever et al., 2014) and attention mechanisms (Bahdanau et al., 2015).

**Paradigm Shifts:**

| Era | Approach | Key Innovation |
|-----|----------|----------------|
| 2014-2016 | Seq2Seq | End-to-end neural generation |
| 2017-2018 | Attention/Transformer | Long-range dependency modeling |
| 2018-2020 | Pre-trained LMs | Transfer learning (BERT, GPT) |
| 2020-Present | Large LLMs + RAG | Knowledge-grounded generation |

Vinyals and Le (2015) demonstrated that neural sequence-to-sequence models could generate conversational responses without explicit dialogue management rules, though early systems suffered from generic responses and factual inconsistencies.

---

## 3. Large Language Models for Dialogue

### 3.1 Transformer Architecture

The Transformer architecture (Vaswani et al., 2017) revolutionized NLP through its self-attention mechanism, enabling parallel processing of sequences and effective modeling of long-range dependencies.

**Key Components:**
- **Multi-Head Self-Attention:** Allows the model to attend to different positions and representation subspaces
- **Positional Encoding:** Injects sequence order information
- **Feed-Forward Networks:** Non-linear transformations per position
- **Layer Normalization:** Stabilizes training

The mathematical formulation of self-attention:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where Q, K, V represent queries, keys, and values, and $d_k$ is the dimension of the keys.

### 3.2 BERT and Bidirectional Understanding

BERT (Devlin et al., 2019) introduced bidirectional pre-training through masked language modeling (MLM) and next sentence prediction (NSP). This approach proved particularly effective for understanding tasks including:

- Intent classification
- Named entity recognition
- Semantic similarity
- Question answering

**Impact on Dialogue Systems:**
BERT's bidirectional representations enabled more nuanced understanding of user queries, particularly for disambiguation and coreference resolution. Chen et al. (2019) demonstrated that BERT significantly improved joint intent classification and slot filling accuracy.

### 3.3 Generative Pre-trained Transformers (GPT)

The GPT family (Radford et al., 2018, 2019; Brown et al., 2020) demonstrated that autoregressive language models, when scaled sufficiently, exhibit emergent capabilities including:

- In-context learning
- Chain-of-thought reasoning
- Instruction following
- Multi-turn dialogue

**GPT-3 and Few-Shot Learning:**
Brown et al. (2020) showed that large language models (175B parameters) could perform tasks with minimal examples, fundamentally changing the paradigm for building conversational systems. Rather than task-specific fine-tuning, systems could leverage prompting and in-context learning.

### 3.4 Open-Source LLMs: Mistral and Llama

The release of capable open-source models has democratized access to LLM technology:

**Mistral 7B (Jiang et al., 2023):**
- 7 billion parameters with performance exceeding larger models
- Sliding window attention for efficient long-context handling
- Apache 2.0 license enabling commercial use
- Strong instruction-following capability with Mistral-Instruct variant

**Llama 2 (Touvron et al., 2023):**
- Models from 7B to 70B parameters
- Trained on 2 trillion tokens
- RLHF-tuned chat variants
- Extensive safety evaluation

These models enable deployment of sophisticated conversational AI systems without proprietary API dependencies—a critical consideration for customer support applications with data privacy requirements.

---

## 4. Retrieval-Augmented Generation (RAG)

### 4.1 Motivation and Overview

Large language models, despite their impressive capabilities, suffer from several limitations when deployed for knowledge-intensive tasks:

1. **Hallucination:** Generation of plausible but factually incorrect information
2. **Knowledge Cutoff:** Training data has a temporal boundary
3. **Lack of Attribution:** Difficulty in tracing information sources
4. **Update Cost:** Retraining is computationally expensive

Retrieval-Augmented Generation (RAG) addresses these limitations by combining parametric knowledge (the LLM) with non-parametric knowledge (external retrieval).

### 4.2 RAG Architecture (Lewis et al., 2020)

Lewis et al. (2020) introduced the RAG framework, which combines a pre-trained seq2seq model with a dense retriever. The architecture consists of:

**Components:**
1. **Query Encoder:** Transforms user query into dense vector
2. **Document Index:** Pre-computed embeddings of knowledge base
3. **Retriever:** Finds relevant documents via similarity search
4. **Generator:** Produces response conditioned on query and retrieved context

**Formulation:**
$$P(y|x) = \sum_{z \in \text{top-k}(p(\cdot|x))} p(z|x) \cdot p(y|x, z)$$

where x is the input query, z are retrieved documents, and y is the generated response.

**RAG Variants:**
- **RAG-Sequence:** Marginalizes over documents for entire sequence
- **RAG-Token:** Marginalizes at each generation step

### 4.3 Dense Passage Retrieval

Karpukhin et al. (2020) demonstrated that learned dense representations significantly outperform traditional sparse methods (BM25) for open-domain question answering.

**Key Insights:**
- Dual-encoder architecture with separate query and passage encoders
- Contrastive learning with in-batch negatives
- Fine-tuning on question-passage pairs

**Embedding Models for RAG:**
| Model | Dimensions | Performance | Use Case |
|-------|------------|-------------|----------|
| all-MiniLM-L6-v2 | 384 | Good | Fast inference, limited resources |
| all-mpnet-base-v2 | 768 | Better | Balanced performance |
| text-embedding-ada-002 | 1536 | Best | Maximum quality |

### 4.4 Vector Databases for RAG

Efficient similarity search is critical for RAG performance. Modern vector databases provide:

- **Approximate Nearest Neighbor (ANN)** search algorithms
- **Persistent storage** with CRUD operations
- **Filtering** capabilities for metadata
- **Scalability** to millions of vectors

**ChromaDB** (used in our implementation) offers:
- Local persistence without external dependencies
- LangChain integration
- Simple Python API
- Suitable for small to medium knowledge bases

### 4.5 Chunking Strategies

How documents are segmented significantly impacts retrieval quality:

| Strategy | Description | Best For |
|----------|-------------|----------|
| Fixed-size | Split by character/token count | Uniform documents |
| Recursive | Split hierarchically (paragraphs → sentences) | Structured text |
| Semantic | Split by topic/meaning boundaries | Long documents |
| Sentence | One sentence per chunk | FAQ-style content |

Optimal chunk size balances:
- **Too small:** Loses context, retrieves fragments
- **Too large:** Dilutes relevance, exceeds context windows

---

## 5. Document-Grounded Dialogue Systems

### 5.1 The Document-Grounding Challenge

Document-grounded dialogue systems must generate responses that are:
1. **Faithful** to source documents
2. **Relevant** to the user query
3. **Coherent** as conversational turns
4. **Complete** in addressing user needs

This represents a more constrained generation task than open-domain dialogue, but one better suited for customer support where accuracy is paramount.

### 5.2 Doc2Dial Dataset and Framework

**Feng et al. (2020)** introduced Doc2Dial, a goal-oriented document-grounded dialogue dataset that has become a benchmark for this task.

**Dataset Characteristics:**
- **Domains:** 4 (DMV, VA, SSA, StudentAid)
- **Dialogues:** 4,793
- **Turns:** 44,149
- **Documents:** 488
- **Average Turns per Dialogue:** 14.3

**Annotation Approach:**
Doc2Dial employs a Wizard-of-Oz methodology where:
1. **User simulator** generates queries based on document content
2. **Agent** responds by grounding in specific document spans
3. **Multi-turn interactions** build toward goal completion

**Key Contributions:**
- First large-scale dataset for document-grounded goal-oriented dialogue
- Span-level grounding annotations
- Diverse dialogue phenomena (coreference, ellipsis, topic switches)

### 5.3 Dialogue Phenomena in Doc2Dial

The Doc2Dial dataset captures several important dialogue phenomena relevant to customer support:

**1. Coreference Resolution:**
```
User: "Tell me about the TechMart Pro Laptop"
Agent: [Laptop information]
User: "How much does IT cost?"  ← Pronoun references laptop
```

**2. Ellipsis:**
```
User: "What's the return policy for electronics?"
Agent: [Electronics return policy]
User: "And for clothing?"  ← Elliptical, inherits "return policy"
```

**3. Topic Switches:**
```
User: [Discussing product features]
User: "What if I need to return IT?"  ← Switches to returns while referencing product
```

**4. Clarification Requests:**
```
User: "My device isn't working"  ← Ambiguous
Agent: "Which device are you having issues with?"
```

### 5.4 Related Datasets

| Dataset | Domain | Dialogues | Grounding | Multi-Turn |
|---------|--------|-----------|-----------|------------|
| Doc2Dial | Government Services | 4,793 | Document spans | Yes |
| QuAC | Wikipedia | 14,000 | Passage | Yes |
| CoQA | Various | 8,000 | Passage | Yes |
| MultiWOZ | Task-Oriented | 10,000 | Database | Yes |
| ShARC | Rules | 32,000 | Rule trees | Limited |

### 5.5 Document-Grounded Generation Models

**Kim et al. (2021)** proposed methods for improving faithfulness in document-grounded generation:

1. **Extractive-then-Generative:** First identify relevant spans, then generate
2. **Copy Mechanisms:** Allow direct copying from source
3. **Faithfulness Rewards:** RL training with factual consistency metrics

**Rashkin et al. (2021)** introduced the AIS (Attributable to Identified Sources) framework for evaluating whether generated statements can be attributed to source documents.

---

## 6. Multi-Turn Conversation Management

### 6.1 Context Modeling Challenges

Multi-turn dialogue requires tracking and utilizing:
- **Entity mentions** across turns
- **User intent evolution**
- **Conversational state**
- **Implicit references**

The challenge intensifies with conversation length due to:
- Growing context window requirements
- Increased ambiguity in references
- Potential topic drift

### 6.2 Dialogue State Tracking

Dialogue State Tracking (DST) maintains a structured representation of the conversation:

**Slot-Value Representation:**
```json
{
  "intent": "order_status",
  "slots": {
    "order_id": "TM-2024-001234",
    "product": "Pro Laptop 15",
    "query_type": "tracking"
  },
  "history": ["product_inquiry", "order_status"]
}
```

**Approaches:**
1. **Classification-based:** Predict slot values from fixed vocabulary
2. **Span-extraction:** Extract values from user utterance
3. **Generative:** Generate slot values with seq2seq model

**TRADE (Wu et al., 2019)** introduced a transferable dialogue state tracker using copy mechanism, achieving strong performance on MultiWOZ.

### 6.3 Coreference Resolution in Dialogue

Coreference resolution—identifying when different expressions refer to the same entity—is critical for multi-turn understanding.

**Approaches:**
1. **Rule-based:** Recency heuristics, gender/number agreement
2. **Neural end-to-end:** Lee et al. (2017) span-ranking model
3. **BERT-based:** Joshi et al. (2019) achieved SOTA on OntoNotes

**Dialogue-Specific Challenges:**
- Deictic references ("this", "that")
- Zero anaphora (omitted subjects)
- Cross-speaker references

### 6.4 Context Compression and Selection

As conversations grow, managing context becomes critical:

**Strategies:**
1. **Truncation:** Keep only recent N turns
2. **Summarization:** Compress history into summary
3. **Retrieval:** Retrieve relevant past turns
4. **Hierarchical Encoding:** Separate utterance and dialogue-level representations

**Xu et al. (2021)** showed that selective context (retrieving relevant history) outperforms full context for long conversations.

---

## 7. Intent Recognition and Slot Filling

### 7.1 Joint Intent Classification and Slot Filling

For task-oriented dialogue, understanding requires both:
- **Intent Classification:** What does the user want? (e.g., "check_order_status")
- **Slot Filling:** What are the parameters? (e.g., order_id="TM-12345")

**Joint Modeling Benefits:**
- Shared representations capture dependencies
- Intent informs slot relevance
- Improved efficiency (single forward pass)

### 7.2 BERT for Intent and Slot Tasks

**Chen et al. (2019)** demonstrated BERT's effectiveness for joint intent classification and slot filling:

**Architecture:**
```
[CLS] Where is my order TM-12345 [SEP]
  ↓
BERT Encoder
  ↓
[CLS] → Intent Classifier → "order_status"
Tokens → Slot Tagger → O O O O B-ORDER_ID
```

**Results on ATIS/SNIPS:**
| Model | Intent Acc | Slot F1 | Sentence Acc |
|-------|------------|---------|--------------|
| Attention BiRNN | 91.1% | 95.2% | 78.9% |
| BERT | 97.5% | 96.1% | 88.2% |
| BERT + CRF | 97.9% | 96.5% | 88.6% |

### 7.3 Handling Ambiguous Queries

Ambiguous queries present significant challenges:

**Types of Ambiguity:**
1. **Lexical:** "Apple" (company vs. fruit)
2. **Syntactic:** "I saw the man with binoculars"
3. **Referential:** "It's not working" (what is "it"?)
4. **Intent:** "I need help" (under-specified)

**Strategies:**
1. **Confidence thresholding:** Request clarification below threshold
2. **Multi-intent detection:** Handle compound queries
3. **Contextual disambiguation:** Use conversation history
4. **Clarification generation:** Produce targeted follow-up questions

**Aliannejadi et al. (2019)** introduced asking clarifying questions in information-seeking conversations, showing improved task success rates.

### 7.4 Few-Shot and Zero-Shot Intent Recognition

Modern LLMs enable intent recognition without task-specific training:

**In-Context Learning:**
```
Classify the customer intent:

Examples:
"Where is my order?" → order_status
"I want to return this" → return_request
"How much is shipping?" → shipping_inquiry

Query: "Can I get a refund?"
Intent:
```

**Performance Comparison:**
| Approach | Training Data | Accuracy |
|----------|---------------|----------|
| Fine-tuned BERT | 10,000 examples | 97% |
| GPT-3 (few-shot) | 10 examples | 89% |
| GPT-4 (zero-shot) | 0 examples | 85% |

---

## 8. Evaluation Metrics and Benchmarks

### 8.1 Automatic Metrics

**Generation Quality:**
| Metric | Description | Limitations |
|--------|-------------|-------------|
| BLEU | N-gram overlap with reference | Ignores semantics |
| ROUGE | Recall-oriented overlap | Same as BLEU |
| BERTScore | Contextual embedding similarity | Computationally expensive |
| METEOR | Includes synonyms, stemming | Still surface-level |

**Retrieval Quality:**
| Metric | Description |
|--------|-------------|
| Recall@k | Fraction of relevant docs in top-k |
| MRR | Mean reciprocal rank of first relevant |
| nDCG | Normalized discounted cumulative gain |

**Dialogue-Specific:**
| Metric | Description |
|--------|-------------|
| Perplexity | Language model confidence |
| Distinct-n | Diversity of n-grams |
| Entity F1 | Accuracy of entity mentions |

### 8.2 Faithfulness Evaluation

For document-grounded systems, faithfulness is critical:

**AIS (Attributable to Identified Sources):**
- Human judges assess if each statement is supported by source
- Expensive but reliable

**Automatic Faithfulness:**
- **NLI-based:** Treat as entailment task
- **QA-based:** Check if source can answer questions about generation
- **Fact verification:** Extract and verify individual claims

### 8.3 Human Evaluation

Human evaluation remains the gold standard:

**Dimensions:**
1. **Fluency:** Is the response grammatical and natural?
2. **Relevance:** Does it address the query?
3. **Faithfulness:** Is it factually correct per documents?
4. **Helpfulness:** Does it resolve the user's need?
5. **Coherence:** Does it fit the conversation flow?

**Methodologies:**
- Likert scale ratings (1-5)
- Pairwise comparisons (A/B testing)
- Task completion rate
- User satisfaction surveys (CSAT, NPS)

### 8.4 Benchmark Datasets

| Benchmark | Task | Size | Metrics |
|-----------|------|------|---------|
| Doc2Dial | Document-grounded dialogue | 4.8K dialogues | F1, BLEU, Exact Match |
| MultiWOZ | Task-oriented dialogue | 10K dialogues | Joint Goal Accuracy |
| QuAC | Conversational QA | 14K dialogues | F1, HEQQ |
| OR-QuAC | Open-retrieval conversational QA | 14K dialogues | Recall, F1 |

---

## 9. Challenges and Future Directions

### 9.1 Current Challenges

**1. Hallucination and Faithfulness:**
Despite RAG, LLMs can still generate unfaithful content, particularly for:
- Numerical data
- Negations
- Complex reasoning chains

**2. Long-Context Handling:**
- Context window limitations
- Attention degradation over long sequences
- Computational costs of extended context

**3. Evaluation Gap:**
- Automatic metrics correlate poorly with human judgment
- Human evaluation is expensive and slow
- Task-specific metrics needed

**4. Personalization:**
- Adapting to individual user preferences
- Maintaining consistency across sessions
- Privacy considerations

**5. Multi-Modal Integration:**
- Incorporating images, documents, screenshots
- Cross-modal reasoning

### 9.2 Emerging Research Directions

**1. Retrieval-Augmented LLMs:**
- **RETRO (Borgeaud et al., 2022):** Retrieval at pre-training time
- **Atlas (Izacard et al., 2022):** Few-shot learning with retrieval
- Tighter integration of retrieval into generation

**2. Long-Context Models:**
- Extended context windows (GPT-4: 128K, Claude: 200K)
- Efficient attention mechanisms (FlashAttention, Linear Attention)
- Memory-augmented architectures

**3. Controllable Generation:**
- Style and tone control
- Faithful generation constraints
- Safety guardrails

**4. Agentic Systems:**
- Tool use and API calling
- Multi-step reasoning
- Autonomous task completion

**5. Evaluation Innovation:**
- LLM-as-judge paradigms
- Automated red-teaming
- Simulation-based evaluation

### 9.3 Industry Trends

**Customer Support AI:**
- Hybrid human-AI systems
- Proactive assistance
- Omnichannel integration
- Real-time personalization

**Enterprise Adoption:**
- On-premise LLM deployment
- Domain-specific fine-tuning
- Compliance and auditability
- Cost optimization

---

## 10. Conclusion

This literature review has surveyed the key technologies and research contributions underlying modern multi-turn conversational AI systems for document-grounded question answering. Several conclusions emerge:

### 10.1 Key Findings

1. **RAG architectures** effectively address LLM limitations by grounding generation in external knowledge, making them well-suited for customer support applications requiring factual accuracy.

2. **Document-grounded dialogue**, as exemplified by Doc2Dial, presents unique challenges including coreference resolution, ellipsis handling, and topic management that require specialized approaches beyond standard QA systems.

3. **Multi-turn conversation management** remains challenging, with current systems relying on combinations of dialogue state tracking, context compression, and history-aware prompting.

4. **Intent recognition** has achieved strong performance with BERT-based models, though handling ambiguous queries remains an open challenge requiring clarification strategies.

5. **Open-source LLMs** like Mistral 7B enable deployment of sophisticated systems without proprietary dependencies, democratizing access to this technology.

### 10.2 Implications for System Design

Based on this review, effective customer support chatbots should incorporate:

- **RAG pipeline** with dense retrieval and domain-specific chunking
- **Doc2Dial-inspired** multi-turn dialogue patterns
- **Hybrid intent recognition** combining classification with clarification
- **Consistent persona** through careful prompt engineering
- **Graceful degradation** when confidence is low or APIs unavailable

### 10.3 Future Outlook

The field continues to evolve rapidly. Key areas to watch include:

- Tighter integration of retrieval into LLM architectures
- Long-context models reducing need for retrieval in some cases
- Improved evaluation methodologies
- Agentic systems capable of multi-step task completion

The convergence of these advances promises increasingly capable and reliable conversational AI systems for customer support and beyond.

---

## 11. References

### Foundational Papers

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30.

2. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *Proceedings of NAACL-HLT 2019*, 4171-4186.

3. Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., ... & Amodei, D. (2020). Language models are few-shot learners. *Advances in Neural Information Processing Systems*, 33, 1877-1901.

### RAG and Retrieval

4. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems*, 33, 9459-9474.

5. Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S., ... & Yih, W. T. (2020). Dense passage retrieval for open-domain question answering. *Proceedings of EMNLP 2020*, 6769-6781.

6. Izacard, G., & Grave, E. (2021). Leveraging passage retrieval with generative models for open domain question answering. *Proceedings of EACL 2021*, 874-880.

### Document-Grounded Dialogue

7. Feng, S., Wan, H., Gunasekara, C., Patel, S., Joshi, S., & Lastras, L. (2020). doc2dial: A goal-oriented document-grounded dialogue dataset. *Proceedings of EMNLP 2020*, 8118-8128.

8. Feng, S., Patel, S., Wan, H., & Joshi, S. (2021). MultiDoc2Dial: Modeling dialogues grounded in multiple documents. *Proceedings of EMNLP 2021*, 6162-6176.

9. Kim, S., Joo, S. H., Kim, D., Shin, J., & Lee, K. (2021). Saving dense retriever from shortcut dependency in conversational search. *Proceedings of EMNLP 2021*, 10159-10170.

### Dialogue Systems

10. Budzianowski, P., Wen, T. H., Tseng, B. H., Casanueva, I., Ultes, S., Ramadan, O., & Gašić, M. (2018). MultiWOZ-a large-scale multi-domain wizard-of-oz dataset for task-oriented dialogue modelling. *Proceedings of EMNLP 2018*, 5016-5026.

11. Wu, C. S., Madotto, A., Hosseini-Asl, E., Xiong, C., Socher, R., & Fung, P. (2019). Transferable multi-domain state generator for task-oriented dialogue systems. *Proceedings of ACL 2019*, 808-819.

12. Hosseini-Asl, E., McCann, B., Wu, C. S., Yavuz, S., & Socher, R. (2020). A simple language model for task-oriented dialogue. *Advances in Neural Information Processing Systems*, 33, 20179-20191.

### Intent Recognition

13. Chen, Q., Zhuo, Z., & Wang, W. (2019). BERT for joint intent classification and slot filling. *arXiv preprint arXiv:1902.10909*.

14. Zhang, J. G., Hashimoto, K., Wu, C. S., Wan, Y., Yu, P. S., Socher, R., & Xiong, C. (2020). Find or classify? dual strategy for slot-value predictions on multi-domain dialog state tracking. *Proceedings of SEM-DIAL 2020*.

### Open-Source LLMs

15. Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., Casas, D. D. L., ... & Sayed, W. E. (2023). Mistral 7B. *arXiv preprint arXiv:2310.06825*.

16. Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., ... & Scialom, T. (2023). Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint arXiv:2307.09288*.

### Evaluation

17. Rashkin, H., Nikolaev, V., Lamm, M., Aroyo, L., Collins, M., Das, D., ... & Tomar, G. S. (2021). Measuring attribution in natural language generation models. *arXiv preprint arXiv:2112.12870*.

18. Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020). BERTScore: Evaluating text generation with BERT. *Proceedings of ICLR 2020*.

### Additional Resources

19. Gao, L., Ma, X., Lin, J., & Callan, J. (2022). Precise zero-shot dense retrieval without relevance labels. *arXiv preprint arXiv:2212.10496*.

20. Borgeaud, S., Mensch, A., Hoffmann, J., Cai, T., Rutherford, E., Millican, K., ... & Sifre, L. (2022). Improving language models by retrieving from trillions of tokens. *Proceedings of ICML 2022*, 2206-2240.

---

## Appendix A: Glossary of Terms

| Term | Definition |
|------|------------|
| **RAG** | Retrieval-Augmented Generation - combining retrieval with generation |
| **LLM** | Large Language Model - neural models with billions of parameters |
| **Doc2Dial** | Document-to-Dialogue dataset for grounded conversations |
| **DST** | Dialogue State Tracking - maintaining conversation state |
| **NLU** | Natural Language Understanding - comprehending user intent |
| **NLG** | Natural Language Generation - producing text responses |
| **Embedding** | Dense vector representation of text |
| **Vector Store** | Database optimized for similarity search |
| **Chunking** | Segmenting documents for retrieval |
| **Hallucination** | Generation of plausible but incorrect information |
| **Coreference** | When expressions refer to the same entity |
| **Ellipsis** | Omission of words recoverable from context |

---

## Appendix B: Dataset Statistics

### Doc2Dial Statistics

| Metric | Value |
|--------|-------|
| Total Dialogues | 4,793 |
| Total Turns | 44,149 |
| Avg Turns/Dialogue | 9.2 |
| Total Documents | 488 |
| Domains | 4 |
| Vocabulary Size | 21,356 |
| Avg Utterance Length | 12.8 words |

### Domain Distribution

| Domain | Dialogues | Documents |
|--------|-----------|-----------|
| DMV | 1,247 | 103 |
| VA Benefits | 1,201 | 127 |
| Social Security | 1,198 | 132 |
| Student Aid | 1,147 | 126 |

---

*Literature Review Version: 1.0*  
*Compiled: December 2025*  
*Course: Natural Language Processing Applications*  
*Institution: BITS Pilani*
