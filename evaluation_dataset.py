EVALUATION_QUESTIONS = [
    {
        "question": "What is retrieval augmented generation?",
        "reference_answer": "Retrieval augmented generation is a technique that retrieves relevant external information and provides it as context to a language model so the generated answer is more grounded and up to date.",
        "expected_sources": ["wiki_retrieval-augmented_generation.txt", "article_retrieval-augmented-generation.txt", "article_state-of-rag-genai.txt"],
    },
    {
        "question": "What is a vector database?",
        "reference_answer": "A vector database stores embeddings and retrieves semantically similar records using vector similarity search.",
        "expected_sources": ["wiki_vector_database.txt"],
    },
    {
        "question": "What is prompt engineering?",
        "reference_answer": "Prompt engineering is the practice of designing and refining prompts to guide an AI model toward useful and accurate outputs.",
        "expected_sources": ["wiki_prompt_engineering.txt"],
    },
    {
        "question": "How do transformers use attention?",
        "reference_answer": "Transformers use attention mechanisms to weigh relationships between tokens, allowing each token to use information from other relevant tokens in the sequence.",
        "expected_sources": ["wiki_attention_(machine_learning).txt", "wiki_transformer_(deep_learning_architecture).txt"],
    },
    {
        "question": "What is a large language model?",
        "reference_answer": "A large language model is a neural network trained on large text corpora to perform language tasks such as generation, summarization, translation, and question answering.",
        "expected_sources": ["wiki_large_language_model.txt"],
    },
    {
        "question": "What are the benefits of RAG?",
        "reference_answer": "RAG can improve factual grounding, provide access to fresh or domain-specific information, reduce hallucinations, and make answers easier to verify through retrieved sources.",
        "expected_sources": ["wiki_retrieval-augmented_generation.txt", "article_retrieval-augmented-generation.txt", "article_state-of-rag-genai.txt"],
    },
    {
        "question": "What are open source RAG frameworks?",
        "reference_answer": "Open source RAG frameworks are software tools that help developers build retrieval augmented generation systems by combining retrieval, indexing, orchestration, and evaluation components.",
        "expected_sources": ["article_best-open-source-rag-frameworks.txt"],
    },
    {
        "question": "What is deep learning?",
        "reference_answer": "Deep learning is a subset of machine learning that uses multilayer neural networks to learn representations and solve tasks such as classification, regression, and generation.",
        "expected_sources": ["wiki_deep_learning.txt"],
    },
    {
        "question": "What is artificial intelligence?",
        "reference_answer": "Artificial intelligence is the capability of computational systems to perform tasks associated with human intelligence, such as reasoning, learning, perception, and decision making.",
        "expected_sources": ["wiki_artificial_intelligence.txt"],
    },
    {
        "question": "How does Top K retrieval work?",
        "reference_answer": "Top K retrieval ranks candidate passages by similarity to a query and returns the K highest scoring passages as context for the generation model.",
        "expected_sources": ["wiki_retrieval-augmented_generation.txt", "wiki_vector_database.txt"],
    },
    {
        "question": "Why can a high Top-K value hurt a RAG system?",
        "reference_answer": "A high Top-K value can add irrelevant context, increase prompt length, raise latency, increase cost, and make the generated answer less focused.",
        "expected_sources": ["wiki_retrieval-augmented_generation.txt", "article_state-of-rag-genai.txt", "article_retrieval-augmented-generation.txt"],
    },
    {
        "question": "Why can a low Top-K value hurt answer quality?",
        "reference_answer": "A low Top-K value can retrieve too little evidence, causing the model to miss important context and produce incomplete or insufficient answers.",
        "expected_sources": ["wiki_retrieval-augmented_generation.txt", "article_retrieval-augmented-generation.txt"],
    },
    {
        "question": "How does RAG reduce hallucinations?",
        "reference_answer": "RAG reduces hallucinations by grounding the language model response in retrieved external documents instead of relying only on the model parameters.",
        "expected_sources": ["wiki_retrieval-augmented_generation.txt", "article_retrieval-augmented-generation.txt", "article_state-of-rag-genai.txt"],
    },
    {
        "question": "Why does RAG help with fresh information?",
        "reference_answer": "RAG helps with fresh information because the system can retrieve updated external documents at query time without retraining the language model.",
        "expected_sources": ["wiki_retrieval-augmented_generation.txt", "article_retrieval-augmented-generation.txt"],
    },
    {
        "question": "What is semantic similarity search?",
        "reference_answer": "Semantic similarity search retrieves items whose meanings are close to the query by comparing their vector embeddings rather than matching exact keywords only.",
        "expected_sources": ["wiki_vector_database.txt", "wiki_retrieval-augmented_generation.txt"],
    },
    {
        "question": "What role do embeddings play in vector databases?",
        "reference_answer": "Embeddings represent text or other data as numerical vectors, allowing vector databases to compare items by distance or similarity in vector space.",
        "expected_sources": ["wiki_vector_database.txt"],
    },
    {
        "question": "What is the difference between retrieval and generation in RAG?",
        "reference_answer": "Retrieval finds relevant documents or passages from a knowledge source, while generation uses those passages as context to produce a natural language answer.",
        "expected_sources": ["wiki_retrieval-augmented_generation.txt", "article_retrieval-augmented-generation.txt"],
    },
    {
        "question": "Why can retrieved context improve answer transparency?",
        "reference_answer": "Retrieved context improves transparency because users can inspect or cite the sources used to produce the answer and verify whether the response is grounded.",
        "expected_sources": ["wiki_retrieval-augmented_generation.txt", "article_state-of-rag-genai.txt"],
    },
    {
        "question": "What is multi-head attention?",
        "reference_answer": "Multi-head attention uses several attention heads in parallel so a transformer can capture different relationships or relevance patterns between tokens.",
        "expected_sources": ["wiki_attention_(machine_learning).txt", "wiki_transformer_(deep_learning_architecture).txt"],
    },
    {
        "question": "Why are transformers important for large language models?",
        "reference_answer": "Transformers are important for large language models because their attention-based architecture can process long token sequences efficiently and learn contextual relationships.",
        "expected_sources": ["wiki_transformer_(deep_learning_architecture).txt", "wiki_large_language_model.txt"],
    },
    {
        "question": "What is chain-of-thought prompting?",
        "reference_answer": "Chain-of-thought prompting is a prompt engineering technique that encourages a language model to produce intermediate reasoning steps before giving a final answer.",
        "expected_sources": ["wiki_prompt_engineering.txt", "wiki_large_language_model.txt"],
    },
    {
        "question": "How can prompt engineering affect LLM output?",
        "reference_answer": "Prompt engineering can affect LLM output by changing the instructions, examples, structure, or constraints given to the model, which can improve relevance and accuracy.",
        "expected_sources": ["wiki_prompt_engineering.txt"],
    },
    {
        "question": "What are common components of a RAG pipeline?",
        "reference_answer": "A RAG pipeline commonly includes document ingestion, chunking, embedding generation, vector indexing, retrieval, prompt construction, and answer generation.",
        "expected_sources": ["article_best-open-source-rag-frameworks.txt", "article_retrieval-augmented-generation.txt", "wiki_retrieval-augmented_generation.txt"],
    },
    {
        "question": "Why is evaluation important for RAG systems?",
        "reference_answer": "Evaluation is important for RAG systems because it measures retrieval quality, answer relevance, faithfulness, latency, and the tradeoff between accuracy and performance.",
        "expected_sources": ["article_best-open-source-rag-frameworks.txt", "wiki_retrieval-augmented_generation.txt"],
    },
    {
        "question": "What does factual grounding mean in RAG?",
        "reference_answer": "Factual grounding means that the generated answer is supported by retrieved documents or passages rather than only by the language model's internal knowledge.",
        "expected_sources": ["wiki_retrieval-augmented_generation.txt", "article_retrieval-augmented-generation.txt"],
    },
    {
        "question": "Why can RAG reduce the need to retrain an LLM?",
        "reference_answer": "RAG can reduce the need to retrain an LLM because new information can be added to the external knowledge base and retrieved at query time.",
        "expected_sources": ["wiki_retrieval-augmented_generation.txt"],
    },
    {
        "question": "What is context precision in RAG evaluation?",
        "reference_answer": "Context precision evaluates whether the retrieved contexts are relevant to the question and useful for producing the answer.",
        "expected_sources": ["article_best-open-source-rag-frameworks.txt"],
    },
    {
        "question": "What is context recall in RAG evaluation?",
        "reference_answer": "Context recall evaluates whether the retrieval step found the necessary information or expected sources needed to answer the question.",
        "expected_sources": ["article_best-open-source-rag-frameworks.txt"],
    },
    {
        "question": "What does faithfulness mean in RAG evaluation?",
        "reference_answer": "Faithfulness measures whether the generated answer is supported by the retrieved context and does not introduce unsupported claims.",
        "expected_sources": ["article_best-open-source-rag-frameworks.txt", "wiki_retrieval-augmented_generation.txt"],
    },
    {
        "question": "Why can answer relevancy be different from faithfulness?",
        "reference_answer": "Answer relevancy checks whether the response addresses the question, while faithfulness checks whether the response is supported by the retrieved context.",
        "expected_sources": ["article_best-open-source-rag-frameworks.txt"],
    },
    {
        "question": "What is approximate nearest neighbor search?",
        "reference_answer": "Approximate nearest neighbor search finds vectors that are close to a query vector efficiently, trading exactness for speed in large vector spaces.",
        "expected_sources": ["wiki_vector_database.txt"],
    },
    {
        "question": "Why are vector embeddings useful for text retrieval?",
        "reference_answer": "Vector embeddings are useful because they encode semantic meaning numerically, enabling retrieval of text passages that are similar in meaning to a query.",
        "expected_sources": ["wiki_vector_database.txt", "wiki_natural_language_processing.txt"],
    },
    {
        "question": "What is a knowledge graph?",
        "reference_answer": "A knowledge graph represents entities and their relationships in a structured graph form that can support search, reasoning, and knowledge representation.",
        "expected_sources": ["wiki_knowledge_graph.txt"],
    },
    {
        "question": "How can knowledge graphs complement RAG?",
        "reference_answer": "Knowledge graphs can complement RAG by adding structured relationships between entities, which can improve retrieval, reasoning, and source organization.",
        "expected_sources": ["wiki_knowledge_graph.txt", "wiki_retrieval-augmented_generation.txt"],
    },
    {
        "question": "What is natural language processing?",
        "reference_answer": "Natural language processing is a field of artificial intelligence focused on enabling computers to process, analyze, and generate human language.",
        "expected_sources": ["wiki_natural_language_processing.txt"],
    },
    {
        "question": "How is NLP related to RAG?",
        "reference_answer": "NLP is related to RAG because RAG uses language understanding, semantic retrieval, and text generation to answer natural language questions.",
        "expected_sources": ["wiki_natural_language_processing.txt", "wiki_retrieval-augmented_generation.txt"],
    },
    {
        "question": "What is representation learning in deep learning?",
        "reference_answer": "Representation learning is the process by which a model learns useful internal features or representations of data for downstream tasks.",
        "expected_sources": ["wiki_deep_learning.txt"],
    },
    {
        "question": "Why are neural networks important in deep learning?",
        "reference_answer": "Neural networks are important in deep learning because they use multiple layers to learn complex patterns and representations from data.",
        "expected_sources": ["wiki_deep_learning.txt"],
    },
    {
        "question": "What is self-attention?",
        "reference_answer": "Self-attention is an attention mechanism where tokens in the same sequence attend to each other to build contextualized representations.",
        "expected_sources": ["wiki_attention_(machine_learning).txt", "wiki_transformer_(deep_learning_architecture).txt"],
    },
    {
        "question": "Why does attention help with long-range dependencies?",
        "reference_answer": "Attention helps with long-range dependencies because it allows a model to directly connect and weight distant tokens in a sequence.",
        "expected_sources": ["wiki_attention_(machine_learning).txt", "wiki_transformer_(deep_learning_architecture).txt"],
    },
    {
        "question": "What is the role of a context window in LLMs?",
        "reference_answer": "The context window defines how much text an LLM can consider at once when processing input and generating output.",
        "expected_sources": ["wiki_large_language_model.txt", "wiki_attention_(machine_learning).txt"],
    },
    {
        "question": "Why can larger context increase computational cost?",
        "reference_answer": "Larger context increases computational cost because the model must process more tokens, and attention computations can become more expensive as input length grows.",
        "expected_sources": ["wiki_attention_(machine_learning).txt", "wiki_transformer_(deep_learning_architecture).txt"],
    },
    {
        "question": "What are hallucinations in large language models?",
        "reference_answer": "Hallucinations are generated statements that sound plausible but are false, unsupported, or not grounded in the provided information.",
        "expected_sources": ["wiki_large_language_model.txt", "wiki_retrieval-augmented_generation.txt"],
    },
    {
        "question": "Why is source verification useful in AI answers?",
        "reference_answer": "Source verification is useful because it lets users check whether an answer is supported by retrieved evidence and identify possible inaccuracies.",
        "expected_sources": ["wiki_retrieval-augmented_generation.txt", "article_state-of-rag-genai.txt"],
    },
    {
        "question": "What is the purpose of chunking documents in RAG?",
        "reference_answer": "Chunking splits documents into smaller passages so retrieval can find focused pieces of relevant context instead of sending entire documents to the model.",
        "expected_sources": ["article_retrieval-augmented-generation.txt", "article_best-open-source-rag-frameworks.txt"],
    },
    {
        "question": "Why does chunk size matter in RAG?",
        "reference_answer": "Chunk size matters because very small chunks may lose context, while very large chunks may include noise and increase prompt length.",
        "expected_sources": ["article_retrieval-augmented-generation.txt", "article_best-open-source-rag-frameworks.txt"],
    },
    {
        "question": "What is the purpose of overlapping chunks?",
        "reference_answer": "Overlapping chunks preserve context near chunk boundaries so important information is less likely to be split away from related text.",
        "expected_sources": ["article_retrieval-augmented-generation.txt", "article_best-open-source-rag-frameworks.txt"],
    },
    {
        "question": "Why can duplicate or redundant retrieved chunks be harmful?",
        "reference_answer": "Duplicate or redundant chunks can waste context space, reduce source diversity, increase latency, and make the prompt less informative.",
        "expected_sources": ["wiki_retrieval-augmented_generation.txt", "article_best-open-source-rag-frameworks.txt"],
    },
    {
        "question": "What is the tradeoff between precision and recall in retrieval?",
        "reference_answer": "Precision favors retrieving mostly relevant passages, while recall favors retrieving as many necessary passages as possible; increasing K often improves recall but can reduce precision.",
        "expected_sources": ["wiki_retrieval-augmented_generation.txt", "article_best-open-source-rag-frameworks.txt"],
    },
    {
        "question": "Why is latency important in a RAG system?",
        "reference_answer": "Latency is important because users need timely answers, and larger retrieval sets or prompts can make the system slower and more costly.",
        "expected_sources": ["article_state-of-rag-genai.txt", "wiki_retrieval-augmented_generation.txt"],
    },
]


def get_questions():
    return [item["question"] for item in EVALUATION_QUESTIONS]


def get_reference_map():
    return {item["question"]: item for item in EVALUATION_QUESTIONS}
