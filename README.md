# RAG (Retrieval Augmented Generation)

https://github.com/user-attachments/assets/2feb802e-71e0-445c-8c9e-05b6a4ef1caf


## 1. What is RAG?
Retrieval-augmented generation (RAG) is a technique for enhancing the accuracy and reliability of generative AI models with facts fetched from external sources.

- [best explanation of RAG](https://www.youtube.com/watch?v=u47GtXwePms)

benefits of RAG:

**building user trust**:
    Retrieval-augmented generation gives models sources they can cite, like footnotes in a research paper, so users can check any claims. That builds trust.

**clear up ambiguity in a user query**:
    If a user asks a question that could have multiple interpretations, a retrieval-augmented model can use the context of the conversation to clarify the question and provide a more accurate answer.

**reduce hallucination**:
    Generative models can sometimes generate text that is not accurate or relevant to the user query. By using retrieval-augmented generation, the model can use facts from external sources to ensure the generated text is accurate and relevant.

**the implementation process is relatively simple**:
    The implementation process for retrieval-augmented generation is relatively simple compared to other techniques for enhancing generative models, such as reinforcement learning or adversarial training.

## 2. What are Embeddings?

**What are embeddings**
Text embeddings are a natural language processing (NLP) technique that converts text into numerical coordinates (called vectors) that can be plotted in an n-dimensional space. This approach lets you treat pieces of text as bits of relational data, which we can then train models on.

**Example**

Embeddings capture semantic meaning and context which results in text with similar meanings having closer embeddings. For example, the sentence "I took my dog to the vet" and "I took my cat to the vet" would have embeddings that are close to each other in the vector space since they both describe a similar context.

**How the simularity is calculated**

Now that the embeddings are generated, create an embedding of the question, and compare it against the collection of embeddings in the dataframe.
The embedding of the question will be a vector (list of float values), which will be compared against the vector of the documents using the dot product. This vector returned from the API is already normalized. The dot product represents the similarity in direction between two vectors.
The values of the dot product can range between -1 and 1, inclusive. If the dot product between two vectors is 1, then the vectors are in the same direction. If the dot product value is 0, then these vectors are orthogonal, or unrelated, to each other. Lastly, if the dot product is -1, then the vectors point in the opposite direction and are not similar to each other.

## 3. How to run
Pre-requisites: docker , docker compose

1 - Clone the repo and cd into the directory
```bash 
    git clone https://github.com/s888n/simpleRAG.git
    cd simpleRAG
```

2 - Create a .env file and add GEMINI_API_KEY=your_api_key (you can get an api key from [here](https://aistudio.google.com/app/apikeys))

3 - run the app
```bash
    docker compose up
```
**That's it**: you can now access the app at http://localhost:8501


## stuff i should learn about

- [the math behind tokens to vectors](https://medium.com/@amallya0523/how-an-llm-understands-input-the-math-under-the-hood-114ac69f96c6)
- [Huggingface](https://huggingface.co/docs)
- optimal chunk size
- [good article](https://retrieval-tutorials.vercel.app/)

## remarks
 - NVIDIA uses LangChain in its reference architecture for retrieval-augmented generation.
- "RAGs can use any kind of database to store data. Vector stores are often discussed in the same literature as RAGs. If you have a large body of data that you want to query based on the LLM prompt, one way to do it is to use a ML model to transform text into N-dimensional vectors (called embeddings). This allows you to use cosine similarity to assess which pieces of text are close in semantic meaning to one another, and so retrieve relevant information to your RAG query." [reddit answer](https://www.reddit.com/r/MachineLearning/comments/1b5l18k/d_types_of_rag_implementations_and_their_benefits/')
## Resources

- [google gemini API](https://ai.google.dev/gemini-api/docs)
- [RAG Paper](https://arxiv.org/pdf/2005.11401)
- [RAG Blog from nvidia](https://ai.facebook.com/blog/retrieval-augmented-generation-of-human-like-text/)
- [Text Splitting](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb)
- [Langchain Docs](https://python.langchain.com/docs/introduction/)
- [Graph RAG](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/)


