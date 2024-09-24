# RAG (Retrieval Augmented Generation)


https://github.com/user-attachments/assets/2feb802e-71e0-445c-8c9e-05b6a4ef1caf


## 1. What is RAG?

- [best explanation of RAG](https://www.youtube.com/watch?v=u47GtXwePms)

## 2. What are Embeddings?

**What are embeddings**
Text embeddings are a natural language processing (NLP) technique that converts text into numerical coordinates (called vectors) that can be plotted in an n-dimensional space. This approach lets you treat pieces of text as bits of relational data, which we can then train models on.

**Example**

Embeddings capture semantic meaning and context which results in text with similar meanings having closer embeddings. For example, the sentence "I took my dog to the vet" and "I took my cat to the vet" would have embeddings that are close to each other in the vector space since they both describe a similar context.

**How the simularity is calculated**

Now that the embeddings are generated, let's create a Q&A system to search these documents. You will ask a question about hyperparameter tuning, create an embedding of the question, and compare it against the collection of embeddings in the dataframe.
The embedding of the question will be a vector (list of float values), which will be compared against the vector of the documents using the dot product. This vector returned from the API is already normalized. The dot product represents the similarity in direction between two vectors.
The values of the dot product can range between -1 and 1, inclusive. If the dot product between two vectors is 1, then the vectors are in the same direction. If the dot product value is 0, then these vectors are orthogonal, or unrelated, to each other. Lastly, if the dot product is -1, then the vectors point in the opposite direction and are not similar to each other.

## 3 How to run
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

## Resources

- [google gemini API](https://ai.google.dev/gemini-api/docs)
- [about RAG](https://arxiv.org/abs/2005.11401)
- [the paper](https://arxiv.org/pdf/2005.11401)
- [text splitting strategies](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb)
- [NVIDIA article](https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation)

