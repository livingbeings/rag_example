# [Source](https://www.youtube.com/watch?v=Ylz779Op9Pw)
## Background of LLM
LLM compressed world knowledge
## LLM limitation :
1. Static world knowledge (limited to its training time)
2. Lack of specialized information 
## What is RAG ?
Augmenting LLM with specialized and mutable knowledge base\
Typical LLM:\
Prompt -> LLM -> Response\
RAG:\
User Query -> RAG Module -> Prompt -> LLM -> Response

In short : RAG Module is prompt generator

In theory this is better than fine tuning LLM because of time needed

## How it works
### Retriever
1. Text Embeddings\
mapping text information into vector where similarity is discribed by distance between its node
2. Knowledge base
- Load Docs
- Chunk Docs : important because LLM only have fixed `context window`
- Embed Chunks : take each of those chunks and translate it to vector inside embedding map
- Load into VDB (Vector Data Base) : search the similarity