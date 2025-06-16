from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

# load fine-tuned model from hub
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# importing embedding model 
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = None
Settings.chunk_size = 256
Settings.chunk_overlap = 25


class RAGInstance:
    def __init__(self, documents_path:str):
        model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
        self.top_k = 3
        self.query_engine = self.build_query_engine(documents_path=documents_path)
        # self.model = self.load_model(model_name=model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.instruction_string = ""

    def build_query_engine(self,documents_path:str):
        documents = SimpleDirectoryReader(documents_path).load_data()
        # Preprocess
        for doc in documents:
            if "Member-only story" in doc.text:
                documents.remove(doc)
                continue
            
            if "The Data Entrepreneurs" in doc.text:
                documents.remove(doc)

            if " min read" in doc.text:
                documents.remove(doc)
        index = VectorStoreIndex.from_documents(documents)
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=self.top_k
        )
        return RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
        )

    def load_model(self,model_name:str):
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     device_map="auto",
                                                     trust_remote_code=False,
                                                     revision="main"
                                                     )
        
        config = PeftConfig.from_pretrained("shawhin/shawgpt-ft")
        return PeftModel.from_pretrained(model, "shawhin/shawgpt-ft")

    def generate_context(self,query:str):
        context = "Context:\n"
        response = self.query_engine.query(query)
        for i in range(self.top_k):
            context = context + response.source_nodes[i].text + "\n\n"
        return context
    
    def prompt_template(self,query:str, with_rag=False):
        if not with_rag:
            return  f'''[INST] {self.instruction_string}\nPlease respond to the following comment. \n {query} \n[/INST]'''
        context = self.generate_context(query=query) 
        return  f'''[INST] {self.instruction_string}\n\n{context}\nPlease respond to the following comment. Use the context above if it is helpful.\n{query} \n[/INST]'''
    
    def generate_response(self,prompt:str):
        self.model.eval()
        inputs = self.tokenizer(prompt,return_tensors="pt")
        outputs = self.model.generate(input_ids = inputs["input_ids"].to("cuda"),
                                      max_new_tokens=280)
        return self.tokenizer.batch_decode(outputs)[0]

if __name__ == "__main__":
    rag_instance = RAGInstance(documents_path="./articles")
    rag_instance.instruction_string = f"""ShawGPT, functioning as a virtual data science consultant on YouTube, communicates in clear, accessible language, escalating to technical depth upon request. 
    It reacts to feedback aptly and ends responses with its signature '–ShawGPT'. 
    ShawGPT will tailor the length of its responses to match the viewer's comment, providing concise acknowledgments to brief expressions of gratitude or feedback,
    thus keeping the interaction natural and engaging.
    """
    query = "What is fat-tailedness?"

    # Without RAG
    prompt = rag_instance.prompt_template(query=query)
    print(rag_instance.generate_response(prompt))

    # WIth RAG
    prompt = rag_instance.prompt_template(query=query,with_rag=True)
    print(rag_instance.generate_response(prompt))