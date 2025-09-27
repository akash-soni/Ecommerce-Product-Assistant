import os 
from langchain_astradb import AstraDBVectorStore
from typing import List
from langchain_core.documents import Document
from prod_assistant.utils.config_loader import load_config
from prod_assistant.utils.model_loader import ModelLoader
from dotenv import load_dotenv  
import sys
from pathlib import Path
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.retrievers import ContextualCompressionRetriever
from prod_assistant.evaluation.ragas_eval import evaluate_context_precision, evaluate_response_relevancy

# Add the project root to the python path for direct script execution
# project_root = Path(__file__).resolve().parent[2]
# sys.path.insert(0, str(project_root))



class Retriever:
    """
    Class to handle retrieval of relevant documents from Astra DB using LangChain.
    """
    def __init__(self):
        """Initialize the Retrieval class by loading environment variables, configuration, and models."""
        print("Initializing Retrieval pipeline...")
        self.model_loader=ModelLoader()
        self.config=load_config()
        self._load_env_variables()
        self.vstore = None
        self.retriever_instance = None


    def _load_env_variables(self):
        """Load environment variables from a .env file.
        """
        load_dotenv()
         
        required_vars = ["GOOGLE_API_KEY", "ASTRA_DB_API_ENDPOINT", "ASTRA_DB_APPLICATION_TOKEN", "ASTRA_DB_KEYSPACE"]
        
        missing_vars = [var for var in required_vars if os.getenv(var) is None]
        
        if missing_vars:
            raise EnvironmentError(f"Missing environment variables: {missing_vars}")

        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.db_api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
        self.db_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        self.db_keyspace = os.getenv("ASTRA_DB_KEYSPACE")

        

    # def load_retriever(self):
    #     """Load the retriever from the vector store."""
    #     # if not self.vstore:
    #     #     collection_name = self.config["astra_db"]["collection_name"]
            
    #     #     self.vstore =AstraDBVectorStore(
    #     #         embedding= self.model_loader.load_embeddings(),
    #     #         collection_name=collection_name,
    #     #         api_endpoint=self.db_api_endpoint,
    #     #         token=self.db_application_token,
    #     #         namespace=self.db_keyspace,
    #     #         )
            
    #     # load top-k data stored in the vector store using retriever    
    #     # if not self.retriever:
    #     #     top_k = self.config["retriever"]["top_k"] if "retriever" in self.config else 3
    #     #     retriever=self.vstore.as_retriever(search_kwargs={"k": top_k})
    #     #     print("Retriever loaded successfully.")
    #     #     return retriever

    #     if not self.vstore:
    #         collection_name = self.config["astra_db"]["collection_name"]
            
    #         self.vstore =AstraDBVectorStore(
    #             embedding= self.model_loader.load_embeddings(),
    #             collection_name=collection_name,
    #             api_endpoint=self.db_api_endpoint,
    #             token=self.db_application_token,
    #             namespace=self.db_keyspace,
    #             )
    #     if not self.retriever:
    #         top_k = self.config["retriever"]["top_k"] if "retriever" in self.config else 3
            
    #         mmr_retriever=self.vstore.as_retriever(
    #             search_type="mmr",
    #             search_kwargs={"k": top_k,
    #                             "fetch_k": 30,
    #                             "lambda_mult": 0.7,
    #                             "score_threshold": 0.1
    #                            }
    #             # search_type="similarity_score_threshold",
    #             # search_kwargs={"k": top_k, "score_threshold": 0.5},
    #                            )
    #         print("Retriever loaded successfully.")
            
    #         llm = self.model_loader.load_llm()
            
    #         compressor=LLMChainFilter.from_llm(llm)
            
    #         self.retriever = ContextualCompressionRetriever(
    #             base_compressor=compressor, 
    #             base_retriever=mmr_retriever
    #         )
            
    #         return self.retriever

    def load_retriever(self):
        """_summary_
        """
        if not self.vstore:
            collection_name = self.config["astra_db"]["collection_name"]
            
            print(collection_name,"\n",self.db_keyspace,"\n",self.db_api_endpoint,"\n",self.db_application_token)

            self.vstore =AstraDBVectorStore(
                embedding= self.model_loader.load_embeddings(),
                collection_name=collection_name,
                api_endpoint=self.db_api_endpoint,
                token=self.db_application_token,
                namespace=self.db_keyspace,
                )
        if not self.retriever_instance:
            top_k = self.config["retriever"]["top_k"] if "retriever" in self.config else 3
            
            mmr_retriever=self.vstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": top_k,
                                "fetch_k": 20,
                                "lambda_mult": 0.7,
                                "score_threshold": 0.6
                               })
            print("Retriever loaded successfully.")
            
            llm = self.model_loader.load_llm()
            
            compressor=LLMChainFilter.from_llm(llm)
            
            self.retriever_instance = ContextualCompressionRetriever(
                base_compressor=compressor, 
                base_retriever=mmr_retriever
            )
            
        return self.retriever_instance


    def call_retriever(self,query):
        """Call the retriever to fetch relevant documents."""
        retriever=self.load_retriever()
        output=retriever.invoke(query)
        return output


# if __name__=='__main__':
#     retriever_obj = Retriever()
#     user_query = "Can you suggest me one plus phone?"
#     results = retriever_obj.call_retriever(user_query)

#     for idx, doc in enumerate(results, 1):
#         print(f"Result {idx}: {doc.page_content}\nMetadata: {doc.metadata}\n")

if __name__=='__main__':
    user_query = "Can you suggest good budget iPhone under 1,00,00 INR?"
    
    retriever_obj = Retriever()
    
    retrieved_docs = retriever_obj.call_retriever(user_query)
    
    def _format_docs(docs) -> list[dict]:
        if not docs:
            return []
        formatted_chunks = []
        for d in docs:
            meta = d.metadata or {}
            formatted = {
                "title": meta.get("product_title", "N/A"),
                "price": meta.get("price", "N/A"),
                "rating": meta.get("rating", "N/A"),
                "reviews": d.page_content.strip()
            }
            formatted_chunks.append(formatted)
        return formatted_chunks

    
    def docs_to_text(formatted_docs: list[dict]) -> list[str]:
        return [
            f"Title: {d['title']}\nPrice: {d['price']}\nRating: {d['rating']}\nReviews:\n{d['reviews']}"
            for d in formatted_docs
        ]


    
    #retrieved_contexts = [_format_docs(doc) for doc in retrieved_docs]
    #print(retrieved_contexts)

    formatted = _format_docs(retrieved_docs) 

    retrieved_contexts = docs_to_text(formatted)   
    
    print(retrieved_contexts)
    
    #this is not an actual output this have been written to test the pipeline
    response="iphone 16 plus, iphone 16, iphone 15, oneplus12, realme 12 are best phones under 1,00,000 INR."
    
    context_score = evaluate_context_precision(user_query,response,retrieved_contexts)
    relevancy_score = evaluate_response_relevancy(user_query,response,retrieved_contexts)

    
    
    print("\n--- Evaluation Metrics ---")
    print("Context Precision Score:", context_score)
    print("Response Relevancy Score:", relevancy_score)