from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from prod_assistant.prompt_library.prompts import PROMPT_REGISTRY, PromptType
#from prompt_library.prompts import PROMPT_REGISTRY, PromptType
from prod_assistant.retriever.retrieval import Retriever
from prod_assistant.utils.model_loader import ModelLoader
from prod_assistant.utils.model_loader import TavilyClientManager
from langgraph.checkpoint.memory import MemorySaver



class AgenticRAG:
    """Agentic RAG pipeline using LangGraph."""

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        rewrites: int # Count of rewrites done

    def __init__(self):
        self.retriever_obj = Retriever()
        self.model_loader = ModelLoader()
        self.llm = self.model_loader.load_llm()
        self.tavily_manager = TavilyClientManager()
        self.checkpointer = MemorySaver()
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

    # ---------- Helpers ----------
    def _format_docs(self, docs) -> str:
        if not docs:
            return "No relevant documents found."
        formatted_chunks = []
        for d in docs:
            meta = d.metadata or {}
            formatted = (
                f"Title: {meta.get('product_title', 'N/A')}\n"
                f"Price: {meta.get('price', 'N/A')}\n"
                f"Rating: {meta.get('rating', 'N/A')}\n"
                f"Reviews:\n{d.page_content.strip()}"
            )
            formatted_chunks.append(formatted)
        return "\n\n---\n\n".join(formatted_chunks)
    

    # ---------- Nodes ----------
    def _ai_assistant(self, state: AgentState):
        print("--- CALL ASSISTANT ---")
        messages = state["messages"]
        last_message = messages[-1].content

        if any(word in last_message.lower() for word in ["price", "review", "product"]):
            return {"messages": [HumanMessage(content="TOOL: retriever")]}
        else:
            prompt = ChatPromptTemplate.from_template(
                "You are a helpful assistant. Answer the user directly.\n\nQuestion: {question}\nAnswer:"
            )
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"question": last_message})
            return {"messages": [HumanMessage(content=response)]}

    def _vector_retriever(self, state: AgentState):
        print("--- RETRIEVER ---")
        query = state["messages"][-1].content
        retriever = self.retriever_obj.load_retriever()
        docs = retriever.invoke(query)

        if not docs:
            return {"messages": [HumanMessage(content="No relevant documents found.")]}

        context = self._format_docs(docs)
        return {"messages": [HumanMessage(content=context)]}
    
    def _tavily_search(self, state: AgentState):
        print("--- TAVILY SEARCH ---")
        client = self.tavily_manager.get_client()
        
        # Always take just the latest user query
        query = state["messages"][-1].content.strip()

        # Enforce Tavily’s max query length
        if len(query) > 400:
            print(f"Truncating query from {len(query)} to 400 chars for Tavily")
            query = query[:400]

        results = client.search(query=query, max_results=2)

        # Extract relevant text (Tavily returns JSON-like dicts)
        snippets = [r.get("content", "") for r in results.get("results", [])]
        return {"messages": [HumanMessage(content="\n".join(snippets))]}

    


    # def _grade_documents(self, state: AgentState) -> Literal["generator", "rewriter", "tavily"]:
    #     print("--- GRADER ---")
    #     question = state["messages"][0].content
    #     docs = state["messages"][-1].content

    #     # if not docs or "No relevant documents found." in docs:
    #     #     return "tavily"

    #     if state.get("rewrites", 0) >= 1:  # limit 1 rewrite
    #         print("Max rewrites reached → going to Tavily")
    #         return "tavily"

    #     prompt = PromptTemplate(
    #         template="""You are a grader. Question: {question}\nDocs: {docs}\n
    #         Are docs relevant to the question? Answer yes or no.""",
    #         input_variables=["question", "docs"],
    #     )
    #     chain = prompt | self.llm | StrOutputParser()
    #     score = chain.invoke({"question": question, "docs": docs})
    #     return "generator" if "yes" in score.lower() else "rewriter"

    # def _grade_documents(self, state: AgentState) -> Literal["generator", "rewriter", "tavily"]:
    #     print("--- GRADER ---")
    #     question = state["messages"][0].content
    #     docs = state["messages"][-1].content

    #     # Case 1: retriever returned nothing
    #     if not docs or "No relevant documents found." in docs:
    #         if state.get("rewrites", 0) >= 1:
    #             print("No docs even after rewrite → going to Tavily")
    #             return "tavily"
    #         else:
    #             print("No docs → try rewrite first")
    #             return "rewriter"

    #     # Case 2: docs exist → ask LLM to check relevance
    #     prompt = PromptTemplate(
    #         template="""You are a grader. Question: {question}\nDocs: {docs}\n
    #         Are docs relevant to the question? Answer yes or no.""",
    #         input_variables=["question", "docs"],
    #     )
    #     chain = prompt | self.llm | StrOutputParser()
    #     score = chain.invoke({"question": question, "docs": docs})

    #     print(f"Grader decision raw → {score}")
    #     normalized = score.strip().lower()

    #     if "yes" in normalized:
    #         return "generator"
    #     else:
    #         # If already rewritten, escalate to Tavily
    #         if state.get("rewrites", 0) >= 1:
    #             print("Docs irrelevant even after rewrite → Tavily")
    #             return "tavily"
    #         else:
    #             return "rewriter"
    def _grade_documents(self, state: AgentState) -> Literal["generator", "rewriter"]:
        print("--- GRADER ---")
        question = state["messages"][0].content
        docs = state["messages"][-1].content

        prompt = PromptTemplate(
            template="""You are a grader. Question: {question}\nDocs: {docs}\n
            Are docs relevant to the question? Answer yes or no.""",
            input_variables=["question", "docs"],
        )
        chain = prompt | self.llm | StrOutputParser()
        score = chain.invoke({"question": question, "docs": docs})
        return "generator" if "yes" in score.lower() else "rewriter"

    # def _generate(self, state: AgentState):
    #     print("--- GENERATE ---")
    #     question = state["messages"][0].content
    #     docs = state["messages"][-1].content
    #     prompt = ChatPromptTemplate.from_template(
    #         PROMPT_REGISTRY[PromptType.PRODUCT_BOT].template
    #     )
    #     chain = prompt | self.llm | StrOutputParser()
    #     response = chain.invoke({"context": docs, "question": question})
    #     return {"messages": [HumanMessage(content=response)]}
    def _generate(self, state: AgentState):
        print("--- GENERATE ---")
        question = state["messages"][0].content
        docs = state["messages"][-1].content
        prompt = ChatPromptTemplate.from_template(
            PROMPT_REGISTRY[PromptType.PRODUCT_BOT].template
        )
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"context": docs, "question": question})
        return {"messages": [HumanMessage(content=response)]}

    # def _rewrite(self, state: AgentState):
        # print("--- REWRITE ---")
        # question = state["messages"][0].content
        # new_q = self.llm.invoke(
        #     [HumanMessage(content=f"Rewrite the query to be clearer: {question}")]
        # )
        # return {"messages": [HumanMessage(content=new_q.content)]}
    

        # print("--- REWRITE ---")
        # question = state["messages"][0].content

        # # Increment rewrite count
        # new_q = self.llm.invoke(
        #     [HumanMessage(content=f"Rewrite the query to be clearer: {question}")]
        # )
        # return {
        #     "messages": [HumanMessage(content=new_q.content)],
        #     "rewrites": state.get("rewrites", 0) + 1,
        # }
    def _rewrite(self, state: AgentState):
        print("--- REWRITE ---")
        question = state["messages"][0].content
        new_q = self.llm.invoke(
            [HumanMessage(content=f"Rewrite the query to be clearer: {question}")]
        )
        return {"messages": [HumanMessage(content=new_q.content)]}
    
    # ---------- Build Workflow ----------
    # def _build_workflow(self):
    #     workflow = StateGraph(self.AgentState)
    #     workflow.add_node("Assistant", self._ai_assistant)
    #     workflow.add_node("Retriever", self._vector_retriever)
    #     workflow.add_node("Generator", self._generate)
    #     workflow.add_node("Rewriter", self._rewrite)
    #     workflow.add_node("TavilySearch", self._tavily_search)
        

    #     workflow.add_edge(START, "Assistant")
    #     workflow.add_conditional_edges(
    #         "Assistant",
    #         lambda state: "Retriever" if "TOOL" in state["messages"][-1].content else END,
    #         {"Retriever": "Retriever", END: END},
    #     )
    #     workflow.add_conditional_edges(
    #         "Retriever",
    #         self._grade_documents,
    #         {
    #             "generator": "Generator",
    #             "rewriter": "Rewriter",
    #             "tavily": "TavilySearch",
    #         },
    #     )
    #     workflow.add_edge("Generator", END)
    #     workflow.add_edge("Rewriter", "Assistant")
    #     workflow.add_edge("Rewriter", "TavilySearch")
    #     workflow.add_edge("TavilySearch", "Generator")
        
    #     return workflow

    # # ---------- Public Run ----------
    # def run(self, query: str, thread_id:str = "default_thread") -> str:
    #     """Run the workflow for a given query and return the final answer."""
    #     result = self.app.invoke({"messages": [HumanMessage(content=query)]}
    #                              config={"configurable":{"thread_id": thread_id}})

    #     return result["messages"][-1].content


    def _build_workflow(self):
        workflow = StateGraph(self.AgentState)
        workflow.add_node("Assistant", self._ai_assistant)
        workflow.add_node("Retriever", self._vector_retriever)
        workflow.add_node("Generator", self._generate)
        workflow.add_node("Rewriter", self._rewrite)

        workflow.add_edge(START, "Assistant")
        workflow.add_conditional_edges(
            "Assistant",
            lambda state: "Retriever" if "TOOL" in state["messages"][-1].content else END,
            {"Retriever": "Retriever", END: END},
        )
        workflow.add_conditional_edges(
            "Retriever",
            self._grade_documents,
            {"generator": "Generator", "rewriter": "Rewriter"},
        )
        workflow.add_edge("Generator", END)
        workflow.add_edge("Rewriter", "Assistant")
        return workflow

    # ---------- Public Run ----------
    def run(self, query: str,thread_id: str = "default_thread") -> str:
        """Run the workflow for a given query and return the final answer."""
        result = self.app.invoke({"messages": [HumanMessage(content=query)]},
                                 config={"configurable": {"thread_id": thread_id}})
        return result["messages"][-1].content

if __name__ == "__main__":
    rag_agent = AgenticRAG()
    answer = rag_agent.run("What is the price of iPhone 15?")
    print("\nFinal Answer:\n", answer)
