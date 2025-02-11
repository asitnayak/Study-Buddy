from colorama import Fore, Style
from sentence_transformers import CrossEncoder
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from pydantic import BaseModel, Field
from langchain import hub
from langgraph.graph.message import add_messages, BaseMessage
from typing import Annotated, Literal, Sequence, List
from typing_extensions import TypedDict
from langchain.tools import Tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph,MessagesState, START, END

from langgraph.prebuilt import tools_condition
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import ToolNode
import os

import warnings
warnings.filterwarnings("ignore", message=".*urllib3.*OpenSSL.*")
warnings.filterwarnings("ignore")

os.environ["OPENAI_API_KEY"] = "your_key"
os.environ["TAVILY_API_KEY"] = "your_key"

class my_graph(MessagesState):
  """
  Represents the state of our graph.

  Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
        messages: Annotated[Sequence[BaseMessage], add_messages]
    """

#   question : Annotated[List[str], add_messages]
  messages: Annotated[Sequence[BaseMessage], add_messages]
  generation : Annotated[List[str], add_messages]
  documents : List[str]

class chat_bot:
    def __init__(self):
        # define persistent directory
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.db_dir = os.path.join(self.current_dir, "db")
        self.persistent_directory = os.path.join(self.db_dir, "chroma_db_with_metadata")

        # define llm
        self.llm=ChatOpenAI(model='gpt-4o-2024-08-06')

        # define the embedding model
        self.embed = OpenAIEmbeddings()

        # load the existing vector store with the embedding function
        self.db = Chroma(persist_directory=self.persistent_directory,
                embedding_function=self.embed)

        self.retriever = self.db.as_retriever(
                                        search_type='similarity',
                                        search_kwargs={"k": 3},
                                    )
        
        self.retriever_tool = create_retriever_tool(self.retriever, name="ncert_retriever",
                                      description="Search and return information from NCERT book data."
                                      )
        # self.retriever_tool = Tool(
        #                 name="ncert_retriever",
        #                 func=self.retriever_tool_func,
        #                 description="Search and return information from NCERT book data."
        #                     )
        self.web_search_tool = TavilySearchResults(k=3)
        
        self.tools = [self.retriever_tool, self.web_search_tool]

        self.retrieve_node = ToolNode(self.tools)

        self.memory = MemorySaver()

        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        self.initial_route_msg = """
            You are an expert at routing a user question to the appropriate resource, either a vectorstore or a websearch.
            The vectorstore contains documents related to NCERT Class 11th Physics, Biology, and Chemistry textbooks. These documents are well-organized and include the content of these books, broken into smaller chunks for efficient retrieval. Use the vectorstore to answer questions specifically related to the topics covered in these textbooks.
            If the user question cannot be answered using the vectorstore (e.g., if it is outside the scope of the textbooks or requires current information), prioritize a web search to provide the most accurate and up-to-date answer.
            Always aim to provide a direct, accurate, and relevant response based on the chosen source.
            """
        
        self.grader_msg = """You are a grader assessing relevance of a retrieved document to a user question. \n
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question or not.
            """
        
        self.rewrite_system_msg = """You a question re-writer that converts an input question to a better version that is optimized \n
            for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning,
            and create a better elaborated version of the provided question.
            """
        
    def retriever_tool_func(self, query: str):
        # Directly use the retriever to get results
        results = self.retriever.get_relevant_documents(query)
        return results

    def route_question(self, state):
        """
            Route question to web search or RAG.
            Args:
                state (dict): The current graph state
            Returns:
                str: Next node to call
        """
        # print("---ROUTE QUESTION---")
        # print(state)
        # print("---ROUTE QUESTION---")
        question = state['messages']
        router_prompt = ChatPromptTemplate.from_messages(
                                                [
                                                    ('system', self.initial_route_msg),
                                                    ('human', '{question}')
                                                ]
                                            )
        question_router = self.llm.bind_tools(self.tools)
        chain = router_prompt | question_router
        response = chain.invoke({'question': question})

        return {"messages": [response]}
    
    def route_retrieval(self, state):
        """
            Route retrieved information to generation or grader.
            Args:
                state (dict): The current graph state
            Returns:
                str: Next node to call
        """
        # print("---ROUTE RETRIEVAL---")
        # print(state)
        # for i in state['messages']:
        #     print(i)
        # print("---ROUTE RETRIEVAL---")
        # if state['messages'][-1].additional_kwargs['tool_calls'][0]['function']['name'] == 'tavily_search_results_json':
        if state['messages'][-1].name == 'tavily_search_results_json':
            return 'generate'
        else:
            # tool_outputs = []
    
            # # Loop through the messages in the state
            # for msg in state['messages']:
            #     if isinstance(msg, ToolMessage) and msg.name == "ncert_retriever":
            #         tool_outputs.append(msg.content)  # Extract the tool's output
            
            # state['documents'] = tool_outputs
            return 'state_update_node'
        
    def state_update_node(self, state):
        """
            Initialises the documents key and updates the retrieved documents from messages key to documents key.
            Args:
                state (dict): The current graph state
            Returns:
                state (dict): Creates and updates documents key with filtered documents
        """
        # print("---STATE UPDATE---")
        # print(state)
        # print("---STATE UPDATE---")
        tool_outputs = []
    
        # Loop through the messages in the state
        for msg in state['messages']:
            if isinstance(msg, ToolMessage) and msg.name == "ncert_retriever":
                tool_outputs.append(msg.content)  # Extract the tool's output
            
        return {"documents": tool_outputs}
        
    def grader_node(self, state):
        """
            Determines whether the retrieved documents are relevant to the question.
            Args:
                state (dict): The current graph state
            Returns:
                state (dict): Updates documents key with only filtered relevant documents
        """
        # print("---GRADE DOCUMENTS---")
        # print(state)
        # print("---GRADE DOCUMENTS---")
        question = None
        for msg in reversed(state["messages"]):  # Start from the latest message
          if isinstance(msg, HumanMessage):
              question =  msg.content
              break
          
        docs = state['documents']
        # print(Fore.YELLOW + f"Total no. of retrieved docs = {len(docs)}" + Style.RESET_ALL)
        class grade_document(BaseModel):
            '''Binary score for relevance check on retreived documents.'''
            binary_score : str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

        structured_llm = self.llm.with_structured_output(grade_document)

        grader_prompt = ChatPromptTemplate.from_messages(
                                    [
                                        ('system', self.grader_msg),
                                        ('human', "Retrieved document: \n\n {document} \n\n User question: {question}")
                                    ]
                                )
        retrieval_grader = grader_prompt | structured_llm

        # Score each doc
        filtered_docs = []
        for idx, d in enumerate(docs):
            # print(f"DOC NUMBER : {idx+1}/{len(docs)}")
            score = retrieval_grader.invoke({"question": question, "document": d})
            grade = score.binary_score

            if grade == 'yes':
            #   print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)

            else:
            #   print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue

        return {'documents': filtered_docs}
    
    def decide_to_generate(self, state):
        """
        Determines whether to generate an answer, or re-generate a question.
        Args:
            state (dict): The current graph state
        Returns:
            str: Binary decision for next node to call
        """
        # print("---DECIDE TO GENERATE---")
        # print(state)
        # print("---DECIDE TO GENERATE---")
        filtered_docs = state['documents']

        if len(filtered_docs) == 0:
                # All documents have been filtered check_relevance
                # We will re-generate a new query
            # print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
            return "transform_query"
        else:
                # We have relevant documents, so generate answer
            # print("---DECISION: GENERATE---")
            return "reranker"
        
    def transform_query(self, state):
        """
            Transform the query to produce a better question.
            Args:
                state (dict): The current graph state
            Returns:
                state (dict): Updates question key with a re-phrased question
        """
        # print("---TRANSFORM QUERY---")
        # print(state)
        # print("---TRANSFORM QUERY---")
        question = state['question'][-1]

        rewrite_prompt = ChatPromptTemplate.from_messages(
                        [
                            ('system', self.rewrite_system_msg),
                            ('human', "Here is an initial question :\n{question}\n\nFormulate a more elaborated improved question.")
                        ]
                    )

        question_rewriter = rewrite_prompt | self.llm | StrOutputParser()

        # Re-write question
        better_question = question_rewriter.invoke({'question': question})

        return {'question': better_question}
        
    def reranker(self, state):
        """
        Re ranks the filtered documents before sending them for generation.
        Args:
            state (dict): The current graph state
        Returns:
            state (dict): Updates documents key with re ranked relevant documents
        """
        # print("---RE-RANKER---")
        # print(state)
        # print("---RE-RANKER---")
        filtered_docs = state['documents']

        question = None
        for msg in reversed(state["messages"]):  # Start from the latest message
          if isinstance(msg, HumanMessage):
              question =  msg.content
              break
          
        pairs = []
        for doc in filtered_docs:
            pairs.append([question, doc])

        scores = self.cross_encoder.predict(pairs)
        scored_docs = zip(scores, filtered_docs)
        reranked_document_cross_encoder = sorted(scored_docs, reverse=True)
        reranked_document_cross_encoder = [i[1] for i in reranked_document_cross_encoder]

        return {'documents':reranked_document_cross_encoder}

    
    def generate(self, state):
        """
            Generate answer
            Args:
                state (messages): The current state
            Returns:
                dict: The updated state with re-phrased question
        """
        # print("---GENERATE---")
        # print(state)
        # print("---GENERATE---")

        last_msg_tool = state['messages'][-1].name
        # print(Fore.YELLOW + f"last node : {last_msg_tool}" + Style.RESET_ALL)

        question = None

        for msg in reversed(state["messages"]):  # Start from the latest message
          if isinstance(msg, HumanMessage):
              question =  msg.content
              break

        # print(Fore.LIGHTGREEN_EX + question + Style.RESET_ALL)

        if last_msg_tool == 'tavily_search_results_json':
            web_results = []
            latest_tool_message = None
            for msg in reversed(state["messages"]):
                if isinstance(msg, ToolMessage):
                    # print(Fore.LIGHTCYAN_EX + f"YES TOOL MESSAGE" + Style.RESET_ALL)
                    latest_tool_message = msg
                    tool_data = json.loads(latest_tool_message.content)
                    for i in tool_data:
                      web_results.append(i['content'])
                    break
            final_doc = "\n\n".join([doc for doc in web_results])
            # print(Fore.YELLOW + f"WEB_results : {len(web_results)}" + Style.RESET_ALL)
            # print(Fore.YELLOW + f"final_doc : {len(final_doc)}" + Style.RESET_ALL)

        elif last_msg_tool == 'ncert_retriever':
            docs = state['documents']
            final_doc = "\n\n".join([doc for doc in docs])

        # print(Fore.LIGHTGREEN_EX + final_doc + Style.RESET_ALL)
        # if web_results == final_doc:
        #     print(Fore.RED + 'sadly yes' +Style.RESET_ALL)

        # prompt
        # prompt = hub.pull("rlm/rag-prompt")
        prompt = ChatPromptTemplate.from_template('''You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use one to three sentences maximum based on the question's requirement and keep the answer concise. If the question can be answered in a single sentence, then use few sentences as required. If the question needs a brief explanation, then streatch the answer till maximum five sentences.
                    Question: {question} 
                    Context: {context} 
                    Answer:
                ''')

        # llm
        # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True)

        # chain
        rag_chain = prompt | self.llm

        # run
        response = rag_chain.invoke({'context':final_doc, 'question': question})
        return {'messages':response}
        
    def __call__(self):
        workflow = StateGraph(my_graph)
        workflow.add_node("agent", self.route_question)
        workflow.add_node("tools", self.retrieve_node)
        workflow.add_node('generate', self.generate)
        workflow.add_node('grader', self.grader_node)
        workflow.add_node('state_update_node', self.state_update_node)
        workflow.add_node('reranker', self.reranker)
        workflow.add_node('transform_query', self.transform_query)

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", tools_condition)
        workflow.add_conditional_edges('tools', self.route_retrieval, {'generate':'generate', 'state_update_node':'state_update_node'})
        workflow.add_edge('state_update_node', 'grader')
        workflow.add_conditional_edges('grader', self.decide_to_generate, {'transform_query':'transform_query', 'reranker':'reranker'})
        workflow.add_edge('transform_query','agent')
        workflow.add_edge('reranker', 'generate')
        workflow.add_edge("generate", END)

        self.app = workflow.compile(checkpointer=self.memory)
        return self.app