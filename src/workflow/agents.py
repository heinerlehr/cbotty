import time
import asyncio
from dataclasses import dataclass

from loguru import logger

import graphviz

from typing import Optional, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_core.prompts import PromptTemplate

from langchain_openai import ChatOpenAI
from langgraph.runtime import Runtime
from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool

from langchain.agents import create_agent


from langchain_tavily import TavilySearch

# Configuration management
from utils.Configuration import Configuration

from chromatools.initdb import querydb


###########################################################################################################
#
# HELPER FUNCTIONS
#
###########################################################################################################

@tool
async def search_web(query: str, num_results: int = 5) -> str:
    """Search the web for information. Useful for current events, news, or general information not in your knowledge base."""
    search_tool = TavilySearch(
        max_results=num_results,
        include_raw_content=False
    )
    
    try:
        results = await search_tool.ainvoke(query)
        # Format results nicely
        formatted = []
        for result in results['results']:
            formatted.append(f"Title: {result['title']}\nContent: {result['content']}\nURL: {result['url']}\n")
        return "\n".join(formatted)
    except Exception as e:
        return f"Search failed: {str(e)}"

def create_prompt_template(prompt_config: dict) -> PromptTemplate:
    """Create a PromptTemplate from config"""
    return PromptTemplate(
        template=prompt_config["template"],
        input_variables=prompt_config["variables"]
    )

###########################################################################################################
#
# States
#
###########################################################################################################
class State(MessagesState):
    question: Optional[str]
    department: Optional[str]
    sentiment: Optional[str]
    response: Optional[str]

@dataclass
class SharedResources:
    """Shared resources available to all agents"""
    system_prompt: str
    config: Configuration
    departments: list[str]
    creative_llm: ChatOpenAI
    sober_llm: ChatOpenAI
    graph: StateGraph

    @classmethod
    def create(cls, config: Configuration) -> "SharedResources":
        """Factory method to create shared resources"""
        # departments
        if not (departments := config('syntheticgen', 'departments')):
            logger.error("No departments found in configuration for SharedResources.")
            raise ValueError("No departments found in configuration for SharedResources.")

        creative_llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        sober_llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

        system_prompt = config('prompts', 'system_prompt', 'template')

        # Build the graph (you'll need to implement this method)
        draw_graph = config.get('draw_graph', False)
        graph = cls.get_graph(draw_graph=draw_graph)

        return cls(
            system_prompt=system_prompt,
            config=config,
            departments=list(departments.keys()),
            creative_llm=creative_llm,
            sober_llm=sober_llm,
            graph=graph,
        )
    
    @classmethod
    def get_graph(cls, draw_graph: bool = True) -> StateGraph:
        """Build and compile the main workflow graph"""
        # You can move your graph building logic here
        graph = StateGraph(MessagesState, context_schema=ContextSchema)
        
        # Add nodes
        graph.add_node("get_sentiment", get_sentiment)
        graph.add_node("department_router", department_router)
        graph.add_node("handle_request", handle_request)
        graph.add_node("customer_agent", customer_agent)

        # Add edges
        graph.add_edge(START, "get_sentiment")
        graph.add_conditional_edges("get_sentiment", 
                                   react_to_sentiment,
                                   ["customer_agent","department_router"])
        graph.add_edge("department_router", "handle_request")
        graph.add_edge("handle_request", END)
        graph.add_edge("customer_agent", END)       

        graph = graph.compile()

        # Get a LangGraph Graph object
        g = graph.get_graph(xray=True)

        try:
            g.draw_mermaid_png(output_file_path="graph.png")
        except Exception:
            # If mermaid isn't available
            dot = graphviz.Digraph()

            # The .nodes attribute is usually a list of node names (strings)
            for node in g.nodes:
                dot.node(str(node), label=str(node))

            # The .edges attribute is typically a list of Edge objects with .source and .target
            for edge in g.edges:
                # Each edge should have .source and .target attributes
                source = getattr(edge, "source", None)
                target = getattr(edge, "target", None)
                if source is not None and target is not None:
                    dot.edge(str(source), str(target))

            # Render to PNG file
            dot.render("graph", format="png", cleanup=True)
        print("Graph written to graph.png")

        return graph


class ContextSchema(TypedDict):
    resources: SharedResources

###########################################################################################################
#
# AGENTS
#
###########################################################################################################

async def get_sentiment(state: State, runtime: Runtime[ContextSchema]) -> State:
    config = runtime.context.config
    llm = runtime.context.creative_llm
    prompt_template = create_prompt_template(config('prompts', 'sentiment_analysis'))
    sentiments = config('sentiments')
    question = state['messages'][-1].content
    prompt = prompt_template.format_prompt(
        question=question,
        sentiments=sentiments
    ).to_string().strip()
    response = await llm.ainvoke(prompt)
    sentiment = response.content.strip().lower()
    return {"question": question, "sentiment": sentiment}

async def react_to_sentiment(state: State, runtime: Runtime[ContextSchema]) -> State:
    # Agent to determine which department should handle the query
    sentiment = state.get("sentiment", "neutral")
    match sentiment:
        case "negative":
            # Send to human support
            return "customer_agent"
        case "neutral" | "positive":
            # Continue with normal routing
            return "department_router"
        case _:
            # Handle unexpected sentiment values
            return "customer_agent"

async def department_router(state: State, runtime: Runtime[ContextSchema]) -> str:
    config = runtime.context.config
    llm = runtime.context.creative_llm

    departments = runtime.context.departments
    if "department" in state.keys() and state['department']:
        previous_department = f"""
        In the last conversation this department was chosen {state['department']}. 
        Please consider routing this question to the same department""".strip()
    else:
        previous_department = ""
    conversation_history = "\n".join([f"{msg.type}: {msg.content}" for msg in state['messages']])

    prompt_template = create_prompt_template(config('prompts', 'department_classification'))
    prompt = prompt_template.format_prompt(
        question=state['question'],
        departments=departments,
        previous_department=previous_department,
        conversation_history=conversation_history
    ).to_string().strip()

    response = await llm.ainvoke(prompt)
    
    if (department := response.content.strip()) not in departments:
        department = "general"
    return {"department": department}

async def customer_agent(state: State, runtime: Runtime[ContextSchema]) -> State:
    # Agent to redirect customer to human support
    config = runtime.context.config
    llm = runtime.context.creative_llm
    prompt_template = create_prompt_template(config('prompts', 'Customer_service_agent'))
    sentiment = state.get("sentiment", "neutral")
    # Use the full conversation:
    conversation_history = "\n".join([f"{msg.type}: {msg.content}" for msg in state['messages']])

    prompt = prompt_template.format_prompt(
        question=state['question'],
        sentiment=sentiment,
        conversation_history=conversation_history
    ).to_string().strip()

    response = await llm.ainvoke(prompt)
    final_response = response.content.strip()
    return {'response': final_response}

async def handle_request(state: State, runtime: Runtime[ContextSchema]) -> State:
    # Main agent to handle the customer request in dependence of the department

    config = runtime.context.config

    department = state.get("department", "general")

    llm = runtime.context.creative_llm

    # Retrieve context from vector store
    question = state['question']
    docs = querydb(config=config, query=question, nresults=5)
    context = "\n\n".join([doc.page_content for doc in docs])
    # Use the full conversation:
    conversation_history = "\n".join([f"{msg.type}: {msg.content}" for msg in state['messages']])

    tools = [search_web]
    agent = create_agent(
        model = llm,
        tools=tools
    )
    system_prompt_template = create_prompt_template(config('prompts', f'{department}_system_prompt'))
    system_prompt = system_prompt_template.format_prompt(
    ).to_string().strip()

    prompt_template = create_prompt_template(config('prompts', f'{department}_agent'))
    prompt = prompt_template.format_prompt(
        question=question,
        context=context,
        conversation_history=conversation_history
    ).to_string().strip()

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ]

    response = await agent.ainvoke({"messages": messages})
    # Extract the actual text content
    if hasattr(response, 'content'):
        final_response = response.content
    elif isinstance(response, dict) and 'messages' in response:
        final_response = response['messages'][-1].content
    else:
        final_response = str(response)

    if "CUSTOMER_AGENT" in final_response:
        return Send("customer_agent", state)

    return {'response': final_response}

###########################################################################################################
#
# Convenience methods
#
###########################################################################################################
class ConversationSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages = []
        self.context = None
    
    def add_message(self, message):
        self.messages.append(message)
    
    def get_messages(self):
        return self.messages.copy()

# Global session storage
sessions = {}

async def invoke(question:str, context: SharedResources, session_id:str):
    """ Invokes the processing graph. """

    # Get or create session
    if session_id not in sessions:
        sessions[session_id] = ConversationSession(session_id)
        sessions[session_id].add_message(SystemMessage(content=context.system_prompt))
    
    session = sessions[session_id]

    # Create state with session history
    session.add_message(HumanMessage(content=question))
    state = State(messages=session.get_messages())

    graph = context.graph
    # Stream the workflow execution
    async for chunk in graph.astream(input=state, context=context):
        # chunk contains the state updates from each node
        yield chunk

        # Extract and store bot response
        if 'handle_request' in chunk and chunk['handle_request'] and 'response' in chunk['handle_request']:
            bot_response = chunk['handle_request']['response']
            session.add_message(AIMessage(content=bot_response))

if __name__ == "__main__":

    async def main():

        session_id = str(time.time())
        config = Configuration()
        context = SharedResources.create(config=config)

        # Streaming execution
        async for update in invoke(question="What's the weather in Boston?", context=context, session_id=session_id):
            # Process each streaming update
            print(f"Update received: {update}")

        async for update in invoke(question="And in New York?", context=context, session_id=session_id):
            # Process each streaming update
            print(f"Update received: {update}")
    
    asyncio.run(main())