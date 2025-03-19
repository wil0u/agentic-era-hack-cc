import os
import warnings

warnings.filterwarnings("ignore")
import vertexai
from vertexai.language_models import ChatModel
from langgraph.graph import StateGraph, END
from typing import Annotated
from typing_extensions import TypedDict

import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
import io
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import pandas as pd
from google.cloud import bigquery
from langchain_google_vertexai import ChatVertexAI
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages.ai import AIMessage
from langchain_core.messages import ToolMessage
import uuid
from typing import Union, Dict, Any
import json
from langchain_core.messages.base import BaseMessage
from langchain_core.tools import tool
from app.prompt_manager import PromptManager 
from langgraph.types import Command, interrupt
from langchain.agents import AgentExecutor, create_react_agent
from typing_extensions import TypedDict, Literal



# import mlflow


# PROJECT_ID = os.environ.get("GOOGLE_CLOUD_REGION", "europe-west1")
# LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "europe-west1")
import os

# Récupérer la variable d'environnement
PROJECT_ID = os.getenv('PROJECT_ID')

# Vérifier si la variable est définie
if PROJECT_ID is None:
    print("Error: PROJECT_ID environment variable is not set")
    exit(1)

print(f"The PROJECT_ID is: {PROJECT_ID}")

def query_result_summary(df):
    def get_info_as_string(df):
        buffer = io.StringIO()
        df.info(buf=buffer)
        return buffer.getvalue()

    formatted_summary = (
        f"Info: \n{get_info_as_string(df)}\n\n"
        f"Description: \n{df.describe(include='all').to_string(header=True).replace('#','')}\n\n"
        f"Valeurs nulles: \n{df.isnull().sum().to_string()}\n\n"
        f"Types de données: \n{df.dtypes.to_string()}\n\n"
        f"Valeurs uniques: \n{df.nunique().to_string()}\n\n"
    )

    return formatted_summary.replace("#","")


def query_result_preview(query_result_df, top_k=20):
    return query_result_df.head(top_k).to_markdown()


def format_model_or_dict(data: Union[BaseModel, Dict[str, Any]], use_ansi: bool = False) -> str:
    # Vérifier si data est un BaseModel
    is_pydantic = isinstance(data, BaseModel)
    data_dict = data.dict() if is_pydantic else data

    if use_ansi:
        # Formatage avec ANSI
        if is_pydantic:
            header = f"\033[1m{data.__class__.__name__}:\033[0m\n"
        elif "role" in data_dict:
            header = f"\n\n\033[1;34m=== {data_dict['role'].upper()} ===\033[0m\n"
        else:
            header = ""
        formatted_items = [f"  \033[1m{key}:\033[0m {value}" for key, value in data_dict.items()]
    else:
        # Formatage en texte brut
        if is_pydantic:
            header = f"{data.__class__.__name__}:\n"
        elif "role" in data_dict:
            header = f"\n\n=== {data_dict['role'].upper()} ===\n"
        else:
            header = ""
        formatted_items = [f"  {key}: {value}" for key, value in data_dict.items()]

    return header + "\n".join(formatted_items)


def stream_graph_updates(initial_state, config):
    # Print a preview of the first message (truncated)
    # first_message = initial_state["messages"][0]
    # print("\n--- First Message Preview ---")
    # print(first_message.content[:500] + ("..." if len(first_message.content) > 500 else ""))
    # print("\n----------------------------")

    # Print the rest of the messages normally
    for message in initial_state["messages"]:
        message.pretty_print()

    # Stream updates from the workflow
    for event in Campaign_Workflow.graph.stream(initial_state, config):
        for value in event.values():
            if type(value["messages"][-1]) is str:
                print(value["messages"][-1])
            else:
                value["messages"][-1].pretty_print()



from typing import Any, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict):
    user_input: Optional[str]
    campaign_brief: Optional[str]
    sql_query: Optional[str]
    sql_query_error: Optional[bool]
    query_result_json: Optional[str]
    stats: Optional[Any]  # modifier noms query_result_stats
    preview: Optional[Any]
    evaluation_result: Any
    adjust: Optional[Any]
    schema_metadata: Optional[Any]
    project_id: Optional[str]
    messages: Annotated[list, add_messages]
    user_feedback: str

class CampaignWorkflow:
    def __init__(self, model, hitl, schema_metadata, project_id, system=""):
        self.schema_metadata = schema_metadata
        self.project_id = project_id
        self.system = system
        self.cpt_loop = 0
        self.limit_loop = 3
        self.hitl=hitl #human in the loop
        # Graph
        self.prompt_manager = PromptManager()
        graph = StateGraph(AgentState)
        graph.add_node("routeur", self.routeur)
        graph.add_node("generate_campaign_strategy", self.generate_campaign_strategy)
        graph.add_node("build_query", self.build_query)
        graph.add_node("execute_query", self.execute_query)
        graph.add_node("evaluate_target", self.evaluate_target)
        graph.add_node("adjust_query", self.adjust_query)
        graph.add_node("final_answer", self.final_answer)

        graph.add_edge("generate_campaign_strategy", "build_query")
        graph.add_edge("build_query", "execute_query")
        graph.add_edge("final_answer", END)
        
        graph.add_conditional_edges(
            "execute_query",
            self.should_adjust_query_error,
            {0: "adjust_query", 1: "evaluate_target", 2: "final_answer"},
        )
        graph.add_conditional_edges(
            "evaluate_target",
            self.should_adjust,
            {True: "adjust_query", False: "final_answer"},
        )

        graph.add_edge("adjust_query", "execute_query")
        # graph.set_entry_point("generate_campaign_strategy")
        graph.set_entry_point("routeur")

        memory = MemorySaver()
        self.graph = graph.compile()
        self.model = model

    def execute_sql_query(query: str) -> str:
        """Exécute une requête SQL et retourne les résultats."""
        try:
            query_result_df = pd.read_gbq(query, project_id="mon-projet-gcp")
            return query_result_df.head(5).to_json(orient="records", indent=2)
        except Exception as e:
            return f"Erreur lors de l'exécution : {str(e)}"
    def routeur(
        self, state: AgentState
    ) -> Command[Literal["generate_campaign_strategy","final_answer"]]:
        class User_Intent(BaseModel):
            ambiguity: bool = Field(
                description="Indicates whether the input is ambiguous and requires clarification."
            )
            response_ambiguity: str = Field(
                description="If ambiguity True, provide a reasoning explaining why it's ambiguous, emphasizing that the goal is to determine an audience based on a marketing campaign."
            )
            generate_brief: bool = Field(
                description="Indicates whether a new campaign brief should be generated based on the user's input."
            )
            # build_query: bool = Field(
            #     description="Indicates whether the query should be built from scratch based on user input."
            # )
            # adjust_query: bool = Field(
            #     description="Indicates whether the existing query should be adjusted based on refined criteria."
            # )


        user_input = state["messages"][-1] if type(state["messages"][-1]) is HumanMessage else None
        print(user_input)
        system_prompt = self.prompt_manager.routeur_prompt()
        template = ChatPromptTemplate(
            [
                ("placeholder", "{messages}"),
                ("system", system_prompt),
            ]
        )
        
        structured_chain = template | self.model.with_structured_output(User_Intent)
        llm_output = structured_chain.invoke(
            {"messages": state["messages"], "user_input": user_input}
        )
        print(llm_output)
        if llm_output.ambiguity:
            state["messages"] = AIMessage(
                content=llm_output.response_ambiguity, name="Routeur_Agent")
            goto = "final_answer"
        else: 
            goto = "generate_campaign_strategy"
        
        # elif llm_output.build_query:
        #     goto = "build_query"
        # elif llm_output.adjust_query:
        #     goto = "adjust_query"
        return Command(
            update={"messages": state["messages"], "user_input": user_input},
            goto=goto,
        )

    def should_adjust(self, state: AgentState):
        return (
            state["evaluation_result"].should_adjust and self.cpt_loop < self.limit_loop
        )

    def should_adjust_query_error(self, state: AgentState):
        if self.cpt_loop >= self.limit_loop:
            return 2
        elif state["sql_query_error"] == True:
            return 0
        else:
            return 1

    def generate_campaign_strategy(self, state: AgentState):
        class CampaignBrief(BaseModel):
            brief: str = Field(
                description="A well-structured and precise campaign request in natural language."
            )

        user_input = state["messages"][-1] if type(state["messages"][-1]) is HumanMessage else None
        system_prompt = self.prompt_manager.generate_campaign_strategy_prompt()
        
        template = ChatPromptTemplate(
            [
                ("placeholder", "{messages}"),
                ("system", system_prompt),
            ]
        )

        structured_chain = template | self.model.with_structured_output(CampaignBrief)
        llm_output = structured_chain.invoke(
            {"messages": state["messages"], "schema_metadata": self.schema_metadata}
        )
        ai_message = AIMessage(
            content=format_model_or_dict(llm_output), name="Brief_Agent"
        )

        return {"campaign_brief": llm_output, "messages": [ai_message], "user_input": user_input}

    def build_query(self, state: AgentState):
        class SQLQueryOutput(BaseModel):
            sql_query: str

        system_prompt = self.prompt_manager.build_query_prompt()

        template = ChatPromptTemplate(
            [
                ("placeholder", "{messages}"),
                ("system", system_prompt),
            ]
        )

        structured_chain = template | self.model.with_structured_output(SQLQueryOutput)

        llm_output = structured_chain.invoke(
            {"messages": state["messages"], "schema_metadata":self.schema_metadata}
        )

        ai_message = AIMessage(
            content=format_model_or_dict(llm_output), name="Sql_Query_Builder_Agent"
        )

        return {"sql_query": llm_output, "messages": [ai_message]}

    def execute_query(self, state: AgentState):
        from langchain.schema import ChatMessage

        class EvaluationResult(BaseModel):
            should_adjust: bool = Field(
                description="Indicates whether the audience should be adjusted or not."
            )
            reasoning: str = Field(
                description="Agent's reasoning explaining the evaluation"
            )
        if isinstance(state["sql_query"], str):
            sql_query = state["sql_query"]
        else:
            sql_query = state["sql_query"].sql_query  
        try:
            query_result_df = pd.read_gbq(
                sql_query,
                project_id=self.project_id,
            )
            query_result_json = query_result_df.to_json(orient="records", indent=2)
            sql_query_error = False

            evaluation_result = EvaluationResult(
                should_adjust=False, reasoning="The query has executed properly."
            )

            stats = query_result_summary(query_result_df)
            preview = query_result_preview(query_result_df, top_k=20)

            tool_message = format_model_or_dict(
                {
                    "role": "tool",
                    "content": format_model_or_dict(
                        {
                            "executed_query": sql_query,
                            "stats": stats,
                            "preview": query_result_df.head(5).to_json(
                                orient="records", indent=2
                            ),
                        }
                    ),
                }
            )

        except Exception as e:
            error_message = str(e)
            print(f"Erreur attrapée : {error_message}")

            sql_query_error = True
            evaluation_result = EvaluationResult(
                should_adjust=True,
                reasoning="The query is not executable. Exception: " + error_message,
            )
            query_result_json = json.dumps({})
            stats = None
            preview = None

            tool_message = format_model_or_dict(
                {"role": "Error", "executed_query": sql_query, "error": error_message}
            )

        tool_message = AIMessage(
            content=tool_message, name="tool_call"
        )

        return {
            "messages": [tool_message],
            "sql_query_error": sql_query_error,
            "query_result_json": query_result_json,
            "evaluation_result": evaluation_result,
            "stats": stats,
            "preview": preview,
        }

    def evaluate_target(self, state: AgentState):
        class EvaluationResult(BaseModel):
            should_adjust: bool = Field(
                description="Indicates whether the audience should be adjusted or not."
            )
            reasoning: str = Field(
                description="Agent's reasoning explaining the evaluation"
            )

        query_result_df = pd.read_json(state["query_result_json"], orient="records")
        system_prompt = self.prompt_manager.evaluate_target_prompt()

        template = ChatPromptTemplate(
            [
                ("placeholder", "{messages}"),
                ("system", system_prompt),
            ]
        )

        structured_chain = template | self.model.with_structured_output(
            EvaluationResult
        )

        llm_output = structured_chain.invoke(
            {"messages": state["messages"], "schema_metadata": self.schema_metadata}
        )

        ai_message = AIMessage(
            content=format_model_or_dict(llm_output), name="Evaluator_Agent"
        )

        return {"evaluation_result": llm_output, "messages": [ai_message]}
    
    @tool
    def human_assistance(query: str) -> str:
        """Request assistance from a human."""
        human_response = interrupt({"query": query})
        return human_response["data"]
    
    @tool 
    def adjust_format(query: str):
        """
        Adjusts and improves the format of the provided SQL query.

        Parameters:
        ----------
        query : str
            The SQL query to be corrected and reformatted.

        Returns:
        -------
        AdjustedQuery :
            An object containing the corrected and improved SQL query.
        """
        class AdjustedQuery(BaseModel):
            sql_query: str = Field(
                description="The corrected SQL query after applying the suggested improvements."
            )

        return AdjustedQuery(sql_query=query) 
    def adjust_query(self, state: AgentState):
        class AdjustedQuery(BaseModel):
            sql_query: str = Field(
                description="The corrected SQL query after applying the suggested improvements."
            )
            reasoning: str = Field(
                description="Agent's reasoning explaining the adjusted query"
            )

        self.cpt_loop += 1
        if self.hitl: 
            system_prompt = self.prompt_manager.adjust_query_prompt_hitl()
            template = ChatPromptTemplate(
                [
                    ("placeholder", "{messages}"),
                    ("user", system_prompt),
                ]
            )

            tools=[self.human_assistance, self.adjust_format]
            agent = create_react_agent(self.model, tools, template)

            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,handle_parsing_errors=True,max_iterations=5,return_intermediate_steps=True)

            llm_output=agent_executor.invoke({"messages": state["messages"], "schema_metadata": self.schema_metadata
             })
            print("SQL Query Adjusted",llm_output[list(llm_output.keys())[-1]][-1][1])
            return {
                 "sql_query": llm_output[list(llm_output.keys())[-1]][-1][1]}
        else : 
            system_prompt = self.prompt_manager.adjust_query_prompt()
            template = ChatPromptTemplate([("placeholder", "{messages}"),("user", system_prompt),])
            structured_chain = template | self.model.with_structured_output(AdjustedQuery)
            llm_output = structured_chain.invoke({"messages": state["messages"], "schema_metadata": self.schema_metadata})
            ai_message = AIMessage(
                content=format_model_or_dict(llm_output), name="Adjuster_Agent"
            )
            return {"sql_query": llm_output,"messages": [ai_message],}


    def final_answer(self, state: AgentState):
        system_prompt = self.prompt_manager.final_answer_prompt()
        template = ChatPromptTemplate(
            [
                ("placeholder", "{messages}"),
                ("user", system_prompt),
            ]
        )

        if "evaluation_result" in state:
            evaluation_result = state["evaluation_result"]
        else:
            evaluation_result = "No evaluation result." 

        structured_chain = template | self.model
        llm_output = structured_chain.invoke(
            {
                "messages": state["messages"],
                "user_input": state["user_input"],
                "last_evaluation": evaluation_result,
            }
        )
        print(llm_output)
        self.cpt_loop = 0
        return {"messages": [llm_output]}



# mlflow.set_tracking_uri("http://localhost:5050")
# mlflow.set_experiment("analysis_test")
# # The logs are located in the "Traces" tab associated with the experiment "screener-Test
# mlflow.langchain.autolog()


with open("app/tables_metadata_4.json", "r", encoding="utf-8") as file:
    metadata = json.load(file)
metadata_str = json.dumps(metadata, indent=4, ensure_ascii=False)
metadata_str = metadata_str.replace("{", "{{").replace("}", "}}")


with open(
    "app/TheLookEcommerce-functionnal-description-v1.txt", "r", encoding="utf-8"
) as fichier:
    description = fichier.read()


metadata = f"{metadata_str} | {description}"
# metadata = f"{description}"


llm = ChatVertexAI(model="gemini-2.0-flash-001", temperature=0)


Campaign_Workflow = CampaignWorkflow(model=llm,hitl=False,project_id=PROJECT_ID,schema_metadata=metadata)

agent = Campaign_Workflow.graph