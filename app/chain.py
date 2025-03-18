# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_google_vertexai import ChatVertexAI
# from google.auth import load_credentials_from_file
# from google.cloud import aiplatform
# from langchain_core.output_parsers.openai_tools import PydanticToolsParser
# LOCATION = "europe-west3"
# LLM = "gemini-1.5-flash-002"

# llm = ChatVertexAI(
#     model_name=LLM,
#     location=LOCATION,
#     temperature=0,
#     max_output_tokens=1024,
# )

# template = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a knowledgeable culinary assistant specializing in providing"
#             "detailed cooking recipes. Your responses should be informative, engaging, "
#             "and tailored to the user's specific requests. Include ingredients, "
#             "step-by-step instructions, cooking times, and any helpful tips or "
#             "variations. If asked about dietary restrictions or substitutions, offer "
#             "appropriate alternatives.",
#         ),
#         MessagesPlaceholder(variable_name="messages"),
#     ]
# )

# chain = template | llm

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
# import mlflow


PROJECT_ID = "retd-283409"
LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "europe-west1")

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
    user_input: str
    campaign_brief: Optional[str]
    sql_query: Optional[str]
    sql_query_error: Optional[bool]
    query_result_json: Optional[str]
    stats: Optional[Any]  # modifier noms query_result_stats
    preview: Optional[Any]
    evaluation_result: Optional[Any]
    adjust: Optional[Any]
    messages: Annotated[list, add_messages]

class CampaignWorkflow:
    def __init__(self, model, schema_metadata, project_id ):
        self.schema_metadata = schema_metadata
        self.project_id = project_id
        self.cpt_loop = 0
        self.limit_loop = 3

        # Graph
        graph = StateGraph(AgentState)

        
        graph.add_node("generate_campaign_strategy", self.generate_campaign_strategy)
        graph.add_node("build_query", self.build_query)
        graph.add_node("execute_query", self.execute_query)
        graph.add_node("evaluate_target", self.evaluate_target)
        graph.add_node("adjust_query", self.adjust_query)
        graph.add_node("final_answer", self.final_answer)
        graph.add_edge("generate_campaign_strategy", "build_query")
        graph.add_edge("build_query", "execute_query")
        graph.add_edge("final_answer", END)

        # graph.add_node("router", self.router)
        # graph.add_conditional_edges("router", self.redo_campaign_strategy
        #     ,{True : "generate_campaign_strategy", False :END})

        # graph.add_conditional_edges("router", self.redo_adjust_strategy
        #     ,{True : "adjust_strategy", False :END})

        graph.add_conditional_edges(
            "execute_query",
            self.should_adjust_query_error,
            {0: "adjust_query", 1: "evaluate_target", 2: "final_answer"},
        )
        graph.add_conditional_edges(
            "evaluate_target", self.should_adjust, {True: "adjust_query", False: "final_answer"}
        )

        graph.add_edge("adjust_query", "execute_query")
        graph.set_entry_point("generate_campaign_strategy")
        memory = MemorySaver()
        self.graph = graph.compile()
        self.model = model

    def should_adjust(self, state: AgentState):
        return (
            state["evaluation_result"].should_adjust
            and self.cpt_loop < self.limit_loop
        )

    def should_adjust_query_error(self, state: AgentState):
        if self.cpt_loop >= self.limit_loop:
            return 2
        elif state["sql_query_error"]==True:
            print("return 0")
            return 0
        else:
            print("return 1")
            return 1

    def generate_campaign_strategy(self, state: AgentState):
        class CampaignBrief(BaseModel):
            brief: str = Field(
                description="A well-structured and precise campaign request in natural language."
            )


        user_input = state["messages"][-1] if type(state["messages"][-1]) is HumanMessage else None

        system_prompt = """You are a marketing campaign strategist for an e-commerce site.  
    Your task is to generate a **clear and precise brief** to define an audience for a targeted campaign.  

    ### Guidelines:
    - **Focus Only on Targeting**: If the request is not related to audience segmentation, inform the user that this is outside the scope.
    - **Ensure Coherence with the Dataset**: Base the brief on the structure and logic of the following schema metadata.
    - The primary objective is to retrieve `user_id`, along with any other relevant columns specified by the user request 
    Don't hesitate to use data from all the available tables


    Use the provided table information and schema to ensure the campaign brief is based on real data.
    ### Metadata Structure: 
    - The tables are qualified with a dataset (e.g., dataset.table). In this case, the dataset is named "thelookecommerce".
    - Tables in the dataset are identified by the `table_id` field.
    - Each table has a `schema` field listing its columns, including `name`, `type`, `key`, and potential relationships.
    The provided schema metadata is as follows:
    {schema_metadata}
    ### Clarity & Precision:
    The generated brief should be well-structured and explicitly define the targeted audience based on the dataset schema and the user input.    """

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
        
        return {"campaign_brief": llm_output, "messages": [ai_message],"user_input":user_input}

    def build_query(self, state: AgentState):
        class SQLQueryOutput(BaseModel):
            sql_query: str

        system_prompt = """You are a BigQuery SQL expert specialized in marketing analytics. Given a marketing campaign brief and an initial user request, your task is to generate a valid SQL query that specifically targets and extracts the right profiles for the marketing campaign.

        **Instructions:**

        - Use the provided schema metadata to determine the appropriate column types before constructing conditions.
        - Retrieve `user_id` and other relevant columns specified by the user request.
        - Ensure the query filters and extracts user profiles based on the criteria relevant to the campaign. The goal is to identify users who match the target audience.
        - If a column has predefined categorical values (e.g., an enumeration or a fixed set of possible values), ensure that user input is correctly mapped to the corresponding stored values before applying filters.
        - Wrap each column name in backticks (`) to denote them as delimited identifiers.
        - Select only the necessary columns based on the user request; never use `SELECT *`.
        - Always include `user_id` in the selected columns. 
        - Ensure that column values used in conditions match the actual data format.
        - When applying filters on columns with categorical values , ensure that the user input values are correctly mapped to the stored values in the table name (for example, 'f' for 'female' and 'm' for 'male'), based on the structure and data of the columns.
        - When filtering by geographic location, always ensure that both `user_country` and `user_state` (if applicable) are used to correctly filter users from the intended countries.
        - The campaign brief should be strictly followed to ensure only the relevant audience is targeted.
        - When the brief mention a limit or a size of n for the population to be targetted, use LIMIT n and force a randomization by adding an order by clause with rand, like this example
             SELECT
              id AS user_id,
            FROM `thelookecommerce.users`
            WHERE gender = 'M'
            ORDER BY RAND()
            LIMIT 100;
    Don't hesitate to use subqueries (fetching data from all the available tables) 
    to ensure the brief and user input are properly respected.
      ### Metadata Structure: 
        - The tables are qualified with a dataset (e.g., dataset.table). In this case, the dataset is named "thelookecommerce".
        - Tables in the dataset are identified by the `table_id` field.
        - Each table has a `schema` field listing its columns, including `name`, `type`, `key`, and potential relationships.
        The provided schema metadata is as follows:
        {schema_metadata}
        """

        template = ChatPromptTemplate(
            [
                ("placeholder", "{messages}"),
                ("system", system_prompt),
            ]
        )

        structured_chain = template | self.model.with_structured_output(SQLQueryOutput)

        llm_output = structured_chain.invoke(
            {"messages": state["messages"], "schema_metadata": self.schema_metadata}
        )
        ai_message = AIMessage(
            content=format_model_or_dict(llm_output), name="Sql_Query_Builder_Agent"
        )
        
        return {"sql_query": llm_output, "messages": [ai_message]}

    def execute_query(self, state: AgentState):

        class EvaluationResult(BaseModel):
            should_adjust: bool = Field(
                description="Indicates whether the audience should be adjusted or not."
            )
            reasoning: str = Field(
                description="Agent's reasoning explaining the evaluation"
            )

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
            tool_dict_message = {
                    "name": "tool",
                    "args": 
                        {
                            "executed_query": sql_query,
                            "stats": stats,
                            "preview": query_result_df.head(5).to_json(
                                orient="records", indent=2
                            ),
                        }
                    ,
                }
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

            
            tool_dict_message = {
                "name": "Error",
                "args": {
                    "executed_query": sql_query,
                    "error": error_message
                }
            }
            tool_message = format_model_or_dict(tool_dict_message)

        return {
            "messages": [tool_message],
            "tool_dict_message": tool_dict_message,
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
        system_prompt = """You are an expert in evaluating marketing campaigns for an e-commerce site.  
        Your role is to determine whether an SQL query correctly extracts the intended audience.

        ### Guidelines:
        - Analyze the SQL query logic.
        - Examine the provided result statistics and first rows of data.
        - Compare the extracted audience with the expected audience from the campaign brief and user input.
        - Use the provided schema metadata to determine the appropriate column types and relationships.
        - If incorrect, explain why and provide a clear reasoning.
        - While evaluating, ensure that user_id is correctly retrieved, as it is the primary information we need to extract.
        ### Metadata Structure: 
        - The tables are qualified with a dataset (e.g., dataset.table). In this case, the dataset is named "thelookecommerce".
        - Tables in the dataset are identified by the `table_id` field.
        - Each table has a `schema` field listing its columns, including `name`, `type`, `key`, and potential relationships.
        The provided schema metadata is as follows:
        {schema_metadata}
     """
        template = ChatPromptTemplate(
            [
                ("placeholder", "{messages}"),
                ("system",system_prompt),
            ]
        )

        structured_chain = template | self.model.with_structured_output(
            EvaluationResult
        )

        llm_output = structured_chain.invoke(
            {"messages": state["messages"], "schema_metadata": self.schema_metadata}
        )

        ai_message = AIMessage(
            content=format_model_or_dict(llm_output)
            , name="Evaluator_Agent"
        )

        return {"evaluation_result": llm_output, "messages": [ai_message]}

    def adjust_query(self, state: AgentState):
        class AdjustedQuery(BaseModel):
            sql_query: str = Field(
                description="The corrected SQL query after applying the suggested improvements."
            )

        self.cpt_loop += 1
        print("self.cpt_loop", self.cpt_loop)
        system_prompt = """You are an expert SQL optimizer for marketing campaigns on an e-commerce site.  
    Your role is to correct and improve SQL queries to ensure they properly filter and extract the intended audience.

    ### Guidelines:
    - Analyze the given SQL query and identify potential issues.
    - Use the provided reasoning to understand why the query may be incorrect or suboptimal.
    - Apply necessary corrections while ensuring logical accuracy and efficiency.
    - Optimize the query for precision, avoiding unnecessary null values or misinterpretations.
    - Ensure the corrected query aligns with the expected audience criteria.
    - While adjusting the query, ensure that user_id is correctly retrieved, as it is a must have
    - Include other informations about the users that are relevant based on the context (user input, brief etc.)
    ### Metadata Structure: 
    - The tables are qualified with a dataset (e.g., dataset.table). In this case, the dataset is named "thelookecommerce".
    - Tables in the dataset are identified by the `table_id` field.
    - Each table has a `schema` field listing its columns, including `name`, `type`, `key`, and potential relationships.
    The provided schema metadata is as follows:
    {schema_metadata}

    """
        template = ChatPromptTemplate(
            [
                ("placeholder", "{messages}"),
                ("user",system_prompt),
            ]
        )

        structured_chain = template | self.model.with_structured_output(AdjustedQuery)

        llm_output = structured_chain.invoke(
            {"messages": state["messages"], "schema_metadata": self.schema_metadata}
        )

        ai_message = AIMessage(
            content=format_model_or_dict(llm_output), name="Adjuster_Agent"
        )
        return {
            "sql_query": llm_output,
            "messages": [ai_message],
        }

    def final_answer(self, state: AgentState):
        # class FinalResponse(BaseModel):
        #     response: str = Field(
        #         description="The final response generated based on the results of previous executions."
        #     )
    #     system_prompt ="""Summarize everything that has been done in the previous executions of the workflow and generate a final, 
    #     clear response for the user. 
    # """
        system_prompt="""Based on the user input {user_input} and the results in the messages, 
        provide a final response to the user """
        template = ChatPromptTemplate(
            [
                ("placeholder", "{messages}"),
                ("user", system_prompt),
            ]
        ) 
        structured_chain = template | self.model
        llm_output = structured_chain.invoke(
            {"messages": state["messages"],"user_input":state["user_input"]}
        )
        # ai_message = AIMessage(
        #     content=llm_output, name="Final Answer"
        # )
        self.cpt_loop = 0
        return { "messages":[llm_output] }



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


#metadata = f"{metadata_str} | {description}"
metadata = f"{description}"


llm = ChatVertexAI(model="gemini-2.0-flash-001", temperature=0)
#llm = ChatVertexAI(model="gemini-2.0-pro-exp-02-05", temperature=0)



Campaign_Workflow = CampaignWorkflow(model=llm,project_id=PROJECT_ID,schema_metadata=metadata)

chain = Campaign_Workflow.graph