import os
from dotenv import find_dotenv, load_dotenv
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)  # load api key
# Uncomment the below to use LangSmith. Not required.
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
# os.environ["LANGCHAIN_TRACING_V2"] = "true"

database_uri = os.getenv("DATABASE_URI")

db = SQLDatabase.from_uri(database_uri)

# Point to the local server
llm = ChatOpenAI(base_url="http://localhost:1234/v1",
                 model="local-model", temperature=0.1, api_key="not-needed")

agent_executor = create_sql_agent(
    llm, db=db, agent_type="zero-shot-react-description", verbose=True)

agent_executor.invoke(
    {
        "input": "What are the twenty most recently updated asset labels, return results of the query."
    }
)
