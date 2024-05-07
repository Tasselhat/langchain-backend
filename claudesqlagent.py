import os
from dotenv import find_dotenv, load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase

# Uncomment the below to use LangSmith. Not required.
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)  # load api key

API_KEY = os.environ.get("ANTHROPIC_API_KEY")

database_uri = os.getenv("DATABASE_URI")

db = SQLDatabase.from_uri(database_uri)

llm = ChatAnthropic(
    model="claude-3-opus-20240229"
)

agent_executor = create_sql_agent(
    llm, db=db, agent_type="zero-shot-react-description", verbose=True)
agent_executor.invoke(
    {
        "input": "What are the twenty most recently updated asset labels, return results of the query."
    }
)
