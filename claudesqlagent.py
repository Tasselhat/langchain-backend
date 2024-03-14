import os

from anthropic import Anthropic
from dotenv import find_dotenv, load_dotenv
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_anthropic import ChatAnthropic

# Uncomment the below to use LangSmith. Not required.
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)  # load api key

API_KEY = os.environ.get("ANTHROPIC_API_KEY")


db = SQLDatabase.from_uri(
    "mysql+pymysql://root:Peanutbutter11@localhost:3306/goals_app")
llm = ChatAnthropic(
    model="claude-3-opus-20240229"
)

agent_executor = create_sql_agent(
    llm, db=db, agent_type="zero-shot-react-description", verbose=True)
agent_executor.invoke(
    {
        "input": "Count the total number of rows in each table in the database"
    }
)
