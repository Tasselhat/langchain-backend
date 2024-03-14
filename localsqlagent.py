from dotenv import find_dotenv, load_dotenv
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)  # load api key
# Uncomment the below to use LangSmith. Not required.
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
# os.environ["LANGCHAIN_TRACING_V2"] = "true"

db = SQLDatabase.from_uri(
    "mysql+pymysql://root:Peanutbutter11@localhost:3306/goals_app")

# Point to the local server
llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")


agent_executor = create_sql_agent(
    llm, db=db, agent_type="openai-tools", verbose=True)
agent_executor.invoke(
    {
        "input": "Count the total number of rows in each table in the database"
    }
)
