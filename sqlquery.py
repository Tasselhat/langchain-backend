from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)  # load api key

# Uncomment the below to use LangSmith. Not required.
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
# os.environ["LANGCHAIN_TRACING_V2"] = "true"

from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("mysql+pymysql://root:Peanutbutter11@localhost:3306/goals_app")
print(db.dialect)
print(db.get_usable_table_names())
db.run("SELECT * FROM asset_labels LIMIT 10;")
print(db.run("SELECT * FROM asset_labels LIMIT 10;"))

from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
chain = create_sql_query_chain(llm, db)
response = chain.invoke({"question": "How many asset_labels are there"})
response
print(
db.run(response)
)
