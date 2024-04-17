import os
from dotenv import find_dotenv, load_dotenv
from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)  # load api key

database_uri = os.getenv("DATABASE_URI")

db = SQLDatabase.from_uri(database_uri)

print(db.dialect)
print(db.get_usable_table_names())
print(db.run("SELECT * FROM abbreviations LIMIT 10;"))

llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
chain = create_sql_query_chain(llm, db)
response = chain.invoke({"question": "How many asset labels are there"})
chain.get_prompts()[0].pretty_print()
print(db.run(response))
