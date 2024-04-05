import json

from dotenv import find_dotenv, load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from langchain.chains import (create_history_aware_retriever,
                              create_retrieval_chain)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import (CharacterTextSplitter,
                                     RecursiveCharacterTextSplitter)
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import (HuggingFaceInstructEmbeddings,
                                            OpenAIEmbeddings)
from langchain_community.utilities import SQLDatabase
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (ChatPromptTemplate, FewShotPromptTemplate,
                                    MessagesPlaceholder, PromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from PyPDF2 import PdfReader

app = Flask(__name__)
cors = CORS(app, headers='Content-Type')


def get_vectorstore_from_url(url):
    # get the text in document form
    loader = WebBaseLoader(url)
    document = loader.load()

    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    # create a vectorstore from the chunks
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())

    return vector_store


def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain


def get_conversational_rag_chain(retriever_chain):

    llm = ChatOpenAI()

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def get_conversational_chain_no_rag():
    output_parser = StrOutputParser()
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    chain = prompt | llm | output_parser

    return chain


@app.route('/websitechat', methods=['POST'])
@cross_origin()
def get_response():
    data = request.get_json()
    website_url = data['websiteURL']
    if website_url:
        vector_store = get_vectorstore_from_url(website_url)
        retriever_chain = get_context_retriever_chain(vector_store)
        conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

        response = conversation_rag_chain.invoke({
            "chat_history": data['messages'],
            "input": data['input']
        })

        print(response)

        return jsonify(response['answer'])
    else:
        chain = get_conversational_chain_no_rag()

        response = chain.invoke({
            "chat_history": data['messages'],
            "input": data['input']
        })

        return jsonify(response)


def agent(input):
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)  # load api key
    # Uncomment the below to use LangSmith. Not required.
    # os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
    # os.environ["LANGCHAIN_TRACING_V2"] = "true"

    db = SQLDatabase.from_uri(
        "mysql+pymysql://root:Peanutbutter11@localhost:3306/goals_app")

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)

    examples = [
        {"input": "List all artists.", "query": "SELECT * FROM Artist;"},
        {
            "input": "Find all albums for the artist 'AC/DC'.",
            "query": "SELECT * FROM Album WHERE ArtistId = (SELECT ArtistId FROM Artist WHERE Name = 'AC/DC');",
        },
        {
            "input": "List all tracks in the 'Rock' genre.",
            "query": "SELECT * FROM Track WHERE GenreId = (SELECT GenreId FROM Genre WHERE Name = 'Rock');",
        },
        {
            "input": "Find the total duration of all tracks.",
            "query": "SELECT SUM(Milliseconds) FROM Track;",
        },
        {
            "input": "List all customers from Canada.",
            "query": "SELECT * FROM Customer WHERE Country = 'Canada';",
        },
        {
            "input": "How many tracks are there in the album with ID 5?",
            "query": "SELECT COUNT(*) FROM Track WHERE AlbumId = 5;",
        },
        {
            "input": "Find the total number of invoices.",
            "query": "SELECT COUNT(*) FROM Invoice;",
        },
        {
            "input": "List all tracks that are longer than 5 minutes.",
            "query": "SELECT * FROM Track WHERE Milliseconds > 300000;",
        },
        {
            "input": "Who are the top 5 customers by total purchase?",
            "query": "SELECT CustomerId, SUM(Total) AS TotalPurchase FROM Invoice GROUP BY CustomerId ORDER BY TotalPurchase DESC LIMIT 5;",
        },
        {
            "input": "Which albums are from the year 2000?",
            "query": "SELECT * FROM Album WHERE strftime('%Y', ReleaseDate) = '2000';",
        },
        {
            "input": "How many employees are there",
            "query": 'SELECT COUNT(*) FROM "Employee"',
        },
        {
            "input": "How many insurance companies start with the letter H?",
            "query": 'SELECT COUNT(*) FROM insurance_companies WHERE "name" LIKE "H%',
        },
    ]

    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        OpenAIEmbeddings(),
        FAISS,
        k=5,
        input_keys=["input"],
    )

    system_prefix = """You are an agent designed to interact with a SQL database.
    Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
    Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
    You can order the results by a relevant column to return the most interesting examples in the database.
    Never query for all the columns from a specific table, only ask for the relevant columns given the question.
    You have access to tools for interacting with the database.
    Only use the given tools. Only use the information returned by the tools to construct your final answer.
    You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

    If the question does not seem related to the database, just return "I'm sorry, I wasn't able to process that request, try rewording the request or providing more detail." as the answer.

    Here are some examples of user inputs and their corresponding SQL queries:"""

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=PromptTemplate.from_template(
            "User input: {input}\nSQL query: {query}"
        ),
        input_variables=["input", "dialect", "top_k"],
        prefix=system_prefix,
        suffix="",
    )

    full_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate(prompt=few_shot_prompt),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent_executor = create_sql_agent(
        llm, db=db, prompt=full_prompt, agent_type="openai-tools", verbose=True)
    return agent_executor.invoke({"input": input})


@app.route('/message', methods=['POST'])
@cross_origin()
def process_agent_message():
    data = request.get_json()
    print(data)
    return jsonify(agent(data['input']))


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


@app.route('/pdfchat', methods=['POST'])
@cross_origin()
def get_chat_response():
    # Files are accessed through request.files, not get_json()
    pdf_files = request.files.getlist(
        'pdf_docs')  # This is how you access files

    # If you have other non-file fields that were appended to FormData
    user_input = request.form.get('input')
    # This will be a JSON string if you sent it as such
    messages_json = request.form.get('messages')

    if messages_json:
        try:
            messages = json.loads(messages_json)
            # Now `messages` is a Python list that you can pass to your langchain
        except json.JSONDecodeError as error:
            return {'error': 'Invalid JSON format for messages'}, 400
    else:
        return {'error': 'Messages are required'}, 400

    raw_text = get_pdf_text(pdf_files)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    retriever_chain = get_context_retriever_chain(vectorstore)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    response = conversation_rag_chain.invoke({
        "chat_history": messages,
        "input": user_input,
    })

    print(response)
    return jsonify(response['answer'])


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
    print("Running on port 5000")
