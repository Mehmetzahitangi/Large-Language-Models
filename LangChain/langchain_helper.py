import os
from dotenv import load_dotenv

from langchain_community.llms import GooglePalm
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
#from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

load_dotenv()

llm = GooglePalm(google_api_key = os.environ['GOOGLE_API_KEY'], temperature=0.5)



instructor_embeddings = HuggingFaceEmbeddings()
vectordb_file_path = 'faiss_index'


def create_vector_db():
    loader = CSVLoader(file_path= "codebasics_faqs.csv" source_column="prompt",encoding='latin') 

    docs = loader.load()

    vectordb = FAISS.from_documents(documents=docs, embedding=instructor_embeddings)
    vectordb.save_local(vectordb_file_path)

def get_qa_chain():
    #load vector database
    vectordb = FAISS.load_local(folder_path=vectordb_file_path, embeddings=instructor_embeddings)

    retriever = vectordb.as_retriever(score_threshold=0.7)

    # to avoid hallucinations
    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""


    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    #chain_type_kwargs = {"prompt": PROMPT}

    chain = RetrievalQA.from_chain_type(
                llm = llm,
                chain_type= 'stuff',
                retriever=retriever,
                input_key='query',
                return_source_documents=True,
                chain_type_kwargs={'prompt':PROMPT}
                )

    return chain
if __name__ == '__main__':
    #create_vector_db()

    chain = get_qa_chain()

    print(chain("Do you provide internship? Do you have EMI option?"))
