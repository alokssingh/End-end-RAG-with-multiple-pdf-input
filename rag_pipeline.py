import os
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from openai import OpenAI
from logger import get_logger
import shutil

openai_api_key = "key"

logger = get_logger("RAG")

class RAGPipeline:
    def __init__(self, docs, model="gpt-4", persist_dir="chroma_store"):
        self.model = model
        self.persist_dir = persist_dir
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir) # cleaning existing db optional 

        logger.info("Building Chroma vector store...")
        #docs = [Document(page_content=chunk) for chunk in chunks]
        self.db = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )
        self.db.persist()
        logger.info("Chroma store built.")

    def retrieve(self, query, top_k=3):
        results = self.db.similarity_search(query, k=top_k)
        return results

    def generate(self, query, history=None):
        
        results = self.retrieve(query)

       #retrieved context chunks
        context_blocks = []
        for i, doc in enumerate(results, start=1):
            page = doc.metadata.get("page", "?")
            dtype = doc.metadata.get("type", "text")
            context_blocks.append(f"### Source {i} (Page {page}, Type: {dtype})\n{doc.page_content}")

       #Format conversation history
        history_text = ""
        if history:
            for user_msg, assistant_reply in history:
                history_text += f"User: {user_msg}\nAssistant: {assistant_reply}\n"

        # Step 3: Construct prompt
        prompt = f"""You are an assistant for renewable energy operators. 
Use the following context and prior conversation history to answer the user's latest question clearly and concisely. 
Always ground your answer in the given context. Do not make up any facts. 
When applicable, cite the page number of the source (e.g., “According to Page 3…” or “(see Page 2)”).


    ### Conversation History
    {history_text}

    ### Context
    {'\n'.join(context_blocks)}

    ### Current Question
    {query}

    ### Answer
    """

        
        client = OpenAI(api_key=openai_api_key)
        logger.info(f"Calling ChatGPT with {len(context_blocks)} chunks and history length: {len(history) if history else 0}")
        
        completion = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return completion.choices[0].message.content
