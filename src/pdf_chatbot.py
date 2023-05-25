from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.chroma import Chroma
from langchain.schema import HumanMessage, AIMessage
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain


class PDFChatbot:
    def __init__(self):
        self.chain = None
        self.chat_history = []
        self.vector_dir = "data/chroma"
        self.collection_name = "pnd-2023"

    def make_chain(self) -> ConversationalRetrievalChain:
        model = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0,
            # verbose=True
        )

        embedding = OpenAIEmbeddings()

        vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=embedding,
            persist_directory=self.vector_dir,
        )

        self.chain = ConversationalRetrievalChain.from_llm(
            model,
            retriever=vector_store.as_retriever(),
            return_source_documents=True,
            # verbose=True,
        )
    
    def ask(self, question: str) -> str:
        question = f"""
        Responda a la siguiente pregunta utilizando como base el documento proporcionado.
        Si requiere complementar la respuesta con fuentes externas, indíquelo apropiadamente.

        Pregunta: {question}
        """

        # Generating the answer
        response = self.chain({"question": question, "chat_history": self.chat_history})

        # Retrieving the answer
        answer = response["answer"]
        source_documents = response["source_documents"]
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))

        return answer, source_documents