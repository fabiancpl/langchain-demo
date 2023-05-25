from dotenv import load_dotenv

import streamlit as st

from pdf_ingest import PDFIngestor
from pdf_chatbot import PDFChatbot


def generate_response(chatbot, question):
    # Asking the chatbot
    answer, source_documents = chatbot.ask(question)

    sources = ""
    for document in source_documents:
        sources += f"Página: {document.metadata['page_number']} - Texto: {document.page_content[:100]}...\n"

    return f"""{answer}
    \n\nFuentes:
    {sources}
    """


def main():
    load_dotenv()

    print("Ingesting PDF into the vector store (if not exists)...")
    PDFIngestor().ingest()

    print("Wating for questions...")
    chatbot = PDFChatbot()
    chatbot.make_chain()

    st.set_page_config(page_title="Langchain Demo")
    
    st.header("¿Cuál será el futuro de Colombia?")
    st.subheader("De acuerdo al Plan Nacional de Desarrollo (2022-2026)")

    question = st.text_input("Habla con el PND:")

    if st.button("Preguntar"):
        response = generate_response(chatbot, question)
        st.text_area("Respuesta:", response, height=400)


if __name__ == "__main__":
    main()