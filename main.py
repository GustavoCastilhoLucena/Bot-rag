import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
import ollama


PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    # Check if the database should be cleared (using the --reset flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()

    if args.reset:
        print("Limpando banco de dados")
        clearDatabase()

    # Create (or update) the data store.
    print("Iniciando o processamento...")
    documentos = carregarDocumentos()
    pedacos = dividirDocumentos(documentos)
    adicionarProChroma(pedacos)
    print("Processamento finalizado.")

    # Query the database.
    queryTexto = args.query_text
    queryRag(queryTexto)

def adicionarProChroma(pedacos: list[Document]):
    db = Chroma(
        persist_directory='./database', embedding_function=conseguirFuncaoEmbeddings()
    )
    pedacosComIds = calcularPaginas(pedacos)
    itensExistentes = db.get(include=[])
    idsExistentes = set(itensExistentes["ids"])
    print(f"Numero de ids existentes no banco de dados: {len(idsExistentes)}")

    novosPedacos = []
    for pedaco in pedacosComIds:
        if pedaco.metadata["id"] not in idsExistentes:
            novosPedacos.append(pedaco)
    if novosPedacos:
        print(f"Adicionando novos documentos: {len(novosPedacos)}")
        novosIdsPedacos = [pedaco.metadata["id"] for pedaco in novosPedacos]
        db.add_documents(novosPedacos, ids=novosIdsPedacos)
    else:
        print("Nenhum documento novo adicionado")
    db = Chroma(
        persist_directory='./database', embedding_function=conseguirFuncaoEmbeddings()
    )
    db.persist()
    pedacosComIds = calcularPaginas(pedacos)
    itensExistentes = db.get(include=[])
    idsExistentes = set(itensExistentes["ids"])
    print(f"Numero de ids existentes no banco de dados: {len(idsExistentes)}")

    novosPedacos = []
    for pedaco in pedacosComIds:
        if pedaco.metadata["id"] not in idsExistentes:
            novosPedacos.append(pedaco)
    if len(novosPedacos):
        print(f"Adicionando novos documentos: {len(novosPedacos)}")
        novosIdsPedacos = [pedaco.metadata["id"] for pedaco in novosPedacos]
        db.add_documents(novosPedacos, ids=novosIdsPedacos)
        db.persist()
    else:
        print("Nenhum documento novo adicionado")

def clearDatabase():
    if os.path.exists('./database/'):
        shutil.rmtree('./database/')

def calcularPaginas(pedacos):
    indexPedacoAtual = 0
    idUltimapagina = None

    for pedaco in pedacos:
        fonte = pedaco.metadata.get("source")
        pagina = pedaco.metadata.get("page")
        idPaginaAtual = f"{fonte}:{pagina}"

        if idPaginaAtual == idUltimapagina:
            indexPedacoAtual += 1
        else:
            indexPedacoAtual = 0

        idUltimapagina = pagina
        idPedaco = f"{idPaginaAtual}:{indexPedacoAtual}"
        pedaco.metadata["id"] = idPedaco

    return pedacos

def conseguirFuncaoEmbeddings():
    embeddings = OllamaEmbeddings(
        model='llama3.2:1b'
    )
    return embeddings

def dividirDocumentos(documentos: list[Document]):
    print("Dividindo os documentos em pedaços...")
    divisorTexto = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False
    )

    pedacos = divisorTexto.split_documents(documentos)
    print(f"Divisão concluída. Total de pedaços gerados: {len(pedacos)}")
    return pedacos

def carregarDocumentos():
    print("Carregando documentos...")
    carregador = PyPDFDirectoryLoader("./data/")
    documentos = carregador.load()
    print(f"Carregamento concluído. Total de documentos carregados: {len(documentos)}")
    return documentos

def queryRag(queryTexto: str):
    funcaoEmbedings = conseguirFuncaoEmbeddings()
    db = Chroma(
        persist_directory='./database', embedding_function=funcaoEmbedings
    )
    resultados = db.similarity_search_with_score(queryTexto, k=5)
    textoContexto = "\n\n---\n\n".join([doc.page_content for doc, _score in resultados])
    templatePrompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = templatePrompt.format(context=textoContexto, question=queryTexto)
    print(f"Prompt enviado ao modelo:\n{prompt}")
    modelo = ollama(model="Llama 3.2")
    textoResposta = modelo.invoke(prompt)
    print("Resposta do modelo:")
    print(textoResposta)


if __name__ == "__main__":
    main()
