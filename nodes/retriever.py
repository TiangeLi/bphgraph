import os
import pickle
from langchain.storage import InMemoryByteStore
from langchain_community.vectorstores import FAISS  
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.retrievers.multi_vector import SearchType
from constants import EMBD, RETRIEVAL_TOP_K

# note: the guideline documents are saved as and loaded from pkl files, but not generated in this repo
# currenly they're being generated in RAG_BPH/md_test.ipynb, using cua.md, aua.md, and eau.md
# documents include raw segmented chunks (by markdown headers) and RAPTOR recursive summaries for each guideline
# guidelines are generated from parsed pdfs and manually cleaned up prior to chunking and summarization

def get_retriever(pickle_directory, top_k=RETRIEVAL_TOP_K):
    with open(f'guideline_pkls/{pickle_directory}/doc_ids.pkl', 'rb') as file:
        doc_ids = pickle.load(file)
    with open(f'guideline_pkls/{pickle_directory}/summary_docs.pkl', 'rb') as file:
        summary_docs = pickle.load(file)
    with open(f'guideline_pkls/{pickle_directory}/docs.pkl', 'rb') as file:
        docs = pickle.load(file)

    if not os.path.isdir(f'guideline_pkls/{pickle_directory}/faiss'):
        vectorstore = FAISS.from_documents(summary_docs, EMBD)
        vectorstore.save_local(f'guideline_pkls/{pickle_directory}/faiss')
    else:
        vectorstore = FAISS.load_local(f'guideline_pkls/{pickle_directory}/faiss', EMBD, allow_dangerous_deserialization=True)

    store = InMemoryByteStore()
    id_key = 'doc_id'
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=id_key,
        search_type=SearchType.similarity,
        search_kwargs={'k': top_k}
    )
    retriever.docstore.mset(list(zip(doc_ids, docs)))
    return retriever

guideline_retrievers = {
    'cua': get_retriever('cua'),
    'aua': get_retriever('aua'),
    'eau': get_retriever('eau')
}