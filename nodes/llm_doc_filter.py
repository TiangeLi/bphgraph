from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from template import surg_abbrevs_table
from langchain_core.runnables import chain
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from nodes.contextual_compressor import compressor_chain
from constants import FILTLLM

headers_to_split_on = [
    ("#", "Title"),
    ("##", "Header 1"),
    ("###", "Header 2"),
    ("####", "Header 3"),
    ("#####", "Header 4"),
    ("######", "Header 5"),
    ("#######", "Header 6")
]

markdown_splitter_with_header = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False
    )


sys_template = \
f"""Use this abbreviations table to help you understand the background information:
{surg_abbrevs_table}"""


hum_template = \
"""BACKGROUND DOCUMENT
{doc_contents}
END OF BACKGROUND DOCUMENT

Query: {question}

Task:
Consider each ENTITY in the Query.
Consider each FOCUS/TOPIC of the BACKGROUND DOCUMENT.
Your ONLY job is to classify YES or NO. 

Rules:
- NO: If NONE of the entities are discussed in the BACKGROUND DOCUMENT.
- YES: if the BACKGROUND DOCUMENT's primary FOCUS contains information pertinent to ANY ONE of the entities in the Query (including any Examples, Equivalents, Brand Names, Abbreviations, etc.)

Remember: ONLY use the BACKGROUND DOCUMENT to inform your decision.
Reply YES or NO"""


filter_prompt = ChatPromptTemplate.from_messages(
    [('system', sys_template),
    ('human', hum_template)])

@chain
def doc_filter_chain(_input: dict):
    queries_dict, document = _input['queries_dict'], _input['document']
    _chain = filter_prompt | FILTLLM | StrOutputParser()

    # split docs and integrate metadata
    split_docs = markdown_splitter_with_header.split_text(document.page_content)
    for s in split_docs:
        s.metadata = {**document.metadata, **{k: v for k, v in s.metadata.items() if k not in document.metadata}}

    # filter based on metadata only. keep YES, continue to filter NO based on full text
    metadata_filter = _chain.batch([
        {
            'question': queries_dict['rephrased'],
            'doc_contents': s.metadata
        }
        for s in split_docs])
    rejected_splits_on_metadata = [s for s, f in zip(split_docs, metadata_filter) if f.strip().lower() == 'no']
    filtered_on_metadata = [s for s, f in zip(split_docs, metadata_filter) if f.strip().lower() == 'yes']
    
    # filter rejected docs, now based on full text
    compressed_filtered_on_doc = []
    if rejected_splits_on_metadata:
        doc_filter = _chain.batch([
            {
                'question': queries_dict['rephrased'],
                'doc_contents': f'Document metadata:\n{s.metadata}\n\nDocument:\n{s.page_content}'
            }
            for s in rejected_splits_on_metadata])
        raw_filtered_on_doc = [s for s, f in zip(rejected_splits_on_metadata, doc_filter) if f.strip().lower() == 'yes']
        compressed_filtered_on_doc = compressor_chain.batch([
            {'question': queries_dict['rephrased'], 
            'document': s} 
            for s in raw_filtered_on_doc])
    
    # reconstruct docs
    filtered_docs = filtered_on_metadata + compressed_filtered_on_doc
    filtered = Document(
        page_content='\n\n'.join([s.page_content for s in filtered_docs]),
        metadata=document.metadata
    )
    return filtered