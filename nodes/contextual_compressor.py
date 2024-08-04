from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import chain
from langchain_core.documents import Document
from constants import COMPRESSORLLM

sys_template = \
"""BACKGROUND DOCUMENT
Document metadata:
{metadata}

Document:
{background}
END OF BACKGROUND DOCUMENT"""

hum_template = \
"""Query: {question}

Task:
Consider the ENTITIES in the Query.
From the BACKGROUND DOCUMENT, extract only the information that pertains to any or all of the ENTITIES in the Query.
Depending on how relevant the BACKGROUND DOCUMENT is, you can return as a whole, or part of the document, as long as you focus on the relevant entities.

Rules:
- Do NOT generate new information that is not in the BACKGROUND DOCUMENT.
- Do NOT include extraneous commentary or suggestions. If an ENTITY is not discussed in the document, do not include it in the response.
- Do NOT discuss entities from the BACKGROUND DOCUMENT that are not in the Query.
- Do NOT mention entities from the Query that are not in the BACKGROUND DOCUMENT.
- Return the most important information that pertains to the ENTITIES in the Query as accurately as possible, in particular including numbers, values, specific details, and other important information.

Return the information in a concise point form, without any additional commentary or suggestions, focusing only on the relevant entities."""


prompt = ChatPromptTemplate.from_messages(
    [
        ('system', sys_template),
        ('human', hum_template)
    ]
)

@chain
def compressor_chain(_input: dict):
    return _input['document']  # for now, just return the document as is, no compression
    question, doc = _input['question'], _input['document']
    _chain = prompt | COMPRESSORLLM | StrOutputParser()
    response = _chain.invoke({'question': question, 'metadata': doc.metadata, 'background': doc.page_content})
    ret = Document(response, metadata=doc.metadata)
    return ret