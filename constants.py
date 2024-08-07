from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

EMBD = OpenAIEmbeddings(model='text-embedding-3-large')

RETRIEVAL_TOP_K = 5

BIG_MODEL = 'gpt-4o-2024-08-06'
SMALL_MODEL = 'gpt-4o-mini-2024-07-18'

# ------------------------------------------------------------------- #
# we go in such painful granularity on the models,
# to future proof easy model swapping on each function
# Obviously doesn't matter right now

# MAIN
CONVLLM = ChatOpenAI(model=BIG_MODEL, temperature=0.5, streaming=True)
SUMMLLM = ChatOpenAI(model=SMALL_MODEL, temperature=0, streaming=True)

# Doc Filter
FILTLLM = ChatOpenAI(model=SMALL_MODEL, temperature=0, streaming=True)

# Router
ROUTERLLM = ChatOpenAI(model=SMALL_MODEL, temperature=0, streaming=True)

# Algo Reader
ALGOLLM = ChatOpenAI(model=SMALL_MODEL, temperature=0, streaming=True)

# Contextual compressor
COMPRESSORLLM = ChatOpenAI(model=SMALL_MODEL, temperature=0, streaming=True) 

# Multi query generator
MULTIQUERYLLM = ChatOpenAI(model=BIG_MODEL, temperature=0, streaming=True)
REPHRASINGLLM = ChatOpenAI(model=BIG_MODEL, temperature=0, streaming=True)
REORGLLM = ChatOpenAI(model=SMALL_MODEL, temperature=0, streaming=True)