import os
from dotenv import load_dotenv
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_APIKEY")
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_APIKEY")
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LS_APIKEY')
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LS_PROJECT_NAME')

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from nodes.routingagent import router_chain
from nodes.algoreader import tx_algo_chain
from nodes.multiretriever import generate_queries, reciprocal_rank_fusion
from nodes.retriever import guideline_retrievers
from nodes.llm_doc_filter import doc_filter_chain

from template import surg_abbrevs_table
from template import template, summary_template, memory_template

from constants import CONVLLM, SUMMLLM

# --- constants ---
#EDUCATION_LEVEL = 'Your answer should be highly sophisticated, at the level of a post doctoral researcher in the field.'
EDUCATION_LEVEL = 'Your answer should be quite succinct and patient oriented, at a grade 12 reading level.'

# --- reducer functions ---
def overwrite(_, new):
    return new

# --- basic graph elements ---
class State(TypedDict):
    prompt: str
    summary: str
    use_guidelines: bool
    guiding_prompt: str
    algo_recs: dict
    multiqueries: Annotated[dict, overwrite]
    raw_retrieved: dict
    filtered_retrieved: dict
    final_synthesized_prompt: ChatPromptTemplate
    response: str
    sources: str
graph_builder = StateGraph(State)

# --- define nodes and edges ---
def router_node(state: State):
    _input = {'question': state['prompt'], 'summary': state['summary']}
    use_guidelines = router_chain.invoke(_input)
    return {'use_guidelines': use_guidelines['b']}
def router_edge(state: State):
    if state['use_guidelines']:
        return '__use_guidelines__'
    else:
        return '__no__'

def expansion_node(state: State):
    _input = {'question': state['prompt'], 'summary': state['summary']}
    _raw = tx_algo_chain.invoke(_input)
    _algo_recs = [v for k, v in _raw.items() if (v and k != 'metadata')]
    _algo_recs_str = ''
    if _algo_recs:
        _algo_recs = '\n\n'.join(_algo_recs)
        _algo_recs_str = 'For the next user query, the recommendations are as follows:\n\n'+\
                        f'{_algo_recs}\n\n'+\
                        'Since different guidelines use different terms to refer to the same treatment, '+\
                        f'always unify them using this reference table:\n\n{surg_abbrevs_table}\n\n'
    guiding_prompt = f'{_algo_recs_str}You should try to combine the information from each guideline where possible, '+\
                     'highlighting only major differences as you go, if applicable.\n'+\
                     'You should give a brief comparison between the guidelines (CUA, AUA, EAU), at the end of your response.'
    algo_recs_options = _raw['metadata']
    return {'guiding_prompt': guiding_prompt, 'algo_recs': algo_recs_options}

def multiquery_node(state: State):
    _input = {'question': state['prompt'], 'summary': state['summary'], 'tx_options': state['algo_recs']}
    queries_dict = generate_queries.invoke(_input)
    return {'multiqueries': queries_dict}

def retrieval_node(state: State):
    _queries_list = [q for q in state['multiqueries'].values() if q]
    retrieved = []
    for retriever in guideline_retrievers.values():
        ret_chain = retriever.map() | reciprocal_rank_fusion
        retrieved.append(ret_chain.invoke(_queries_list))
    raw_retrieved = reciprocal_rank_fusion(retrieved)
    return {'raw_retrieved': raw_retrieved}

def filter_node(state: State):
    _inputs = [{'document': doc, 'queries_dict': state['multiqueries']} for doc in state['raw_retrieved']]
    filter = doc_filter_chain.map().invoke(_inputs)
    filtered_retrieved = [doc for doc in filter if doc.page_content]
    return {'filtered_retrieved': filtered_retrieved}

def prompt_synthesis_node(state: State):
    if not state['use_guidelines']:
        synthesized = [
                ('system', template),
                ('ai', memory_template),
                ('human', state['prompt'])
            ]
    else:
        synthesized = [
                ('system', template),
                ('ai', memory_template),
                ('system', state['guiding_prompt']),
                ('human', state['prompt'])
            ]
        if state['multiqueries'].get('reorganized'):
            synthesized.extend([
                ('ai', f'Based on the query and context, I think the user is asking: {state['multiqueries']["reorganized"]}'),
                ('human', f'Hidden Instructions: If there are conflicting recommendations, place your HIGHEST PRIORITY ON PATIENT SAFETY (e.g. risk of significant bleeding). '+\
                          f'YOU MUST DO THIS AND NOT FORGET THIS. Always take a moment to think before you respond. You must NOT make unsafe recommendations.\n\n'+\
                          f'Education Level Hint: {EDUCATION_LEVEL}'+\
                          '\n\nOne more thing: for any CLAIM you make, you MUST provide a reference in this format at the end of the claim: <doc_#x>, <doc_#y>, ...')
            ])
    synthesized = ChatPromptTemplate.from_messages(synthesized)
    return {'final_synthesized_prompt': synthesized}
    
def chat_node(state: State):
    if not state['use_guidelines']:
        sources = '```None```'
    else:
        sources = [f'```<document_#{i+1}>\nSOURCE: {d.metadata.get('Title', d.metadata)}\n\n{d.page_content}\n</document_#{i+1}>```' for i, d in enumerate(state['filtered_retrieved'])]
        sources = '\n\n'.join(sources)
    chain = state['final_synthesized_prompt'] | CONVLLM | StrOutputParser()
    response = chain.invoke({'context': sources, 'summary': state['summary']})
    return {'response': response}

def summary_node(state: State):
    if not state['use_guidelines']:
        return {'summary': state['summary'], 'sources': 'n/a'}
    messages = [
        ('ai', memory_template),
        ('human', state['prompt']),
        ('ai', state['response']),
        ('system', summary_template)
    ]
    chain = ChatPromptTemplate.from_messages(messages) | SUMMLLM | StrOutputParser()
    summary = chain.invoke({'summary': state['summary']})
    if state['filtered_retrieved']:
        sources = [d.metadata for d in state['filtered_retrieved'] if d.metadata.get('Header 1')]
        final_sources = []
        for i, s in enumerate(sources):
            guideline = s['Title']
            if 'AUA' in guideline: guideline = 'AUA Guidelines'
            elif 'EAU' in guideline: guideline = 'EAU Guidelines'
            elif 'CUA' in guideline: guideline = 'CUA Guidelines'
            else: guideline = ''
            formatted_string = f'{guideline} | ' + "".join([f"{v.strip('*')} | " for k, v in s.items() if k != 'Title'])
            final_sources.append(f'{i+1}. | {formatted_string}')
        raptor_num_tracker = {}
        raptor_sources = [d.metadata for d in state['filtered_retrieved'] if not d.metadata.get('Header 1')]
        for r in raptor_sources:
            guideline = r['Title']
            if 'AUA' in guideline: guideline = 'AUA Guidelines'
            elif 'EAU' in guideline: guideline = 'EAU Guidelines'
            elif 'CUA' in guideline: guideline = 'CUA Guidelines'
            else: guideline = ''
            raptor_num_tracker[guideline] = raptor_num_tracker.get(guideline, 0) + 1
        raptor_num_str = ' | '.join([f'{k}: {v}' for k,v in raptor_num_tracker.items()])
        final_sources_str = f'{"\n".join(final_sources)}\n\nRAPTOR Documents: {raptor_num_str if raptor_num_str else "n/a"}'
    else:
        final_sources_str = 'n/a'
    return {'summary': summary, 'sources': final_sources_str}

# --- add nodes ---
graph_builder.add_node('router', router_node)
graph_builder.add_node('expander', expansion_node)
graph_builder.add_node('multiquery', multiquery_node)
graph_builder.add_node('retrieval', retrieval_node)
graph_builder.add_node('filter', filter_node)
graph_builder.add_node('prompt_synthesis', prompt_synthesis_node)
graph_builder.add_node('chat', chat_node)
graph_builder.add_node('summarizer', summary_node)

# --- add edges ---
graph_builder.add_edge(START, 'router')
graph_builder.add_conditional_edges(
    source='router',
    path=router_edge,
    path_map={'__use_guidelines__': 'expander', '__no__': 'prompt_synthesis'}
)
graph_builder.add_edge('expander', 'multiquery')
graph_builder.add_edge('multiquery', 'retrieval')
graph_builder.add_edge('retrieval', 'filter')
graph_builder.add_edge('filter', 'prompt_synthesis')
graph_builder.add_edge('prompt_synthesis', 'chat')
graph_builder.add_edge('chat', 'summarizer')
graph_builder.add_edge('summarizer', END)

# --- compile graph ---
graph = graph_builder.compile()