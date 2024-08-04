import os
import json
from dotenv import load_dotenv
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_APIKEY")
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_APIKEY")
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LS_APIKEY')
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LS_PROJECT_NAME')

DISPLAYS_TOPIC_EXPANSION = True if int(os.getenv('DISPLAYS_TOPIC_EXPANSION')) else False

from collections import deque

import streamlit as st
from streamlit import session_state as ss

from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate, AIMessagePromptTemplate
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from template import template, summary_template, memory_template, AI_GREETING_MSG, surg_abbrevs_table
from st_utils import  StState

from multiretriever import generate_queries, reciprocal_rank_fusion
from routingagent import router_chain
from algoreader import tx_algo_chain
from constants import EMBD, CONVLLM, SUMMLLM
from llm_doc_filter import doc_filter_chain
from contextual_compressor import compressor_chain

# sources sorting function
def custom_sort_key(item):
    # Provide a default value (e.g., float('inf')) for missing keys
    return (
        item.get('Title', 'NO_KEY'),
        item.get('Header 1', 'NO_KEY'),
        item.get('Header 2', 'NO_KEY'),
        item.get('Header 3', 'NO_KEY'),
        item.get('Header 4', 'NO_KEY'),
        item.get('Header 5', 'NO_KEY')
    )

# ------------------------------------------------------------------- #
import pickle
from langchain.storage import InMemoryByteStore
from langchain_community.vectorstores import FAISS  
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.retrievers.multi_vector import SearchType



# note: the guideline documents are saved as and loaded from pkl files, but not generated in this repo
# currenly they're being generated in RAG_BPH/md_test.ipynb, using cua.md, aua.md, and eau.md
# documents include raw segmented chunks (by markdown headers) and RAPTOR recursive summaries for each guideline
# guidelines are generated from parsed pdfs and manually cleaned up prior to chunking and summarization

def get_retriever(pickle_directory, top_k=5):
    with open(f'pkl/{pickle_directory}/doc_ids.pkl', 'rb') as file:
        doc_ids = pickle.load(file)
    with open(f'pkl/{pickle_directory}/summary_docs.pkl', 'rb') as file:
        summary_docs = pickle.load(file)
    with open(f'pkl/{pickle_directory}/docs.pkl', 'rb') as file:
        docs = pickle.load(file)

    vectorstore = FAISS.from_documents(summary_docs, EMBD)
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

StState('cua_retreiver', default=get_retriever('cua'))
StState('aua_retreiver', default=get_retriever('aua'))
StState('eau_retreiver', default=get_retriever('eau'))

# ------------------------------------------------------------------- #

MAX_MSGS_IN_MEMORY = 10

page_name = 'Chat BPH'
st.set_page_config(page_title=page_name)
st.title(page_name)

# ------------------------------------------------------------------- #
st_showbtns = StState('showbtns', default=True)

system_prompt = deque([SystemMessagePromptTemplate.from_template(template)])
summ_prompt = deque([HumanMessagePromptTemplate.from_template(summary_template)])
memm_prompt = deque([AIMessagePromptTemplate.from_template(memory_template)])

st_ui_msgs = StState('ui_msgs', default=[{"role": "ai", "content": AI_GREETING_MSG}])
st_llm_msgs = StState('llm_msgs', default=deque([], maxlen=MAX_MSGS_IN_MEMORY))
st_convo_summary = StState('convo_summary', default='(none so far. this is a brand new conversation.)')

st_prompt = StState('prompt', default='')

example_qs = [
        "Which treatment is best if I want to preserve sexual function?",
        "Which treatments are suitable while remaining on anticoagulation?",
        "What are the main side effects of TURP, Rezum, and Greenlight?",
        "For a 55cc prostate, compare surgical options in a table in terms of catheter use, retention risk, and retreatment rates.",
    ]
st_generated_qs = StState('generated_qs', default=example_qs)

st_education_level = StState('education_level', default='Your answer should be quite succinct and patient oriented, at a grade 12 reading level.')

# ------------------------------------------------------------------- #

def assign_prompt(prompt):
    ss.prompt = prompt

for message in ss.ui_msgs:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
    if message.get('sources', None):
        if message['sources'] != 'n/a':
            with st.expander('Sources'):
                st.caption(message['sources'])
        else:
            st.caption('Sources: n/a')

if ss.showbtns:
    btns = st.empty()
    with btns.container():
        for q in ss.generated_qs:
            st.button(q, on_click=assign_prompt, args=(q,), use_container_width=True)


if prompt := st.chat_input("") or ss.prompt:
    prompt = ss.prompt or prompt
    prompt = prompt.strip()
    ss.ui_msgs.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    

    ss.prompt = ''
    if ss.showbtns:
        btns.empty()
        ss.showbtns = False


    with st.chat_message("ai"):
        
        with st.status('Reading Guidelines...', expanded=True) as status:

            st.write('Generating Topic Expansion...')
            routed = router_chain.invoke({"question":prompt, 'summary': ss.convo_summary}).lower().strip()
            if routed == 'no':
                context = 'n/a'
                ret = None
                queries_dict = {}  # no changes or query expansion. 
                algo_ans_dict = {}
                guiding_prompt = None
            else:  # routed == 'yes' or routed == any other string

                _algo_ans: dict = tx_algo_chain.invoke({'question': prompt, 'summary': ss.convo_summary})
                algo_ans_dict = {k: v for k, v in _algo_ans.items() if (v and k != 'metadata')}
                algo_ans_str = '\n\n'.join([v for k, v in algo_ans_dict.items() if v]) 
                if algo_ans_dict:
                    recommendations_str = 'For the next user query, the recommendations are as follows:\n\n'+\
                        f'{algo_ans_str}\n\n'+\
                        f'Since different guidelines use different terms to refer to the same treatment, always unify them using this reference table:\n\n{surg_abbrevs_table}'+\
                        '\n\n'
                else:
                    recommendations_str = ''
                
                guiding_prompt = SystemMessage(content=
                    f'{recommendations_str}You should try to combine the information from each guideline where possible, highlighting only major differences as you go, if applicable.\n'+\
                    'You should give a brief comparison between the guidelines (CUA, AUA, EAU), at the end of your response.')


                queries_dict = generate_queries.invoke({'question': prompt, 'tx_options': _algo_ans['metadata'], 'summary': ss.convo_summary})

                if DISPLAYS_TOPIC_EXPANSION:  # only for debugging
                    st.dataframe({k:v for k, v in queries_dict.items() if k != 'original'})

                st.write('Retrieving Documents...')
                queries_list = [q for q in queries_dict.values() if q]
                cua_ret = (ss.cua_retreiver.map() | reciprocal_rank_fusion).invoke(queries_list)
                aua_ret = (ss.aua_retreiver.map() | reciprocal_rank_fusion).invoke(queries_list)
                eau_ret = (ss.eau_retreiver.map() | reciprocal_rank_fusion).invoke(queries_list)
                ret = reciprocal_rank_fusion([cua_ret, aua_ret, eau_ret])
                print(len(ret))

                st.write('Compressing and Synthesizing Query...')
                filtered_ret = doc_filter_chain.map().invoke([{
                    'document': r,
                    'queries_dict': queries_dict}
                    for r in ret])
                ret = [f for f in filtered_ret if f.page_content]# [r for r, f in zip(ret, filtered_ret) if f.lower().strip() == 'yes']
                print(len(ret))


                compressed = [{'compressed': doc.page_content, 'metadata': doc.metadata} for doc in ret]  # we're not actually compressing here, just passing through. TODO: remove.
                #compressed = compressor_chain.map().invoke([{ 'question': queries_dict['rephrased'], 'document': r, 'summary': ss.convo_summary} for r in ret])
                c = [f'```Document #{i+1}\nSOURCE: {r['metadata'].get('Title', r['metadata'])}\n\n{r['compressed']}```' for i, r in enumerate(compressed)]
                context = "\n\n".join(c)

            print('\n', routed, algo_ans_dict, len(ret) if ret else 0, len([r for r in ret if r.metadata.get('Header 1')]) if ret else 0, '\n')

            status.update(label='Complete!', state='complete', expanded=False)


        msgs = system_prompt+memm_prompt+ss.llm_msgs
        if guiding_prompt: msgs.append(guiding_prompt)

        msgs.append(HumanMessage(content=prompt))
        if queries_dict.get('reorganized'):
            msgs.append(AIMessage(content=f'Based on the query and context, I think the user is asking: {queries_dict["reorganized"]}'))
            _steer_msg = f'Hidden Instructions: If there are conflicting recommendations, place your HIGHEST PRIORITY ON PATIENT SAFETY (e.g. risk of significant bleeding). '+\
                f'YOU MUST DO THIS AND NOT FORGET THIS. Always take a moment to think before you respond. You must NOT make unsafe recommendations.\n\n'+\
                f'Query: {ss.education_level}'
            msgs.append(HumanMessage(content=_steer_msg))

        formated = ChatPromptTemplate.from_messages(msgs)

#        ss.llm_msgs.append(HumanMessage(content=prompt))
        
        stream = CONVLLM.stream(formated.format_prompt(context=context, summary=ss.convo_summary).to_messages())
        stresult = ''
        container = st.empty()
        for chunk in stream:
            stresult += chunk.content
            container.code(stresult)
        response = stresult
        st.markdown(response)
        # -------- response = st.write_stream(stream)

        #ss.ui_msgs.append({"role": "ai", "content": response})
#        ss.llm_msgs.append(AIMessage(content=response))


    with st.spinner('Getting Sources...'):
        last_msgs = deque([HumanMessage(content=prompt), AIMessage(content=response)])
        to_summ = ChatPromptTemplate.from_messages(memm_prompt+last_msgs+summ_prompt)
        ss.convo_summary = SUMMLLM(to_summ.format_prompt(summary=ss.convo_summary).to_messages()).content

        if ret:
            sources = [r.metadata for r in ret if r.metadata.get('Header 1')]
            sources_sorted = sources # sorted(sources, key=custom_sort_key)    # sorting by metadata key is disabled for now, sorted by reciprocal rank fusion
            final_sources = []
            for i, r in enumerate(sources_sorted):
                guideline = r['Title']
                if 'AUA' in guideline: guideline = 'AUA Guidelines'
                elif 'EAU' in guideline: guideline = 'EAU Guidelines'
                elif 'CUA' in guideline: guideline = 'CUA Guidelines'
                else: guideline = ''
                formatted_string = f'{guideline} | ' + "".join([f"{value.strip('*')} | " for key, value in r.items() if key != 'Title'])
                final_sources.append(f'{i+1}. | {formatted_string}')

            raptor_num_tracker = {}
            raptor_sources = [r.metadata for r in ret if not r.metadata.get('Header 1')]
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

        
        ss.ui_msgs.append({"role": "ai", "content": response, 'sources': final_sources_str})
        if final_sources_str != 'n/a':
            with st.expander('Sources'):
                st.caption(final_sources_str)
        else:
            st.caption('Sources: n/a')


edu_toggle = st.toggle("Clinician Mode")
if edu_toggle:
    ss.education_level = 'Your answer should be highly sophisticated, at the level of a post doctoral researcher in the field.'
else:
    ss.education_level = 'Your answer should be quite succinct and patient oriented, at a grade 12 reading level.'

print()
print()
print(ss.convo_summary)