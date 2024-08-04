import os
from dotenv import load_dotenv
load_dotenv()
USE_LOCAL_SERVER = True if int(os.getenv('USE_LOCAL_SERVER')) else False

if USE_LOCAL_SERVER:
    from main_graph import graph
    app = graph
else:
    from langserve import RemoteRunnable
    app = RemoteRunnable("http://localhost:8000/chat/")

import streamlit as st
import asyncio
import re

st.set_page_config(layout="wide")


def replace_doc_placeholders(input_string, readable_sources):
    link_trans_table = str.maketrans({
        ' ': '-',
        '.': '-',
        ':': '-',
        '(': '',
        ')': '',
        '\'': '',
        '"': '',
        '/': '-',
    })
    headings = [s.metadata[list(s.metadata.keys())[-1]] if len(s.metadata)>1 else '' for s in readable_sources]
    headings_list = [f'[Source {num+1}](#{heading.lower().strip().strip(':-.').translate(link_trans_table).replace('---', '-').replace('--', '-').strip('-')})' if heading else '[]' for num, heading in enumerate(headings)]
    def replacement(match):
        doc_number = match.group(1)
        return headings_list[int(doc_number)-1]
    pattern = r'<doc_#(\d+)>'
    return re.sub(pattern, replacement, input_string)

async def rungraph(inputs, runnable):
    cols = st.columns(2)
    percent_complete = 0
    final_sources = ''
    chat_response = ''
    readable_sources = []
    readable_sources_str = ''
    current_node = ''

    with cols[0]:
        status_container = st.empty()
        container = st.empty()
    with cols[1]:
        sources_container = st.empty()
        detailed_container = st.empty()

    async for output in runnable.astream_log(inputs, include_types=["llm"]):
        for op in output.ops:
            if current_node in ['router', 'expander', 'multiquery']:
                percent_complete = (min(33, percent_complete + 1))
                status_container.progress(percent_complete, text='Generating Topic Expansion...')
            elif current_node == 'retrieval':
                percent_complete = (min(66, percent_complete + 1))
                status_container.progress(percent_complete, text='Retrieving Knowledge...')
            elif current_node in ['filter', 'prompt_synthesis']:
                percent_complete = (min(75, percent_complete + 1))
                status_container.progress(percent_complete, text='Compressing and Synthesizing Query...')
            elif current_node in ['chat', 'summarizer']:
                status_container.empty()
            if type(op['value'])==dict and op['value'].get('metadata') and op['value']['metadata'].get('langgraph_node'):
                if current_node == 'chat' and current_node != op['value']['metadata']['langgraph_node']:
                    chat_response = chat_response.replace('[],','').replace('[].', '').replace('[]', '')
                    container.markdown(chat_response)
                current_node = op['value']['metadata']['langgraph_node']
            elif op["path"].startswith("/logs/") and op["path"].endswith("/streamed_output/-") and current_node == 'chat':
                chat_response += op["value"].content
                chat_response = replace_doc_placeholders(chat_response, readable_sources)
                container.markdown(chat_response)
            elif op["path"] == "/streamed_output/-" and current_node == 'summarizer':
                final_sources = op['value']['summarizer']['sources']
            elif op["path"] == "/streamed_output/-" and current_node == 'filter':
                try:
                    readable_sources = (op['value']['filter']['filtered_retrieved'])
                    readable_sources_str = '\n\n---\n\n'.join([f'{s.page_content}\n\nSource: {s.metadata}' for s in readable_sources if len(s.metadata) > 1])
                except:
                    current_node = ''

    if final_sources != 'n/a':
        with sources_container.expander('Sources'):
            st.caption(final_sources)
    else:
        sources_container.caption('Sources: '+final_sources)
    with detailed_container.expander('Detailed Sources'):
        st.caption(readable_sources_str)
                

input_text = st.text_input(label='')
if input_text:
    try:
        with st.spinner('Generating...'):
            asyncio.run(rungraph(inputs={'prompt': input_text}, runnable=app))    
    except Exception as e:
        st.error(f"Error: {e}")


