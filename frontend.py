import streamlit as st
import asyncio
import re
import requests
import json
import os
from dotenv import load_dotenv
load_dotenv()

backend_url = os.getenv('BACKEND_URL')

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
    headings = [s['metadata'][list(s['metadata'].keys())[-1]] if len(s['metadata'])>1 else '' for s in readable_sources]
    headings_list = [f'[Source {num+1}](#{heading.lower().strip().strip(':-.').translate(link_trans_table).replace('---', '-').replace('--', '-').strip('-')})' if heading else '[]' for num, heading in enumerate(headings)]
    def replacement(match):
        doc_number = match.group(1)
        return headings_list[int(doc_number)-1]
    pattern = r'<doc_#(\d+)>'
    return re.sub(pattern, replacement, input_string)

async def rungraph(inputs):
    cols = st.columns(2)
    curr_node = ''
    content = ''
    final_sources = ''
    readable_sources = []
    readable_sources_str = ''
    chat_response = ''  

    with cols[0]:
        status_container = st.empty()
        container = st.empty()
    with cols[1]:
        sources_container = st.empty()
        detailed_container = st.empty()
    
    with requests.post(backend_url, json=inputs, stream=True) as resp:
        for line in resp.iter_lines():
            line = json.loads(line)
            if 'node' in line:
                if curr_node == 'chat' and line['node'] != 'chat':
                    chat_response = chat_response.replace(', []', '').replace('. []', '').replace('[]', '')
                    container.markdown(chat_response)
                curr_node = line['node']
            elif 'content' in line:
                content = line['content']

            if curr_node == 'expander':
                status_container.progress(15, text='Generating Topic Expansion...')
            elif curr_node == 'multiquery':
                status_container.progress(30, text='Generating Topic Expansion...')
            elif curr_node == 'retrieval':
                status_container.progress(57, text='Retrieving Knowledge...')
            elif curr_node in ['filter', 'prompt_synthesis']:
                status_container.progress(80, text='Compressing and Synthesizing Query...')
            elif curr_node == 'chat':
                status_container.empty()

            if curr_node == 'chat' and content:
                chat_response += content
                chat_response = replace_doc_placeholders(chat_response, readable_sources)
                chat_response = chat_response.replace('[],', '').replace('[].', '')
                container.markdown(chat_response)
                content = ''
            elif curr_node == 'summarizer' and content:
                final_sources = content
                content = ''
            elif curr_node == 'filter' and content:
                readable_sources = content
                readable_sources_str = '\n\n---\n\n'.join([f'{s['page_content']}\n\nSource: {s['metadata']}' for s in readable_sources if len(s['metadata']) > 1])
                content = ''
        
            if final_sources != 'n/a':
                with sources_container.expander('Sources'):
                    st.caption(final_sources)
            else:
                sources_container.caption('Sources: '+final_sources)

            with detailed_container.expander('Detailed Sources'):
                st.caption(readable_sources_str)


input_text = st.text_input(label='Ask anything about BPH')
if input_text:
    try:
        with st.spinner('Generating...'):
            asyncio.run(rungraph(inputs={'prompt': input_text}))    
    except Exception as e:
        st.error(f"Error: {e}")


