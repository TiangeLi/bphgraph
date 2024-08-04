import json
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
from constants import ALGOLLM

algorithms = {
    '<30cc': {
        'aua': ['HoLEP', 'ThuLEP', 'PVP', 'TUIP', 'TURP', 'TIPD', 'TUVP'],
        'cua': ['TUIP', 'Urolift', 'M/B-TURP'],
        'eau': ['TUIP', 'TURP']
    },
    '30-80cc': {
        'aua': ['HoLEP', 'ThuLEP', 'PVP', 'TURP', 'PUL', 'WVTT', 'RWT', 'TIPD', 'TUVP'],
        'cua': ['AEEP', 'Greenlight PVP', 'M/B-TURP', 'Urolift', 'Rezum', 'Aquablation', 'iTIND', 'TUMT'],
        'eau': ['Laser enucleation', 'Laser vaporisation', 'TURP', 'PU Lift / Urolift', 'Bipolar enucleation']
    },
    '>80cc': {
        'aua': ['Simple Prostatectomy (Open, Laparoscopic, Robotic)', 'HoLEP', 'ThuLEP'],
        'cua': ['Simple Prostatectomy (Open, Laparoscopic, Robotic)', 'AEEP', 'Greenlight PVP', 'Aquablation'],
        'eau': ['Open prostatectomy', 'HoLEP', 'ThuLEP', 'laser vaporisation', 'TURP', 'bipolar enucleation']
    },
    'q_s': { # sexual preservation (erectile & ejaculatory)
        'aua': ['PUL', 'WVTT']
    },
    'q_m': { # medically complicated (anesthesia risk)
        'aua': ['HoLEP', 'ThuLEP', 'PVP'],
        'cua': ['Urolift', 'Rezum', 'iTIND', 'TUMT'],
        'eau': ['PU Lift / Urolift']
    },
    'q_b': { # medically complicated (bleeding risk)
        'aua': ['HoLEP', 'ThuLEP', 'PVP'],
        'cua': ['AEEP', 'Greenlight PVP'],
        'eau': ['laser enucleation', 'laser vaporisation']
    }
}


sys_template = \
"""Last updated conversation context:
{summary}

Given a user query and the conversation context, extract the following information using the following template, using ONLY the provided answer choices:

```json
{{
    "size": "prostate size (choices: <30cc, 30-80cc, >80cc; none if not applicable)",
    "q_s": "is the user concerned about or interested in preservation of sexual function, including erectile and/or ejaculatory function? (choices: yes, no; none if not applicable)", 
    "q_m": "is the user medically complicated, i.e. unfit or cannot have anesthesia for any reason? (choices: yes, no; none if not applicable)",
    "q_b": "is the user at risk for bleeding or post procedural hematuria, such as patients on anticoagulation or antiplatelet therapy? (choices: yes, no; none if not applicable)",
}}
```

Remember to return only the json template, using only the categories and choices provided. Do not include any additional information or context in your response."""

prompt = ChatPromptTemplate.from_messages(
    [
        ('system', sys_template),
        ('human', "Query: {question}")
    ]
)

def get_recs_sentence(header: str, input: dict):
    lines = [f'{algo.upper()} Guidelines recommend: {recs}' for algo, recs in input.items()]
    return f'{header} {', '.join(lines)}.' if lines else None

@chain
def tx_algo_chain(_input):
    question, summary = _input['question'], _input['summary']
    _chain = prompt | ALGOLLM | StrOutputParser()
    _llmresp = _chain.invoke({'question': question, 'summary': summary})
    _llmresp = json.loads(_llmresp.strip('```json\n').strip('```'))
    ans = {
        'size': algorithms.get(_llmresp['size'], {}),
        'q_s': algorithms['q_s'] if _llmresp['q_s'].lower().strip() == 'yes' else {},
        'q_m': algorithms['q_m'] if _llmresp['q_m'].lower().strip() == 'yes' else {},
        'q_b': algorithms['q_b'] if _llmresp['q_b'].lower().strip() == 'yes' else {}
    }
    ret = {
        'size': get_recs_sentence(header=f'Based solely on the patient\'s prostate size of {_llmresp['size']}:', input=ans['size']),
        'q_s': get_recs_sentence(header='Based solely on the patient\'s interest in preservation of sexual function (including erectile & ejaculatory function):', input=ans['q_s']),
        'q_m': get_recs_sentence(header='Based solely on the patient\'s medical complexity (i.e. unfit or cannot have anesthesia):', input=ans['q_m']),
        'q_b': get_recs_sentence(header='Based solely on the patient\'s risk for bleeding / hematuria (e.g. patients on anticoagulation or antiplatelet therapy):', input=ans['q_b']),
        'metadata': {k:v for k, v in ans.items() if v}
    }

    return ret