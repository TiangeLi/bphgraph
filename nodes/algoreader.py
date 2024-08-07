from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain

from typing_extensions import TypedDict, Annotated, Literal, Union

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

Given the user query and conversation context, extract information into a structured format as provided."""


class AlgorithmRecommendations(TypedDict):
    size: Annotated[Literal['None', '<30cc', '30-80cc', '>80cc'], ..., 'the size of the prostate if available in the query']
    q_s: Annotated[bool, ..., "is the user concerned about or interested in preservation of sexual function, including erectile and/or ejaculatory function?", ]
    q_m: Annotated[bool, ..., "is the user medically complicated (excluding bleeding risk), i.e. unfit or cannot have anesthesia for any reason?"]
    q_b: Annotated[bool, ..., "is the user at risk for bleeding or post procedural hematuria, such as patients on anticoagulation or antiplatelet therapy?"]



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
    _chain = prompt | ALGOLLM.with_structured_output(schema=AlgorithmRecommendations, method='json_schema', strict=True)
    _llmresp = _chain.invoke({'question': question, 'summary': summary})
    ans = {
        'size': algorithms.get(_llmresp['size'], {}),
        'q_s': algorithms['q_s'] if _llmresp['q_s'] else {},
        'q_m': algorithms['q_m'] if _llmresp['q_m'] else {},
        'q_b': algorithms['q_b'] if _llmresp['q_b'] else {}
    }
    ret = {
        'size': get_recs_sentence(header=f'Based solely on the patient\'s prostate size of {_llmresp['size']}:', input=ans['size']),
        'q_s': get_recs_sentence(header='Based solely on the patient\'s interest in preservation of sexual function (including erectile & ejaculatory function):', input=ans['q_s']),
        'q_m': get_recs_sentence(header='Based solely on the patient\'s medical complexity (i.e. unfit or cannot have anesthesia):', input=ans['q_m']),
        'q_b': get_recs_sentence(header='Based solely on the patient\'s risk for bleeding / hematuria (e.g. patients on anticoagulation or antiplatelet therapy):', input=ans['q_b']),
        'metadata': {k:v for k, v in ans.items() if v}
    }

    return ret