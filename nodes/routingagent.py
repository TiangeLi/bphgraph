from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from constants import ROUTERLLM

router_template = \
f"""Given the user query, reply either YES or NO.

NO: if the user query is general/unrelated to BPH and does not require information from a BPH knowledge base
YES: if the user query is or could be about benign prostate hyperplasia (BPH), including definitions, symptoms, diagnosis, testing, management, treatment, related medications/medical therapy, surgical therapy, complications, side effects, or other topics broadly related to BPH.
If in doubt, reply YES.

Conversation summary:
```{{summary}}```"""


router_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', router_template),
        ('human', 'Query: {question}')
    ]

)
router_chain = router_prompt | ROUTERLLM | StrOutputParser()