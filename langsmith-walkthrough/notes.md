# LangSmith Walkthrough
My notes from completing this LangChain tutorial: [LangSmith Walkthrough](https://python.langchain.com/docs/langsmith/walkthrough/)

## Rough Notes

This Python script:
```python
import os
from uuid import uuid4
from rich import print
from langsmith import Client

from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_openai import ChatOpenAI

unique_id = uuid4().hex[0:8]

os.environ["LANGCHAIN_TRACING_V2"]  = "***"
os.environ["LANGCHAIN_PROJECT"]     = "***"
os.environ["LANGCHAIN_ENDPOINT"]    = "***"
os.environ["LANGCHAIN_API_KEY"]     = "***"
os.environ["OPENAI_API_KEY"]        = "***"

client = Client()

# Fetches the latest version of this prompt
prompt = hub.pull("wfh/langsmith-agent-prompt:5d466cbc")

llm = ChatOpenAI(
    model="gpt-3.5-turbo-16k",
    temperature=0,
)

tools = [
    DuckDuckGoSearchResults(
        name="duck_duck_go"
    ),
]

llm_with_tools = llm.bind_tools(tools)

runnable_agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)
```

Builds these components:
```bash
---LLM_WITH_TOOLS---
RunnableBinding(
    bound=ChatOpenAI(
        client=<openai.resources.chat.completions.Completions object at 0x00000254D25AF530>,
        async_client=<openai.resources.chat.completions.AsyncCompletions object at 0x00000254D2634800>,
        model_name='gpt-3.5-turbo-16k',
        temperature=0.0,
        openai_api_key=SecretStr('**********'),
        openai_proxy=''
    ),
    kwargs={
        'tools': [
            {
                'type': 'function',
                'function': {
                    'name': 'duck_duck_go',
                    'description': 'A wrapper around Duck Duck Go Search. Useful for when you need to answer questions about current events. Input should be a search query. Output is a JSON array of the query results',
                    'parameters': {'type': 'object', 'properties': {'query': {'description': 'search query to look up', 'type': 'string'}}, 'required': ['query']}
                }
            }
        ]
    }
)
(learn-langchain) PS C:\projects\learn-langchain\langsmith-walkthrough> python .\tutorial.py
---LLM----
ChatOpenAI(
    client=<openai.resources.chat.completions.Completions object at 0x000002E22EFDB740>,
    async_client=<openai.resources.chat.completions.AsyncCompletions object at 0x000002E22F0588C0>,
    model_name='gpt-3.5-turbo-16k',
    temperature=0.0,
    openai_api_key=SecretStr('**********'),
    openai_proxy=''
)
---TOOLS----
[DuckDuckGoSearchResults(name='duck_duck_go')]
---RUNNABLE_AGENT---
RunnableSequence(
    first=RunnableParallel(steps={'input': RunnableLambda(...), 'agent_scratchpad': RunnableLambda(...)}),
    middle=[
        ChatPromptTemplate(
            input_variables=['agent_scratchpad', 'input'],
            input_types={
                'agent_scratchpad': typing.List[typing.Union[langchain_core.messages.ai.AIMessage, langchain_core.messages.human.HumanMessage, langchain_core.messages.chat.ChatMessage, langchain_core.messages.system.SystemMessage,langchain_core.messages.function.FunctionMessage, langchain_core.messages.tool.ToolMessage]]
            },
            metadata={'lc_hub_owner': 'wfh', 'lc_hub_repo': 'langsmith-agent-prompt', 'lc_hub_commit_hash': '5d466cbc8466b1157dc921acb77125a564ae99e712fcde28f550f657149d32ea'},
            messages=[
                SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template='You are a helpful assistant.')),
                HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}')),
                MessagesPlaceholder(variable_name='agent_scratchpad')
            ]
        ),
        RunnableBinding(
            bound=ChatOpenAI(
                client=<openai.resources.chat.completions.Completions object at 0x000001BB5A7176E0>,
                async_client=<openai.resources.chat.completions.AsyncCompletions object at 0x000001BB5A798920>,
                model_name='gpt-3.5-turbo-16k',
                temperature=0.0,
                openai_api_key=SecretStr('**********'),
                openai_proxy=''
            ),
            kwargs={
                'tools': [
                    {
                        'type': 'function',
                        'function': {
                            'name': 'duck_duck_go',
                            'description': 'A wrapper around Duck Duck Go Search. Useful for when you need to answer questions about current events. Input should be a search query. Output is a JSON array of the query results',    
                            'parameters': {'type': 'object', 'properties': {'query': {'description': 'search query to look up', 'type': 'string'}}, 'required': ['query']}
                        }
                    }
                ]
            }
        )
    ],
    last=OpenAIToolsAgentOutputParser()
)
```

## Definitions

#### [Chains](https://python.langchain.com/docs/modules/chains/)
Chains refer to sequences of calls - whether to an LLM, a tool, or a data preprocessing step. The way you create chains with LangChain is with something called LCEL

#### [LCEL](https://python.langchain.com/docs/expression_language/) (LangChain Expression Language)
LCEL is a declarative way to easily compose chains together.

Simple example
```lcel
chain = prompt | model | output_parser
```

Less simple example
```lcel
composed_chain_with_lambda = (
    chain
    | (lambda input: {"joke": input})
    | analysis_prompt
    | model
    | StrOutputParser()
)
```


## Weird findings
Found several things that were likely supposed to behave differently
 - [LangChain Teacher](https://langchain-teacher-lcel.streamlit.app/?ref=blog.langchain.dev) is broken. You send a single response, and it returns a python traceback.
 - The tutorial references an agent prompt that's supposed to be available "in the [Hub here](https://smith.langchain.com/hub/wfh/langsmith-agent-prompt)." but it is not. Or rather, it requires authentication? I'm not sure yet.
 - There's an error with the `pip install` commands in the tutorial. A duplicate `--quiet` flag:
```
%pip install --upgrade --quiet  langchain langsmith langchainhub --quiet
%pip install --upgrade --quiet  langchain-openai tiktoken pandas duckduckgo-search --quiet
```