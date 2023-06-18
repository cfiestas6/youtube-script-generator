import os
import streamlit as st

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
from api_key import OPENAI_API_KEY

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# ---------------
# App framework
# ---------------
st.title('Youtube GPT Script Generator')
prompt = st.text_input('Enter the topic of the video here:')

# ---------------
# Prompt templates
# ---------------
title_template = PromptTemplate(
        input_variables = ['topic'],
        template = 'Write a youtube video title about {topic}'
)

script_template = PromptTemplate(
        input_variables = ['title', 'wikipedia_research'],
        template = 'Write a youtube video script based on this title : {title}. While leverging this wikipedia research: {wikipedia_research}'
)

# ---------------
# Memory
# ---------------

title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

# ---------------
# LLMs
# ---------------
llm = OpenAI(temperature=0.5)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

wiki = WikipediaAPIWrapper()

if prompt:
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title=title, wikipedia_research=wiki_research)
    st.write(title)
    st.write(script)

    with st.expander('Title history'):
        st.info(title_memory.buffer)
    with st.expander('Script history'):
        st.info(script_memory.buffer)
    with st.expander('Wikipedia research'):
        st.info(wiki_research)
