#--------------------------------------------------------#
# Imports
#--------------------------------------------------------#
import streamlit as st
import json
import os
import uuid
import zipfile

import openai
import pinecone
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import Pinecone

#--------------------------------------------------------#
# Page Config
#--------------------------------------------------------#

st.set_page_config(
  page_title="Safe Docs Bot",
  page_icon="🪄",
  layout="wide",
)

#--------------------------------------------------------#
# Functions
#--------------------------------------------------------#

index_name = "safe"
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
PINECONE_API_KEY = os.environ['PINECONE_API_KEY']

pinecone.init(api_key=PINECONE_API_KEY, environment="gcp-starter")
index = pinecone.Index(index_name)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
docsearch = Pinecone.from_existing_index(index_name, embeddings)

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=docsearch.as_retriever()
)

#--------------------------------------------------------#
# Main Body
#--------------------------------------------------------#

# Create the title at the top of page
st.title('Safe Docs Bot 🪄')
st.subheader('Everything you need to know to start building on Safe')

with st.form("my_form"):
    text = st.text_area("Enter a question about Safe protocol:")
    submitted = st.form_submit_button("Submit")
    if submitted:
      st.info(qa.run(text))

