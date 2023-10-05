#--------------------------------------------------------#
# Imports
#--------------------------------------------------------#

from langchain.vectorstores import DeepLake
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import streamlit as st
import json
import argparse
import os
import openai

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

activeloop_username = 'kofi'
repo_name= 'safe-contracts'

activeloop_dataset_path = f"hub://{activeloop_username}/{repo_name}"

# Create an instance of OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

# Create an instance of DeepLake with the specified dataset path and embeddings
db = DeepLake(
    dataset_path=activeloop_dataset_path,
    read_only=True,
    embedding_function=embeddings,
)

# Create a retriever from the DeepLake instance
retriever = db.as_retriever()
# Set the search parameters for the retriever
retriever.search_kwargs["distance_metric"] = "cos"
retriever.search_kwargs["fetch_k"] = 100
retriever.search_kwargs["maximal_marginal_relevance"] = True
retriever.search_kwargs["k"] = 10
# Create a ChatOpenAI model instance
model = ChatOpenAI(model="gpt-4")
# Create a RetrievalQA instance from the model and retriever
qa = RetrievalQA.from_llm(model, retriever=retriever)

#--------------------------------------------------------#
# Main Body
#--------------------------------------------------------#

# Create the title at the top of page
st.title('Safe Contracts Bot 🪄')
st.subheader('A bot that knows everything about the Safe contracts')

with st.form("my_form"):
    text = st.text_area("Enter a question about Safe protocol:")
    submitted = st.form_submit_button("Submit")
    if submitted:
      st.info(qa.run(text))

