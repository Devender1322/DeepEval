from deepeval.metrics import KnowledgeRetentionMetric
from deepeval.test_case import ConversationalTestCase,LLMTestCase

from langchain import FAISS, PromptTemplate
from langchain_openai import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from deepeval.dataset import EvaluationDataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from deepeval.utils import set_is_running_deepeval
import json
import pytest
from deepeval.test_case import ConversationalTestCase
from deepeval import assert_test
import os
import deepeval
import pandas as pd
from langchain.evaluation import load_dataset
# Set the OpenAI API Key environment variable.
#os.environ["OPENAI_API_KEY"] = "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"
#os.environ["OPENAI_API_KEY"]="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"
#os.environ["OPENAI_API_KEY"]="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"
os.environ["OPEN_API_KEY"]="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"

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, add_start_index=True)
loader = PyPDFLoader("DatasetCovid19.pdf")
db = FAISS.from_documents(loader.load_and_split(text_splitter), OpenAIEmbeddings())
dataset = EvaluationDataset()
with open('cot.json', 'r') as f:
    data = json.load(f)

# Access the entries
entries = data['entrie']
# Prepare QA chain
PROMPT_TEMPLATE = """You are the Medical Assistant, a helpful AI assistant.
  Your task is to answer common questions on Covid-19.
  You will be given a question and relevant contexts from the Covid19 report.
  Please provide short and clear answers based on the provided context. Be polite and helpful.

Context:
{context}

Question:
{question}

Your answer:
"""

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0,)
prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["question", "context"])
covid_qa_chain = RetrievalQA.from_llm(llm=llm, retriever=db.as_retriever(), prompt=prompt)

# Test that everything works
#covid_qa_chain.run({"query": "When and why was a lockdown implemented in India?"})


context= [
            "Lockdown was implemented on March, 2020 in India.Coronaviruses are a large family of respiratory viruses that includes COVID-19, Middle East Respiratory Syndrome (MERS), and Severe Acute Respiratory Syndrome (SARS).",
            "Coronaviruses cause diseases in animals and humans. They often circulate among camels, cats, and bats, and can sometimes evolve and infect people.",
            "Its symptoms depend on the virus, but in humans common signs include mild respiratory infections, like the common cold, fever, cough, shortness of breath, and breathing difficulties",
            "In more severe cases, infection can cause pneumonia, severe acute respiratory syndrome, kidney failure and even death.",
            "There are two hypotheses as to COVID-19's origins: exposure to an infected animal or a laboratory leak. There is not enough evidence to support either argument.",
            "The novel coronavirus (SARS-CoV-2) that causes COVID-19 first emerged in the Chinese city of Wuhan in 2019 and was declared a pandemic by the World Health Organization (WHO).",
            "The 2020 lockdown in India left tens of millions of migrant workers unemployed. With factories and workplaces shut down, many migrant workers were left with no livelihood. They thus decided to walk hundreds of kilometers to go back to their native villages, accompanied by their families in many cases. In response, the central and state governments took various measures to help them. The central government then announced that it had asked state governments to set up immediate relief camps for the migrant workers returning to their native states,and later issued orders protecting the rights of the migrants."
            
        ]
retrievalContext=[ 
                   "Coronaviruses are a large family of respiratory viruses that includes COVID-19, Middle East Respiratory Syndrome (MERS), and Severe Acute Respiratory Syndrome (SARS)",
                    " Online classes were made mandatory by schools and institutions to provide education remotely amidst the pandemic,Vaccinations were being provided in various centers and healthy lifestyle was being promoted",
                    "Its symptoms depend on the virus, but in humans common signs include mild respiratory infections, like the common cold, fever, cough, shortness of breath, and breathing difficulties and could be fatal also, The migrant workers suffered a lot during the lockdown imposed by the government."

                 ]

test_cases = []
for entry in entries:
    input_question = entry.get("input", None)
    actual_output = covid_qa_chain.run({"query":input_question})
    print("Question:",input_question)
    print("Response:",actual_output)
    test_case = [LLMTestCase(
        input=input_question,
        actual_output=actual_output)]
    """test_cases.append(test_case)

dataset = EvaluationDataset(test_cases=test_case)
conversational_test_case = ConversationalTestCase(messages=test_case)
@pytest.mark.parametrize(
    "conversational_test_case",
    dataset,
)

def test_customer_chatbot(test_case:conversational_test_cases):
    
    knowledge_rentention_metric= KnowledgeRetentionMetric(threshold=0.5,model="gpt-3.5-turbo",include_reason=True)
    assert_test(test_case, [knowledge_rentention_metric])  



@deepeval.on_test_run_end
def function_to_be_called_after_test_run():
    print("Test finished!") """







conversational_test_case = ConversationalTestCase(messages=test_case)
metric = KnowledgeRetentionMetric(threshold=0.5,model="gpt-3.5-turbo",include_reason=True)

metric.measure(conversational_test_case)
#assert_test(test_case=conversational_test_case,metrics=[metric])
print(metric.score)
print(metric.reason) 