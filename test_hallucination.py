
from langchain import FAISS, PromptTemplate
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from deepeval import assert_test
from deepeval.metrics import  GEval, HallucinationMetric
from deepeval.test_case import LLMTestCase,LLMTestCaseParams
import os
from langchain_community.document_loaders import PyPDFLoader

# Set the OpenAI API Key environment variable.
#os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_API_KEY"] = "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"

# Prepare vector store (FAISS) with IPPC report
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, add_start_index=True)
loader = PyPDFLoader("D:/vsfiles/DatasetCovid19.pdf")
db = FAISS.from_documents(loader.load_and_split(text_splitter), OpenAIEmbeddings())

# Prepare QA chain
PROMPT_TEMPLATE = """You are the Medical Assistant, a helpful AI assistant made by Giskard.
  Your task is to answer common questions on Covid-19.
  You will be given a question and relevant excerpts from the Covid19 report.
  Please provide short and clear answers based on the provided context. Be polite and helpful.

Context:
{context}

Question:
{question}

Your answer:
"""

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)
prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["question", "context"])
covid_qa_chain = RetrievalQA.from_llm(llm=llm, retriever=db.as_retriever(), prompt=prompt)

# Test that everything works
covid_qa_chain.run({"query": "When and why was a lockdown implemented in India?"})

context= [
            "Coronaviruses are a large family of respiratory viruses that includes COVID-19, Middle East Respiratory Syndrome (MERS), and Severe Acute Respiratory Syndrome (SARS).",
            "Coronaviruses cause diseases in animals and humans. They often circulate among camels, cats, and bats, and can sometimes evolve and infect people.",
            "Its symptoms depend on the virus, but in humans common signs include mild respiratory infections, like the common cold, fever, cough, shortness of breath, and breathing difficulties",
            "In more severe cases, infection can cause pneumonia, severe acute respiratory syndrome, kidney failure and even death.",
            "There are two hypotheses as to COVID-19's origins: exposure to an infected animal or a laboratory leak. There is not enough evidence to support either argument.",
            "The novel coronavirus (SARS-CoV-2) that causes COVID-19 first emerged in the Chinese city of Wuhan in 2019 and was declared a pandemic by the World Health Organization (WHO).",
            "The 2020 lockdown in India left tens of millions of migrant workers unemployed. With factories and workplaces shut down, many migrant workers were left with no livelihood. They thus decided to walk hundreds of kilometers to go back to their native villages, accompanied by their families in many cases. In response, the central and state governments took various measures to help them. The central government then announced that it had asked state governments to set up immediate relief camps for the migrant workers returning to their native states,and later issued orders protecting the rights of the migrants.",
            "On 16 March 2020, the union government ordered the closure of schools and colleges, Exams were postponed or cancelled across the country. E-learning as well as online classes were made compulsory. Only a few educational institutions in India have been able to effectively adapt to e-learning and remote learning, which was further impacted by serious electricity issues and lack of internet connectivity.",
            "There were various COVID-19 vaccines available namely Pfizer-BioNTech and Moderna mRNA vaccines, Johnson & Johnson’s Janssen (J&J/Janssen) viral vector vaccine (expired as of May 6, 2023, and is no longer available in the U.S. ), Novavax protein subunit vaccine, etc.Pfizer-BioNTech's and Moderna’s had a lot of success, Ayurvedic medicines, yoga, exercise as well as good food to increase immunity was being included in lifestyle of people."  ]

retrieval_context= [
"Coronaviruses are a large family of respiratory viruses that includes COVID-19, Middle East Respiratory Syndrome (MERS), and Severe Acute Respiratory Syndrome (SARS)., Its symptoms depend on the virus, but in humans common signs include mild respiratory infections, like the common cold, fever, cough, shortness of breath, and breathing difficulties and could be fatal also, The migrant workers suffered a lot during the lockdown imposed by the government",
" Online classes were made mandatory by schools and institutions to provide education remotely amidst the pandemic,Vaccinations were being provided in various centers and healthy lifestyle was being promoted."
]

#An updated version of the class exists in the langchain-openai package and should be used instead. To use it run `pip install -U langchain-openai` and import as `from langchain_openai import OpenAIEmbeddings`.
"""
"""
#The hallucination metric determines 
# whether your LLM generates factually correct information by comparing the actual_output to the provided context.

input= "What are the symptoms of coronavirus?" #Pass
input2= "How many planets are there in the solar system?" #Fail


metric = HallucinationMetric(threshold=0.5,model="gpt-3.5-turbo", include_reason="true")
test_case = LLMTestCase(
        input=input2,
        actual_output=covid_qa_chain.run(input2),
        context=context
    )
assert_test(test_case, [metric])
print(covid_qa_chain.run(input2))
print(metric.score)
print(metric.reason)

