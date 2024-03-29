import os
from langchain import FAISS, PromptTemplate
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from deepeval import assert_test
from deepeval.metrics import GEval, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
import pandas as pd
import matplotlib.pyplot as plt

# Set the OpenAI API Key environment variable.
os.environ["OPENAI_API_KEY"] = "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"

# Prepare vector store (FAISS) with IPPC report
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, add_start_index=True)
loader = PyPDFLoader("DatasetCovid19.pdf")
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

# Define reporting functionality
def generate_report(metric_results):
    # Convert metric results to DataFrame
    df = pd.DataFrame(metric_results)

    # Create and save pie chart
    pass_fail_counts = df['Status'].value_counts()
    colors = ['#4CAF50', '#FF0000']
    explode = (0.1, 0)
    fig, ax = plt.subplots(figsize=(12, 6))
    wedges, _, autotexts = ax.pie(pass_fail_counts, labels=pass_fail_counts.index, autopct='%1.1f%%', colors=colors,
                                   textprops=dict(color="w"), explode=explode)
    ax.legend(wedges, pass_fail_counts.index, title="Status", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.setp(autotexts, size=10, weight="bold")
    ax.set_title('Pass/Fail Distribution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('pie_chart_3d.png')

    # Create and save HTML report
    with open("metrics_report.html", "w") as file:
        # Write HTML content
        file.write("<!DOCTYPE html>\n")
        file.write("<html>\n<head>\n<title>Metrics Report</title>\n</head>\n<body>\n")
        # Write table header
        file.write("<table>\n<tr><th>Test Case</th><th>Metric</th><th>Score</th><th>Reason</th><th>Input</th>"
                   "<th>Actual Output</th><th>Status</th></tr>\n")
        # Write table rows
        for index, row in df.iterrows():
            file.write(f"<tr><td>{row['Test Case']}</td><td>{row['Metric']}</td><td>{row['Score']}</td>"
                       f"<td>{row['Reason']}</td><td>{row['Input']}</td><td>{row['Actual Output']}</td>"
                       f"<td>{row['Status']}</td></tr>\n")
        file.write("</table>\n</body>\n</html>")

# Define context and retrieval_context
context = [
    "Coronaviruses are a large family of respiratory viruses that includes COVID-19, Middle East Respiratory Syndrome (MERS), and Severe Acute Respiratory Syndrome (SARS).",
    "Coronaviruses cause diseases in animals and humans. They often circulate among camels, cats, and bats, and can sometimes evolve and infect people.",
    "Its symptoms depend on the virus, but in humans common signs include mild respiratory infections, like the common cold, fever, cough, shortness of breath, and breathing difficulties",
    "In more severe cases, infection can cause pneumonia, severe acute respiratory syndrome, kidney failure and even death.",
    "There are two hypotheses as to COVID-19's origins: exposure to an infected animal or a laboratory leak. There is not enough evidence to support either argument.",
    "The novel coronavirus (SARS-CoV-2) that causes COVID-19 first emerged in the Chinese city of Wuhan in 2019 and was declared a pandemic by the World Health Organization (WHO).",
    "The 2020 lockdown in India left tens of millions of migrant workers unemployed. With factories and workplaces shut down, many migrant workers were left with no livelihood. They thus decided to walk hundreds of kilometers to go back to their native villages, accompanied by their families in many cases. In response, the central and state governments took various measures to help them. The central government then announced that it had asked state governments to set up immediate relief camps for the migrant workers returning to their native states,and later issued orders protecting the rights of the migrants.",
    "On 16 March 2020, the union government ordered the closure of schools and colleges, Exams were postponed or cancelled across the country. E-learning as well as online classes were made compulsory. Only a few educational institutions in India have been able to effectively adapt to e-learning and remote learning, which was further impacted by serious electricity issues and lack of internet connectivity.",
    "There were various COVID-19 vaccines available namely Pfizer-BioNTech and Moderna mRNA vaccines, Johnson & Johnson’s Janssen (J&J/Janssen) viral vector vaccine (expired as of May 6, 2023, and is no longer available in the U.S. ), Novavax protein subunit vaccine, etc.Pfizer-BioNTech's and Moderna’s had a lot of success, Ayurvedic medicines, yoga, exercise as well as good food to increase immunity was being included in lifestyle of people."
]

retrieval_context = [
    "Coronaviruses are a large family of respiratory viruses that includes COVID-19, Middle East Respiratory Syndrome (MERS), and Severe Acute Respiratory Syndrome (SARS)., Its symptoms depend on the virus, but in humans common signs include mild respiratory infections, like the common cold, fever, cough, shortness of breath, and breathing difficulties and could be fatal also, The migrant workers suffered a lot during the lockdown imposed by the government",
    " Online classes were made mandatory by schools and institutions to provide education remotely amidst the pandemic,Vaccinations were being provided in various centers and healthy lifestyle was being promoted."
]

# Define input for testing
input_query = "What are the symptoms of coronavirus?"

# Define metric for testing
metric = AnswerRelevancyMetric(threshold=0.5, model="gpt-3.5-turbo", include_reason=True)

# Create test case
test_case = LLMTestCase(
    input=input_query,
    actual_output=covid_qa_chain.run(input_query),
    retrieval_context=retrieval_context
)

# Assert the test case
assert_test(test_case, [metric])

# Generate report
generate_report([{
    'Test Case': test_case.input,
    'Metric': metric.__class__.__name__,
    'Score': metric.score,
    'Reason': metric.reason,
    'Input': test_case.input,
    'Actual Output': test_case.actual_output,
    'Status': 'Pass' if metric.score > metric.threshold else 'Fail'
}])

# Print output
print(covid_qa_chain.run(input_query))
print(metric.score)
print(metric.reason)
