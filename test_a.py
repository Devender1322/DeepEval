from langchain import FAISS, PromptTemplate
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from deepeval import assert_test
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
import os
import json
import pandas as pd
import matplotlib.pyplot as plt


# Set the OpenAI API Key environment variable.
# os.environ["OPENAI_API_KEY"] = "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"
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

# An updated version of the class exists in the langchain-openai package and should be used instead. To use it run `pip install -U langchain-openai` and import as `from langchain_openai import OpenAIEmbeddings`.
"""
"""
# The hallucination metric determines
# whether your LLM generates factually correct information by comparing the actual_output to the provided context.

input = "What are the symptoms of coronavirus?"  # Pass
input2 = "How many planets are there in the solar system?"  # Fail


metric = HallucinationMetric(threshold=0.5, model="gpt-3.5-turbo", include_reason="true")
test_case = LLMTestCase(
    input=input2,
    actual_output=covid_qa_chain.run(input2),
    context=context
)
assert_test(test_case, [metric])
print(covid_qa_chain.run(input2))
print(metric.score)
print(metric.reason)

# Save metric results
metric_results = {
    'Test Case': test_case.id,
    'Metric': metric.__class__.__name__,
    'Score': metric.score,
    'Reason': metric.reason,
    'Input': test_case.input,
    'Actual Output': test_case.actual_output,
}

# Generate HTML report
def create_report(metric_results):
    table_columns = ['Test Case', 'Metric', 'Score', 'Reason', 'Input', 'Actual Output']
    report_df = pd.DataFrame(metric_results, columns=table_columns)

    # Generate HTML report
    with open("metrics_report.html", "w") as file:
        file.write("""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Metrics Report</title>
                <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
                <style>
                    body {
                        font-family: "Times New Roman", Times, serif;
                        background-color: #f8f9fa;
                        color: #495057;
                        margin: 0;
                    }

                    .container {
                        max-width: 100%;
                        margin: auto;
                        background-color: #ffffff;
                        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                        border-radius: 10px;
                        padding: 20px;
                    }

                    header {
                        background-color: #343a40;
                        padding: 30px;
                        color: #fff;
                        text-align: center;
                        border-radius: 10px 10px 0 0;
                    }

                    h1, h2 {
                        color: #00C0FF;
                    }

                    table {
                        width: 100%;
                        border-collapse: collapse;
                        margin-top: 20px;
                    }

                    table, th, td {
                        border: 2px solid #dee2e6;
                    }

                    th, td {
                        padding: 12px;
                        text-align: left;
                    }

                    th {
                        background-color: #4A5275;
                        color: #fff;
                        text-align: center;
                    }

                    img {
                        max-width: 100%;
                        height: auto;
                        margin-top: 20px;
                        border-radius: 10px;
                        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    }

                    .filter-container {
                        margin-top: 20px;
                        padding: 20px;
                        background-color: #e9ecef;
                        border-radius: 10px;
                    }

                    .filter-container label {
                        font-weight: bold;
                        margin-right: 10px;
                    }

                    .filter-container select, .filter-container input[type="text"] {
                        padding: 10px;
                        border-radius: 5px;
                        border: 1px solid #ced4da;
                        outline: none;
                    }

                    .filter-container button {
                        padding: 10px 20px;
                        background-color: #4A5275;
                        color: #fff;
                        border: none;
                        border-radius: 5px;
                        cursor: pointer;
                    }

                    .filter-container button:hover {
                        background-color: #343a40;
                    }

                    .pass-status {
                        color: green;
                        font-weight: bold;
                    }

                    .fail-status {
                        color: red;
                        font-weight: bold;
                    }
                    .divider {
                        border-top: 2px solid #dee2e6;
                        margin-top: 20px;
                        margin-bottom: 20px;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <header>
                        <h1 class="display-4">Metrics Report&#128240</h1>
                    </header>
                    <section id="test-cases">
                        <h2 class="mt-4">Test Results&#128202</h2>
                        <div class="divider"></div>
                        <table class="table table-striped" id="metrics-table">
                            <thead>
                                <tr>
                                    <th>Test Case</th>
                                    <th>Metric</th>
                                    <th>Score</th>
                                    <th>Reason</th>
                                    <th>Input</th>
                                    <th>Actual Output</th>
                                </tr>
                            </thead>
                            <tbody>
        """)

        for index, row in report_df.iterrows():
            file.write(f"<tr>")
            for col in table_columns:
                file.write(f"<td>{row[col]}</td>")
            file.write(f"</tr>")

        file.write("""
                            </tbody>
                        </table>
                    </section>
                </div>
                <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
                <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.10.2/dist/umd/popper.min.js"></script>
                <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
            </body>
            </html>
        """)

# Call the function to create the report
create_report([metric_results])
