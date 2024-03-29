from langchain import FAISS, PromptTemplate
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase
import os
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

# Define context and input for test cases
context= [
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

# Create test cases
test_cases = [
    LLMTestCase(
        input="What are the symptoms of coronavirus?",
        actual_output=covid_qa_chain.run("What are the symptoms of coronavirus?"),
        context=context
    ),
    LLMTestCase(
        input="How many planets are there in the solar system?",
        actual_output=covid_qa_chain.run("How many planets are there in the solar system?"),
        context=context
    )
]

# Measure metrics
metrics_results = []
metric = HallucinationMetric(threshold=0.5)
for test_case in test_cases:
    metric.measure(test_case)
    metrics_results.append({
        'Test Case': test_case.input,
        'Metric': metric.__class__.__name__,
        'Score': metric.score,
        'Reason': metric.reason,
        'Input': test_case.input,
        'Actual Output': test_case.actual_output,
        'Status': 'Pass' if metric.score > 0.5 else 'Fail'
    })

# Create HTML report
    table_columns = ['Serial No.', 'Test Case', 'Metric', 'Score', 'Reason', 'Input', 'Actual Output', 'Status']
    report_df = pd.DataFrame(metrics_results, columns=table_columns)
    report_df.drop(columns=['Serial No.'], inplace=True)

    # Generate sequential serial numbers
    report_df.insert(0, 'Serial No.', range(1, len(report_df) + 1))

    overall_pass_count = (report_df['Status'] == 'Pass').sum()
    overall_fail_count = (report_df['Status'] == 'Fail').sum()
    total_test_cases = len(report_df)

# Save the HTML report to a file
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
    <div class="filter-container">
        <form id="filter-form">
            <label for="column-select" style="color: black; font-size: 20px;">Where</label>
            <select id="column-select">
                <option value="Serial No.">Serial No.</option>
                <option value="Test Case">Test Case</option>
                <option value="Metric">Metric</option>
                <option value="Score">Score</option>
                <option value="Reason">Reason</option>
                <option value="Input">Input</option>
                <option value="Actual Output">Actual Output</option>
                <option value="Status">Status</option>
            </select>
            <select id="operator-select">
                <option value="is">is</option>
                <option value="is not">is not</option>
                <option value="contains">contains</option>
            </select>
            <input type="text" id="value-input" placeholder="Value...">
            <button type="button" onclick="filterTable()">Apply Filter</button>
        </form>
    </div>
    <section id="test-cases">
        <h2 class="mt-4">Test Results&#128202</h2>
        <div class="divider"></div>
        <table class="table table-striped" id="metrics-table">
            <thead>
                <tr>
                    <th>Serial No.</th>
                    <th>Test Case</th>
                    <th>Metric</th>
                    <th>Score</th>
                    <th>Reason</th>
                    <th>Input</th>
                    <th>Actual Output</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
""")

    pass_fail_counts = report_df.groupby('Metric')['Status'].value_counts().unstack(fill_value=0)

    for index, row in report_df.iterrows():
        status_class = "pass-status" if row['Status'] == 'Pass' else "fail-status"
        file.write(f"<tr>")
        for col in table_columns:
            if col == 'Status':
                file.write(f"<td class='{status_class}'>{row[col]}</td>")
            else:
                file.write(f"<td>{row[col]}</td>")
        file.write(f"</tr>")

    file.write("""
            </tbody>
        </table>
    </section>
""")

    # Generate summary box HTML
    file.write("""
    <div class="divider"></div>       
    <section id="summary-box"  style="color: black;">
        <h1 class="mt-4">Summary:</h1>
        <div class="divider"></div>
        <div>
            <p>Total Test Cases: {total_test_cases}</p>
            <p>Overall Pass/Fail Counts: Pass - {overall_pass_count}, Fail - {overall_fail_count}</p>
            <p>Pass/Fail Counts by Metric:</p>
            <ul>
    """.format(total_test_cases=total_test_cases, overall_pass_count=overall_pass_count, overall_fail_count=overall_fail_count))
    for metric in pass_fail_counts.index:
        pass_count = pass_fail_counts.loc[metric, 'Pass']
        fail_count = pass_fail_counts.loc[metric, 'Fail']
        file.write("""
                <li>{metric}: Pass - {pass_count}, Fail - {fail_count}</li>
        """.format(metric=metric, pass_count=pass_count, fail_count=fail_count))
    file.write("""
            </ul>
        </div>
    </section>
    <div class="divider"></div>      
""")

    # Create pie charts
    metrics = report_df['Metric'].unique()
    for metric in metrics:
        metric_df = report_df[report_df['Metric'] == metric]
        pass_fail_counts = metric_df['Status'].value_counts()
        pass_color = 'green'
        fail_color = 'red'

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(pass_fail_counts, labels=pass_fail_counts.index, autopct='%1.1f%%',
               colors=[pass_color, fail_color],
               textprops=dict(color="w"), shadow=True)
        ax.set_title(f'{metric} Pass/Fail Distribution', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{metric.lower()}_pie_chart.png')
        plt.close()

        file.write(f"""
        <section id='{metric.lower().replace(" ", "_")}-pie-chart'>
            <h2 class="mt-4">{metric} Pass/Fail Distribution</h2>
            <img src='{metric.lower()}_pie_chart.png' alt='{metric} Pass/Fail Distribution' class="img-fluid">
        </section>
        """)

        # Calculate pass/fail summary
        pass_fail_summary = metric_df['Status'].value_counts()
        fail_count = pass_fail_summary.get('Fail', 0)

        file.write(f"""
        <section id='{metric.lower().replace(" ", "_")}-pass-fail-summary'>
            <h3 class="mt-4">{metric} Pass/Fail Summary</h3>
            <table class="table table-striped" id="{metric.lower().replace(" ", "_")}-summary-table">
                <thead>
                    <tr>
                        <th>Status</th>
                        <th>Count</th>
                    </tr>
                </thead>
                <tbody>
            <tr>
                <td class="pass-status">Pass</td>
                <td>{pass_count}</td>
            </tr>
            <tr>
                <td class="fail-status">Fail</td>
                <td>{fail_count}</td>
            </tr>
        </tbody>
    </table>
</section>
""".format(metric=metric, pass_count=pass_count, fail_count=fail_count))

    file.write("""
        </div>
        <script>
            function filterTable() {
                var column = document.getElementById("column-select").value;
                var operator = document.getElementById("operator-select").value;
                var value = document.getElementById("value-input").value.trim().toLowerCase(); // Trim whitespace
                var table = document.getElementById("metrics-table");
                var rows = table.getElementsByTagName("tr");

                if (value === "") {
                    // If value is empty, do nothing
                    return;
                }

                for (var i = 1; i < rows.length; i++) { // Start from index 1 to skip header row
                    var cells = rows[i].getElementsByTagName("td");
                    var cellText = cells[column === 'Serial No.' ? 0 : column === 'Test Case' ? 1 : column === 'Metric' ? 2 : column === 'Score' ? 3 : column === 'Reason' ? 4 : column === 'Input' ? 5 : column === 'Actual Output' ? 6 : 7].innerText.toLowerCase();
                    if (operator === "is" && cellText !== value) {
                        rows[i].style.display = "none";
                    } else if (operator === "is not" && cellText === value) {
                        rows[i].style.display = "none";
                    } else if (operator === "contains" && !cellText.includes(value)) {
                        rows[i].style.display = "none";
                    } else {
                        rows[i].style.display = "";
                    }
                }
            }
        </script>
    </div>
    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.10.2/dist/umd/popper.min.js"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
</body>
</html>
""")


print("HTML report generated successfully.")
