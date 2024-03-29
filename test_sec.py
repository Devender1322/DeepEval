import deepeval
from deepeval.dataset import EvaluationDataset
from deepeval import assert_test
from deepeval.metrics import HallucinationMetric, BiasMetric, LatencyMetric, CostMetric
from deepeval.test_case import LLMTestCase
import pytest
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from langchain_community.document_loaders import PyPDFLoader

# Set the OpenAI API Key environment variable.
#os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_API_KEY"] = "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"
# Assuming EvaluationDataset and other necessary classes are imported properly

# Instantiate your EvaluationDataset
dataset = EvaluationDataset()

# Read the JSON file
with open('D:/vsfiles/medicalDataset.json', 'r') as f:
    data = json.load(f)

# Access the entries
entries = data['entrie']
context = [
    "Coronaviruses are a large family of respiratory viruses that includes COVID-19, Middle East Respiratory Syndrome (MERS), and Severe Acute Respiratory Syndrome (SARS).",
    "Coronaviruses cause diseases in animals and humans. They often circulate among camels, cats, and bats, and can sometimes evolve and infect people.",
    "Its symptoms depend on the virus, but in humans common signs include mild respiratory infections, like the common cold, fever, cough, shortness of breath, and breathing difficulties",
    "In more severe cases, infection can cause pneumonia, severe acute respiratory syndrome, kidney failure and even death.",
    "There are two hypotheses as to COVID-19's origins: exposure to an infected animal or a laboratory leak. There is not enough evidence to support either argument.",
    "The novel coronavirus (SARS-CoV-2) that causes COVID-19 first emerged in the Chinese city of Wuhan in 2019 and was declared a pandemic by the World Health Organization (WHO).",
    "The 2020 lockdown in India left tens of millions of migrant workers unemployed. With factories and workplaces shut down, many migrant workers were left with no livelihood. They thus decided to walk hundreds of kilometers to go back to their native villages, accompanied by their families in many cases. In response, the central and state governments took various measures to help them. The central government then announced that it had asked state governments to set up immediate relief camps for the migrant workers returning to their native states,and later issued orders protecting the rights of the migrants."
]

test_cases = []
for entry in entries:
    input = entry.get("input", None)
    actual_output = entry.get("actual_output", None)       ## Get from LLM application by run() method
    expected_output = entry.get("expected_output", None)   ##actual_output=chatbot.run(prompt)
    latency = entry.get("latency", None)    # The latency metric measures whether the completion time of your LLM (application) is efficient and meets the expected time limits.
    cost = entry.get("cost", None)   # The cost metric is another performance metric offered by deepeval, and measures whether the token cost of your LLM (application) is economically acceptable.
    id = entry.get("id", None)
    context = context

    test_case = LLMTestCase(
        input=input,
        actual_output=actual_output,
        expected_output=expected_output,
        context=context,
        latency=latency,
        cost=cost,
        id=id
    )
    test_cases.append(test_case)

dataset = EvaluationDataset(test_cases=test_cases)


@pytest.mark.parametrize(
    "test_case",
    dataset,
)
def test_customer_chatbot(test_case: LLMTestCase):
    hallucination_metric = HallucinationMetric(threshold=0.7)
    bias_metric = BiasMetric(threshold=0.2)
    latency_metric = LatencyMetric(max_latency=0.5)
    cost_metric = CostMetric(max_cost=10)
    assert_test(test_case, [hallucination_metric, bias_metric, latency_metric, cost_metric])


@deepeval.on_test_run_end
def function_to_be_called_after_test_run():
    metric_results = []
    metrics = [
        HallucinationMetric(threshold=0.7),
        BiasMetric(threshold=0.2),
        LatencyMetric(max_latency=0.5),
        CostMetric(max_cost=10)
    ]

    for metric in metrics:
        for test_case in dataset:
            metric.measure(test_case)

            if isinstance(metric, (BiasMetric, LatencyMetric, CostMetric)):
                # For BiasMetric, LatencyMetric, and CostMetric, use a different Pass/Fail logic
                status = 'Pass' if metric.score <= metric.threshold else 'Fail'
            else:
                # For other metrics, use the threshold-based Pass/Fail logic
                status = 'Pass' if metric.score > metric.threshold else 'Fail'

            metric_results.append({
                'Test Case': test_case.id,
                'Metric': metric.__class__.__name__,
                'Score': metric.score,
                'Reason': metric.reason,
                'Input': test_case.input,
                'Actual Output': test_case.actual_output,
                # 'Context': ', '.join(test_case.context),
                'Status': status
            })

    # Create and save pie chart
    create_pie_chart(metric_results)

    # Create and save HTML report
    log_report_table(metric_results)
    print("Test finished!")

# Adding reporting functionality
def create_pie_chart(metric_results):
    df = pd.DataFrame(metric_results)
    pass_fail_counts = df['Status'].value_counts()

    # Custom colors for Pass (green) and Fail (red)
    pass_color = 'green'
    fail_color = 'red'

    fig, ax = plt.subplots(figsize=(12, 6))
    wedges, texts, autotexts = ax.pie(pass_fail_counts, labels=pass_fail_counts.index, autopct='%1.1f%%',
                                      colors=[pass_color, fail_color],
                                      textprops=dict(color="w"), shadow=True)

    ax.legend(wedges, pass_fail_counts.index, title="Status", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.setp(autotexts, size=10, weight="bold")

    ax.set_title('Pass/Fail Distribution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('pie_chart_3d.png')


def log_report_table(metrics_results):
    table_columns = ['Serial No.', 'Test Case', 'Metric', 'Score', 'Reason', 'Input', 'Actual Output', 'Status']
    report_df = pd.DataFrame(metrics_results, columns=table_columns)
    report_df.drop(columns=['Serial No.'], inplace=True)

    # Generate sequential serial numbers
    report_df.insert(0, 'Serial No.', range(1, len(report_df) + 1))

    overall_pass_count = (report_df['Status'] == 'Pass').sum()
    overall_fail_count = (report_df['Status'] == 'Fail').sum()
    total_test_cases = len(report_df)

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
                            <td>{pass_fail_summary['Pass']}</td>
                        </tr>
                        <tr>
                            <td class="fail-status">Fail</td>
                            <td>{pass_fail_summary['Fail']}</td>
                        </tr>
                    </tbody>
                </table>

            </section>
            """)

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
            <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.10.2/dist/umd/popper.min.js"></script>
            <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
        </body>
        </html>
        """)


if __name__ == "__main__":
    # Run tests and generate reports
    pytest.main([__file__, '-v'])
