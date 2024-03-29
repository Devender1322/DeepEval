import deepeval
from deepeval.dataset import EvaluationDataset
from deepeval import assert_test, evaluate
from deepeval.metrics import HallucinationMetric, BiasMetric, LatencyMetric, CostMetric
from deepeval.test_case import LLMTestCase
import pytest
import json
import pandas as pd
import matplotlib.pyplot as plt

# Assuming EvaluationDataset and other necessary classes are imported properly

# Instantiate your EvaluationDataset
dataset = EvaluationDataset()

# Read the JSON file
with open('D:/vsfiles/medicalDataset.json', 'r') as f:
    data = json.load(f)

# Access the entries
entries = data['entrie']
context= [
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
    latency=entry.get("latency",None)    #The latency metric measures whether the completion time of your LLM (application) is efficient and meets the expected time limits.
    cost=entry.get("cost",None)   #The cost metric is another performance metric offered by deepeval, and measures whether the token cost of your LLM (application) is economically acceptable.
    id=entry.get("id",None)
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
    metrics = [
        HallucinationMetric(threshold=0.7),
        BiasMetric(threshold=0.2),
        LatencyMetric(max_latency=0.5),
        CostMetric(max_cost=10),
        # AnswerRelevancyMetric(threshold=0.5),
    ]
    assert_test(test_case, metrics)

    metric_results = []

    for metric in metrics:
        for test_case in test_cases:
            metric.measure(test_case)
            metric_results.append({
                'Test Case': test_case.id,
                'Metric': metric.__class__.__name__,
                'Score': metric.score,
                'Reason': metric.reason,
                'Input': test_case.input,
                'Actual Output': test_case.actual_output,
                'Context': ', '.join(test_case.context),
                'Status': 'Pass' if metric.score > 0.5 else 'Fail'
            })
    return metric_results



@deepeval.on_test_run_end
def function_to_be_called_after_test_run():
    print("Test finished!")


#metric = HallucinationMetric(threshold=0.7)

#metric.measure(test_case)

#5 all pass 6 all fail
    
def create_pie_chart(metric_results):
    df = pd.DataFrame(metric_results)

    # Determine pass/fail status based on some condition (you can customize this)
    df['Status'] = df['Score'] > 0.5  # Change the condition as needed

    # Count pass and fail values
    pass_fail_counts = df['Status'].value_counts()

    # Create a pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(pass_fail_counts, labels=pass_fail_counts.index, autopct='%1.1f%%', colors=['#4CAF50', '#FFC107'])
    plt.title('Pass/Fail Distribution')
    plt.tight_layout()

    # Save the chart as an image
    plt.savefig('pie_chart.png')

def log_report_table(metrics_results):
    table_columns = ['Test Case', 'Metric', 'Score', 'Reason', 'Input', 'Actual Output', 'Context', 'Status']
    report_df = pd.DataFrame(metrics_results, columns=table_columns)

    # Save the HTML report to a file
    with open("Main_report.html", "w") as file:
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
                    font-family: 'Arial', sans-serif;
                    background-color: #f8f9fa;
                    color: #495057;
                    margin: 0;
                }

                .container {
                    max-width: 800px;
                    margin: 0 auto;
                    background-color: #ffffff;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    padding: 20px;
                    border-radius: 10px;
                    margin-top: 20px;
                }

                header {
                    background-color: #343a40;
                    padding: 20px;
                    color: #fff;
                    text-align: center;
                    border-radius: 10px 10px 0 0;
                }

                h1, h2 {
                    color: #007bff;
                }

                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 20px;
                }

                table, th, td {
                    border: 1px solid #dee2e6;
                }

                th, td {
                    padding: 12px;
                    text-align: left;
                }

                th {
                    background-color: #007bff;
                    color: #fff;
                }

                img {
                    max-width: 100%;
                    height: auto;
                    margin-top: 20px;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                }
            </style>
        </head>
        <body>
            <div class='container'>
                <header><h1 class="display-4">Metrics Report</h1></header>
                <section id='test-cases'>
                    <h2 class="mt-4">Test Cases</h2>
                    """)

        file.write(report_df.to_html(index=False, classes='table table-striped', escape=False))

        # Embed the pie chart
        file.write("""
                </section>
                <section id='pass-fail-distribution'>
                    <h2 class="mt-4">Pass/Fail Distribution</h2>
                    <img src='pie_chart.png' alt='Pass/Fail Distribution' class="img-fluid">
                </section>
            </div>
            <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.10.2/dist/umd/popper.min.js"></script>
            <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
        </body>
        </html>
        """)

def main():
    # test_case = create_test_case()
    metric_results = test_customer_chatbot([test_case])
    create_pie_chart(metric_results)
    log_report_table(metric_results)

if __name__ == "__main__":
    main()