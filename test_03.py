import deepeval
from deepeval.dataset import EvaluationDataset
from deepeval import assert_test
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
    colors = ['#4CAF50', '#FF0000']

    # Add shadow to create a 3D effect
    shadow = True

    # Explode the slices for better separation
    explode = (0.1, 0)

    fig, ax = plt.subplots(figsize=(12, 6))
    wedges, texts, autotexts = ax.pie(pass_fail_counts, labels=pass_fail_counts.index, autopct='%1.1f%%', colors=colors,
                                      textprops=dict(color="w"), shadow=shadow, explode=explode)

    ax.legend(wedges, pass_fail_counts.index, title="Status", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.setp(autotexts, size=10, weight="bold")

    ax.set_title('Pass/Fail Distribution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('pie_chart_3d.png')


def log_report_table(metrics_results):
    table_columns = ['Test ID', 'Metric', 'Score', 'Reason', 'Input', 'Actual Output', 'Status']
    report_df = pd.DataFrame(metrics_results, columns=table_columns)

    with open("metricsTables_report.html", "w") as file:
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
                    max-width: 100%;  /* Set max-width to 100% for full-width */
                    margin: 0;
                    background-color: #ffffff;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    padding: 20px;
                    border-radius: 10px;
                    margin-top: 20px;
                }

                header {
                    background-color: #4A5275;
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
                    border: 2px solid #000;
                }

                table, th, td {
                    border: 2px solid #000;
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
            </style>
        </head>
        <body>
            <div class='container'>
                <header><h1 class="display-4">Metrics Report &#128240</h1></header>
                """)

        # Iterate through each metric and create a separate table for each
        for metric_name, metric_df in report_df.groupby('Metric'):
            file.write(f"<section id='test-cases'>")
            file.write(f"<h2 class='mt-4'>{metric_name}</h2>")
            file.write(metric_df.to_html(index=False, classes='table table-striped', escape=False))
            file.write(f"</section>")

        file.write("""
                <section id='pass-fail-distribution'>
                    <h2 class="mt-4">Pass/Fail Distribution</h2>
                    <img src='pie_chart_3d.png' alt='Pass/Fail Distribution' class="img-fluid">
                </section>
            </div>
            <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.10.2/dist/umd/popper.min.js"></script>
            <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
        </body>
        </html>
        """)


if __name__ == "__main__":
    # Run tests and generate reports
    pytest.main([__file__, '-v'])
