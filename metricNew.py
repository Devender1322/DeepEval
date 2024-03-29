import pandas as pd
import matplotlib.pyplot as plt
from deepeval.metrics import HallucinationMetric, BiasMetric, LatencyMetric, CostMetric
from deepeval.test_case import LLMTestCase

def create_test_case():
    return LLMTestCase(
        input="How many evaluation metrics does DeepEval offer?",
        actual_output="14+ evaluation metrics",
        context=["DeepEval offers 14+ evaluation metrics"],
        latency=0.8,
        cost=12,
        id="test_111"
    )

def measure_metrics(test_cases):
    metrics = [
        HallucinationMetric(threshold=0.7),
        BiasMetric(threshold=0.02),
        LatencyMetric(max_latency=0.5),
        CostMetric(max_cost=15)
    ]

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

def create_pie_chart(metric_results):
    df = pd.DataFrame(metric_results)

    # Determine pass/fail status based on some condition (you can customize this)
    df['Status'] = df['Score'] > 0.5  # Change the condition as needed

    # Count pass and fail values
    pass_fail_counts = df['Status'].value_counts()

    # Create a pie chart with custom colors
    plt.figure(figsize=(10, 6))
    plt.pie(pass_fail_counts, labels=pass_fail_counts.index, autopct='%1.1f%%', colors=['#4CAF50', '#FF5733'])
    plt.title('Pass/Fail Distribution', fontsize=16, color='#007bff')
    plt.tight_layout()

    # Save the chart as an image
    plt.savefig('pie_chart.png')

def log_report_table(metrics_results):
    table_columns = ['Test Case', 'Metric', 'Score', 'Reason', 'Input', 'Actual Output', 'Context', 'Status']
    report_df = pd.DataFrame(metrics_results, columns=table_columns)

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
            <script src="https://code.jquery.com/jquery-3.6.4.min.js"></script>
            <style>
                body {
                    font-family: 'Arial', sans-serif;
                    background-color: #f8f9fa;
                    color: #495057;
                    margin: 0;
                    padding: 20px;
                }

                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: #ffffff;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    border-radius: 10px;
                    margin-top: 20px;
                    overflow: hidden;
                }

                header {
                    background-color: #007bff;
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

                .img-container {
                    text-align: center;
                    margin-top: 20px;
                }

                img {
                    max-width: 70%;
                    height: auto;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                }

                .dropdown-container {
                    margin-top: 20px;
                    text-align: center;
                }

                select {
                    padding: 10px;
                    font-size: 16px;
                }
            </style>
        </head>
        <body>
            <div class='container'>
                <header><h1 class="display-4">Metrics Report</h1></header>
                
                <!-- Dropdown for selecting metrics -->
                <div class="dropdown-container">
                    <label for="metricDropdown">Select Metric:</label>
                    <select id="metricDropdown">
                        <option value="all">All</option>
                        <option value="HallucinationMetric">HallucinationMetric</option>
                        <option value="BiasMetric">BiasMetric</option>
                        <option value="LatencyMetric">LatencyMetric</option>
                        <option value="CostMetric">CostMetric</option>
                    </select>
                </div>
                
                <section id='test-cases'>
                    <h2 class="mt-4">Test Cases</h2>
                    """)

        file.write(report_df.to_html(index=False, classes='table table-striped', escape=False, table_id='metricsTable'))

        # Embed the pie chart
        file.write("""
                </section>
                <section id='pass-fail-distribution'>
                    <h2 class="mt-4">Pass/Fail Distribution</h2>
                    <div class="img-container">
                        <img src='pie_chart.png' alt='Pass/Fail Distribution' class="img-fluid">
                    </div>
                </section>
            </div>
            
            <script>
                $(document).ready(function() {
                    // Function to filter table based on selected metric
                    $("#metricDropdown").change(function() {
                        var selectedMetric = $(this).val();
                        
                        if (selectedMetric === 'all') {
                            $('#metricsTable tbody tr').show();
                        } else {
                            $('#metricsTable tbody tr').hide();
                            $('#metricsTable tbody tr[data-metric="' + selectedMetric + '"]').show();
                        }
                    });
                });
            </script>
            
            <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.10.2/dist/umd/popper.min.js"></script>
            <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
        </body>
        </html>
        """)

def main():
    test_case = create_test_case()
    metric_results = measure_metrics([test_case])
    create_pie_chart(metric_results)
    log_report_table(metric_results)

if __name__ == "__main__":
    main()
