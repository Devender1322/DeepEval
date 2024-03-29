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
            metric_results.append([test_case.id, metric.__class__.__name__, metric.score, metric.reason])

    return metric_results

def create_pie_chart(metric_results):
    df = pd.DataFrame(metric_results, columns=['Test Case', 'Metric', 'Score', 'Reason'])

    # Determine pass/fail status based on some condition (you can customize this)
    df['Status'] = df['Score'] > 0.5  # Change the condition as needed

    # Create a pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(df['Status'].value_counts(), labels=df['Status'].value_counts().index, autopct='%1.1f%%', colors=['#4CAF50', '#FFC107'])
    plt.title('Pass/Fail Distribution')
    plt.tight_layout()

    # Save the chart as an image
    plt.savefig('pie_chart.png')

def log_report_table(metrics_results):
    table_columns = ['Test Case', 'Metric', 'Score', 'Reason']
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
            <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
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

                    <label for="select-metric">Select Metric:</label>
                    <select id="select-metric" class="form-control">
                        <option value="HallucinationMetric">Hallucination Metric</option>
                        <option value="BiasMetric">Bias Metric</option>
                        <option value="LatencyMetric">Latency Metric</option>
                        <option value="CostMetric">Cost Metric</option>
                    </select>

                    <table id="metrics-table" class="table table-bordered table-striped">
                        <thead>
                            <tr>
                                <th>Test Case</th>
                                <th>Metric</th>
                                <th>Score</th>
                                <th>Reason</th>
                            </tr>
                        </thead>
                        <tbody>
                            <!-- Table content goes here -->
                        </tbody>
                    </table>
                </section>

                <section id='pass-fail-distribution'>
                    <h2 class="mt-4">Pass/Fail Distribution</h2>
                    <img src='pie_chart.png' alt='Pass/Fail Distribution' class="img-fluid">
                </section>
            </div>
            <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.10.2/dist/umd/popper.min.js"></script>
            <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
            <script>
                $(document).ready(function() {
                    // Initial load of the table
                    updateTable('HallucinationMetric');

                    // Event listener for dropdown change
                    $('#select-metric').on('change', function() {
                        var selectedMetric = $(this).val();
                        updateTable(selectedMetric);
                    });

                    function updateTable(metric) {
                        // Fetch data based on the selected metric (you need to implement this)
                        var metricData = fetchDataForMetric(metric);

                        // Update the table content
                        var tableBody = $('#metrics-table tbody');
                        tableBody.empty();

                        $.each(metricData, function(index, row) {
                            var tableRow = $('<tr>');
                            $.each(row, function(_, value) {
                                tableRow.append($('<td>').text(value));
                            });
                            tableBody.append(tableRow);
                        });
                    }

                    // Fetch data based on the selected metric (you need to implement this function)
                    function fetchDataForMetric(metric) {
                        // Implement logic to fetch data for the selected metric
                        // You might want to make an AJAX request to the server or use predefined data
                        // For simplicity, return a dummy array here
                        return [
                            ["test_111", metric, 0.8, "Reason 1"],
                            ["test_222", metric, 0.6, "Reason 2"],
                            // ... more rows ...
                        ];
                    }
                });
            </script>
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
