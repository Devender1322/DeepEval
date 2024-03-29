import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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
    # Count the number of passing and failing test cases
    pass_count = sum(1 for result in metric_results if result[2] >= 0.7)
    fail_count = sum(1 for result in metric_results if result[2] < 0.7)

    # Create a pie chart with passing and failing percentages
    labels = ['Pass', 'Fail']
    values = [pass_count, fail_count]

    plt.figure(figsize=(8, 6))
    wedges, texts, autotexts = plt.pie(values, labels=labels, autopct='%1.1f%%', colors=['#4CAF50', '#FFC107'])
    
    plt.title('Pass/Fail Distribution')
    plt.tight_layout()

    # Save the chart as an image
    plt.savefig('pie_chart.png')

def animate_pie_chart(metric_results):
    # Count the number of passing and failing test cases
    pass_count = sum(1 for result in metric_results if result[2] >= 0.7)
    fail_count = sum(1 for result in metric_results if result[2] < 0.7)

    # Create a pie chart with passing and failing percentages
    labels = ['Pass', 'Fail']
    values = [pass_count, fail_count]

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%', colors=['#4CAF50', '#FFC107'])

    ax.legend(wedges, labels, title="Status", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    ax.set_title('Pass/Fail Distribution')

    def update(frame):
        # Pause for 0.5 seconds between frames
        plt.pause(0.5)
        return []

    ani = FuncAnimation(fig, update, frames=range(10), repeat=False, blit=True)
    return ani.to_jshtml()

def log_report_table(metrics_results, animated_pie_chart):
    table_columns = ['Test Case', 'Metric', 'Score', 'Reason']
    report_df = pd.DataFrame(metrics_results, columns=table_columns)

    # Save the HTML report to a file
    with open("metrics_report.html", "w") as file:
        file.write(f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Metrics Report</title>
            <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
            <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body {{
                    font-family: 'Arial', sans-serif;
                    background-color: #f8f9fa;
                    color: #495057;
                    margin: 0;
                }}

                .container {{
                    max-width: 800px;
                    margin: 0 auto;
                    background-color: #ffffff;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    padding: 20px;
                    border-radius: 10px;
                    margin-top: 20px;
                }}

                header {{
                    background-color: #343a40;
                    padding: 20px;
                    color: #fff;
                    text-align: center;
                    border-radius: 10px 10px 0 0;
                }}

                h1, h2 {{
                    color: #007bff;
                }}

                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 20px;
                }}

                table, th, td {{
                    border: 1px solid #dee2e6;
                }}

                th, td {{
                    padding: 12px;
                    text-align: left;
                }}

                th {{
                    background-color: #007bff;
                    color: #fff;
                }}

                img {{
                    max-width: 100%;
                    height: auto;
                    margin-top: 20px;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                }}
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
                            {report_df.to_html(index=False, escape=False)}
                        </tbody>
                    </table>
                </section>

                <section id='pass-fail-distribution'>
                    <h2 class="mt-4">Pass/Fail Distribution</h2>
                    {animated_pie_chart}
                </section>
            </div>
            <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.10.2/dist/umd/popper.min.js"></script>
            <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
            <script>
                $(document).ready(function() {{
                    // Event listener for dropdown change
                    $('#select-metric').on('change', function() {{
                        var selectedMetric = $(this).val();
                        updateTable(selectedMetric);
                    }});

                    function updateTable(metric) {{
                        // Fetch data based on the selected metric
                        var metricData = fetchDataForMetric(metric);

                        // Update the table content
                        var tableBody = $('#metrics-table tbody');
                        tableBody.empty();

                        $.each(metricData, function(index, row) {{
                            var tableRow = $('<tr>');
                            $.each(row, function(_, value) {{
                                tableRow.append($('<td>').text(value));
                            }});
                            tableBody.append(tableRow);
                        }});
                    }}

                    // Fetch data based on the selected metric
                    function fetchDataForMetric(metric) {{
                        // Implement logic to fetch data for the selected metric
                        // You might want to make an AJAX request to the server or use predefined data
                        // For simplicity, return a dummy array here
                        return {report_df.to_dict(orient='records')};
                    }}
                }});
            </script>
        </body>
        </html>
        """)

def main():
    test_case = create_test_case()
    metric_results = measure_metrics([test_case])
    create_pie_chart(metric_results)
    animated_pie_chart = animate_pie_chart(metric_results)
    log_report_table(metric_results, animated_pie_chart)

if __name__ == "__main__":
    main()
