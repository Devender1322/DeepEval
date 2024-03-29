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

    # Create a pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(pass_fail_counts, labels=pass_fail_counts.index, autopct='%1.1f%%', colors=['#4CAF50', '#FFC107'])
    plt.title('Pass/Fail Distribution')
    plt.tight_layout()

    # Save the chart as an image
    plt.savefig('pie_chart.png')

def log_report_table(metrics_results):
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
