from deepeval.metrics import HallucinationMetric, BiasMetric, LatencyMetric, CostMetric
from deepeval.test_case import LLMTestCase
import wandb
import wandb.apis.reports as wr
import pandas as pd

def initialize_wandb():
    wandb.init(project="LLMtest")

def create_test_case():
    return LLMTestCase(
        input="How many evaluation metrics does DeepEval offer?",
        actual_output="14+ evaluation metrics",
        context=["DeepEval offers 14+ evaluation metrics"],
        latency=0.8,
        cost=12,
        id="test_111"
    )

def create_report():
    return wr.Report(
        project="LLMtest",
        title='Metrics Report',
        description="Evaluation metrics report for LLM test"
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

def log_report_table(metrics_results):
    table_columns = ['Test Case', 'Metric', 'Score', 'Reason']
    report_df = pd.DataFrame(metrics_results, columns=table_columns)
    styled_report = (
        report_df.style
        .set_table_styles([
            {'selector': 'thead th', 'props': [('background-color', '#4285f4'), ('color', 'white')]},
            {'selector': 'tbody tr:hover', 'props': [('background-color', '#f5f5f5')]},
            {'selector': 'tbody td', 'props': [('border', '1px solid #ddd'), ('padding', '8px')]},
            {'selector': 'table', 'props': [('border-collapse', 'collapse'), ('width', '100%')]},
        ])
    )

    # Save the HTML report to a file
    with open("metrics_report.html", "w") as file:
        file.write(styled_report.to_html(index=False))

    wandb.log({"metrics_reportq": wandb.Html(styled_report.to_html(index=False))})

# def log_report_table(metrics_results):
#     table_columns = ['Test Case', 'Metric', 'Score', 'Reason',]
#     report_table = wandb.Table(data=metrics_results, columns=table_columns)
    
#     wandb.log({"metrics_report": report_table})

def finish_wandb_run():
    wandb.finish()

def main():
    initialize_wandb()

    test_case = create_test_case()

    report = create_report()

    metric_results = measure_metrics([test_case])

    log_report_table(metric_results)
    finish_wandb_run()

if __name__ == "__main__":
    main()
