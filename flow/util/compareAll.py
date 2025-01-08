#!/usr/bin/env python3
import os
import subprocess
import argparse
import json
import logging
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict
from log import Log


@dataclass
class DesignMetrics:
    """store all metrics in one design"""

    design_configs: str
    cell_count: float = 0
    cell_area: float = 0
    tns: float = 0
    wns: float = 0
    power: float = 0


class DesignResults:
    """manage ppa results of all design"""

    def __init__(self):
        self.designs: Dict[str, List[DesignMetrics]] = {
            "gcd": [],
            "aes": [],
            "ibex": [],
        }

    def add_result(self, design_name: str, metrics: DesignMetrics) -> None:
        """add DesignMetrics to specific design"""
        if design_name not in self.designs:
            self.designs[design_name] = []
        self.designs[design_name].append(metrics)

    def get_design_platforms(self, design_name: str) -> List[str]:
        """get pdks of specific design"""
        return [metrics.platform for metrics in self.designs[design_name]]

    def get_metric_values(self, design_name: str, metric_name: str) -> List[float]:
        """get specific metric of specific design"""
        return [getattr(metrics, metric_name) for metrics in self.designs[design_name]]

    def get_all_metrics_for_design(self, design_name: str) -> List[DesignMetrics]:
        """get all metrics of specific design"""
        return self.designs[design_name]


class DesignAnalyzer:
    def __init__(self, logs_dir: str):
        self.logs_dir = logs_dir
        self.results = DesignResults()
        self.golden_results = DesignResults()

    def get_directories(self, path: str) -> List[str]:
        return [
            entry
            for entry in os.listdir(path)
            if os.path.isdir(os.path.join(path, entry))
        ]

    def analyze_design(
        self, platform: str, design: str, specific_design: str = ""
    ) -> bool:
        if specific_design and design != specific_design:
            my_log.warning(
                f"the specific_design is {specific_design}, so skip {design}"
            )
            return False

        design_config = f"{platform}/{design}"
        report_json_path = f"{self.logs_dir}/{design_config}/base/6_report.json"
        step_json_path = f"{self.logs_dir}/{design_config}/base/step.json"

        if not os.path.exists(report_json_path) or not os.path.exists(step_json_path):
            my_log.warning(f"No 6_report.json or step.json in {design_config}")
            return False

        # get the golden results when wanna compare with golden
        if not args.golden and args.compare:
            golden_report_json_path = (
                f"{golden_logs_dir}/{design_config}/base/6_report.json"
            )
            golden_step_json_path = f"{golden_logs_dir}/{design_config}/base/step.json"
            if not os.path.exists(golden_report_json_path) or not os.path.exists(
                golden_step_json_path
            ):
                my_log.warning(
                    f"No 6_report.json or step.json in {design_config} which stored at {golden_logs_dir}"
                )
                return False
            # extract golden metrics and stored in golden results
            self._extract_golden_metrics(
                design, design_config, golden_report_json_path, golden_step_json_path
            )

        self._extract_metrics(design, design_config, report_json_path, step_json_path)
        return True

    def _extract_metrics(
        self,
        design: str,
        design_config: str,
        report_json_path: str,
        step_json_path: str,
    ):
        my_log.debug(
            f"design_config: {design_config}, {report_json_path}, {step_json_path}"
        )

        with open(report_json_path, "r") as report_json_file:
            my_log.info(f"@@extract relevant data of {design_config} in 6_report.json")
            report_data = json.load(report_json_file)

            with open(step_json_path, "r") as step_json_file:
                my_log.info(f"@@extract relevant data of {design_config} in step.json")
                step_data = json.load(step_json_file)

                metrics = DesignMetrics(
                    design_configs=design_config,
                    cell_count=report_data["finish__design__instance__count__stdcell"],
                    cell_area=report_data["finish__design__instance__area__stdcell"],
                    wns=report_data["finish__timing__setup__ws"],
                    tns=report_data["finish__timing__setup__tns"],
                    power=float(step_data["final"]["report_power"]),
                )

                self.results.add_result(design, metrics)

    def _extract_golden_metrics(
        self,
        design: str,
        design_config: str,
        report_json_path: str,
        step_json_path: str,
    ):
        my_log.debug(
            f"design_config: {design_config}, {report_json_path}, {step_json_path}"
        )

        with open(report_json_path, "r") as report_json_file:
            my_log.info(f"@@extract relevant data of {design_config} in 6_report.json")
            report_data = json.load(report_json_file)

            with open(step_json_path, "r") as step_json_file:
                my_log.info(f"@@extract relevant data of {design_config} in step.json")
                step_data = json.load(step_json_file)

                metrics = DesignMetrics(
                    design_configs=design_config,
                    cell_count=report_data["finish__design__instance__count__stdcell"],
                    cell_area=report_data["finish__design__instance__area__stdcell"],
                    wns=report_data["finish__timing__setup__ws"],
                    tns=report_data["finish__timing__setup__tns"],
                    power=float(step_data["final"]["report_power"]),
                )

                self.golden_results.add_result(design, metrics)


class ResultsPlotter:
    def __init__(self, results: DesignResults, golden_results: DesignResults):
        self.results = results
        self.golden_results = golden_results
        self.metric_labels = {
            "cell_count": "Cell Counts",
            "cell_area": "Cell Areas/μm^2",
            "tns": "Total Negative Slack (TNS)/ps",
            "wns": "Worst Negative Slack (WNS)/ps",
            "power": "Total Power/pw",
        }

    def plot_all_metrics(self, design_name: str):
        metrics_list = self.results.get_all_metrics_for_design(design_name)
        if not metrics_list:
            my_log.warning(f"No metrics found for design {design_name}")
            return

        my_log.info(f"@@plot all results for {design_name}")
        for metric_name in self.metric_labels.keys():
            self._plot_metric_separately(design_name, metric_name)
            if args.compare and not args.golden:
                self._plot_metric_separately_with_golden(design_name, metric_name)

    def _plot_metric_separately(self, design_name: str, metric_name: str):
        # vars setup
        metrics_list = self.results.get_all_metrics_for_design(design_name)
        design_configs = [metric.design_configs for metric in metrics_list]
        values = [getattr(metric, metric_name) for metric in metrics_list]
        my_log.debug(f"design configs: {design_configs}")
        my_log.debug(f"metric_name: {metric_name}, values: {values}")
        # plot figure
        plt.figure(figsize=(8, 5))
        plt.plot(
            design_configs,
            values,
            marker="o",
            linestyle="-",
            label=self.metric_labels[metric_name],
            color="blue",
        )
        # calc for annotation
        minMetric = min(values)
        minIndex = values.index(minMetric)
        maxMetric = max(values)
        maxIndex = values.index(maxMetric)
        meanMetric = sum(values) / len(values)

        my_log.debug(
            f"{metric_name} statistics - min: {minMetric:.4f}, max: {maxMetric:.4f}, mean: {meanMetric:.4f}"
        )

        # annotate max, min and mean value on figure
        # max
        plt.annotate(
            f"max: ({design_configs[maxIndex]}, {maxMetric:.4f})",
            xy=(maxIndex, maxMetric),
            xytext=(maxIndex - 0.2, maxMetric),
        )

        # min
        plt.annotate(
            f"min: ({design_configs[minIndex]}, {minMetric:.4f})",
            xy=(minIndex, minMetric),
            xytext=(minIndex - 0.2, minMetric),
        )

        # mean
        plt.axhline(
            y=meanMetric, color="orange", linestyle="--", alpha=0.5, label="mean"
        )
        plt.annotate(
            f"mean: {meanMetric:.4f}",
            xy=(len(design_configs) / 2 - 1, meanMetric),
            xytext=(len(design_configs) / 2 - 1, meanMetric),
        )

        plt.xlabel("Design Configurations")
        plt.ylabel(self.metric_labels[metric_name])
        plt.title(f"{design_name}: {metric_name} between each pdk")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # save figure
        pics_dir = f"{analyze_dir}/{design_name}"
        if not os.path.exists(pics_dir):
            os.mkdir(pics_dir)

        plt.savefig(
            f"{pics_dir}/{metric_name}.png",
            bbox_inches="tight",
            dpi=300,
        )

        plt.close()

    def _plot_metric_separately_with_golden(self, design_name: str, metric_name: str):
        metrics_list = self.results.get_all_metrics_for_design(design_name)
        design_configs = [metric.design_configs for metric in metrics_list]
        values = [getattr(metric, metric_name) for metric in metrics_list]

        plt.figure(figsize=(10, 6))
        # plot current results
        plt.plot(
            design_configs,
            values,
            marker="o",
            linestyle="-",
            label=f"Current {self.metric_labels[metric_name]}",
            color="blue",
        )

        minMetric = min(values)
        minIndex = values.index(minMetric)
        maxMetric = max(values)
        maxIndex = values.index(maxMetric)
        meanMetric = sum(values) / len(values)

        my_log.debug(
            f"Current {metric_name} statistics - min: {minMetric:.4f}, max: {maxMetric:.4f}, mean: {meanMetric:.4f}"
        )

        # golden data
        golden_metrics_list = self.golden_results.get_all_metrics_for_design(
            design_name
        )
        golden_values = [getattr(metric, metric_name) for metric in golden_metrics_list]
        # plot golden results
        plt.plot(
            design_configs,
            golden_values,
            marker="s",
            linestyle="--",
            label=f"Golden {self.metric_labels[metric_name]}",
            color="red",
        )

        golden_minMetric = min(golden_values)
        golden_minIndex = golden_values.index(golden_minMetric)
        golden_maxMetric = max(golden_values)
        golden_maxIndex = golden_values.index(golden_maxMetric)
        golden_meanMetric = sum(golden_values) / len(golden_values)

        my_log.debug(
            f"Golden {metric_name} statistics - min: {golden_minMetric:.4f}, max: {golden_maxMetric:.4f}, mean: {golden_meanMetric:.4f}"
        )
        # calc diff_percentage
        diff_percentage = [(v - g) / g * 100 for v, g in zip(values, golden_values)]
        avg_diff = sum(diff_percentage) / len(diff_percentage)
        my_log.info(f"Average difference: {avg_diff:.2f}%")

        # annotate golden results
        plt.scatter(golden_maxIndex, golden_maxMetric, color="darkred", s=50)
        plt.annotate(
            f"Golden max: ({design_configs[golden_maxIndex]}, {golden_maxMetric:.4f})",
            xy=(golden_maxIndex, golden_maxMetric),
            xytext=(golden_maxIndex - 0.2, golden_maxMetric * 1.15),
            arrowprops=dict(
                facecolor="darkred",
                shrink=0.05,
                width=1,
                headwidth=8,
            ),
            bbox=dict(boxstyle="round,pad=0.5", fc="mistyrose", alpha=0.5),
        )

        plt.scatter(golden_minIndex, golden_minMetric, color="darkgreen", s=50)
        plt.annotate(
            f"Golden min: ({design_configs[golden_minIndex]}, {golden_minMetric:.4f})",
            xy=(golden_minIndex, golden_minMetric),
            xytext=(golden_minIndex - 0.2, golden_minMetric * 0.85),
            arrowprops=dict(
                facecolor="darkgreen",
                shrink=0.05,
                width=1,
                headwidth=8,
            ),
            bbox=dict(boxstyle="round,pad=0.5", fc="mistyrose", alpha=0.5),
        )

        plt.axhline(
            y=golden_meanMetric,
            color="red",
            linestyle="--",
            alpha=0.3,
            label="Golden Mean",
        )

        # annotate Mean diff
        plt.annotate(
            f"Mean diff: {avg_diff:.2f}%",
            xy=(len(design_configs) - 1, (meanMetric + golden_meanMetric) / 2),
            xytext=(
                len(design_configs) - 1,
                (meanMetric + golden_meanMetric) / 2 * 1.1,
            ),
            bbox=dict(boxstyle="round,pad=0.5", fc="lightgray", alpha=0.5),
        )

        # annotate current results
        plt.scatter(maxIndex, maxMetric, color="blue", s=50)
        plt.annotate(
            f"Current max: ({design_configs[maxIndex]}, {maxMetric:.2f})",
            xy=(maxIndex, maxMetric),
            xytext=(maxIndex - 0.2, maxMetric * 1.1),
            arrowprops=dict(
                facecolor="blue",
                shrink=0.05,
                width=1,
                headwidth=8,
            ),
            bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", alpha=0.5),
        )

        plt.scatter(minIndex, minMetric, color="green", s=50)
        plt.annotate(
            f"Current min: ({design_configs[minIndex]}, {minMetric:.2f})",
            xy=(minIndex, minMetric),
            xytext=(minIndex - 0.2, minMetric * 0.9),
            arrowprops=dict(
                facecolor="green",
                shrink=0.05,
                width=1,
                headwidth=8,
            ),
            bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", alpha=0.5),
        )

        plt.axhline(
            y=meanMetric, color="blue", linestyle="--", alpha=0.3, label="Current Mean"
        )

        plt.xlabel("Design Configurations")
        plt.ylabel(self.metric_labels[metric_name])
        plt.title(f"{design_name}: {metric_name} Comparison")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # save figures
        pics_dir = f"{analyze_dir}/{design_name}"
        if not os.path.exists(pics_dir):
            os.mkdir(pics_dir)
        comparison_suffix = "_vs_golden"
        plt.savefig(
            f"{pics_dir}/{metric_name}{comparison_suffix}.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()


class DesignRunner:
    def __init__(self, current_dir: str):
        self.current_dir = current_dir
        self.util_dir = f"{current_dir}/util"
        self.run_all_script = f"{self.util_dir}/run_all.csh"
        self.run_all_golden_script = f"{self.util_dir}/run_all_golden.csh"

    def run_all_cases(self):
        my_log.info("@@run util/run_all.csh")
        try:
            result = subprocess.run(
                f"csh -f {self.run_all_script}",
                shell=True,
                check=True,
                text=True,
                capture_output=True,
            )
            my_log.info(result.stdout)
            my_log.error(result.stderr)
        except subprocess.CalledProcessError as e:
            my_log.error(f"Error: {e}")
            my_log.error(e.stderr)

    def run_all_cases_with_golden(self):
        my_log.info("@@run util/run_all_golden.csh")
        try:
            result = subprocess.run(
                f"csh -f {self.run_all_golden_script}",
                shell=True,
                check=True,
                text=True,
                capture_output=True,
            )
            my_log.info(result.stdout)
            my_log.error(result.stderr)
        except subprocess.CalledProcessError as e:
            my_log.error(f"Error: {e}")
            my_log.error(e.stderr)


def main():
    # check for comparation
    if args.compare and not os.path.exists(f"{current_dir}/logs_golden"):
        my_log.error(f"No golden results: {current_dir}/logs_golden")
        return

    # STEP1:run all cases if necessary
    if not args.only_analyze:
        my_log.info("@@run all cases")
        runner = DesignRunner(current_dir)
        if args.golden:
            runner.run_all_cases_with_golden()
        else:
            runner.run_all_cases()

    # STEP2:analyze PPA for each cases
    analyzer = DesignAnalyzer(logs_dir)
    platforms = analyzer.get_directories(logs_dir)

    for platform in platforms:
        designs = analyzer.get_directories(f"{logs_dir}/{platform}")
        for design in designs:
            analyzer.analyze_design(platform, design, args.design)

    # STEP3:plot the PPA results for each cases
    plotter = ResultsPlotter(analyzer.results, analyzer.golden_results)
    if args.design:  # specific design
        plotter.plot_all_metrics(args.design)
    else:
        for design_name in analyzer.results.designs.keys():
            if analyzer.results.get_all_metrics_for_design(design_name):
                plotter.plot_all_metrics(design_name)


if __name__ == "__main__":
    # global vars setup #
    # argument
    parser = argparse.ArgumentParser(
        description="You can run all cases with run_all.csh and compare the ppa results with golden."
    )
    parser.add_argument(
        "--only-analyze",
        action="store_true",
        help="Run the script in analysis-only mode",
    )
    parser.add_argument(
        "--golden",
        action="store_true",
        help="Run all script with standard openroad binary for getting golden results",
    )
    parser.add_argument(
        "--compare", action="store_true", help="Compare the ppa results with golden"
    )
    parser.add_argument(
        "--design",
        type=str,
        choices=["gcd", "ibex", "aes"],
        help="Specific the design you wanna analyze",
    )
    parser.add_argument(
        "--result",
        default="ppa_analyze",
        help="Specific the path you wanna store the results",
    )

    args = parser.parse_args()

    # orfs logs
    current_dir = os.getcwd()

    if args.golden:
        logs_dir = f"{current_dir}/logs_golden"
    else:
        logs_dir = f"{current_dir}/logs"

    golden_logs_dir = f"{current_dir}/logs_golden"
    analyze_dir = f"{logs_dir}/{args.result}"

    if not os.path.exists(analyze_dir):
        os.makedirs(analyze_dir)
    else:
        analyze_dir = analyze_dir + "_1"
        os.makedirs(analyze_dir)

    logs_path = f"{analyze_dir}/{args.result}.log"
    # analyze logs
    my_log = Log(logs_path, level=logging.DEBUG)
    my_log.info(f"@@analyze logs save at {logs_path}")

    main()
