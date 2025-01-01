#!/usr/bin/env python3
import os
import subprocess
import argparse
import json
import matplotlib.pyplot as plt


def get_directories(path):
    # Use os.listdir to list all entries and filter only directories
    return [
        entry for entry in os.listdir(path) if os.path.isdir(os.path.join(path, entry))
    ]


def main(only_analyze, spcific_design):
    # S1.SETUP
    print("@@Setup")
    CURRENT_DIR = os.getcwd()
    LOGS_DIR = CURRENT_DIR + "/logs"
    UTIL_DIR = CURRENT_DIR + "/util"
    RUN_ALL_SCRIPTS = UTIL_DIR + "run_all.csh"
    designs = {}  # TODO: {designs: [design-platform1, xx,]}
    design_configs = []
    cell_counts = []
    cell_areas = []
    tns = []
    wns = []
    power = []

    # S2.RUN ALL CASES in util/each_platform
    #   1.RUN + ANALYZE
    #   2.ANALYZE
    if only_analyze == False:
        print("@@Run util/run_all.csh")
        try:
            result = subprocess.run(
                f"csh -f {RUN_ALL_SCRIPTS}",
                shell=True,
                check=True,  # Raise an exception if the script fails
                text=True,  # Return output as a string (Python 3.7+)
                capture_output=True,  # Capture stdout and stderr
            )
            # Print the script's output
            print("run_all.csh\n", result.stdout)
            print("run_all.csh errors:", result.stderr)
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            print("stderr:", e.stderr)

    # S3.Analyze the results
    print("@@Analyze the Results")
    # S3.1 get all results
    plateforms = get_directories(LOGS_DIR)
    for p in plateforms:
        designs = get_directories(LOGS_DIR + "/" + p)
        for d in designs:
            if spcific_design != "" and d != spcific_design:
                print(f"[WARN]: the specific_design is {spcific_design}, so skip {d}")
                continue
            design_config = p + "/" + d
            report_json_path = LOGS_DIR + "/" + design_config + "/base/6_report.json"
            step_json_path = LOGS_DIR + "/" + design_config + "/base/step.json"
            # NOTE: Only analyze the results which contains 6_report.json and step.json
            if (
                os.path.exists(report_json_path) == False
                or os.path.exists(step_json_path) == False
            ):
                print(f"[WARN]: No 6_report.json or step.json in {design_config}")
                continue
            design_configs.append(design_config)
            print(f"design_config: {design_config}, {report_json_path}")
            with open(report_json_path, "r") as report_json_file:
                print(f"@@extract relevant data of {design_config} in 6_report.json")
                report_json_data = json.load(report_json_file)
                cell_counts.append(
                    report_json_data["finish__design__instance__count__stdcell"]
                )
                cell_areas.append(
                    report_json_data["finish__design__instance__area__stdcell"]
                )
                wns.append(report_json_data["finish__timing__setup__ws"])
                tns.append(report_json_data["finish__timing__setup__tns"])

            with open(step_json_path, "r") as step_json_file:
                print(f"@@extract relevant data of {design_config} in step.json")
                step_json_data = json.load(step_json_file)
                power.append(step_json_data["final"]["report_power"])
    # S3.2 Plot all results
    print(f"@@plot all {len(design_configs)} results")
    # Metrics and their labels
    metrics = [
        (cell_counts, "Cell Counts"),
        (cell_areas, "Cell Areas"),
        (tns, "Total Negative Slack (TNS)"),
        (wns, "Worst Negative Slack (WNS)"),
        (power, "Total Power"),
    ]

    # Plot individual line charts for each metric
    for metric_data, metric_label in metrics:
        plt.figure(figsize=(8, 5))
        plt.plot(
            design_configs,
            metric_data,
            marker="o",
            linestyle="-",
            label=metric_label,
            color="blue",
        )
        plt.xlabel("Design Configurations")
        plt.ylabel(metric_label)
        plt.title(f"{metric_label}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Control the script's behavior using flags."
    )

    # Add a flag for analysis-only mode
    parser.add_argument(
        "--only-analyze",
        action="store_true",
        help="Run the script in analysis-only mode",
    )

    parser.add_argument(
        "--design",
        type=str,
        choices=["gcd", "asap7", "nangate45hd"],
        help="Specific the design you wanna analyze",
    )

    # Parse arguments
    args = parser.parse_args()

    # Pass the flag to the main function
    main(args.only_analyze, args.design)
