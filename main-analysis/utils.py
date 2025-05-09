import numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def get_indiv_level_roc(data, individuals, indiv_strategy):
    samples_per_individual = len(data["true_labels"]) // individuals

    signals = np.array(data["signal_values"])
    labels = np.array(data["true_labels"])

    signals_reshaped = signals.reshape(individuals, samples_per_individual)

    if indiv_strategy == "indiv_mean":
        indiv_signals = np.mean(signals_reshaped, axis=1)
    elif indiv_strategy == "indiv_median":
        indiv_signals = np.median(signals_reshaped, axis=1)
    elif indiv_strategy == "indiv_outlier":
        q1 = np.quantile(signals_reshaped, 0.25, axis=1, keepdims=True)
        q3 = np.quantile(signals_reshaped, 0.75, axis=1, keepdims=True)
        outside_q = (signals_reshaped < q1) | (signals_reshaped > q3)
        indiv_signals = np.mean(signals_reshaped, axis=1, where=outside_q)
    elif indiv_strategy == "indiv_mle":
        epsilon = 1e-12  # to prevent log(0)
        min_val, max_val = np.min(signals_reshaped), np.max(signals_reshaped)
        log_probs = np.log(np.clip(signals_reshaped, epsilon, 1.0)) if 0.0 <= min_val and max_val <= 1.0 else signals_reshaped
        indiv_signals  = np.sum(log_probs, axis=1, keepdims=True)
    else:
        raise ValueError("Unknown individual strategy")

    indiv_labels = labels[::samples_per_individual]
    indiv_labels = np.array(indiv_labels, dtype=bool)

    fpr, tpr, thresholds = roc_curve(indiv_labels, indiv_signals)
    return fpr, tpr

def get_indiv_level_fixed_tpr(data, individuals, indiv_strategy):
    fpr, tpr = get_indiv_level_roc(data, individuals, indiv_strategy)
    return np.max(tpr[fpr <= 0.0])

class AblationStudy:
    base_fixed_fpr = {
            "TPR@1%FPR": 0.0,
            "TPR@0.1%FPR": 0.0,
            "TPR@0.01%FPR": 0.0,
            "TPR@0%FPR": 0.0,
        }

    def __init__(self, parameters, datas, importants, config):
        self.n_results = len(datas)
        assert len(datas) == len(importants)

        if "order" in config.keys():
            priority = {key: index for index, key in enumerate(config["order"])}
            sorted_pairs = sorted(zip(importants, datas), key=lambda x: priority.get(x[0], float('inf')))

            importants_sorted, datas_sorted = zip(*sorted_pairs)

            importants = list(importants_sorted)
            datas = list(datas_sorted)

        importants = [str(imp) for imp in importants]

        self.parameters = parameters
        self.datas = datas
        self.importants = importants
        self.config = config

    def _format_table(self, d) -> str:
        # Make best result bold
        for k, v in d.items():
            best = max(float(bajs) for bajs in v)
            if best > 0.0:
                for i in range(len(v)):
                    if float(v[i]) >= best:
                        v[i] = r"\textbf{" + v[i] + "}"

        s = "% " + " & ".join(self.importants) + "\n\n"
        s = s + r"\multirow{7}{*}{\texttt{" + self.parameters["target_model"] + "}}\n"
        s = s + r"& \multicolumn{1}{r|}{\textit{sample-level}}\\" + "\n"

        s = s + r"    & \multicolumn{1}{r|}{1.00\%} & " + " & ".join(d["TPR@1%FPR"]) + r"\\" + "\n"
        s = s + r"    & \multicolumn{1}{r|}{0.10\%} & " + " & ".join(d["TPR@0.1%FPR"]) + r"\\" + "\n"
        s = s + r"    & \multicolumn{1}{r|}{0.01\%} & " + " & ".join(d["TPR@0.01%FPR"]) + r"\\" + "\n"
        s = s + r"    & \multicolumn{1}{r|}{0.00\%} & " + " & ".join(d["TPR@0%FPR"]) + r"\\" + "\n"

        s = s + r"\cmidrule{2-3}" + "\n"
        s = s + r"& \multicolumn{1}{r|}{\textit{individual-level}}\\" + "\n"

        s = s + r"    & \multicolumn{1}{r|}{0.00\%} & " + " & ".join(d["indiv_tpr"]) + r"\\" + "\n"

        return s

    def make_table(self, save_dir):
        """
        Make a latex table for the specific result
        """
        filename = f"{save_dir}/table.tex"
        d = {
            "TPR@1%FPR": [],
            "TPR@0.1%FPR": [],
            "TPR@0.01%FPR": [],
            "TPR@0%FPR":[],
            "indiv_tpr": []
        }
        for i in range(self.n_results):
            fixed_fpr = self.datas[i]['fixed_fpr']
            if fixed_fpr is None:
                fixed_fpr = AblationStudy.base_fixed_fpr
            for fpr in ["TPR@1%FPR", "TPR@0.1%FPR", "TPR@0.01%FPR", "TPR@0%FPR"]:
                d[fpr].append(f"{fixed_fpr[fpr] * 100:.2f}")

            if "indivs" in self.parameters.keys():
                indiv_tpr = get_indiv_level_fixed_tpr(self.datas[i], self.parameters["indivs"], self.config["indiv_strategy"])
            else:
                indiv_tpr = get_indiv_level_fixed_tpr(self.datas[i], self.config["ds_indivs"][self.parameters["dataset"]], self.config["indiv_strategy"])
            d["indiv_tpr"].append(f"{indiv_tpr * 100:.2f}")

        s = self._format_table(d)
        with open(filename, "w") as f:
            f.write(s)


    def make_roc_plot(self, save_dir):
        filename = f"{save_dir}/ROC.png"

        for i in range(self.n_results):
            label = self.importants[i]
            data = self.datas[i]
            fpr, tpr = np.array(data["fpr"]), np.array(data["tpr"])
            if fpr is None or tpr is None:
                plt.fill_between([0, 1], [0, 0], alpha=0.15)
                plt.plot([0, 1], [0, 0], label=label)
            else:
                outside = (fpr < 1e-5)
                fpr, tpr = fpr[~outside], tpr[~outside]
                plt.fill_between(fpr, tpr, alpha=0.15)
                plt.plot(fpr, tpr, label=label)

        # Plot baseline (random guess)
        range01 = np.linspace(0, 1)
        plt.plot(range01, range01, "--", label="Random guess")

        # Set plot parameters
        plt.yscale("log")
        plt.xscale("log")
        plt.xlim(left=1e-5)
        plt.ylim(bottom=1e-5)
        plt.tight_layout()
        plt.grid()
        plt.legend(loc="lower right")

        plt.xlabel("False positive rate (FPR)")
        plt.ylabel("True positive rate (TPR)")
        if "title_format" in self.config.keys():
            plt.title(self.config["title_format"](self.parameters))
        plt.savefig(fname=filename, dpi=1000, bbox_inches="tight")
        plt.clf()

    def make_indiv_roc_plot(self, save_dir):
        filename = f"{save_dir}/ROC_indiv.png"

        for i in range(self.n_results):
            label = self.importants[i]
            data = self.datas[i]

            if "indivs" in self.parameters.keys():
                fpr, tpr = get_indiv_level_roc(self.datas[i], self.parameters["indivs"], self.config["indiv_strategy"])
            else:
                fpr, tpr = get_indiv_level_roc(self.datas[i], self.config["ds_indivs"][self.parameters["dataset"]], self.config["indiv_strategy"])
            
            if fpr is None or tpr is None:
                plt.fill_between([0, 1], [0, 0], alpha=0.15)
                plt.plot([0, 1], [0, 0], label=label)
            else:
                plt.fill_between(fpr, tpr, alpha=0.15)
                plt.plot(fpr, tpr, label=label)

        # Plot baseline (random guess)
        range01 = np.linspace(0, 1)
        plt.plot(range01, range01, "--", label="Random guess")

        # Set plot parameters
        plt.tight_layout()
        plt.grid()
        plt.legend(bbox_to_anchor =(0.5,-0.27), loc="lower center")

        plt.xlabel("False positive rate (FPR)")
        plt.ylabel("True positive rate (TPR)")
        if "title_format" in self.config.keys():
            plt.title(self.config["title_format"](self.parameters))
        plt.savefig(fname=filename, dpi=1000, bbox_inches="tight")
        plt.clf()

def objects_to_ablations(objects, config) -> list[AblationStudy]:
    ablations = {}
    def key(parameters):
        return tuple((k,v) for k,v in parameters.items())

    for data, parameters, important in objects:
        k = key(parameters)
        if k in ablations.keys():
            ablations[k]["data"].append(data)
            ablations[k]["important"].append(important)
        else:
            ablations[k] = {
                "data" : [data],
                "parameters": parameters,
                "important": [important]
            }

    return [
        AblationStudy(
            d["parameters"], d["data"], d["important"], config
        )
        for d in ablations.values()]