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

def get_indiv_level_fixed_fpr(data, individuals, indiv_strategy):
    fpr, tpr = get_indiv_level_roc(data, individuals, indiv_strategy)
    return np.max(tpr[fpr <= 0.0])

def get_fixed_fpr(fixed_fpr):
    if fixed_fpr is None:
        return {
            "TPR@1%FPR": 0.0,
            "TPR@0.1%FPR": 0.0,
            "TPR@0.01%FPR": 0.0,
            "TPR@0%FPR": 0.0,
        }
    return fixed_fpr

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
        self.aggregated = False

        # Sort importants
        if "order" in config.keys():
            priority = {key: index for index, key in enumerate(config["order"])}
        else:
            priority = {key: index for index, key in enumerate(sorted(importants))}
        sorted_pairs = sorted(zip(importants, datas), key=lambda x: priority.get(x[0], float('inf')))
        importants_sorted, datas_sorted = zip(*sorted_pairs)
        importants = list(importants_sorted)
        datas = list(datas_sorted)
        importants = [str(imp) for imp in importants]

        self.parameters = parameters
        self.importants = importants
        self.config = config

        self.fpr, self.tpr = [], []
        self.fixed_fpr = []
        self.indiv_fpr, self.indiv_tpr = [], []
        self.indiv_fixed_fpr = []
        for i in range(self.n_results):
            data = datas[i]
            if data["fpr"] is not None and data["tpr"] is not None:
                self.fpr.append(np.array(data["fpr"]))
                self.tpr.append(np.array(data["tpr"]))
            else:
                self.fpr.append(np.zeros(1))
                self.tpr.append(np.zeros(1))
            
            self.fixed_fpr.append(get_fixed_fpr(data["fixed_fpr"]))

            if "indivs" in parameters.keys():
                self.indiv_fpr, self.indiv_tpr = get_indiv_level_roc(self.datas[i], self.parameters["indivs"], self.config["indiv_strategy"])
                self.indiv_fixed_fpr = get_indiv_level_fixed_fpr(self.datas[i], self.parameters["indivs"], self.config["indiv_strategy"])
            else:
                self.indiv_fpr, self.indiv_tpr = get_indiv_level_roc(self.datas[i], self.config["ds_indivs"][self.parameters["dataset"]], self.config["indiv_strategy"])
                self.indiv_fixed_fpr = get_indiv_level_fixed_fpr(self.datas[i], self.config["ds_indivs"][self.parameters["dataset"]], self.config["indiv_strategy"])

    def _table_to_string_bold(self) -> str:
        max_sample = {k: max(f[k] for f in self.fixed_fpr) for k in self.fixed_fpr[0].keys()}
        max_sample = {k: (v if v > 0.0 else np.inf) for k, v in max_sample.items()}
        max_indiv = max(self.indiv_fixed_fpr)
        if max_indiv <= 0.0:
            max_indiv = np.inf
        
        # Make best result bold
        new_fixed_fpr = []
        for i in range(self.n_results):
            d = dict()
            for k, v in self.fixed_fpr[i].items():
                s = f"{v:.2f}"
                if v == max_sample[k] and self.n_results > 1:
                    s = r"\textbf{" + s + "}"
                if self.aggregated:
                    s = s + " $\pm" + f"{self.fixed_fpr_var[i][k]:.2f}" + "$"

                d[k] = s
            new_fixed_fpr.append(d)

        new_indiv_fixed_fpr = []
        for i in range(self.n_results):
            s = str(self.indiv_fixed_fpr[i])
            if self.indiv_fixed_fpr[i] == max_indiv and self.n_results > 1:
                s = r"\textbf{" + s + "}"
            if self.aggregated:
                s = s + " $\pm" + f"{self.indiv_fixed_fpr_var[i]:.2f}" + "$"
            new_indiv_fixed_fpr.append(s)

        return new_fixed_fpr, new_indiv_fixed_fpr

    def make_table(self, save_dir):
        """
        Make a latex table for the specific result
        """
        filename = f"{save_dir}/table.tex"

        s = "% " + " & ".join(self.importants) + "\n\n"

        fixed_fpr, indiv_fixed_fpr = self._table_to_string_bold()
        for i in range(self.n_results):
            s = s + f'& {fixed_fpr[i]["TPR@0.1%FPR"]:<16} & {fixed_fpr[i]["TPR@0.01%FPR"]:<16} & {indiv_fixed_fpr[i]:<16} \n'

        with open(filename, "w") as f:
            f.write(s)

    def make_roc_plot(self, save_dir):
        filename = f"{save_dir}/ROC.png"

        for i in range(self.n_results):
            label = self.importants[i]
            fpr, tpr = self.fpr[i], self.tpr[i]
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
            fpr, tpr = self.indiv_fpr, self.indiv_tpr
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