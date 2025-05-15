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
            "TPR@10%FPR": 0.0,
            "TPR@1%FPR": 0.0,
            "TPR@0.1%FPR": 0.0,
            "TPR@0.01%FPR": 0.0,
            "TPR@0%FPR": 0.0,
        }
    return fixed_fpr

class AblationStudy:

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
                indiv_fpr, indiv_tpr = get_indiv_level_roc(data, self.parameters["indivs"], self.config["indiv_strategy"])
                self.indiv_fpr.append(indiv_fpr)
                self.indiv_tpr.append(indiv_tpr)
                self.indiv_fixed_fpr.append((data, self.parameters["indivs"], self.config["indiv_strategy"]))
            else:
                indiv_fpr, indiv_tpr = get_indiv_level_roc(data, self.config["ds_indivs"][self.parameters["dataset"]], self.config["indiv_strategy"])
                self.indiv_fpr.append(indiv_fpr)
                self.indiv_tpr.append(indiv_tpr)
                self.indiv_fixed_fpr.append(get_indiv_level_fixed_fpr(data, self.config["ds_indivs"][self.parameters["dataset"]], self.config["indiv_strategy"]))

    @classmethod
    def aggregate(cls, ablations):
        self = cls.__new__(cls)
        self.aggregated = True
        self.parameters = ablations[0].parameters.copy()
        self.parameters["random_seed"] = "total"
        self.n_results = ablations[0].n_results
        self.config = ablations[0].config
        self.importants = ablations[0].importants

        n_ablations = len(ablations)
        
        if n_ablations == 1:
            self.fixed_fpr = ablations[0].fixed_fpr
            self.indiv_fixed_fpr = ablations[0].indiv_fixed_fpr

            self.fixed_fpr_var = [{
            "TPR@10%FPR": 0.0,
            "TPR@1%FPR": 0.0,
            "TPR@0.1%FPR": 0.0,
            "TPR@0.01%FPR": 0.0,
            "TPR@0%FPR": 0.0,
            } for _ in range(self.n_results)]
            self.indiv_fixed_fpr_var = [0.0 for _ in range(self.n_results)]
        else:
            metrics = ["TPR@10%FPR", "TPR@1%FPR", "TPR@0.1%FPR", "TPR@0.01%FPR", "TPR@0%FPR"]

            # Initialize mean and variance accumulators
            self.fixed_fpr = []
            self.fixed_fpr_var = []
            self.indiv_fixed_fpr = []
            self.indiv_fixed_fpr_var = []

            for i in range(self.n_results):
                # Collect all values for each metric across ablations
                per_metric_values = {m: [] for m in metrics}
                indiv_values = []

                for abl in ablations:
                    for m in metrics:
                        per_metric_values[m].append(abl.fixed_fpr[i][m])
                    indiv_values.append(abl.indiv_fixed_fpr[i])

                # Compute mean and variance
                fixed_mean = {m: float(np.mean(per_metric_values[m])) for m in metrics}
                fixed_var = {m: float(np.var(per_metric_values[m])) for m in metrics}

                indiv_mean = float(np.mean(indiv_values))
                indiv_var = float(np.var(indiv_values))

                self.fixed_fpr.append(fixed_mean)
                self.fixed_fpr_var.append(fixed_var)
                self.indiv_fixed_fpr.append(indiv_mean)
                self.indiv_fixed_fpr_var.append(indiv_var)

        return self

    def _table_to_string_bold(self) -> str:
        max_sample = {k: max(f[k] for f in self.fixed_fpr) for k in self.fixed_fpr[0].keys()}
        max_sample = {k: (f"{v*100:.2f}" if v > 0.00005 else "?") for k, v in max_sample.items()}
        max_indiv = f"{max(self.indiv_fixed_fpr)*100:.2f}"
        if max_indiv == "0.00":
            max_indiv = "?"
        
        # Make best result bold
        new_fixed_fpr = []
        for i in range(self.n_results):
            d = dict()
            for k, v in self.fixed_fpr[i].items():
                s = f"{v*100:.2f}"
                if s == max_sample[k] and self.n_results > 1:
                    s = "\\B{" + s + "}"
                if self.aggregated:
                    s = s + " \\spm{" + f"{self.fixed_fpr_var[i][k]*100:.2f}" + "}"

                d[k] = s
            new_fixed_fpr.append(d)

        new_indiv_fixed_fpr = []
        for i in range(self.n_results):
            s = f"{self.indiv_fixed_fpr[i]*100:.2f}"
            if self.indiv_fixed_fpr[i] == 1.0 and self.n_results > 1:
                s = "\\B{100.0}"
            elif s == max_indiv and self.n_results > 1:
                s = "\\B{" + s + "}"
            if self.aggregated:
                s = s + " \\spm{" + f"{self.indiv_fixed_fpr_var[i]*100:.2f}" + "}"
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
            s = s + f'& {fixed_fpr[i]["TPR@0.1%FPR"]:<20} & {fixed_fpr[i]["TPR@0.01%FPR"]:<20} & \scl {indiv_fixed_fpr[i]:<24} \n'

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
            fpr, tpr = self.indiv_fpr[i], self.indiv_tpr[i]
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

def aggregate_on_random_seed(ablations) -> list[AblationStudy]:
    groups = {}
    def key(parameters):
        return tuple((k,v) for k,v in parameters.items() if k != "random_seed")

    for ablation in ablations:
        k = key(ablation.parameters)
        if k in groups.keys():
            groups[k].append(ablation)
        else:
            groups[k] = [ablation]

    return [
        AblationStudy.aggregate(l)
        for l in groups.values()]