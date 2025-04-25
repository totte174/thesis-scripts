import numpy as np, matplotlib.pyplot as plt

def get_individual_level_tpr(data, individuals, indiv_strategy):
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
    else:
        raise ValueError("Unknown individual strategy")

    indiv_labels = labels[::samples_per_individual]
    indiv_labels = np.array(indiv_labels, dtype=bool)

    korv = np.min(indiv_signals[~indiv_labels])
    indiv_pred = np.array(indiv_signals < korv, dtype=int)
    tpr = np.sum(indiv_pred[indiv_labels]) / (individuals//2)
    return tpr


class AblationStudy:
    def __init__(self, parameters, datas, importants, config):
        self.n_results = len(datas)
        assert len(datas) == len(importants)

        if "order" in config.keys():
            # prepare for the worst code you've ever seen
            bajs = [korv for korv in config["order"] if korv in importants]
            bajs = [importants.index(imp) for imp in bajs]

            importants = list(np.array(importants)[bajs])
            datas = list(np.array(datas)[bajs])

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
            for bajs in ["TPR@1%FPR", "TPR@0.1%FPR", "TPR@0.01%FPR", "TPR@0%FPR"]:
                d[bajs].append(f"{self.datas[i]['fixed_fpr'][bajs] * 100:.2f}")

            if "indivs" in self.parameters.keys():
                indiv_tpr = get_individual_level_tpr(self.datas[i], self.parameters["indivs"], self.config["indiv_strategy"])
            else:
                indiv_tpr = get_individual_level_tpr(self.datas[i], self.config["ds_indivs"][self.parameters["dataset"]], self.config["indiv_strategy"])
            d["indiv_tpr"].append(f"{indiv_tpr * 100:.2f}")

        
        
        s = self._format_table(d)
        with open(filename, "w") as f:
            f.write(s)


    def make_roc_plot(self, save_dir):
        filename = f"{save_dir}/ROC.png"

        for i in range(self.n_results):
            label = self.importants[i]
            data = self.datas[i]
            plt.fill_between(data["fpr"], data["tpr"], alpha=0.15)
            plt.plot(data["fpr"], data["tpr"], label=label)

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
        plt.legend(bbox_to_anchor =(0.5,-0.27), loc="lower center")

        plt.xlabel("False positive rate (FPR)")
        plt.ylabel("True positive rate (TPR)")
        #plt.title("ROC Curve")
        plt.savefig(fname=filename, dpi=1000, bbox_inches="tight")
        plt.clf()



