import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_folder",
        default=r"/home/dml/project/detect_val_tools/workdir",
        help="the path of pr and confidence",
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default=r"/home/dml/project/detect_val_tools/workdir",
        help="path of saving figures",
    )
    parser.add_argument("--show", action="store_true", default=False)
    args = parser.parse_args()

    Path(args.save_folder).mkdir(parents=True, exist_ok=True)
    num_curves = len(list(Path(args.result_folder).rglob("*.pkl")))
    colors = cm.gist_rainbow(np.linspace(0, 1, num_curves*2))

    print("draw pr")
    plt.figure(figsize=(16, 10))
    plt.title("PR")
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.axis('square')
    for i, conf_file in enumerate(Path(args.result_folder).rglob("*.pkl")):
        with open(conf_file,'rb') as f:
            data = pickle.load(f)
        px,p,r,f1,py = data
        plt.plot(px, np.stack(py,1).mean(1), color=colors[i], label=conf_file.parent.name)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{args.save_folder}/PR.png")
    print("save pr")
    
    
    
    print("draw pr thresh")
    plt.figure(figsize=(16, 10))
    plt.title("PR thresh")
    plt.xlabel("thresh")
    plt.ylabel("precision/recall")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.axis('square')
    for i, conf_file in enumerate(Path(args.result_folder).rglob("**/*.pkl")):
        with open(conf_file,'rb') as f:
            data = pickle.load(f)
        px,p,r,f1,py = data
        plt.plot(
            px, np.stack(p,1).mean(1), color=colors[i*2], label=f"{conf_file.parent.name}-prec"
        )
        plt.plot(
            px, np.stack(r,1).mean(1), color=colors[i*2+1], label=f"{conf_file.parent.name}-rec"
        )
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{args.save_folder}/PR-thresh.png")
    print("save pr-thresh")




if __name__ == "__main__":
    main()
