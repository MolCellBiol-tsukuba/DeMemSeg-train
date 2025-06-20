import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Optional
import matplotlib.colors as mcolors


def load_and_concat_csvs(path_list: List[str]) -> pd.DataFrame:
    if len(path_list) == 0:
        raise ValueError("path_list is empty")
    elif len(path_list) == 1:
        return pd.read_csv(path_list[0])
    elif len(path_list) > 1:
        print(f"Loading {len(path_list)} CSV files...")
        df_list = []
        for path in path_list:
            try:
                df = pd.read_csv(path)
                df.columns = df.columns.str.strip()
                df_list.append(df)
            except Exception as e:
                print(f"Error reading {path}: {e}")
        return pd.concat(df_list, ignore_index=True)

def plot_each_strain_vs_wt_with_marginals(
    path_list: List[str],
    strain_wt: str = "YSIY874",
    wt_path: str = "../data/mmdet_results_PSM_exps_CorrectAnnotation_100percent_mask-rcnn_r50_fpn_2x_coco_epoch_24.csv",
    x_var: str = "Perimeter",
    y_var: str = "Roundness",
    facet_var: str = "StrainName",
    point_alpha: float = 0.7,
    point_size: float = 2,
    title_prefix: str = "",
    fig_width: Optional[float] = 20,
    fig_height: float = 6.0,
    save: bool = False,
    save_dir: Optional[str] = None,
    show_plot: bool = False,
    colormap: str = "viridis",
    max_points: Optional[int] = 10000,
    xlim: Optional[tuple] = (0, 350),
    ylim: Optional[tuple] = (0, 1)
):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    df = load_and_concat_csvs(path_list)
    df_wt = pd.read_csv(wt_path)
    df_wt.columns = df_wt.columns.str.strip()
    df = pd.concat([df, df_wt], ignore_index=True)

    df = df[df[facet_var].notna()]
    if y_var == "Roundness" and y_var in df.columns:
        df[y_var] = np.where(df[y_var] > 1, 2 - df[y_var], df[y_var])

    strains = sorted(s for s in df[facet_var].unique() if s != strain_wt)
    cmap = plt.get_cmap(colormap)
    color_dict = {strain: mcolors.to_hex(cmap(i / len(strains))) for i, strain in enumerate(strains)}

    for strain in strains:
        df_wt_plot = df[df[facet_var] == strain_wt].dropna(subset=[x_var, y_var])
        df_strain = df[df[facet_var] == strain].dropna(subset=[x_var, y_var])

        if max_points:
            df_wt_plot = df_wt_plot.sample(n=min(max_points, len(df_wt_plot)), random_state=42)
            df_strain = df_strain.sample(n=min(max_points, len(df_strain)), random_state=42)

        color = color_dict[strain]
        title = f"{title_prefix}{strain} vs {strain_wt}"

        sns.set_style("white")
        joint_data = pd.concat([
            df_wt_plot.assign(Group="WT"),
            df_strain.assign(Group=strain)
        ])

        g = sns.jointplot(
            data=joint_data,
            x=x_var,
            y=y_var,
            hue="Group",
            kind="scatter",
            palette={"WT": "black", strain: color},
            marginal_kws=dict(fill=True, alpha=0.4),
            alpha=point_alpha,
            height=fig_height,
            ratio=5
        )

        g.fig.suptitle(title, y=1.02)
        g.ax_joint.grid(False)
        g.ax_joint.set_xlim(xlim)
        g.ax_joint.set_ylim(ylim)

        # 💡 legend を右外に出して背景を透明に
        handles, labels = g.ax_joint.get_legend_handles_labels()
        legend = g.ax_joint.legend(
            handles=handles,
            labels=labels,
            loc="center left",
            bbox_to_anchor=(1.2, 0.5),
            frameon=True
        )
        legend.set_title("Strain")
        legend.get_frame().set_facecolor('none')  # 背景透明
        legend.get_frame().set_edgecolor('black')

        if save:
            if save_dir is None:
                raise ValueError("save_dir must be provided if save=True")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{strain}_vs_{strain_wt}_joint.png")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()
