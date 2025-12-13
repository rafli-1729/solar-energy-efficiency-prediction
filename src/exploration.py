import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
from scipy.stats import boxcox, yeojohnson


# To plot correlation coefficient for each feature with each others
def plot_numerical_correlation(
    X: pd.DataFrame,
    numerical_features: list[str],
    ax=None,
    figsize=(15, 12),
    save_path: str = None,
):
    """
    Computes and visualizes the Pearson correlation matrix for numerical features
    using a heatmap. Supports subplot mode and clean individual saving.
    Automatically shows annotation if <= 9 features.
    """
    corr_matrix = X[numerical_features].corr(method="pearson")

    # Aktifkan angka jika heatmap kecil (kurang dari 10 fitur)
    annot = len(numerical_features) < 10
    if annot:
        fmt = ".2f"
        annot_kws = {"size" : figsize[0]}
    else:
        fmt, annot_kws = "", {}

    if save_path is not None:
        tmp_fig, tmp_ax = plt.subplots(figsize=figsize)

        sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                    linewidths=.5, cbar_kws={'label': 'Correlation Coefficient'},
                    annot=annot, fmt=fmt, ax=tmp_ax, annot_kws=annot_kws)

        tmp_ax.set_title(
            'Numerical Features Correlation Heatmap (Pearson)',
            fontsize=20, fontweight='bold'
        )
        tmp_ax.set_xticklabels(tmp_ax.get_xticklabels(), rotation=45, ha='right', fontsize=figsize[0])
        tmp_ax.set_yticklabels(tmp_ax.get_yticklabels(), rotation=0, fontsize=figsize[0])

        tmp_fig.tight_layout()
        tmp_fig.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.close(tmp_fig)

    standalone = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        standalone = True

    sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                linewidths=.5, cbar_kws={'label': 'Correlation Coefficient'},
                annot=annot, fmt=".2f" if annot else "", ax=ax)

    ax.set_title('Numerical Features Correlation Heatmap (Pearson)',
                 fontsize=14, fontweight='bold')

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    if standalone:
        plt.tight_layout()
        plt.show()

    return ax


def compare_features_boxplot(
    df: pd.DataFrame, cols: list[str],
    ax=None, colors=['#1f77b4', '#d62728'], save_path=None,
    figsize: tuple[int] = (12, 7)
):
    if len(cols) != 2:
        raise ValueError("cols length must be exactly 2")

    col1, col2 = cols
    corr_val = df[col1].corr(df[col2])

    tmp_fig, tmp_ax = plt.subplots(figsize=figsize)

    melted = df[[col1, col2]].melt(var_name='Feature', value_name='Value')
    palette = {col1: colors[0], col2: colors[1]}

    sns.boxplot(data=melted, x='Feature', y='Value', ax=tmp_ax, palette=palette)
    tmp_ax.set_title(f'Boxplot Comparison: {col1} vs {col2}\nCorr = {corr_val:.4f}',
                    fontsize=15, fontweight='bold')
    tmp_ax.grid(True, axis='y', alpha=0.3)

    if save_path or ax is None:
        tmp_fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if ax is not None:
        ax.clear()

        # replot ke axis subplot
        sns.boxplot(data=melted, x='Feature', y='Value', ax=ax, palette=palette)
        ax.set_title(f'Boxplot Comparison: {col1} vs {col2}\nCorr = {corr_val:.4f}',
                     fontsize=14, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)

    plt.close(tmp_fig)
    return corr_val


def compare_features_distribution(
    df: pd.DataFrame, cols: list[str],
    ax=None, colors=['#1f77b4', '#d62728'],
    save_path: str = None,
    figsize: tuple[int] = (12, 7)
):
    if len(cols) != 2:
        raise ValueError("cols length must be exactly 2")

    col1, col2 = cols
    corr_val = df[col1].corr(df[col2])

    tmp_fig, tmp_ax = plt.subplots(figsize=figsize)

    sns.histplot(
        df[col1], kde=True, stat='density', alpha=0.35,
        ax=tmp_ax, color=colors[0], label=col1
    )
    sns.histplot(
        df[col2], kde=True, stat='density', alpha=0.35,
        ax=tmp_ax, color=colors[1], label=col2
    )

    tmp_ax.set_title(
        f'Distribution Comparison: {col1} vs {col2}\nCorr = {corr_val:.4f}',
        fontsize=14, fontweight='bold'
    )
    tmp_ax.set_xlabel('Value')
    tmp_ax.legend()
    tmp_ax.grid(True, alpha=0.3)

    if save_path:
        tmp_fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if ax is not None:
        ax.clear()

        sns.histplot(
            df[col1], kde=True, stat='density', alpha=0.35,
            ax=ax, color=colors[0], label=col1
        )

        sns.histplot(
            df[col2], kde=True, stat='density', alpha=0.35,
            ax=ax, color=colors[1], label=col2
        )

        ax.set_title(
            f'Distribution Comparison: {col1} vs {col2}\nCorr = {corr_val:.4f}',
            fontsize=14, fontweight='bold'
        )
        ax.set_xlabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.close(tmp_fig)
    return corr_val


def check_transformations(
    series: pd.Series, target, save_path: str = None,
    transforms=["original", "log", "sqrt"],
    bins=50, colors: list[str] = ['#1f77b4', '#d62728']
):
    valid_transforms = ["original", "log", "sqrt", "cbrt","boxcox", "yeojohnson"]

    for t in transforms:
        if t not in valid_transforms:
            raise ValueError(f"Transform '{t}' belum didukung. Pilihan: {valid_transforms}")

    max_cols = 3
    n = len(transforms)

    n_cols = min(n, max_cols)
    n_rows = math.ceil(n / max_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

    axes = np.array(axes).reshape(-1)
    for idx, (ax, tname) in enumerate(zip(axes, transforms)):

        # === ORIGINAL ===
        if tname == "original":
            transformed = series.copy()
            title = "Original"

        elif tname == "log":
            transformed = np.log(series[series > 0])
            title = "Log Transform"

        elif tname == "sqrt":
            transformed = np.sqrt(series.clip(lower=0))
            title = "Square Root"

        elif tname == "cbrt":
            transformed = np.cbrt(series.clip(lower=0))
            title = "Cube Root"

        elif tname == "boxcox":
            positive = series[series > 0]
            transformed, _ = boxcox(positive)
            title = "Box-Cox Transform"

        elif tname == "yeojohnson":
            transformed, _ = yeojohnson(series)
            title = "Yeo-Johnson Transform"

        sns.histplot(
            transformed, bins=bins, kde=True, ax=ax,
            color=colors[idx % 2], edgecolor='black', alpha=0.6
        )

        ax.set_title(
            f"{title}\nSkew: {pd.Series(transformed).skew():.2f}",
            fontsize=14, fontweight='bold'
        )
        ax.set_xlabel(target)
        ax.grid(alpha=0.3)

    for j in range(len(transforms), len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(
        f"{target} Distribution – Transform Comparison",
        fontsize=20, fontweight='bold', y=1.01
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)

    plt.show()

# ======================= ASTRONOMICAL FUNCTIONS ======================= #

def plot_monthly_seasonality(
    df: pd.DataFrame,
    target: str,
    ax=None,
    figsize=(12, 7),
    save_path: str = None
):
    """
    Plot monthly seasonality of energy output and GHI.
    Supports both subplot mode and standalone saving.
    """
    data = df.copy()
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data['Month'] = data['Timestamp'].dt.month

    monthly_stats = data.groupby('Month')[[target, 'GHI']].mean()

    if save_path is not None:
        tmp_fig, tmp_ax1 = plt.subplots(figsize=figsize)
        tmp_ax2 = tmp_ax1.twinx()

        sns.lineplot(x=monthly_stats.index, y=monthly_stats[target],
                     ax=tmp_ax1,color='blue',marker='o',label=target)
        sns.lineplot(x=monthly_stats.index,y=monthly_stats['GHI'],
                     ax=tmp_ax2,color='orange',linestyle='--',marker='s',label='GHI')

        tmp_ax1.set_title("Monthly Average: Seasonality Check", weight='bold')
        tmp_ax1.set_ylabel(target)
        tmp_ax2.set_ylabel("GHI")
        tmp_ax1.grid(True, alpha=0.3)

        h1, l1 = tmp_ax1.get_legend_handles_labels()
        h2, l2 = tmp_ax2.get_legend_handles_labels()
        tmp_ax1.legend(h1 + h2, l1 + l2, loc="upper left")

        tmp_fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(tmp_fig)


    standalone = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        standalone = True

    ax1 = ax
    ax2 = ax1.twinx()

    sns.lineplot(x=monthly_stats.index, y=monthly_stats[target],
                 ax=ax1, color='blue', marker='o', label=target)
    sns.lineplot(x=monthly_stats.index, y=monthly_stats['GHI'],
                 ax=ax2, color='orange', linestyle='--', marker='s', label='GHI')

    ax1.set_title("Monthly Average: Seasonality Check", weight='bold')
    ax1.set_ylabel(target)
    ax2.set_ylabel("GHI")
    ax1.grid(True, alpha=0.3)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")

    if standalone:
        plt.tight_layout()
        plt.show()

    return ax


def plot_diurnal_cycle(
    df: pd.DataFrame,
    target: str,
    ax=None,
    figsize=(12, 7),
    save_path: str = None
):
    """
    Plot normalized diurnal cycle for TARGET, GHI, and DNI.
    Supports both standalone plot and clean individual saving.
    """
    data = df.copy()
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data['Hour'] = data['Timestamp'].dt.hour

    hourly_stats = data.groupby('Hour')[[target, 'GHI', 'DNI']].mean()
    hourly_norm = (hourly_stats - hourly_stats.min()) / (hourly_stats.max() - hourly_stats.min())

    if save_path is not None:
        tmp_fig, tmp_ax = plt.subplots(figsize=figsize)

        sns.lineplot(data=hourly_norm, x=hourly_norm.index, y=target, ax=tmp_ax, label=target, color='blue')
        sns.lineplot(data=hourly_norm, x=hourly_norm.index, y='GHI', ax=tmp_ax, label='GHI', color='orange')
        sns.lineplot(data=hourly_norm, x=hourly_norm.index, y='DNI', ax=tmp_ax, label='DNI', linestyle=':', color='green')

        peak_hour = hourly_stats['GHI'].idxmax()
        tmp_ax.axvline(peak_hour, color='red', linestyle='--', label=f"Peak {peak_hour}:00")

        tmp_ax.set_title("Diurnal Cycle – Solar Noon Check", weight='bold')
        tmp_ax.set_ylabel("Normalized Value (0-1)")
        tmp_ax.legend()
        tmp_ax.grid(True, alpha=0.3)

        tmp_fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(tmp_fig)

    standalone = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        standalone = True

    sns.lineplot(data=hourly_norm, x=hourly_norm.index, y=target, ax=ax, label=target, color='blue')
    sns.lineplot(data=hourly_norm, x=hourly_norm.index, y='GHI', ax=ax, label='GHI', color='orange')
    sns.lineplot(data=hourly_norm, x=hourly_norm.index, y='DNI', ax=ax, label='DNI', linestyle=':', color='green')

    peak_hour = hourly_stats['GHI'].idxmax()
    ax.axvline(peak_hour, color='red', linestyle='--', label=f"Peak {peak_hour}:00")

    ax.set_title("Diurnal Cycle – Solar Noon Check", weight='bold')
    ax.set_ylabel("Normalized Value (0-1)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if standalone:
        plt.tight_layout()
        plt.show()

    return ax


def plot_two_feature_scatter(
    df: pd.DataFrame,
    feature1: str,
    feature2: str,
    target: str,
    ax=None,
    colors=['#1f77b4', '#d62728'],
    figsize=(12, 7),
    save_path: str = None,
    sample: int = 5000
):
    # Filter sample for performance
    data = df.sample(min(sample, len(df)), random_state=42)

    if save_path is not None:
        tmp_fig, tmp_ax = plt.subplots(figsize=figsize)

        sns.scatterplot(data=data, x=feature1, y=target, ax=tmp_ax, alpha=0.30,
                        label=f'{feature1} vs {target}', color=colors[0])

        sns.scatterplot(data=data, x=feature2, y=target, ax=tmp_ax, alpha=0.30,
                        label=f'{feature2} vs {target}', color=colors[1])

        tmp_ax.set_title(f"{feature1} & {feature2} vs {target}", weight='bold')
        tmp_ax.set_xlabel("Feature Value")
        tmp_ax.set_ylabel(target)
        tmp_ax.legend()

        tmp_fig.tight_layout()
        tmp_fig.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.close(tmp_fig)

    standalone = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        standalone = True

    sns.scatterplot(data=data, x=feature1, y=target, ax=ax, alpha=0.30,
                    label=f'{feature1} vs {target}', color=colors[0])
    sns.scatterplot(data=data, x=feature2, y=target, ax=ax, alpha=0.30,
                    label=f'{feature2} vs {target}', color=colors[1])

    ax.set_title(f"{feature1} & {feature2} vs {target}", weight='bold')
    ax.set_xlabel("Feature Value")
    ax.set_ylabel(target)
    ax.legend()

    if standalone:
        plt.tight_layout()
        plt.show()

    return ax



# ============================ MONTHLY ANALYSIS ============================ #


def _prepare_month(df):
    """Internal helper: add Month_Name column with correct order."""
    month_map = {
        1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
        7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'
    }
    order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

    df = df.copy()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Month'] = df['Timestamp'].dt.month
    df['Month_Name'] = df['Month'].map(month_map)

    return df, order


def _prep_daylight(df):
    """Return two filtered datasets: day_data and high_sun_data."""
    day_data = df[df['GHI'] > 10].copy()
    high_sun_data = df[df['GHI'] > 200].copy()
    high_sun_data['raw_efficiency'] = high_sun_data['% Baseline'] / high_sun_data['GHI']
    return day_data, high_sun_data


def plot_monthly_boxplot(
    df: pd.DataFrame,
    col: str,
    title: str,
    ylabel: str,
    palette='viridis',
    ax=None,
    figsize=(12, 7),
    save_path: str = None
):
    data, order = _prepare_month(df)

    if save_path:
        tmp_fig, tmp_ax = plt.subplots(figsize=figsize)

        sns.boxplot(
            data=data,
            x='Month_Name',
            y=col,
            order=order,
            palette=palette,
            ax=tmp_ax
        )

        tmp_ax.set_title(title, fontsize=14, weight='bold')
        tmp_ax.set_ylabel(ylabel)
        tmp_ax.grid(True, axis='y', alpha=0.3)

        tmp_fig.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.close(tmp_fig)

    standalone = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        standalone = True

    sns.boxplot(
        data=data,
        x='Month_Name',
        y=col,
        order=order,
        palette=palette,
        ax=ax
    )

    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_ylabel(ylabel)
    ax.grid(True, axis='y', alpha=0.3)

    if standalone:
        plt.tight_layout()
        plt.show()

    return ax


def plot_feature_vs_target(
    df, x, y,
    filter_mode=None, ax=None,
    title="", xlabel="",
    ylabel="", color="blue",
    reg_color="black",
    figsize=(12, 7),
    scatter_alpha=0.1,
    save_path=None
):
    # Apply filtering using your helper
    if filter_mode == "day":
        df, _ = _prep_daylight(df)
        df = df  # day_data
    elif filter_mode == "highsun":
        _, df = _prep_daylight(df)
        df = df  # high_sun_data

    # SAVE section
    if save_path:
        fig_tmp, ax_tmp = plt.subplots(figsize=figsize)

        sns.scatterplot(df, x=x, y=y, ax=ax_tmp,
                        alpha=scatter_alpha, color=color)
        sns.regplot(df, x=x, y=y, ax=ax_tmp, scatter=False,
                    color=reg_color, line_kws={'linestyle': '--'})

        ax_tmp.set_title(title, weight='bold')
        ax_tmp.set_xlabel(xlabel)
        ax_tmp.set_ylabel(ylabel)
        ax_tmp.grid(True, alpha=0.3)

        fig_tmp.tight_layout()
        fig_tmp.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig_tmp)

    # NORMAL PLOT
    standalone = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        standalone = True

    sns.scatterplot(df, x=x, y=y, ax=ax,
                    alpha=scatter_alpha, color=color)
    sns.regplot(df, x=x, y=y, scatter=False, ax=ax,
                color=reg_color, line_kws={'linestyle': '--'})

    ax.set_title(title, weight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    if standalone:
        plt.tight_layout()
        plt.show()

    return ax


def plot_categorical_impact(
    df,
    category: str,
    target: str,
    sort_by: str = "median",      # "median" | "Q3"
    filter_mode: str = "day",      # None | "day" | "highsun"
    palette: str = "Spectral",
    ax=None,
    figsize=(12, 7),
    save_path: str = None
):
    df1, df2 = _prep_daylight(df)
    if filter_mode == "day":
        df = df1
    elif filter_mode == "highsun":
        df = df2

    if sort_by == "Q3":
        order_vals = df.groupby(category)[target].quantile(0.75)
    else:
        order_vals = df.groupby(category)[target].median()

    order_sorted = order_vals.sort_values(ascending=False).index.tolist()

    standalone = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        standalone = True
    else:
        fig = ax.figure

    sns.boxplot(data=df, x=category, y=target, order=order_sorted, palette=palette, ax=ax)

    ax.set_title(f"{category} Impact on {target} (Sorted by {sort_by})", weight='bold')
    ax.set_xlabel(category)
    ax.set_ylabel(target)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.3)

    if save_path:
        plt.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if standalone:
        plt.tight_layout()
        plt.show()

    return ax


# ================================= TIME SERIES ANALYSIS ================================= #

def plot_timeseries(
    df,
    column,
    start=None,
    end=None,
    ax=None,
    figsize=(15,4),
    color=None,
    alpha=0.6,
    ylabel=None,
    title=None,
    grid=True,
    save_path=None,
):
    df_range = df.loc[start:end] if (start or end) else df

    # Detect if we need to create a new figure
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Plot data
    ax.plot(df_range[column], alpha=alpha, color=color)
    ax.set_ylabel(ylabel if ylabel else column)
    ax.set_title(title if title else f"{column} Over Time", weight='bold')
    if grid:
        ax.grid(True)

    if save_path:
        # Create a temporary figure
        temp_fig = plt.figure(figsize=figsize)
        temp_ax = temp_fig.add_subplot(111)

        # Copy artists from original axis
        for artist in ax.get_children():
            try:
                artist_figure = artist.figure
                artist.axes = temp_ax
                temp_ax._children.append(artist)
            except:
                pass

        temp_ax.set_xlim(ax.get_xlim())
        temp_ax.set_ylim(ax.get_ylim())
        temp_ax.set_title(ax.get_title(), weight='bold')
        temp_ax.set_xlabel(ax.get_xlabel())
        temp_ax.set_ylabel(ax.get_ylabel())
        temp_ax.grid(grid)

        temp_fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(temp_fig)

    # Show only if standalone
    if standalone:
        plt.tight_layout()
        plt.show()

    return ax


def plot_overlay_timeseries(
    df,
    columns,
    start=None,
    end=None,
    ax=None,
    figsize=(15,5),
    colors=None,
    alpha=0.5,
    ylabel="Value",
    title=None,
    grid=True,
    save_path=None,
):
    # Slice range
    df_range = df.loc[start:end] if (start or end) else df

    # Determine if function owns the figure
    standalone = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        standalone = True
    else:
        fig = ax.figure

    # Default colors based on Matplotlib cycle
    if colors is None:
        colors = [None] * len(columns)

    # Plot each column
    for col, c in zip(columns, colors):
        ax.plot(df_range[col], alpha=alpha, label=col, color=c)

    # Labels and titles
    ax.set_ylabel(ylabel)
    ax.set_title(title if title else "Overlay Time Series", weight='bold')
    if grid:
        ax.grid(True)
    ax.legend()

    # Save figure
    if save_path:
        plt.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    # Show only if standalone
    if standalone:
        plt.tight_layout()
        plt.show()

    return ax


def boxplot_by_group(
    df, columns,
    group_col,
    start=None,
    end=None,
    figsize=(12, 6),
    fontsize=12,
    save_path=None,
    sharey=False
):
    df = df.loc[start:end] if (start or end) else df

    n_cols = len(columns)
    fig, axes = plt.subplots(1, n_cols, figsize=figsize, sharey=sharey)

    # if only 1 column, axes is not iterable
    if n_cols == 1:
        axes = [axes]

    for ax, col in zip(axes, columns):
        df.boxplot(column=col, by=group_col, ax=ax)
        ax.set_title(f"{col} Distribution by {group_col}", fontsize=fontsize, weight='bold')
        ax.set_xlabel(group_col, fontsize=fontsize)
        ax.set_ylabel(col, fontsize=fontsize)
        ax.grid(True, alpha=0.3)

    plt.suptitle("")  # remove default grouped title
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axes