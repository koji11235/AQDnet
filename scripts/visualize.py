import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import palettable

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr



def lmplot_label2prediction(extonionnet_object, label='pKa', prediction='pKa_predicted', 
                            write_to_png=False, output=None, title=None, xlim=None, ylim=None, auto_lim_detect=False, ax=None):
    
    label_values = extonionnet_object.prediction_result[label]
    prediction_values = extonionnet_object.prediction_result[prediction]

    # set style and color palette -----------------------------------------------------------------------
    plt.style.use('seaborn-whitegrid')
    current_palette = palettable.matplotlib.Plasma_5.mpl_colors

    # set output file name -----------------------------------------------------------------------
    if output is None:
        output = f"correlation_{label}2{prediction}.png"

    # set plot range (xlim and ylim) -----------------------------------------------------------------------
    if auto_lim_detect is True:
        margin = 0.05 * abs(np.concatenate([label_values, prediction_values]).max() - np.concatenate([label_values, prediction_values]).min())
        lim = (np.concatenate([label_values, prediction_values]).min() - margin, np.concatenate([label_values, prediction_values]).max() + margin)
        xlim = lim
        ylim = lim

    # calculate linear regression statistics -----------------------------------------------------------------------
    lm = LinearRegression()
    lm.fit(label_values[:, np.newaxis], prediction_values)
    reg_coef = lm.coef_[0]
    reg_intercept = lm.intercept_
    r2 = lm.score(label_values[:, np.newaxis], prediction_values)
    
    # plot scatter and regression line -----------------------------------------------------------------------
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    ax.scatter(label_values, prediction_values, alpha=0.3, color=current_palette[0])
    ax.plot(label_values, lm.predict(label_values[:, np.newaxis]), color=current_palette[0])
    # plot annotation in the box -----------------------------------------------------------------------
    ax.text(
        0.05, 0.95, f"R^2 = {r2:.2f}\ny = {reg_coef:.2f}x {reg_intercept:+.2f}",
        linespacing = 1.5, horizontalalignment='left', verticalalignment='top',
        transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='white', alpha = 1, edgecolor='grey'), 
    )   
    # plot other part of figure -----------------------------------------------------------------------
    xlabel = label.replace('_', ' ')
    ylabel = prediction.replace('_', ' ')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return ax 


def pieplot_dataset(y_file, output, annotations=["Training", "Validation", "Untrained"], 
                          ligand2protein_file="../Data/ligand2protein.csv"):
    
    ligand2protein = pd.read_csv(ligand2protein_file, index_col=0)
    y = pd.read_csv(y_file, index_col=0)
    # y["Ligand"] = [idx[:-5] for idx in y.index]
    y = pd.merge(y, ligand2protein, on='Ligand')
    y_protein_grouped_size = y.groupby(by=["Protein"], as_index=True).size().sort_index(ascending=True, level="Protein")
    y_ligand_grouped_size = y.groupby(by=["Protein","Ligand"], as_index=True).size().sort_index(ascending=True, level="Protein")


    protein_size = y_protein_grouped_size.values
    protein_index = y_protein_grouped_size.index
    protein_label = [protein+"\n"+str(size) for  protein, size in zip(protein_index, protein_size)]

    ligand_size = y_ligand_grouped_size.values
    ligand_index = y_ligand_grouped_size.index
    ligand_label = [lig+"\n"+str(size) for (protein, lig), size in zip(ligand_index, ligand_size)]

    nligand_per_protein_list = collections.Counter([protein for protein, ligand in y_ligand_grouped_size.index]).values()
    ligand_colors = [cm.Set2(ligand_color/len(nligand_per_protein_list), alpha=1-ligand_color/6) 
                        for ligand_color, num_lig in enumerate(nligand_per_protein_list) 
                            for ligand_color in range(1, num_lig+1)]
    protein_colors = [cm.Set2(protein_color/len(nligand_per_protein_list), alpha=1) 
                        for protein_color, num_lig in enumerate(nligand_per_protein_list)]

    ax = fig.add_subplot(1, 3, i+1)
    ax.axis('equal')

    pie_protein, texts_protein = ax.pie(x=protein_size, labels=protein_label, colors=protein_colors,
                                        radius=1.2, labeldistance=1.1, startangle=90, counterclock=False)
    for t in texts_protein:
        t.set_horizontalalignment('center')
        t.set_linespacing(1.5)
    plt.setp( pie_protein, width=0.3, edgecolor='white')

    pie_ligand, texts_ligand = ax.pie(ligand_size,  labels=ligand_label, colors=ligand_colors,
                                        radius=1.2-0.3, labeldistance=0.8,  startangle=90, counterclock=False)
    for t in texts_ligand:
        t.set_horizontalalignment('center')
        t.set_linespacing(1.5)
    plt.setp(pie_ligand, width=0.4, edgecolor='white')
    plt.margins(0,0)
    ax.text(0, 0, f"{annotation}\nTotal: {ligand_size.sum()}", fontsize=16, linespacing = 3, ha='center', va='center')

    fig.savefig(output, format="png", dpi=600)

def prepare_piechart_layer1(data, groupby=['Protein']):
    grouped_size = data.groupby(by=groupby, as_index=True).size().sort_index(ascending=True, level=groupby[0])
    size = grouped_size.values
    index = grouped_size.index
    label = [idx+"\n"+str(size) for  idx, size in zip(index, size)]
    colors = [cm.Set2(color/len(index), alpha=1) for color in range(len(index))]
    return size, label, color 






def lmplot(data, x='pKa', y='pKa_predicted', hue=None, color=None, label=None, c=None, alpha=0.3,
           xlim=None, ylim=None, auto_lim_detect=False, ax=None, 
           horizontalalignment='left', verticalalignment='top'):

    if xlim is None:
        margin = 0.025 * abs(np.concatenate([data[x], data[y]]).max() - np.concatenate([data[x], data[y]]).min())
        xlim = (data[x].min() - margin, data[x].max() + margin)

    # set plot range (xlim and ylim) -----------------------------------------------------------------------
    if auto_lim_detect is True:
        margin = 0.025 * abs(np.concatenate([data[x], data[y]]).max() - np.concatenate([data[x], data[y]]).min())
        lim = (np.concatenate([data[x], data[y]]).min() - margin, np.concatenate([data[x], data[y]]).max() + margin)
        xlim = lim
        ylim = lim

    # calculate linear regression statistics -----------------------------------------------------------------------
    has_2samples = len(data[x]) >= 2
    if has_2samples:
        lm = LinearRegression()
        lm.fit(data[x].values.reshape([-1,1]), data[y])
        reg_coef = lm.coef_[0]
        reg_intercept = lm.intercept_
        pcc = pearsonr(data[x], data[y])[0] 
        rmse = np.sqrt(mean_squared_error(data[x], data[y]))

    # plot scatter and regression line -----------------------------------------------------------------------
    ax = plt.gca()
    ax.set_title(label) # MUST-take argument
    if c is None:
        ax.scatter(data[x], data[y], alpha=alpha, color=color)
    else:
        ax.scatter(data[x], data[y], alpha=alpha, color=color, c=data[c]) # MUST-take argument
    if has_2samples:
        ax.plot(xlim, lm.predict(np.array(xlim)[:, np.newaxis]), color=color)
        # plot annotation in the box -----------------------------------------------------------------------
        ax.text(
            0.05, 0.95, f"PCC = {pcc:.2f} \nrmse = {rmse:.2f} \ny = {reg_coef:.2f}x {reg_intercept:+.2f}",
            linespacing = 1.5, horizontalalignment=horizontalalignment, verticalalignment=verticalalignment,
            transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='white', alpha = 1, edgecolor='grey'), 
        )   
    xlabel = x.replace('_', ' ')
    ylabel = y.replace('_', ' ')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return ax 

def lmplot_facetgrid(data, x='pKa', y='pKa_predicted', col=None, row=None, hue=None, col_wrap=None, xlim=None, ylim=None, height=5, aspect=1.0):

    # set style and color palette -----------------------------------------------------------------------
    plt.style.use('seaborn-whitegrid')

    if xlim is None:
        margin = 0.025 * abs(data[x].max() - data[x].min())
        xlim = (data[x].min() - margin, data[x].max() + margin)

    if ylim is None:
        margin = 0.025 * abs(data[x].max() - data[x].min())
        ylim = (data[x].min() - margin, data[x].max() + margin)

    # TODO: must handle Nan with other way 
    # this go away if FacetGrid dropna argument was turned into False
    # data.fillna(0, inplace=True) # facetgrid.map_dataframe returns error if NaN in the dataframe.

    grid_plot = sns.FacetGrid(data, col=col, row=row, hue=hue, col_wrap=col_wrap, 
                              height=height, aspect=aspect,
                              palette=palettable.matplotlib.Plasma_5.mpl_colors, dropna=False)
    grid_plot.map_dataframe(lmplot, x=x, y=y, xlim=xlim, ylim=ylim)
    return grid_plot

# def visualize_preds_by_ligand(preds, regex='smina_(.{4})_'):
#     retrieve_ligand = RetrieveLigand(regex=regex)
#     fold2ligand = preds.reset_index()\
#         .groupby('fold')\
#         .agg({'file_name': retrieve_ligand})\
#         .rename({'file_name': 'ligand'}, axis=1)
    
#     preds['ligand'] = preds['fold']\
#         .map(dict(zip(fold2ligand.index, fold2ligand['ligand'])))
    
#     grid_plot = lmplot_facetgrid(
#         data=preds, 
#         x='pKa_true', y='pKa_preds', 
#         col='ligand', hue='ligand', 
#         col_wrap=4
#     )
#     return grid_plot


def visualize_preds_by_ligand(preds, regex='smina_(.{4})_'):
    preds['ligand'] = preds.index.str.extract(regex)[0].values
    
    grid_plot = lmplot_facetgrid(
        data=preds, 
        x='pKa_true', y='pKa_preds', 
        col='ligand', hue='ligand', 
        col_wrap=4
    )
    return grid_plot


def retrieve_ligand(col):
    ligand = np.unique(col.str.extract('smina_(.{4})_').values)[0]
    return ligand


class RetrieveLigand(object):
    def __init__(self, regex='smina_(.{4})_'):
        self.regex = regex
        
    def __call__(self, col):
        ligand = np.unique(col.str.extract('smina_(.{4})_').values)[0]
        return ligand


def plot_training_log(data, xaxis="epoch", 
                      yaxis_train="mse", yaxis_validation="val_mse", 
                      label="pKa", xlabel="epoch", ylabel='Mean Squared Error', 
                      ax=None, ylim=None, rolling_mean=None, color=None):
    # set style and color palette -----------------------------------------------------------------------
    plt.style.use('seaborn-whitegrid')
    palette = palettable.matplotlib.Plasma_5.mpl_colors
    
    # plot the train log and and the validation log ----------------------------------------------------------
    ax = plt.gca()
        
    data_x = data[xaxis]
    if rolling_mean is None:
        data_y_tr = data[yaxis_train]
        data_y_val = data[yaxis_validation]
    elif type(rolling_mean) is int:
        data_y_tr = data[yaxis_train].rolling(rolling_mean).mean()
        data_y_val = data[yaxis_validation].rolling(rolling_mean).mean()
    else:
        raise ValueError('The rolling_mean argument must be either int or None')
    
    ax.plot(data_x, data_y_tr, color=palette[0], label="Train")
    ax.plot(data_x, data_y_val, color=palette[2], label="Validation")

    # set other part of figure -----------------------------------------------------------------------
    if ylim is not None:
        #ylim = (0, data[yaxis_validation][int(len(data)/3):-1].median()*2)
        ax.set_ylim(ylim[0],ylim[1])
    
    ax.set_title(label)
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax

def visualize_cv_log(log_path_file, rolling_mean=10):
    # prepare concatenated log dataframe
    log_paths = pd.read_csv(log_path_file, index_col=0)
    log_df = pd.DataFrame()
    for fold, (_, log) in log_paths.iterrows():
        log_to_add = pd.read_csv(log, index_col=0)
        log_to_add['fold'] = fold
        log_df = log_df.append(log_to_add, ignore_index=True)

    # visualize facet grid
    col = 'fold'
    hue = 'fold'
    col_wrap = 4

    xaxis = "epoch"
    yaxis_train = "mse"
    yaxis_validation = "val_mse"

    grid_plot = sns.FacetGrid(log_df, col=col, hue=hue, col_wrap=col_wrap, 
                              dropna=False, sharex=False, sharey=False)
    grid_plot.map_dataframe(
        plot_training_log, 
        xaxis=xaxis, 
        yaxis_train=yaxis_train,
        yaxis_validation=yaxis_validation,
        rolling_mean=rolling_mean
    )
    return grid_plot