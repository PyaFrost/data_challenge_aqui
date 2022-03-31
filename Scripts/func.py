"""

Functions to perform data treatments
author : Les Loustiques
created on : 04/03/2022

==========

Table of contents :

    | :meth:`replace_string <func.replace_string>`
    | :meth:`check_NAF <func.check_NAF>`
    | :meth:`clean_data <func.clean_data>` merge_one_hot_encoding
    | :meth:`merge_one_hot_encoding <func.merge_one_hot_encoding>` 
    
    TO DO
    
------------------------------------------------------------------

Load only this module :

    >>> import sys
    >>> sys.path.append('Scripts')
    >>> import func as fc
    
------------------------------------------------------------------

"""

import sys
import time
import numpy as np
import pandas as pd
import pickle
import unidecode
from joblib import load, dump
from tqdm.notebook import tqdm

import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings("ignore")

sys.path.append('../Sources')

from params_data import params, paires

#-------------------------------------------------------------------------------------------------------------------------------------------
def replace_string(df, col_name, params) :
    """
    Replace wrong strings in a column of a dataframe using default params
    
    :param df: dataframe
    :param col_name: string name of the column to treat
    :param params: dict of params
    :return df: dataframe treated
    """
    
    # On retire les accents et les espaces inutils (début et fin) et on fixe la casse en majuscule dans la colonne
    tmp = [unidecode.unidecode(elt).strip().upper() for elt in df[col_name]]
    # On récupère les individus mal orthographiés et leur correction
    string_false = params['dict_ville'][col_name]['false']
    string_true = params['dict_ville'][col_name]['true']
    # Pour chaque erreur, on applique la correction
    for i in range(len(string_false)) :
        ind_false = np.where(np.array(tmp) == string_false[i])[0].tolist()
        for j in ind_false :
            tmp[j] = string_true[i]
    # On met à jour la colonne corrigée dans le dataframe
    df[col_name] = tmp
    
    return(df)

#-------------------------------------------------------------------------------------------------------------------------------------------
def check_NAF(df, col_name):
    """
    Check the structure of a column content
    
    :param df: dataframe
    :param col_name: string name of the column to treat
    """
    
    # On génère une liste de chiffres et de lettres sous format caractère
    num_str = [str(i) for i in range(10)]
    alphabet = [chr(i).upper() for i in range(ord('a'), ord('z') + 1)]
    # On initialise une variable de comptage qui compte le nombre d'incohérences
    count_false = 0
    # On calcule le nombre d'incohérences selon des conditions
    for i in range(len(df[col_name].unique())):
        elt = df[col_name][i].upper()
        if len([i for i in elt[0:4] if i in num_str]) < 4:
            count_false += 1
        elif elt[4] not in alphabet: 
            count_false += 1
    # On affiche le bilan général du nombre d'incohérences
    if (count_false == 0):
        print('Aucune donnée incohérente')
    else:
         print(f'Il y a {count_false} donnée(s) incohérente(s)')
            
#-------------------------------------------------------------------------------------------------------------------------------------------
def clean_data(df_clt, df_job, new_obs_job=False, new_obs_clt=False):
    """
    Pipeline to clean tatami's data
    
    :param df_clt: dataframe of client data
    :param df_job: dataframe of jobbeur data
    :new_obs_job: string choice to treat new observation from a jobbeur
    :new_obs_clt: string choice to treat new observation from a clt
    :return df_clt: dataframe of client data cleaned
    :return df_job: dataframe of jobbeur data cleaned
    """
    
    time_total = time.time()
    print('------------------- NETTOYAGE COMPLET DES DONNEES -------------------\n')
    time.sleep(0.1)
    
    print('Suppression des colonnes de la table client ...', end=' ')
    start_time = time.time()
    df_clt = df_clt.drop(params['client']['col_to_del'], axis=1)
    print('fini en {}s '.format(round(time.time() - start_time, 3)))
    time.sleep(0.1)
    
    if not new_obs_clt :
        print('Suppression des lignes de la table client ...', end=' ')
        start_time = time.time()
        df_clt = df_clt.iloc[:-2]
        print('fini en {}s '.format(round(time.time() - start_time, 3)))
        time.sleep(0.1)
    
    print('Suppression des colonnes de la table jobbeur ...', end=' ')
    start_time = time.time()
    df_job = df_job.drop(params['jobbeur']['col_to_del'], axis=1)
    print('fini en {}s '.format(round(time.time() - start_time, 3)))
    time.sleep(0.1)
    
    if not new_obs_job :
        print('Suppression des lignes de la table jobbeur ...', end=' ')
        start_time = time.time()
        df_job = df_job.iloc[:-1]
        print('fini en {}s '.format(round(time.time() - start_time, 3)))
        time.sleep(0.1)
    
    print('Nettoyage des doublons dans les variables client et correction ...', end=' ')
    start_time = time.time()
    df_clt = replace_string(df_clt, 'Métier du poste', params['client'])
    df_clt = replace_string(df_clt, 'Localisation du poste', params['client'])
    if 'Poste avec du déplacement (en %) si 75 ramené a 100%' in df_clt.columns :
        ind = np.where(df_clt['Poste avec du déplacement (en %) si 75 ramené a 100%'] == 75)[0]
        df_clt.loc[ind, 'Poste avec du déplacement (en %) si 75 ramené a 100%'] = 100
    print('fini en {}s '.format(round(time.time() - start_time, 3)))
    time.sleep(0.1)
    
    print('Nettoyage des doublons dans les variables jobbeur ...', end=' ')
    start_time = time.time()
    df_job = replace_string(df_job, 'VILLE', params['jobbeur'])
    df_job = replace_string(df_job, 'Dernier poste occupé (ou actuel)', params['jobbeur'])
    df_job = replace_string(df_job, 'Mission recherchée : Exemple n°1 de poste (métier + secteur)', params['jobbeur'])
    print('fini en {}s '.format(round(time.time() - start_time, 3)))
    time.sleep(0.1)
    
    print('Complétion des données dans les variables jobbeur ...', end=' ')
    start_time = time.time()
    df_job['Vos compétences 2'] = df_job['Vos compétences 2'].replace(np.nan, 'Non renseigné')
    df_job['Vos compétences 3'] = df_job['Vos compétences 3'].replace(np.nan, 'Non renseigné')
    df_job['CODE POSTAL'] = df_job['CODE POSTAL'].astype('str')
    print('fini en {}s '.format(round(time.time() - start_time, 3)))
    time.sleep(0.1)
    
    print('\nOpération complète terminée en {}s'.format(round(time.time() - time_total, 3)))
    
    return(df_clt, df_job)

#-------------------------------------------------------------------------------------------------------------------------------------------
def merge(df_1, df_2):
    """
    Computes Merge algorithms on two dataframes 
    
    :param df_1: first dataframe
    :param df_2: second dataframe
    :return col_del: list of column names deleted after encoding
    :return df_dummies: dataframe with OneHotEncoded columns
    """

    # On fait une copie des df d'entrée
    df1 = df_1.copy()
    df2 = df_2.copy()    
    # On place une clé commune à nos deux tables
    df1['key'] = 1
    df2['key'] = 1
    # On fait fait le produit cartésien de toutes les lignes suivant la clé
    merge_df = pd.merge(df1, df2, on ='key').drop('key', axis=1)    
    
    return(merge_df)

#-------------------------------------------------------------------------------------------------------------------------------------------
def labellisation(df_merge, verbose=True):
    """
    Labellises merged dataframe according to business rules 
    
    :param df_merge: dataframe
    :param verbose: boolean choice of progression bar visualisation 
    :return df_merge: dataframe with labelled rows
    """
        
    df_for_label = df_merge.copy()
    df_for_label['y'] = 0
    
    print('{} \033[1met\033[0m {}'.format('niveau de rémunération', 'Niveau de rémunération mensuelle brute souhaitée'))
    ind = df_for_label[df_for_label['niveau de rémunération'] > df_for_label['Niveau de rémunération mensuelle brute souhaitée']].index.values
    df_for_label.loc[ind, 'y'] += 1

    for paire in paires :
        print('{} \033[1met\033[0m {}'.format(paires[paire]['item'][0], paires[paire]['item'][1]))
        if paires[paire]['correction?'] == 1 :
            for elt_to_corr in paires[paire]['corr'] :
                ind_to_corr = np.where(df_for_label[paires[paire]['item'][1]] == elt_to_corr[1])
                df_for_label[paires[paire]['item'][1]].iloc[ind_to_corr] = elt_to_corr[0]
        list_pair_1 = df_for_label[paires[paire]['item'][0]].unique()
        if verbose : 
            loop = tqdm(list_pair_1)
        else :
            loop = list_pair_1
        for elt_pair_1 in loop :
            if (elt_pair_1 in df_for_label[paires[paire]['item'][0]].values) & (elt_pair_1 in df_for_label[paires[paire]['item'][1]].values) :
                ind_rows = np.where((df_for_label[paires[paire]['item'][0]] == elt_pair_1) & (df_for_label[paires[paire]['item'][1]] == elt_pair_1))[0]
                df_for_label.loc[ind_rows, 'y'] += paires[paire]['poids']
    df_merge['y'] = df_for_label['y']
    
    return(df_merge)

#-------------------------------------------------------------------------------------------------------------------------------------------
def shaping_for_model(df_merge, load_enc=True, code_filename='encoder', save_enc=True) :
    """
    TO DO
    """
    
    X = df_merge.iloc[:, :-1]
    y = df_merge['y'].values
    
    print('Encodage des colonnes ...', end=' ')
    start_time = time.time()
    if load_enc :
        encoder = load('Sources/'+code_filename)
        X_encoded = encoder.transform(X)
    else :
        encoder = ce.OneHotEncoder(use_cat_names=True)
        X_encoded = encoder.fit_transform(X)
        if save_enc :
            with open("Sources/encoder", "wb") as f :
                pickle.dump(encoder, f)
    print('fini en {}s '.format(round(time.time() - start_time, 3)))
    
    print('Normalisation de la variable réponse ...', end=' ')
    start_time = time.time()
    scaler = MinMaxScaler()
    y_scaled = scaler.fit_transform(y.reshape(-1, 1)).ravel()
    print('fini en {}s '.format(round(time.time() - start_time, 3)))
    
    return(X_encoded, y_scaled)

#-------------------------------------------------------------------------------------------------------------------------------------------
def model_construction(X, y, model) :
    """
    TO DO
    """
    
    print("\nEntraînement du modèle ...", end=' ')
    start_time = time.time()
    model_trained = model
    model_trained.fit(X, y)
    print('fini en {}s '.format(round(time.time() - start_time, 3)))
    time.sleep(0.1)
    print("\nR² obtenu : {}".format(round(r2_score(model_trained.predict(X), y), 2))) 
    
    return(model_trained, X, y)

#-------------------------------------------------------------------------------------------------------------------------------------------
def pipeline_modelisation(df_clt, df_job, load_enc=True, code_filename='encoder', save_enc=True, 
                          model=RandomForestRegressor(n_jobs=-1), save_model=True, verbose=False) :
    """
    TO DO
    """
    
    df_clt_clean, df_job_clean = clean_data(df_clt, df_job)

    print('\n------------------- ADAPTATION POUR MODELISATION --------------------\n')
    time.sleep(0.1)
    time_total = time.time()
    print('Merge des deux dataframes ...', end=' ')
    start_time = time.time()
    df_merge = merge(df_clt_clean, df_job_clean)
    print('fini en {}s '.format(round(time.time() - start_time, 3)))
    time.sleep(0.1)

    print('\nLabellisation des données :')
    start_time = time.time()
    df_merge = labellisation(df_merge, verbose = False)
    print('Fini en {}s '.format(round(time.time() - start_time, 3)))
    time.sleep(0.1)

    print('\nOneHotEncoding et Standardisation variable réponse :')
    time.sleep(0.1)
    X, y = shaping_for_model(df_merge)

    print('\nOpération complète terminée en {}s'.format(round(time.time() - time_total, 3)))
    time.sleep(0.1)

    print('\n---------------------- ENTRAINEMENT DU MODELE -----------------------')
    time.sleep(0.1)
    
    time_total = time.time()
    model_trained, X, y = model_construction(X, y, model=model)
    if save_model :
        filename = 'Sources/model.sav'
        pickle.dump(model_trained, open(filename, 'wb'))
        print('Modèle sauvegardé dans {}.'.format(filename))
        time.sleep(0.1)

    print('\nOpération complète terminée en {}s'.format(round(time.time() - time_total, 3)))    
    
    return(model_trained, X, y)

#-------------------------------------------------------------------------------------------------------------------------------------------
def matching_new_job(new_job, df_clt) :
    """
    TO DO
    """
    
    df_clt_clean, df_job_clean = clean_data(df_clt, new_job, new_obs_job=True)
    df_merge = merge(df_clt_clean, df_job_clean)
    
    loaded_model = pickle.load(open('Sources/model.sav', 'rb'))
    encoder = load('Sources/encoder')
    X_encoded = encoder.transform(df_merge)
    matchs = loaded_model.predict(X_encoded)
    df_matchs = pd.DataFrame(np.array([X_encoded.index, matchs]).transpose(), columns=['index', 'matching'])
    ind_best_match = df_matchs.sort_values(by='matching', ascending=False)['index'].values
    df_best_match = encoder.inverse_transform(X_encoded.loc[ind_best_match])
    res = df_clt.loc[ind_best_match]
    res['match_value'] = df_matchs.loc[ind_best_match, 'matching']
    return(res)

#-------------------------------------------------------------------------------------------------------------------------------------------
def matching_new_clt(new_clt, df_job) :
    """
    TO DO
    """
    
    df_clt_clean, df_job_clean = clean_data(new_clt, df_job, new_obs_clt=True)
    df_merge = merge(df_clt_clean, df_job_clean)
    
    loaded_model = pickle.load(open('Sources/model.sav', 'rb'))
    encoder = load('Sources/encoder')
    X_encoded = encoder.transform(df_merge)
    matchs = loaded_model.predict(X_encoded)
    df_matchs = pd.DataFrame(np.array([X_encoded.index, matchs]).transpose(), columns=['index', 'matching'])
    ind_best_match = df_matchs.sort_values(by='matching', ascending=False)['index'].values
    df_best_match = encoder.inverse_transform(X_encoded.loc[ind_best_match])
    res = df_job.loc[ind_best_match]
    res['match_value'] = df_matchs.loc[ind_best_match, 'matching']
    return(res)
