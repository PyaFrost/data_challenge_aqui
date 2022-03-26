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
import unidecode
from tqdm.notebook import tqdm

sys.path.append('../Sources')

from params_data import params

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
def clean_data(df_clt, df_job):
    """
    Pipeline to clean tatami's data
    
    :param df_clt: dataframe of client data
    :param df_job: dataframe of jobbeur data
    :return df_clt: dataframe of client data cleaned
    :return df_job: dataframe of jobbeur data cleaned
    """
    
    time_total = time.time()
    print('------------------- NETTOYAGE COMPLET DES DONNEES -------------------\n')
    
    print('Suppression des colonnes de la table client ...', end=' ')
    start_time = time.time()
    df_clt = df_clt.drop(params['client']['col_to_del'], axis=1)
    print('fini en {}s '.format(round(time.time() - start_time, 3)))
    
    print('Suppression des lignes de la table client ...', end=' ')
    start_time = time.time()
    df_clt = df_clt.iloc[:-2]
    print('fini en {}s '.format(round(time.time() - start_time, 3)))
    
    print('Suppression des colonnes de la table jobbeur ...', end=' ')
    start_time = time.time()
    df_job = df_job.drop(params['jobbeur']['col_to_del'], axis=1)
    print('fini en {}s '.format(round(time.time() - start_time, 3)))
    
    print('Suppression des lignes de la table jobbeur ...', end=' ')
    start_time = time.time()
    df_job = df_job.iloc[:-1]
    print('fini en {}s '.format(round(time.time() - start_time, 3)))
    
    print('Nettoyage des doublons dans les variables client et correction ...', end=' ')
    start_time = time.time()
    df_clt = replace_string(df_clt, 'Métier du poste', params['client'])
    df_clt = replace_string(df_clt, 'Localisation du poste', params['client'])
    if 'Poste avec du déplacement (en %) si 75 ramené a 100%' in df_clt.columns :
        ind = np.where(df_clt['Poste avec du déplacement (en %) si 75 ramené a 100%'] == 75)[0]
        df_clt.loc[ind, 'Poste avec du déplacement (en %) si 75 ramené a 100%'] = 100
    print('fini en {}s '.format(round(time.time() - start_time, 3)))
    
    print('Nettoyage des doublons dans les variables jobbeur ...', end=' ')
    start_time = time.time()
    df_job = replace_string(df_job, 'VILLE', params['jobbeur'])
    df_job = replace_string(df_job, 'Dernier poste occupé (ou actuel)', params['jobbeur'])
    df_job = replace_string(df_job, 'Mission recherchée : Exemple n°1 de poste (métier + secteur)', params['jobbeur'])
    print('fini en {}s '.format(round(time.time() - start_time, 3)))
    
    print('Complétion des données dans les variables jobbeur ...', end=' ')
    start_time = time.time()
    df_job['Vos compétences 2'] = df_job['Vos compétences 2'].replace(np.nan, 'Non renseigné')
    df_job['Vos compétences 3'] = df_job['Vos compétences 3'].replace(np.nan, 'Non renseigné')
    df_job['CODE POSTAL'] = df_job['CODE POSTAL'].astype('str')
    print('fini en {}s '.format(round(time.time() - start_time, 3)))
    
    print('\nOpération complète terminée en {}s'.format(round(time.time() - time_total, 3)))
    
    return(df_clt, df_job)

#-------------------------------------------------------------------------------------------------------------------------------------------
def merge_one_hot_encoding(df_1, df_2):
    """
    Computes Merge and OneHotEncoding algorithms on two dataframes 
    
    :param df_1: first dataframe
    :param df_2: second dataframe
    :return col_del: list of column names deleted after encoding
    :return df_dummies: dataframe with OneHotEncoded columns
    """
    
    time_total = time.time()
    print('--------------- MISE EN FORME DES DONNEES POUR MODELE ---------------\n')
    # On fait une copie des df d'entrée
    df1 = df_1.copy()
    df2 = df_2.copy()
    # On récupère le nom de toutes les colonnes dans les deux df
    names = df1.columns.tolist() + df2.columns.tolist()
    # On stocke les noms de colonnes communes aux deux 2
    col_commune = [elt for elt in df1.columns if elt in df2.columns]
    # Si il y a des colonnes communes, on prépare les extensions qui seront créés avec dummies
    # Important sinon ces colonnes seront mal traitées par la suite
    if len(col_commune) > 0 : 
        print('Gestion des colonnes communes ...', end=' ')
        start_time = time.time()
        extensions = ['_x', '_y', '_z']
        for elt in col_commune :
            n_del = 0
            # Tant qu'il y a des colonnes communes, on la supprime des noms
            while elt in names :
                n_del += 1
                names.remove(elt)
            # Pour chaque suppression on replace la colonne avec son extension
            for i in range(n_del) :
                names.append(elt + extensions[i])
        print('fini en {}s '.format(round(time.time() - start_time, 3)))
    
    print('Merge des deux dataframes ...', end=' ')
    start_time = time.time()
    # On place une clé commune à nos deux tables
    df1['key'] = 1
    df2['key'] = 1
    # On fait fait le produit cartésien de toutes les lignes suivant la clé
    merge_df = pd.merge(df1, df2, on ='key').drop('key', axis=1)
    print('fini en {}s '.format(round(time.time() - start_time, 3)))
    
    print('Encodage des variables qualitatives ...')
    start_time = time.time()
    # On applique le OneHotEncoding de pandas
    df_dummies = pd.get_dummies(merge_df)
    col_del = []    
    pbar = tqdm(total=len(names))
    # Pour chaque variable on supprime la dernière catégorie des colonnes créées
    for name in names :
        i = 0
        for elt in df_dummies.columns :
            # On récupère la dernière colonne portant le nom de base
            if name in elt :
                i += 1
                col_to_del = elt
        # S'il y a plus d'une colonne avec le même nom de base, on supprime la dernière stockée
        if i > 1 :
            df_dummies = df_dummies.drop(col_to_del, axis=1)
            # On récupère l'information des colonnes supprimées pour vérification
            col_del.append(col_to_del)
        pbar.update(1)
    pbar.close()
    
    print('Opération complète terminée en {}s'.format(round(time.time()-time_total, 3)))
    
    return(merge_df, col_del, df_dummies)
