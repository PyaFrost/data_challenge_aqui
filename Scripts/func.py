"""

Fonction to perform data treatment
author : Les Loustiques
date : 04/03/2022

-----------

Table of contents :

    |:meth replace_string: <function func.replace_string>|

-----------

"""

import numpy as np
import pandas as pd
import unidecode 

############################################################################################################################################
def replace_string(df, col_name, params) :
    """
    Replace wrong strings in a column of a dataframe using defaults params
    
    :param df: dataframe
    :param col_name: string name of the column to treat
    :param params: dict of params
    :return df: dataframe treated
    """
    
    # On retire les accents et les espaces inutils (début et fin) et on fixe la casse en majuscule dans la colonne
    tmp = [unidecode.unidecode(elt).strip().upper() for elt in df[col_name]]
    # On récupère les individus mal orthographiés et leur correction
    string_false = params[col_name]['false']
    string_true = params[col_name]['true']
    # Pour chaque erreur, on applique la correction
    for i in range(len(string_false)) :
        ind_false = np.where(tmp == string_false[i])[0].tolist()
        for j in ind_false :
            tmp[j] = string_true[i]
    # On met à jour la colonne corrigée dans le dataframe
    df[col_name] = tmp
    
    return(df)

############################################################################################################################################