# Version 2 de helper.py.
# Descripcion: funciones de ayuda para el desarrollo de modelos de score crediticio

# Librerias
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif


# Clase para calculo de IV y WOEs para variables categoricas, numericas y combinadas
class informationValue_WOEs:
    def __init__(self,datos:pd.DataFrame,nombreTarget:str):
        self.datos = datos
        self.target = nombreTarget

    def var_categoricas_numericas(self,dropCol:list=None):
        '''
        Comprueba el tipo de variables en un dataFrame y devuelve una tupla con dos listas con las variables
        numericas y categoricas (en ese orden).

        dropCol: lista de variables que no deben ser consideradas en la clasificacion.
        '''
        numericas = []
        categoricas = []

        if dropCol is None:
            dropCol = []
        else:
            columnas_invalidas = [col for col in dropCol if col not in self.datos.columns]
            if columnas_invalidas:
                raise ValueError(f"Las columnas {columnas_invalidas} no existen en el DataFrame.")

        for variable in self.datos.drop(columns=dropCol).columns:
            if self.datos[variable].dtype in ('float64', 'int64'):
                numericas.append(variable)
            else:
                categoricas.append(variable)

        return numericas, categoricas
    
    def calcularBins(self, variable:str, max_depth:int=3,min_samples_leaf:float=0.05):
        '''
        Calcula las categorias de una variable numerica (para clasificar un target) usando 
        arboles de decision. 

        variable: variable a la que se le calcularan las categorias
        '''
        tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        tree.fit(self.datos[[variable]], self.datos[self.target])
        
        limites = tree.tree_.threshold[tree.tree_.children_left != -1]
        limites = np.sort(limites)
        limites = np.concatenate(([-np.inf], limites, [np.inf]))

        return limites

    def calcular_IV_WOE_numericas(self, variable:str, bins):
        """
        Calcula el Information Value (IV) y WOEs para una variable numérica, utilizando un conjunto de 
        bins (calculado con la funcion calcularBins).
        
        Args:
            variable (str): Nombre de la variable a analizar.
            bins (list): Lista con los límites de los bins para categorizar la variable.
        
        Returns:
            iv_df (pd.DataFrame): DataFrame con la variable y su IV total.
            grouped (pd.DataFrame): DataFrame detallado con métricas por bin.
        """
        datos_copia = self.datos.copy()
        datos_copia['categoria'] = pd.cut(datos_copia[variable], bins=bins, duplicates='drop')
        grouped = datos_copia.groupby('categoria',observed=False)[self.target].agg(['count', 'sum'])
        grouped = grouped.rename(columns={'sum':'good'})
        grouped['bad'] = grouped['count'] - grouped['good']
        grouped['good_dist'] = grouped['good'] / grouped['good'].sum()
        grouped['bad_dist'] = grouped['bad'] / grouped['bad'].sum()
        grouped['woe'] = np.log(grouped['good_dist'] / grouped['bad_dist'])
        grouped['iv'] = (grouped['good_dist'] - grouped['bad_dist']) * grouped['woe']
        grouped['iv'] = np.where((grouped['bad'] == 0) | (grouped['good'] == 0), 0, grouped['iv'])
        grouped['iv_total'] = grouped['iv'].sum()
        grouped['dist'] = grouped['count'] / grouped['count'].sum()
        grouped['good_cat'] = grouped['good'] / grouped['count']
        grouped['bad_cat'] = grouped['bad'] / grouped['count']
        grouped['good_total'] = grouped['good'].sum() / grouped['count'].sum()
        iv = grouped['iv'].sum()

        grouped = grouped.reset_index()
        grouped = grouped.rename(columns={variable:'categoria'})

        grouped['variable'] = variable
        grouped = grouped[['variable','categoria','count','good','bad','good_dist','bad_dist','woe','iv','iv_total','dist','good_cat','bad_cat','good_total']]

        iv_df = pd.DataFrame({'variable':[variable],'IV':[iv]})

        return iv_df, grouped
    
    def calcular_IV_WOE_categoricas(self,variable:str):
        """
        Calcula el Information Value (IV) y WOEs para una variable categorica.
        
        Args:
            variable (str): Nombre de la variable a analizar.
        
        Returns:
            iv_df (pd.DataFrame): DataFrame con la variable y su IV total.
            grouped (pd.DataFrame): DataFrame detallado con métricas por categoria.
        """
        datos_copia = self.datos.copy()
        grouped = datos_copia.groupby(variable,observed=False)[self.target].agg(['count','sum'])
        grouped = grouped.rename(columns={'sum':'good'})
        grouped['bad'] = grouped['count'] - grouped['good']
        grouped['good_dist'] = grouped['good'] / grouped['good'].sum()
        grouped['bad_dist'] = grouped['bad'] / grouped['bad'].sum()
        grouped['woe'] = np.log(grouped['good_dist'] / grouped['bad_dist'])
        grouped['iv'] = (grouped['good_dist'] - grouped['bad_dist']) * grouped['woe']
        grouped['iv'] = np.where((grouped['bad'] == 0) | (grouped['good'] == 0), 0, grouped['iv'])
        grouped['iv_total'] = grouped['iv'].sum()
        grouped['dist'] = grouped['count'] / grouped['count'].sum()
        grouped['good_cat'] = grouped['good'] / grouped['count']
        grouped['bad_cat'] = grouped['bad'] / grouped['count']
        grouped['good_total'] = grouped['good'].sum() / grouped['count'].sum()
        iv = grouped['iv'].sum()

        grouped = grouped.reset_index()
        grouped = grouped.rename(columns={variable:'categoria'})

        grouped[['variable']] = variable
        grouped = grouped[['variable','categoria','count','good','bad','good_dist','bad_dist','woe','iv','iv_total','dist','good_cat','bad_cat','good_total']]

        iv_df = pd.DataFrame({'variable':[variable],'IV':[iv]})

        return iv_df, grouped

    def calcular_IV_WOE_combinadas(self, lista_variables, max_depth: int = 3, min_samples_leaf: float = 0.05):
        """
        Genera combinaciones de reglas entre dos variables usando un árbol de decisión y las categoriza.

        Argumentos:
            lista_variables (list): Lista con dos nombres de columnas que se quieren combinar.
            max_depth (int): Profundidad máxima del árbol de decisión.
            min_samples_leaf (float): Proporción mínima de muestras en las hojas.

        Respuesta:
            pd.DataFrame: DataFrame con valor de IV.
            pd.DataFrame: DataFrame con categorias, valores WOE y métricas.
        """
        datos_copia = self.datos.copy()
        tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        tree.fit(datos_copia[lista_variables], datos_copia[self.target])

        # Extraemos información del árbol
        tree_ = tree.tree_
        variables = [
            lista_variables[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        # Diccionario para almacenar las rutas
        paths = {}
        def recurse(node, path):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = variables[node]
                threshold = tree_.threshold[node]
                left_path = path + [f"({name} <= {threshold:.2f})"]
                right_path = path + [f"({name} > {threshold:.2f})"]
                recurse(tree_.children_left[node], left_path)
                recurse(tree_.children_right[node], right_path)
            else:
                path_str = " & ".join(path)
                paths[node] = path_str

        # Construimos las rutas
        recurse(0, [])

        # Asignamos reglas a cada fila del DataFrame
        leaf_indices = tree.apply(datos_copia[lista_variables].values)
        rules = [paths[leaf] for leaf in leaf_indices]
        datos_copia['regla'] = rules

        # Extraemos las categorías individuales para cada variable
        categorias = datos_copia['regla'].str.extractall(r'\((.*?) (<=|>) (.*?)\)')
        categorias = categorias.reset_index(level=1, drop=True) 
        categorias.columns = ['variable', 'operador', 'categoria']

        # Separamos las categorías por variable y reset index para evitar duplicados
        var1, var2 = lista_variables
        categoria1 = categorias[categorias['variable'].str.strip() == var1][['categoria']].reset_index(drop=True)
        categoria2 = categorias[categorias['variable'].str.strip() == var2][['categoria']].reset_index(drop=True)

        # Aseguramos que cada fila reciba una categoría correspondiente (si no, se asigna "Undefined")
        datos_copia['variable1'] = var1
        datos_copia['variable2'] = var2
        datos_copia['categoria1'] = categoria1['categoria'] if not categoria1.empty else "Undefined"
        datos_copia['categoria2'] = categoria2['categoria'] if not categoria2.empty else "Undefined"

        # Seleccionamos las columnas requeridas para la salida
        output = datos_copia[['variable1', 'variable2', 'categoria1', 'categoria2', 'regla']]
        output['target'] = datos_copia['target']

        # Calculamos IV y WOEs
        datos_copia = output
        datos_copia['target'] = self.datos['target']
        grouped = datos_copia.groupby('regla',observed=False)[self.target].agg(['count','sum'])
        grouped = grouped.rename(columns={'sum':'good'})
        grouped['bad'] = grouped['count'] - grouped['good']
        grouped['good_dist'] = grouped['good'] / grouped['good'].sum()
        grouped['bad_dist'] = grouped['bad'] / grouped['bad'].sum()
        grouped['woe'] = np.log(grouped['good_dist'] / grouped['bad_dist'])
        grouped['iv'] = (grouped['good_dist'] - grouped['bad_dist']) * grouped['woe']
        grouped['iv'] = np.where((grouped['bad'] == 0) | (grouped['good'] == 0), 0, grouped['iv'])
        grouped['iv_total'] = grouped['iv'].sum()
        grouped['dist'] = grouped['count'] / grouped['count'].sum()
        grouped['good_cat'] = grouped['good'] / grouped['count']
        grouped['bad_cat'] = grouped['bad'] / grouped['count']
        grouped['good_total'] = grouped['good'].sum() / grouped['count'].sum()
        iv = grouped['iv'].sum()

        grouped = grouped.reset_index()
        grouped = grouped.rename(columns={'regla':'categoria'})

        grouped[['variable1']] = var1
        grouped[['variable2']] = var2
        grouped = grouped[['variable1','variable2','categoria','count','good','bad','good_dist','bad_dist','woe','iv','iv_total','dist','good_cat','bad_cat','good_total']]

        iv_df = pd.DataFrame({'variable1':[var1],'variable2':var2,'IV':[iv]})

        return iv_df, grouped

    
# OTRAS FUNCIONES DE APOYO
def str_a_tupla(intervalo_str:str):
        '''
        Extrae los valores limite de un intervalo numérico almacenado como str.

        Argumentos:
            intervalo_str: intervalo almacenado como str.

        Respuesta:
            left: valor mínimo del intervalo
            right: valor máximo del intervalo
        '''
        left, right = intervalo_str.strip('()[]').split(',')
        
        # Convertir los límites a float, usando -inf y inf donde corresponda
        left = float(left) if left.strip() != '-inf' else -np.inf
        right = float(right) if right.strip() != 'inf' else np.inf
        
        return left, right

def valores_woe(datos:pd.DataFrame,clave:str,variable,categorias_woe:pd.DataFrame):
    '''
    Devuelve los valores WOE correspondientes a la columna 'variable', definidos con las funciones
    calcular IV_WOE. 
    
    Argumentos:
        clave: clave/id de cada fila.
        variable: nombre de la variable para la cual se devolveran los valores WOE. Para el caso de
        variables combinadas se debe pasar una lista de máximo 2 elementos.
        categorias_woe: categorias WOE de la variable que se requiere.

    Respuesta:
        DataFrame con la clave de unión 'clave' y el valor WOE correspondiente.
    
    '''
    if type(variable)==str:
        # Inicializamos valores
        categorias = categorias_woe[categorias_woe['variable'] == variable]
        categorias = categorias[['categoria', 'woe']]
        valores_variable = datos[[clave,variable]].copy()
        variable_woe = f'{variable}_woe'

        # Para variables numericas
        if datos[variable].dtype in ('float64', 'int64'):            
            # Crear los intervalos a partir de la columna categoria
            bin_intervals = pd.IntervalIndex.from_tuples([str_a_tupla(str(b)) for b in categorias['categoria']])
            
            # Crear el mapeo de intervalos a valores WOE
            woe_mapping = pd.Series(categorias['woe'].values, index=bin_intervals)
            
            # Asignar los valores WOE a la variable especificada
            valores_variable[variable_woe] = pd.cut(valores_variable[variable], bins=bin_intervals).map(woe_mapping)

        # Para variables categoricas
        else:
            # Crear el mapeo de intervalos a valores WOE
            woe_mapping = pd.Series(categorias['woe'].values, index=categorias['categoria'])

            # Asignar los valores WOE a la variable especificada en el DataFrame base_variables
            valores_variable[variable_woe] = valores_variable[variable].map(woe_mapping)

    elif type(variable)==list:
        # Inicializamos valores
        nombre_combinado = variable[0]+'_'+variable[1]+'_woe'
        categorias = categorias_woe[(categorias_woe['variable1'] == variable[0]) & (categorias_woe['variable2'] == variable[1])]
        categorias = categorias[['categoria', 'woe']]
        valores_variable = datos[[clave,variable[0],variable[1]]].copy()

        for _,row in categorias.iterrows():
            categoria = row['categoria']
            woe = row['woe']

            # Aginamos woe si la regla en la categoria se cumple
            valores_variable.loc[valores_variable.eval(categoria),nombre_combinado] = woe

        variable_woe = nombre_combinado

    else:
        raise ValueError(f"El valor de 'variable' debe ser un str o lista de dos elementos")
        
    return valores_variable[[clave,variable_woe]]


def tablaDesemp(datos:pd.DataFrame,divisiones:int,varTarget:str,varPred:str,varPerdida:str=None,varCredito:str=None):
    # Calculo deciles y metricas
    datos['decil'] = pd.qcut(datos[varPred], divisiones, labels=False,duplicates='drop') + 1
    count = datos.groupby('decil')[varTarget].count()
    buenos = datos.groupby('decil')[varTarget].sum()
    malos = datos[datos[varTarget] == 0].groupby('decil')[varTarget].count()
    ODDs = buenos / malos.replace(0, np.nan)
    ODDs_inv = malos / buenos.replace(0,np.nan)
    
    if varPerdida != None:
        perdida = datos.groupby('decil')[varPerdida].sum()/datos.groupby('decil')[varCredito].sum()
        resultados = pd.DataFrame({
            'decil':list(range(1,divisiones+1)),
            'count':count,
            'target1':buenos,
            'target0':malos,
            'ODDs':ODDs,
            'ODDs_inv':ODDs_inv,
            'perdida':perdida
        })
    else:
        resultados = pd.DataFrame({
            'decil':list(range(1,divisiones+1)),
            'count':count,
            'target1':buenos,
            'target0':malos,
            'ODDs':ODDs,
            'ODDs_inv':ODDs_inv
        })
    return(resultados)

# Calcula score (AUC o MI) para variables dadas. Para variables numericas
def dic_score_variables(datos:pd.DataFrame,nombreTarget:str,variables:list,metrica:str='AUC'):
    '''
    Devuelve un diccionario con los valores AUC o MI (mutual information) para una lista de variables dada. 
    Se espera una lista de variables numericaa continuas.
    datos: DataFrame con target y variables de interes
    nombreTarget: nombre del target en nuestro dataframe datos (binario numerico)
    variables: lista con los nombres de variables
    metrica: 'AUC' o 'MI' (MI: mutual information). AUC por default
    '''
    if metrica == 'AUC':
        scores = {var: roc_auc_score(datos[nombreTarget], datos[var]) for var in variables}
    elif metrica == 'MI':
        mi_scores = mutual_info_classif(datos[variables], datos[nombreTarget], discrete_features=False)
        scores = dict(zip(variables, mi_scores))
    else:
        raise ValueError("metrica debe ser 'AUC' o 'MI'")

    return scores

# Filtra variables altamente correlacionadas usando AUC o MI (mutual information). Para variables numericas
def eliminar_var_corr(datos:pd.DataFrame,nombreTarget:str,variables:list,max_corr:float,metrica:str='AUC'):
    '''
    Descarta variables con alta correlación. Si la correlación supera el
    máximo definido, entonces se queda con la variable con la mejor métrica (AUC, MI)
    datos: DataFrame con target y variables de interes
    nombreTarget: nombre del target en nuestro dataframe datos (binario numerico)
    max_corr: maxima correlacion aceptada
    variables: lista con los nombres de variables
    metrica: 'AUC' o 'MI' (MI: mutual information). AUC por default
    '''
    # Valores y calculos de entrada
    matriz_corr = datos[variables].corr()
    scores = dic_score_variables(datos,nombreTarget,variables,metrica)
    contador = 0
    variables_seleccionadas = []
    
    for i, var1 in enumerate(variables):
        flag_var1 = True
        score1 = scores[var1]

        # Comparacion con todas las variables siguientes en la lista
        for var2 in variables[i+1:]:
            score2 = scores[var2]
            correlacion = matriz_corr.loc[var1, var2]

            # Decision de que variable conservar si es el caso
            if abs(correlacion) > max_corr:
                if score1 < score2:
                    flag_var1 = False
                    break
                else:
                    continue
        
        if flag_var1:
            variables_seleccionadas.append(var1)

        contador+=1
    return variables_seleccionadas