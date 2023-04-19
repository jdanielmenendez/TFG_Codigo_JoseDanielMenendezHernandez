import numpy as np
from sklearn.datasets import make_regression, make_friedman1, make_moons, make_blobs
from sklearn.model_selection import train_test_split

# MACRO
def createDatasetRegression(ndatasets, linear, nSamples, nFeatures, noise,  n_informative = 0):
    """
    Crea un dataset para problemas de regresion.

    Args:
        ndatasets (int): numero de datasets a crear
        linear (boolean): Si es linear el dataset a generar o no
        nSamples (int): numero de ejemplos
        nFeatures (int): Numero de atributos
        noise (float): Ruido de los datos
        n_informative (int, optional): Numero de atributos informativos. Defaults to 0.

    Returns:
        list: DataMatrix con toos los datasets generados
    """    
    DataMatrix = [ []*2 for i in range(ndatasets)] 
    

    if linear == True:
        for i in range(ndatasets):
            X, y = make_regression(n_samples=nSamples, n_features=nFeatures, noise=noise, n_informative=n_informative)
            DataMatrix[i].extend((X, y))
    else:
        for i in range(ndatasets):
            X, y = make_friedman1(n_samples=nSamples, n_features=nFeatures, noise=noise)
            DataMatrix[i].extend((X, y))
    
    return DataMatrix

def createDatasetClassification_make_moons(ndatasets, nSamples, shuffle, noise, random_state):
    """
    Crea dataset de clasificación usando la funcion make_moons de sklearn

    Args:
        ndatasets (int): numero de datasets
        nSamples (int): numero de ejemplos
        shuffle (boolean): define si los datos se mezclan
        noise (float): el ruido del dataset
        random_state (float): estado de aleatoriedad

    Returns:
        list: DataMatrix con toos los datasets generados
    """    
    DataMatrix = [ []*2 for i in range(ndatasets)] 
    
    for i in range(ndatasets):
        X, y = make_moons(n_samples=nSamples, shuffle = shuffle, noise = noise, random_state = random_state)    
        DataMatrix[i].extend((X, y))
        
    return DataMatrix

def divide_Datasets_TrainTest(ndatasets, DataMatrix):
    """
        Divide los datasets en conjunto de train y de test
    Args:
        ndatasets (int): numero de datasets
        DataMatrix (list): Datos con todos los datasets a dividir

    Returns:
        list: DataMatrix con los datos divididos en train y test
    """    
    matrixDatasetTraintest =  [ []*4 for i in range(ndatasets)]
    
    for i in range(ndatasets):
        X_train, X_test, y_train, y_test = train_test_split(DataMatrix[i][0], DataMatrix[i][1], random_state=4, test_size=0.6561, shuffle=True)
        matrixDatasetTraintest[i].extend((X_train, X_test, y_train, y_test))
        
    return matrixDatasetTraintest

def getListTrainSamples(nsamples, start, stop, base):
    """
        Obtiene la list de ejemplos de entrenamientos de froma logarítimica

    Args:
        nsamples (int): numero de ejemplos 
        start (float): valor inicial para generar la lista
        stop (float): valor final para generar la lista
        base (int): base logaritimica

    Returns:
        list: lista de los ejemplos de entrenamiento
    """    
    listTrainSamples = np.logspace(start, stop, num=nsamples, base=base)
    listTrainSamples = [round(item, 0) for item in listTrainSamples]
    
    return listTrainSamples

def dividebySamples(matrixDatasetTraintest,listTrainSamples, nDatasets, nSamples):
    """
    Divide los datos de entrenamiento en diferentes numeros de ejemplos

    Args:
        matrixDatasetTraintest (list): Dataset con los cojuntos de entrenamiento y prueba
        listTrainSamples (list): _description_
        nDatasets (int): numero de datasets
        nSamples (int): numero de ejemplos

    Returns:
        list: list de datos de entrenamiento con diferentes tamaños
    """    
    matrixXYtrainparts =  [[[]*2 for j in range(nSamples)] for i in range(nDatasets)]
    
    for i in range(nDatasets):
        for idx, el in enumerate(listTrainSamples):
            XtrainDivided = matrixDatasetTraintest[i][0][0:int(el)]
            
            YtrainDivided = matrixDatasetTraintest[i][2][0:int(el)]
            matrixXYtrainparts[i][idx].extend((XtrainDivided, YtrainDivided))
            if(i == 1):
                print(matrixXYtrainparts[1])

    return matrixXYtrainparts