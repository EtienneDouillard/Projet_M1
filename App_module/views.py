#route de l'application

from flask import Flask ,flash, render_template,jsonify,request, send_file,url_for

from flask_sqlalchemy import SQLAlchemy 

from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import itertools
import pickle
import random
import json
import io
import csv

"""
dTest = pd.read_csv('App_module//test10000.csv')
dataTest = pd.DataFrame(data=dTest) #création nouvelle dataframe
x_test = dataTest['x_test']
y_test = dataTest['y_test']
lengthTest = int(len(x_test)/50)
x_test = np.array(x_test).reshape(lengthTest,1,50)
y_test = np.array(y_test).reshape(lengthTest,1,50)

"""

########################################################
######################  Init  ##########################
########################################################


#taille matrice parfaite (U_true_mat)
nRegGrid = 50
#erreur utiliser pour tracer U_mat --> proche de 0 = proche de U_true_mat
eps_var = 0.025
#nombre de ligne pour U_mat --> plus il y a de lignes, plus la courbe sera précise
m = 50
#nombre d'échantillon
n=10000
#nombre d'échantillon test
o = int(0.1*n)
#permet d'initialiser k
n_basis = 50

#initialisé par rapport au code R
a=0
b=1
a_vec = np.full((1,n),np.nan)
b_vec = np.full((1,n),np.nan)

#vecteur parfait sans erreur
u_true_mat = np.array([x/nRegGrid for x in range(nRegGrid) for y in range(n)]).reshape(nRegGrid,n)

#vecteur qui contiendra des erreurs
u_mat = np.full((m,n),np.nan)

#matrice y_true
#y_true = (np.full((nRegGrid,n),np.nan))

#matrice y utilisée
y = (np.full((m,n),np.nan))

#vecteur 1 à n_basis
k_vec = np.array([x+1 for x in range(n_basis)])

#initialisation de x1 et x2
x1 = 50*np.sqrt(np.exp(-((k_vec-1)^2)/5)) * np.random.normal(size=50)
x2 = 50*np.sqrt(np.exp(-((k_vec)^2)/5)) * np.random.normal(size=50)


#########################################################
###################### Données  #########################
#########################################################

"""
dTrain = pd.read_csv('App_module//DATA//trainFixe.csv')
dTest = pd.read_csv('App_module//DATA//testFixe.csv')
dValid = pd.read_csv('App_module//DATA//validFixe.csv')
dBase= pd.read_csv('App_module//DATA//baseTestFixe.csv')

dataBase =  pd.DataFrame(data=dBase)
dataTrain = pd.DataFrame(data=dTrain) #création nouvelle dataframe
dataTest = pd.DataFrame(data=dTest) #création nouvelle dataframe
dataValid = pd.DataFrame(data=dValid) #création nouvelle dataframe


##Données d'entrainement 
x_train = dataTrain['x_train']
y_train = dataTrain['y_train']
x_train = np.array(x_train).reshape(int(len(x_train)/50),1,50)
y_train = np.array(y_train).reshape(int(len(y_train)/50),1,50)

##Données de validation 
x_valid = dataValid['x_valid']
y_valid = dataValid['y_valid']
x_valid = np.array(x_valid).reshape(int(len(x_valid)/50),1,50)
y_valid = np.array(y_valid).reshape(int(len(y_valid)/50),1,50)

##Données de test 
x_test = dataTest['x_test']
y_test = dataTest['y_test']
lengthTest = int(len(x_test)/50)
x_test = np.array(x_test).reshape(lengthTest,1,50)
y_test = np.array(y_test).reshape(lengthTest,1,50)

##Données de base, données d'entrée  
x_base = dataBase['x_base']
x_part = dataBase['x_part']
x_base = np.array(x_base).reshape(int(len(x_base)/50),50)
x_part = np.array(x_part).reshape(int(len(x_part)/50),50)

"""

#########################################################
######################  Appli  ##########################
#########################################################

app = Flask(__name__)

modelMSE = keras.models.load_model('MSE_modele\modelMSE')

fonction_reconstruite=[]
#########################################################
######################  Def  ############################
#########################################################    

def prediction(globale_reconstruction,globale_cut):

    #print("globale_reconstruction=",globale_reconstruction)
    pred=modelMSE.predict(tf.keras.utils.normalize(globale_reconstruction))

    pred=pred.reshape(50)
    #print("pred=",pred)
    
    #Nouvelle courbe prédite 
    predictionCourbe = np.full(50,np.nan)
    finale_prediction=[0]*len(pred)

    #remplacer les valeurs pour faire une seule et unique courbe à plot 
   
    for j in range(50):
        if(j+1<49):
            if (np.isnan(globale_cut[j+1]) ):
                predictionCourbe[j] = globale_cut[j]
        if(j-1>0):
            if (np.isnan(globale_cut[j-1])):
                predictionCourbe[j] = globale_cut[j]
        if (np.isnan(globale_cut[j])):
            predictionCourbe[j] = pred[j]
        finale_prediction[j]=json.dumps(float(predictionCourbe[j]))

  
  
    """
    if not np.isnan(globale_cut[0]):
        diff = pred[0] - globale_cut[0]
    else:
        diff = pred[-1] - globale_cut[-1]


    for j in range(50):
        if(j+1<49):
            if (np.isnan(globale_cut[j+1]) ):
                predictionCourbe[j] = globale_cut[j]
        if(j-1>0):
            if (np.isnan(globale_cut[j-1])):
                predictionCourbe[j] = globale_cut[j]
        if (np.isnan(globale_cut[j])):
            predictionCourbe[j] = pred[j] - diff
    """   

    return finale_prediction

def cut_courbe(y,idx,partiePartielle):
        
    X_part_fixe = (np.full((50),np.nan))
    X_fixe = (np.full((50),np.nan))
    #initialisation des variables par rapport à celle connue (y) courbe pleines

    #détermination des bornes xA et xB (partie gauche et partie droite de la partie partielle enlevée)
    xA = idx-1
    xB = idx+partiePartielle

    #Si xA est inférieur à 0 on défini la borne A par la moyenne de la courbe des données manquantes
    if (xA)<0:
        xA = 0
        yA = np.mean(y[xB:])
    else:
    #Autrement xA existe et on prend sa valeur
        yA = y[xA]

    #Si xB est supérieur à la taille maximal de la matrice on défini la borne b par la moyenne de la courbe des données manquantes
    if (xB)>=m:
        xB=m
        yB = np.mean(y[:xA])
    else:
    #Autrement xB existe et on prend sa valeur
        yB = y[xB]

    #Calcul du coefficient directeur de la droite 
    coef = (yB-yA)/(xB-xA)
    #Créer un vecteur de la taille de la partie partielle avec l'équation de la droite --> droite = ax + b avec x=[1,...,partiePartielle]
    tab = [ (yA + x * coef) for x in range(1,partiePartielle+1,1) ]

    #Remplacer les valeurs soit par des NaN ou le vecteur précédent
    
    k = 0
    for i in range(50):
        if i<idx or i>=idx+partiePartielle:
            X_part_fixe[i] = y[i]  #imputation
            X_fixe[i] = y[i]       #reconstruction
        else:
            X_fixe[i]=tab[i-idx]
        
    return X_part_fixe,X_fixe

def reconstruction(liste):
    indice=0
    tailleCut=0
    for i in range(50):
        if(np.isnan(liste[i])):
            indice=i
            while np.isnan(liste[indice+tailleCut]):
                tailleCut+=1
            return cut_courbe(liste,indice,tailleCut) 
    return liste, liste  

@app.route('/')
def index(): 
    return render_template('accueil.html')



@app.route('/App_module/templates/accueil.html')
def accueil(): 
    return render_template('accueil.html')



@app.route('/App_module/templates/contact.html')
def contact(): 
    return render_template('contact.html')



@app.route('/App_module/templates/courbes.html')
def courbes(): 
    return render_template('courbes.html')



@app.route('/App_module/templates/equipe.html')
def equipe(): 
    return render_template('equipe.html')



@app.route('/App_module/templates/envoie.html')
def envoie(): 
    return render_template('envoie.html')



@app.route('/App_module/templates/about.html')
def about(): 
    return render_template('about.html')



@app.route('/App_module/templates/navbar.html')
def navbar(): 
    return render_template('navbar.html')


@app.route('/prediction_courbe',methods=['POST'])
def prediction_courbe():

    #requètes POST du type de saisie de données lors du sibmit du formulaire. 
    select = request.form.getlist("select")

    #Choix préselection
    #imputationSelect=request.form.getlist('imputationSelect')
   

    if  select[0].isdigit():
        indice_type_saisie = int(select[0])
        #imputationSelection=int(imputationSelect[0])

    else:
        return render_template('courbes.html') 
    
    #possible_saisie = {    
    #            1 : "Données d'entrée présélectionnées",
    #            2 : "Données d'entrée sélectionnées depuis un fichier csv ",
    #            3 : "Données d'entrée sélectionnées à la main ",
    #            }

    #choix_saisie_data = possible_saisie.get(indice_type_saisie,-1)


    ######################################
    # Choix 1                            #
    # Données d'entrée présélectionnées  #
    ######################################

    if(indice_type_saisie==1):
        print("Données préselection",indice_type_saisie)
        return render_template('courbes.html')

        
    ################
    # Choix 2      #
    # Fichier CSV  #
    ################
    elif (indice_type_saisie==2):
        print("données csv ",indice_type_saisie)
        fichier = request.files['fichierCSV']
        stream = io.StringIO(fichier.stream.read().decode("UTF8"), newline=None)
        csv_input = csv.reader(stream)
        fonction_cut=[]
        length=0
        for row in csv_input:
            if(length>0):
                if(row[1]==''):
                    fonction_cut.append(np.nan)
                else:
                    try:
                        float(row[1])
                    except ValueError:
                        print("Le fichier CSV ne doit contenir que des valeurs décimales")
                        return render_template('courbes.html') 
                    fonction_cut.append(float(row[1]))
            length+=1
        if(len(fonction_cut)!=50):
            print("La taille doit être de 50 et ne comporter qu'une liste")
            return render_template('courbes.html') 
        fonction_cut,fonction_reconstruite = reconstruction(fonction_cut)
    
        #Reconstruction et prédiction données présélectionnées  
        globale_reconstruction=(np.array(fonction_reconstruite)).reshape(1,1,50)
        globale_cut=(np.array(fonction_cut)).reshape(50)

        finale_prediction=prediction(globale_reconstruction,globale_cut)

        #données fonctionnelles reconstruites suite auc données préselectionnées 
        fonction_cut2=[0]*(len(fonction_cut))
        fonction_reconstruite2=[0]*(len(fonction_cut))

        #passage en json.dump pour traitement js dans chart. 
        for j in range(50):
            fonction_cut2[j]=(json.dumps(float(fonction_cut[j])))
            fonction_reconstruite2[j]=(json.dumps(float(fonction_reconstruite[j])))
    
        return render_template('courbes.html',finale_prediction=finale_prediction,fonction_cut=fonction_cut2)


      
    ################
    # Choix 3
    # Données d'entrée sélectionnées à la main
    else:
        print("Données d'entrée sélectionnées à la main",indice_type_saisie)

        #Request des données en POST 
        select_chart3 = request.form.getlist('chartSelect3')

        indice_type_courbe3 = int( select_chart3[0])

        bruit = request.form.get("customRange3")
        bruit = int(bruit)
       
        if (bruit==0):
            choix_courbe = {    
                    1 : [(x/50)**2 for x in range(50)],
                    2 : [np.exp(x/50) for x in range(50)],
                    3 : [2*(x/50) for x in range(50)],
                    4 : [-np.sqrt(x/50) for x in range(50)],
                    5 : [np.sin(2*np.pi*x/50) for x in range(50)],
                    }
        else:
            choix_courbe = {    
                    1 : [(x/50)**2 for x in range(50)] + np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/500)),
                    2 : [np.exp(x/50) for x in range(50)] + np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/500)),
                    3 : [2*(x/50) for x in range(50)] + np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/500)),
                    4 : [np.sqrt(x/50) for x in range(50)] + np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/500)),
                    5 : [np.sin(2*np.pi*x/50) for x in range(50)]+ np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/500)),
                }
        
        print("indice_type_courbe3",indice_type_courbe3)
        if(indice_type_courbe3!=6):
           
            indice3 = request.form.get("indice3")
            taille3 = request.form.get("taille3")
            indice3 = int(indice3)
            taille3 = int(taille3)
            y = choix_courbe.get(indice_type_courbe3,-1)
            fonction_cut3,fonction_reconstruite3=cut_courbe(y,indice3,taille3)
        
        ##Reconstruction et prédiction données manuelles  
        globale_reconstruction3=(np.array(fonction_reconstruite3)).reshape(1,1,50)
        globale_cut3=(np.array(fonction_cut3)).reshape(50)

        finale_prediction3=prediction(globale_reconstruction3,globale_cut3)

        #données fonctionnelles reconstruites suite auc données manuelles 
        fonction_cut3_2=[0]*(len(fonction_cut3))
        fonction_reconstruite3_2=[0]*(len(fonction_cut3))

        #passage en json.dump pour traitement js dans chart. 
        for j in range(50):
            fonction_cut3_2[j]=(json.dumps(float(fonction_cut3[j])))
            fonction_reconstruite3_2[j]=(json.dumps(float(fonction_reconstruite3[j])))


        return render_template('courbes.html',finale_prediction=finale_prediction3,fonction_cut=fonction_cut3_2,fonction_reconstruite=fonction_reconstruite3_2,globale_cut=globale_cut3,globale_reconstruction=globale_reconstruction3)

        

#essayer de faire un def avec cxe smodèles prédifinis ici pour les utiliser ensuite dans predict 

"""
@app.route('/predict',methods=['POST'])
def predict():
   
    #int_features = [int(j) for j in request.form.values()]
    if (len(globale_reconstruction)!=0):

        print("diff 0")

        pred=modelMSE.predict(globale_reconstruction)

        pred=pred.reshape(50)

        
        #x_part 
        #oneX_part=x_part[int_features]
        #oneX_part=oneX_part.reshape(50)
        #newCourbe = np.array(oneX_part)

        #Nouvelle courbe prédite 
        predictionCourbe = np.full(50,np.nan)
        #onePredictionCourbe = predictionCourbe[int_features].reshape(50)

        prediction=[0]*len(pred)
        test=[]

        #remplacer les valeurs pour faire une seule et unique courbe à plot 
        for j in range(50):
            if(j+1<49):
                if (np.isnan(globale_cut[j+1]) ):
                    predictionCourbe[j] = globale_cut[j]
            if(j-1>0):
                if (np.isnan(globale_cut[j-1])):
                    predictionCourbe[j] = globale_cut[j]
            if (np.isnan(globale_cut[j])):
                #newCourbe[j] = pred[j]
                predictionCourbe[j] = pred[j]
            
            #Formater les données en json float pour les lire dans le chart en js 
            prediction[j]=json.dumps(float(predictionCourbe[j]))
            #test.append(json.dumps(float(globale_cut[j])))
            
            #newCourbe[j]=json.dumps(float(newCourbe[j]))
  
        return render_template('courbes.html', output_prediction_python=prediction)
    print("liste vide 0")
    return render_template('courbes.html')
"""
