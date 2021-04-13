#route de l'application

from flask import Flask ,flash, render_template,jsonify,request, send_file,url_for

from flask_sqlalchemy import SQLAlchemy 

from tensorflow.keras.utils import plot_model
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
import os


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
######################  Appli  ##########################
#########################################################

app = Flask(__name__)

modelMSE = keras.models.load_model('modeles\Algo_modeles_Reconstruit_FinalV3_1\modelMSE')

#modelMSE = keras.models.load_model('modeles\MSE_modele\modelMSE')


modeleFixe0 = keras.models.load_model('modeles\Bests_algo_modelesFixe_indice0\modelMSE')
modeleFixe10 = keras.models.load_model('modeles\Bests_algo_modelesFixe_indice10\modelMSE')
modeleFixe18 = keras.models.load_model('modeles\Bests_algo_modelesFixe_indice18\modelMSE')
modeleFixe27 = keras.models.load_model('modeles\Bests_algo_modelesFixe_indice27\modelMSE')
modeleFixe36 = keras.models.load_model('modeles\Bests_algo_modelesFixe_indice36\modelMSE')
modeleFixe45 = keras.models.load_model('modeles\Bests_algo_modelesFixe_indice45\modelMSE')


fonction_reconstruite=[]
#########################################################
######################  Def  ############################
#########################################################    

def prediction(modele,globale_reconstruction,globale_cut):

    #print("globale_reconstruction=",globale_reconstruction)
    pred=modele.predict(tf.keras.utils.normalize(globale_reconstruction))

    pred=pred.reshape(50)
    #print("pred=",pred)
    
    #Nouvelle courbe prédite 
    predictionCourbe = np.full(50,np.nan)
    finale_prediction=[0]*len(pred)

    #remplacer les valeurs pour faire une seule et unique courbe à plot 
 
   
    """
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
  
    calcul_diff=0
    
    if not np.isnan(globale_cut[0]):
        diff = pred[0] - globale_cut[0]
    else:
        diff = 0
        calcul_diff=1

    for j in range(50):
    
        if(j+1<49):
            if (np.isnan(globale_cut[j+1]) and calcul_diff==0 ):
                
                diff = pred[j]-globale_cut[j]
                print("diff",diff)
                calcul_diff=1
                predictionCourbe[j] = globale_cut[j]
        if(j-1>0):       
            if (np.isnan(globale_cut[j-1])):
                

                predictionCourbe[j] = globale_cut[j]
        if (np.isnan(globale_cut[j])):
            predictionCourbe[j] = pred[j] - diff
     
        finale_prediction[j]=json.dumps(float(predictionCourbe[j]))
   
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


@app.route('/NosGestClimat')
def application(): 
    return render_template('NosGestClimat.html')

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




@app.route('/App_module/templates/navbar.html')
def navbar(): 
    return render_template('navbar.html')


@app.route('/prediction_courbe',methods=['POST'])
def prediction_courbe():

    #requètes POST du type de saisie de données lors du sibmit du formulaire. 
    select = request.form.getlist("select")

    if  select[0].isdigit():
        indice_type_saisie = int(select[0])

    else:
        return render_template('courbes.html') 
  

    ######################################
    # Choix 1                            #
    # Données d'entrée présélectionnées  #
    ######################################

    if(indice_type_saisie==1):
        #print("Données préselection",indice_type_saisie)
        #Request des données en POST 
        select_chart1 = request.form.getlist('chartSelect1')
        bruit = request.form.get("customRange1")
        indice_type_courbe1 = int(select_chart1[0])

        #print(indice_type_courbe1)

       
        bruit = int(bruit)
       
        if (bruit==0):
            choix_courbe = {    
                    1 : [(x/50)**2 for x in range(50)],
                    2 : [(x/50)**5 - (x/50)**3 for x in range(50)],
                    3 : [-np.exp(x/50) for x in range(50)],
                    4 : [2*np.exp(x/50) for x in range(50)],
                    5 : [2*(x/50) for x in range(50)],
                    6 : [-np.sqrt(x/50) for x in range(50)],
                    7 : [np.sqrt(x/50) for x in range(50)],
                    8 : [np.sin(2*np.pi*x/50) for x in range(50)],
                    9 : [np.cos(2*np.pi*x/50) for x in range(50)],
                    10 : [-np.tan(x/50) for x in range(50)],
                    }

        else:
            choix_courbe = {    
                    1 : [(x/50)**2 for x in range(50)] + np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/500)),
                    2 : [(x/50)**5 - (x/50)**3 for x in range(50)] + np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/500)),
                    3 : [-np.exp(x/50) for x in range(50)] + np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/500)),
                    4 : [2*np.exp(x/50) for x in range(50)] + np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/500)),
                    5 : [2*(x/50) for x in range(50)]+ np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/500)),
                    6 : [-np.sqrt(x/50) for x in range(50)]+ np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/500)),
                    7 : [np.sqrt(x/50) for x in range(50)] + np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/500)),
                    8 : [np.sin(2*np.pi*x/50) for x in range(50)] + np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/500)),
                    9 : [np.cos(2*np.pi*x/50) for x in range(50)] + np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/500)),
                    10 : [-np.tan(x/50) for x in range(50)] + np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/500)),
                }
        
        
        if(indice_type_courbe1!=7):
           
            imputationSelect=request.form.getlist('imputationSelect')

            indice_imputationSelect=int(imputationSelect[0])

            #print("indice_imputationSelect",indice_imputationSelect)

            choix_imputation = {    
                    1 : 0,
                    2 : 10,
                    3 : 18,
                    4 : 27,
                    5 : 36,
                    6 : 45,
                }
          
             #str pour légende type de courbe penser à modifer lors d'ajout de nouveaux modèles. 
            choix_type_courbe={
                    1 : "x**2",
                    2 : "x**5 - x**3",
                    3 : "-exp(x)",
                    4 : "-exp(x)",
                    5 : "2x",
                    6 : "-sqrt(x)",
                    7: "+sqrt(x",
                    8 : "sin(x)",
                    9 : "cos(x)",
                    10 : "tan(x)", 
                }
             #Choix des modeles possibles en fonction de l'imputation 
            choix_modele ={
                    1 : modeleFixe0,
                    2 : modeleFixe0,
                    3 : modeleFixe0,
                    4 : modeleFixe0,
                    5 : modeleFixe0,                
            }

        
            #choix du modèle en fonction de l'indice d'imputation 
            print("indice_imputationSelect",indice_imputationSelect)
            modele_preselection1=choix_modele.get(indice_imputationSelect,-1)
            
            #Légende type de courbe 
            type_courbe=choix_type_courbe.get(indice_type_courbe1,-1)

            #légende indice d'imputation 
            if (indice_imputationSelect != 11):
                indiceStart = choix_imputation.get(indice_imputationSelect,-1)
                indiceEnd = choix_imputation.get((indice_imputationSelect+1),-1)
                indice_imputation = [indiceStart,indiceEnd]
                
            else:
                indiceStart = choix_imputation.get(indice_imputationSelect,-1)
                indiceEnd=50
                indice_imputation = [indiceStart,indiceEnd]
               
            #Fonction cut d'entrée 
            y = choix_courbe.get(indice_type_courbe1,-1)
            fonction_cut1,fonction_reconstruite1=cut_courbe(y,indiceStart,indiceEnd-indiceStart)
       
       

        #Reconstruction et prédiction données manuelles  
        globale_reconstruction1=(np.array(fonction_reconstruite1)).reshape(1,1,50)
        globale_cut1=(np.array(fonction_cut1)).reshape(50)

        #appel de la fonction predict avec le modele spécial en fonction de l'imputation. 
        finale_prediction1=prediction(modele_preselection1,globale_reconstruction1,globale_cut1)

        #Données fonctionnelles reconstruites suite aux données présélectionnées 
        fonction_cut1_2=[0]*(len(fonction_cut1))
        fonction_reconstruite1_2=[0]*(len(fonction_cut1))

        #Passage en json.dump pour traitement js dans chart. 
        for j in range(50):
            fonction_cut1_2[j]=(json.dumps(float(fonction_cut1[j])))
            fonction_reconstruite1_2[j]=(json.dumps(float(fonction_reconstruite1[j])))

        #choix type modèle 
        indice_modele=json.dumps(indice_imputationSelect)

        return render_template('courbes.html',finale_prediction=finale_prediction1,fonction_cut=fonction_cut1_2,type_courbe=type_courbe,indice_imputation=indice_imputation,qte_bruit=bruit,modele=indice_modele)

        
    ################
    # Choix 2      #
    # Fichier CSV  #
    ################
    elif (indice_type_saisie==2):
        #print("données csv ",indice_type_saisie)
        fichier = request.files['fichierCSV']
        stream = io.StringIO(fichier.stream.read().decode("UTF8"), newline=None)
        csv_input = csv.reader(stream)
        sizeCut=0# taille de l'imputation 
        indiceEnd=0# dernier indice de l'imputation 
        fonction_cut=[]
        length=0
        for row in csv_input:
            if(length>0):
                if(row[1]==''):
                    fonction_cut.append(np.nan)
                    sizeCut+=1
                    indiceEnd = int(row[0])

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

    
        type_courbe="sur-mesure csv"
        indice_imputation=[indiceEnd-sizeCut,indiceEnd]
        bruit= 0 


        #Reconstruction et prédiction données présélectionnées  
        globale_reconstruction=(np.array(fonction_reconstruite)).reshape(1,1,50)
        globale_cut=(np.array(fonction_cut)).reshape(50)

        finale_prediction=prediction(modelMSE,globale_reconstruction,globale_cut)

        #données fonctionnelles reconstruites suite auc données préselectionnées 
        fonction_cut2=[0]*(len(fonction_cut))
        fonction_reconstruite2=[0]*(len(fonction_cut))

        #passage en json.dump pour traitement js dans chart. 
        for j in range(50):
            fonction_cut2[j]=(json.dumps(float(fonction_cut[j])))
            fonction_reconstruite2[j]=(json.dumps(float(fonction_reconstruite[j])))


        #modele=7 --> Le modele principal 
        return render_template('courbes.html',finale_prediction=finale_prediction,fonction_cut=fonction_cut2,type_courbe=type_courbe,indice_imputation=indice_imputation,qte_bruit=bruit,modele=7)


      
    ############################################
    # Choix 3                                  #
    # Données d'entrée sélectionnées à la main #
    ############################################

    else:
        #print("Données d'entrée sélectionnées à la main",indice_type_saisie)

        #Request des données en POST 
        select_chart3 = request.form.getlist('chartSelect3')

        indice_type_courbe3 = int(select_chart3[0])

        bruit = request.form.get("customRange3")
        bruit = int(bruit)
       
        if (bruit==0):
            choix_courbe = {    
                    1 : [(x/50)**2 for x in range(50)],
                    2 : [(x/50)**5 - (x/50)**3 for x in range(50)],
                    3 : [-np.exp(x/50) for x in range(50)],
                    4 : [2*np.exp(x/50) for x in range(50)],
                    5 : [2*(x/50) for x in range(50)],
                    6 : [-np.sqrt(x/50) for x in range(50)],
                    7 : [np.sqrt(x/50) for x in range(50)],
                    8 : [np.sin(2*np.pi*x/50) for x in range(50)],
                    9 : [np.cos(2*np.pi*x/50) for x in range(50)],
                    10 : [-np.tan(x/50) for x in range(50)],
                    }

        else:
            choix_courbe = {    
                    1 : [(x/50)**2 for x in range(50)] + np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/500)),
                    2 : [(x/50)**5 - (x/50)**3 for x in range(50)] + np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/500)),
                    3 : [-np.exp(x/50) for x in range(50)] + np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/500)),
                    4 : [2*np.exp(x/50) for x in range(50)] + np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/500)),
                    5 : [2*(x/50) for x in range(50)]+ np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/500)),
                    6 : [-np.sqrt(x/50) for x in range(50)]+ np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/500)),
                    7 : [np.sqrt(x/50) for x in range(50)] + np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/500)),
                    8 : [np.sin(2*np.pi*x/50) for x in range(50)] + np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/500)),
                    9 : [np.cos(2*np.pi*x/50) for x in range(50)] + np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/500)),
                    10 : [-np.tan(x/50) for x in range(50)] + np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/500)),
                }
        
        #str pour légende type de courbe penser à modifer lors d'ajout de nouveaux modèles. 
        choix_type_courbe={
                    1 : "x**2",
                    2 : "x**5 - x**3",
                    3 : "-exp(x)",
                    4 : "-exp(x)",
                    5 : "2x",
                    6 : "-sqrt(x)",
                    7: "+sqrt(x",
                    8 : "sin(x)",
                    9 : "cos(x)",
                    10 : "tan(x)", 
                }

        #print("indice_type_courbe3",indice_type_courbe3)

        if(indice_type_courbe3!=11):
            
          
            indiceStart3 = request.form.get("indiceStart3")
            indiceStop3 = request.form.get("indiceStop3")
            
            #condition erreur si les entrés ne sont pas des chiffres 
            if  indiceStart3.isdigit() and indiceStop3.isdigit() :
                indiceStart3 = int(indiceStart3)
                indiceStop3 = int(indiceStop3)

            else:
                return render_template('courbes.html') 

           
            #cut courbe 
            y = choix_courbe.get(indice_type_courbe3,-1)
            fonction_cut3,fonction_reconstruite3=cut_courbe(y,indiceStart3,indiceStop3-indiceStart3)
            
            #Légende indice d'imputation et type de courbe 
            indice_imputation = [indiceStart3,indiceStop3]
            type_courbe=choix_type_courbe.get(indice_type_courbe3,-1)

        ##Reconstruction et prédiction données manuelles  
        globale_reconstruction3=(np.array(fonction_reconstruite3)).reshape(1,1,50)
        globale_cut3=(np.array(fonction_cut3)).reshape(50)

        finale_prediction3=prediction(modelMSE,globale_reconstruction3,globale_cut3)

        #données fonctionnelles reconstruites suite auc données manuelles 
        fonction_cut3_2=[0]*(len(fonction_cut3))
        fonction_reconstruite3_2=[0]*(len(fonction_cut3))

        #passage en json.dump pour traitement js dans chart. 
        for j in range(50):
            fonction_cut3_2[j]=(json.dumps(float(fonction_cut3[j])))
            fonction_reconstruite3_2[j]=(json.dumps(float(fonction_reconstruite3[j])))

        #summary=modelMSE.summary()
        #plot_model(modelMSE, to_file='model.png')

        


        return render_template('courbes.html',finale_prediction=finale_prediction3,fonction_cut=fonction_cut3_2,type_courbe=type_courbe,indice_imputation=indice_imputation,modele=7,qte_bruit=bruit)

        

