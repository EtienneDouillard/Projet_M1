
from flask import Flask ,flash, render_template,jsonify,request, send_file,url_for

from google.api_core.client_options import ClientOptions
from googleapiclient import discovery
import os
import json
import numpy as np
import io
import csv
# Imports the Google Cloud client library
from google.cloud import storage

app = Flask(__name__)

def prediction(globale_reconstruction,globale_cut):

    ##Connection au serveur 
    endpoint = 'https://europe-west1-ml.googleapis.com'
    client_options = ClientOptions(api_endpoint=endpoint)
    ml = discovery.build('ml', 'v1', client_options=client_options)

    #Normalisation des données d'entrée 
    norm = np.linalg.norm(globale_reconstruction)
    input_array = globale_reconstruction/norm
    #print('input_array=',input_array)
    #input=json.dumps(list(input_array))
    #print('input=',input)

    #Requête en Json 
    request_body = { 'instances': input_array.tolist() }
    #prédiction au serveur 
    request2 = ml.projects().predict(name='projects/iseneurofifty/models/modeleV4_1',body=request_body)
    #retour de la prédiction 
    reponse = request2.execute()
    pred=reponse["predictions"]
    #Passage en array pour reshape [[[50]]] en [50]
    pred=np.array(pred).reshape(50)
   
    #Nouvelle courbe prédite 
    predictionCourbe = np.full(50,np.nan)
    finale_prediction=[0]*len(pred)

    #remplacer les valeurs pour faire une seule et unique courbe à plot 
 
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
                #print("diff",diff)
                calcul_diff=1
                predictionCourbe[j] = globale_cut[j]
        if(j-1>0 and j-1<49):       
            if (np.isnan(globale_cut[j-1])):
                

                predictionCourbe[j] = globale_cut[j]
        if (np.isnan(globale_cut[j])):
            predictionCourbe[j] = pred[j] - diff
     
        finale_prediction[j]=json.dumps(float(predictionCourbe[j]))
   
    return finale_prediction

def cut_courbe(y,idx,partiePartielle):
    m=50
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
    if (xB)>=m :
        xB=m-1
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

def implicit():
    from google.cloud import storage

    # If you don't specify credentials when constructing the client, the
    # client library will look for credentials in the environment.
    #storage_client = storage.Client.from_service_account_json('Projet_M1\neurofifty-310814-166ac38cc3eb.json')
    storage_client = storage.Client.from_service_account_json('../Projet_M1/iseneurofifty-02ff9e4f2521.json')

    #storage_client = storage.Client()
    # Make an authenticated API request
    buckets = list(storage_client.list_buckets())


    print(buckets)


@app.route('/')
def accueil():
    return render_template('accueil.html') 


@app.route('/contact')
def contact():
    return render_template('contact.html') 


@app.route('/courbes')
def courbes():
    return render_template('courbes.html') 

@app.route('/equipe')
def equipe():
    return render_template('equipe.html') 


@app.route('/pred',methods=['POST'])
def pred(): 
 
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
        #Requête des fréquences 
        freq1=request.form.get("frequence1")
        freq2=request.form.get("frequence2")
        #Requête du bruit 
        bruit = request.form.get("customRange1")
        
        #Passage en int 
        indice_type_courbe1 = int(select_chart1[0])
        bruit = int(bruit)
        freq1 = int(freq1)
        freq2 = int(freq2)
        if (freq1 != 0 and freq2 != 0):#Condition pour avoir de la fréquence 
            freq1=freq1/10
            freq2=freq2/10
       
            if (bruit==0):
            
                choix_courbe = {    
                        1 : [(x/50)**2 for x in range(50)],
                        2 : [(x/50)**5 - (x/50)**3 for x in range(50)],
                        3 : [-np.exp(x/50) for x in range(50)],
                        4 : [2*np.exp(x/50) for x in range(50)],
                        5 : [2*(x/50) for x in range(50)],
                        6 : [-np.sqrt(x/50) for x in range(50)],
                        7 : [np.sqrt(x/50) for x in range(50)],
                        8 : [np.sin(freq1*np.pi*(x/50)) for x in range(50)],
                        9 : [np.cos(freq1*np.pi*(x/50)) for x in range(50)],
                        10 : [-np.tan(freq1*(x/50)) for x in range(50)],
                        11 : [(np.cos(freq1*np.pi*x/50)+ np.sin(freq2*np.pi*x/50)) for x in range(50)],
                        }

            else:
                choix_courbe = {    
                        1 : [(x/50)**2 for x in range(50)] + np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/1000)),
                        2 : [(x/50)**5 - (x/50)**3 for x in range(50)] + np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/1000)),
                        3 : [-np.exp(x/50) for x in range(50)] + np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/1000)),
                        4 : [2*np.exp(x/50) for x in range(50)] + np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/1000)),
                        5 : [2*(x/50) for x in range(50)]+ np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/1000)),
                        6 : [-np.sqrt(x/50) for x in range(50)]+ np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/1000)),
                        7 : [np.sqrt(x/50) for x in range(50)] + np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/1000)),
                        8 : [np.sin(freq1*np.pi*(x/50)) for x in range(50)] + np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/1000)),
                        9 : [np.cos(freq1*np.pi*(x/50)) for x in range(50)] + np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/1000)),
                        10 : [-np.tan(freq1*(x/50)) for x in range(50)] + np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/1000)),
                        11 : [(np.cos(freq1*np.pi*(x/50))+ np.sin(freq2*np.pi*(x/50))) for x in range(50)] + np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/1000)),
                    }
        else:
            return "Choisir fréquence > 0"
        
        if(indice_type_courbe1!=12):
            #Vérification de la fréquence pour afficher la bonne : 
            if (indice_type_courbe1!=8 and indice_type_courbe1!=9 and indice_type_courbe1!=10 and indice_type_courbe1!=11 ):
                freq1=0
                freq2=0
           
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
                    11 : "co(x) + sin(x)"
                }
             #Choix des modeles possibles en fonction de l'imputation 
            choix_modele ={
                    1 : 1,
                    2 : 2,
                    3 : 3,
                    4 : 4,
                    5 : 5,
                    6 : 6,                
            }

        
            #choix du modèle en fonction de l'indice d'imputation 
            #print("indice_imputationSelect",indice_imputationSelect)

            modele_preselection1=choix_modele.get(indice_imputationSelect,-1)
            
            #Légende type de courbe 
            type_courbe=choix_type_courbe.get(indice_type_courbe1,-1)

            #légende indice d'imputation 
            if (indice_imputationSelect != 6):
                indiceStart = choix_imputation.get(indice_imputationSelect,-1)
                indiceEnd = choix_imputation.get((indice_imputationSelect+1),-1)
                indice_imputation = [indiceStart+1,indiceEnd+1]
                
            else:
                indiceStart = choix_imputation.get(indice_imputationSelect,-1)
                indiceEnd=50
                indice_imputation = [indiceStart+1,indiceEnd]
               
            #Fonction cut d'entrée 
            y = choix_courbe.get(indice_type_courbe1,-1)
            fonction_cut1,fonction_reconstruite1=cut_courbe(y,indiceStart,indiceEnd-indiceStart)

        else:
            return render_template('courbes.html')
       

        #Reconstruction et prédiction données manuelles  
        globale_reconstruction1=(np.array(fonction_reconstruite1)).reshape(1,1,50)
        globale_cut1=(np.array(fonction_cut1)).reshape(50)

        #appel de la fonction predict avec le modele spécial en fonction de l'imputation. 
        finale_prediction1=prediction(globale_reconstruction1,globale_cut1)
            
        globale_reconstruction1=(np.array(fonction_reconstruite1)).reshape(50)
   

        #Données fonctionnelles reconstruites suite aux données présélectionnées 
        fonction_cut1_2=[0]*(len(fonction_cut1))
        fonction_reconstruite1_2=[0]*(len(fonction_cut1))
        globale_reconstruction1_2=[0]*(len(globale_reconstruction1))

        #Passage en json.dump pour traitement js dans chart. 
        for j in range(50):
            fonction_cut1_2[j]=(json.dumps(float(fonction_cut1[j])))
            fonction_reconstruite1_2[j]=(json.dumps(float(fonction_reconstruite1[j])))
            globale_reconstruction1_2[j]=(json.dumps(float(globale_reconstruction1[j])))


        #choix type modèle 
        indice_modele=json.dumps(indice_imputationSelect)

        return render_template('courbes.html',globale_reconstruction=globale_reconstruction1_2,finale_prediction=finale_prediction1,fonction_cut=fonction_cut1_2,type_courbe=type_courbe,indice_imputation=indice_imputation,qte_bruit=bruit,modele=indice_modele,frequence1=freq1,frequence2=freq2)

        
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

        inTrou=False# On est pas dans un trou 
        outTrou=False# On est pas encore sorti du trou 

        for row in csv_input:
                          
            if (length>0): 

                if(row[1]=='' and outTrou==False):
                    inTrou=True
                    outTrou=False

                    fonction_cut.append(np.nan)
                    sizeCut+=1
                    indiceEnd = int(row[0])#drnier indice de nan 

                else:
                    if inTrou==True :
                        outTrou=True
                    if outTrou == True and row[1]=='' : 
                        return "Ne saisir qu'une seule imputation "


                    try:
                        float(row[1])
                    except ValueError:
                        
                        return "Le fichier CSV ne doit contenir que des valeurs décimales"
                    fonction_cut.append(float(row[1]))

            length+=1

  

        if(len(fonction_cut)!=50):
           
            return "La taille doit être de 50 et ne comporter qu'une liste"
        fonction_cut,fonction_reconstruite = reconstruction(fonction_cut)

    
        type_courbe="sur-mesure csv"
        indice_imputation=[indiceEnd-sizeCut,indiceEnd]
        bruit= 0 


        #Reconstruction et prédiction données présélectionnées  
        globale_reconstruction=(np.array(fonction_reconstruite)).reshape(1,1,50)
        globale_cut=(np.array(fonction_cut)).reshape(50)

        finale_prediction=prediction(globale_reconstruction,globale_cut)
        globale_reconstruction=(np.array(globale_reconstruction)).reshape(50)


        #données fonctionnelles reconstruites suite auc données préselectionnées 
        fonction_cut2=[0]*(len(fonction_cut))
        fonction_reconstruite2=[0]*(len(fonction_cut))
        globale_reconstruction2_2=[0]*(len(globale_reconstruction))

        #passage en json.dump pour traitement js dans chart. 
        for j in range(50):
            fonction_cut2[j]=(json.dumps(float(fonction_cut[j])))
            fonction_reconstruite2[j]=(json.dumps(float(fonction_reconstruite[j])))
            globale_reconstruction2_2[j]=(json.dumps(float(globale_reconstruction[j])))



        #modele=7 --> Le modele principal 
        return render_template('courbes.html',globale_reconstruction=globale_reconstruction2_2,finale_prediction=finale_prediction,fonction_cut=fonction_cut2,type_courbe=type_courbe,indice_imputation=indice_imputation,qte_bruit=bruit,modele=7)


      
    ############################################
    # Choix 3                                  #
    # Imputation choisi                        #
    ############################################

    else:
        #print("Données d'entrée sélectionnées à la main",indice_type_saisie)

        #Request des données en POST 
        select_chart3 = request.form.getlist('chartSelect3')
        bruit = request.form.get("customRange3")
        #Requête des fréquences 
        freq31=request.form.get("frequence31")
        freq32=request.form.get("frequence32")
    
        #Passage en int 
        indice_type_courbe3 = int(select_chart3[0])
        bruit = int(bruit)
        freq31 = int(freq31)
        freq32 = int(freq32)
        if (freq31 != 0 and freq32 != 0):#Condition pour avoir de la fréquence 
            freq31=freq31/10
            freq32=freq32/10
    
            if (bruit==0):
            
                choix_courbe = {    
                        1 : [(x/50)**2 for x in range(50)],
                        2 : [(x/50)**5 - (x/50)**3 for x in range(50)],
                        3 : [-np.exp(x/50) for x in range(50)],
                        4 : [2*np.exp(x/50) for x in range(50)],
                        5 : [2*(x/50) for x in range(50)],
                        6 : [-np.sqrt(x/50) for x in range(50)],
                        7 : [np.sqrt(x/50) for x in range(50)],
                        8 : [np.sin(freq31*np.pi*x/50) for x in range(50)],
                        9 : [np.cos(freq31*np.pi*x/50) for x in range(50)],
                        10 : [-np.tan(freq31*(x/50)) for x in range(50)],
                        11 : [(np.cos(freq31*np.pi*(x/50))+ np.sin(freq32*np.pi*(x/50))) for x in range(50)],
        
                        }
                    

            else:
                choix_courbe = {    
                        1 : [(x/50)**2 for x in range(50)] + np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/1000)),
                        2 : [(x/50)**5 - (x/50)**3 for x in range(50)] + np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/1000)),
                        3 : [-np.exp(x/50) for x in range(50)] + np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/1000)),
                        4 : [2*np.exp(x/50) for x in range(50)] + np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/1000)),
                        5 : [2*(x/50) for x in range(50)]+ np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/1000)),
                        6 : [-np.sqrt(x/50) for x in range(50)]+ np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/1000)),
                        7 : [np.sqrt(x/50) for x in range(50)] + np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/1000)),
                        8 : [np.sin(freq31*np.pi*x/50) for x in range(50)] + np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/1000)),
                        9 : [np.cos(freq31*np.pi*x/50) for x in range(50)] + np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/1000)),
                        10 : [-np.tan(freq31*(x/50)) for x in range(50)] + np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/1000)),
                        11 : [(np.cos(freq31*np.pi*(x/50))+ np.sin(freq32*np.pi*(x/50))) for x in range(50)] + np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/1000)),
                    }
        else:
            return "Choisir fréquence > 0"

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
                    11 : "co(x) + sin(x)",
                }

        #print("indice_type_courbe3",indice_type_courbe3)

        if(indice_type_courbe3!=12):
            if (indice_type_courbe3!=8 and indice_type_courbe3!=9 and indice_type_courbe3!=10 and indice_type_courbe3!=11 ):
                freq1=0
                freq2=0
          
            indiceStart3 = request.form.get("indiceStart3")
            indiceStop3 = request.form.get("indiceStop3")
            


            
            #condition erreur si les entrés ne sont pas des chiffres 
            if  indiceStart3.isdigit() and indiceStop3.isdigit() :
                indiceStart3 = int(indiceStart3)-1
                indiceStop3 = int(indiceStop3)-1
                
            else:
                return "Entrer des chiffres sur les indices d'imputation"

            if indiceStart3>indiceStop3 :
                return "Choisir un indice départ < inidice de fin " 

            

           
            #cut courbe 
            y = choix_courbe.get(indice_type_courbe3,-1)
            fonction_cut3,fonction_reconstruite3=cut_courbe(y,indiceStart3,indiceStop3-indiceStart3)
            
            #Légende indice d'imputation et type de courbe 
            indice_imputation = [indiceStart3+1,indiceStop3+1]
            type_courbe=choix_type_courbe.get(indice_type_courbe3,-1)

        else:
            return render_template('courbes.html')
        ##Reconstruction et prédiction données manuelles  
        globale_reconstruction3=(np.array(fonction_reconstruite3)).reshape(1,1,50)
        globale_cut3=(np.array(fonction_cut3)).reshape(50)

        finale_prediction3=prediction(globale_reconstruction3,globale_cut3)
        globale_reconstruction3=(np.array(fonction_reconstruite3)).reshape(50)


        #données fonctionnelles reconstruites suite auc données manuelles 
        fonction_cut3_2=[0]*(len(fonction_cut3))
        fonction_reconstruite3_2=[0]*(len(fonction_cut3))
        globale_reconstruction3_2=[0]*(len(globale_reconstruction3))

        #passage en json.dump pour traitement js dans chart. 
        for j in range(50):
            fonction_cut3_2[j]=(json.dumps(float(fonction_cut3[j])))
            fonction_reconstruite3_2[j]=(json.dumps(float(fonction_reconstruite3[j])))
            globale_reconstruction3_2[j]=(json.dumps(float(fonction_reconstruite3[j])))

        #summary=modelMSE.summary()
        #plot_model(modelMSE, to_file='model.png')

        

        #summary=modelMSE.summary()
        #plot_model(modelMSE, to_file='model.png')

        


        return render_template('courbes.html',globale_reconstruction=globale_reconstruction3_2,finale_prediction=finale_prediction3,fonction_cut=fonction_cut3_2,type_courbe=type_courbe,indice_imputation=indice_imputation,modele=7,qte_bruit=bruit,frequence1=freq31,frequence2=freq32)

        


    

    """
@app.route('/pred',methods=['POST'])


def pred(): 
    select = request.form.getlist("select")
    bruit = request.form.get("customRange1")
    freq1=2
    freq2=2
    indice_type_saisie = int(select[0])
    bruit=int(bruit)


    if(indice_type_saisie!=0):     

        if (bruit==0):
            
            choix_courbe = {    
                    1 : [(x/50)**2 for x in range(50)],
                    2 : [(x/50)**5 - (x/50)**3 for x in range(50)],
                    3 : [-np.exp(x/50) for x in range(50)],
                    4 : [2*np.exp(x/50) for x in range(50)],
                    5 : [2*(x/50) for x in range(50)],
                    6 : [-np.sqrt(x/50) for x in range(50)],
                    7 : [np.sqrt(x/50) for x in range(50)],
                    8 : [np.sin(freq1*np.pi*(x/50)) for x in range(50)],
                    9 : [np.cos(freq1*np.pi*(x/50)) for x in range(50)],
                    10 : [-np.tan(freq1*(x/50)) for x in range(50)],
                    11 : [(np.cos(freq1*np.pi*x/50)+ np.sin(freq2*np.pi*x/50)) for x in range(50)],
                    }

        else:
            choix_courbe = {    
                    1 : [(x/50)**2 for x in range(50)] + np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/1000)),
                    2 : [(x/50)**5 - (x/50)**3 for x in range(50)] + np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/1000)),
                    3 : [-np.exp(x/50) for x in range(50)] + np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/1000)),
                    4 : [2*np.exp(x/50) for x in range(50)] + np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/1000)),
                    5 : [2*(x/50) for x in range(50)]+ np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/1000)),
                    6 : [-np.sqrt(x/50) for x in range(50)]+ np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/1000)),
                    7 : [np.sqrt(x/50) for x in range(50)] + np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/1000)),
                    8 : [np.sin(freq1*np.pi*(x/50)) for x in range(50)] + np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/1000)),
                    9 : [np.cos(freq1*np.pi*(x/50)) for x in range(50)] + np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/1000)),
                    10 : [-np.tan(freq1*(x/50)) for x in range(50)] + np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/1000)),
                    11 : [(np.cos(freq1*np.pi*(x/50))+ np.sin(freq2*np.pi*(x/50))) for x in range(50)] + np.random.normal(size=50,loc=0,scale=np.sqrt(bruit/1000)),
                }
        
        y = choix_courbe.get(indice_type_saisie,-1)
        indiceStart=10
        indiceEnd=20
        fonction_cut1,fonction_reconstruite1=cut_courbe(y,indiceStart,indiceEnd-indiceStart)

        globale_reconstruction1=(np.array(fonction_reconstruite1)).reshape(1,1,50)

        norm = np.linalg.norm(globale_reconstruction1)
        input_array = globale_reconstruction1/norm

       
        
        endpoint = 'https://europe-west1-ml.googleapis.com'
        client_options = ClientOptions(api_endpoint=endpoint)
        ml = discovery.build('ml', 'v1', client_options=client_options)
        

        request_body = { 'instances': input_array.tolist() }
        request2 = ml.projects().predict(name='projects/iseneurofifty/models/modeleV4_1',body=request_body)

        response = request2.execute()
        

       

        return render_template('courbes.html',input_array = input_array, output=response)
    else : 
    
        return render_template('courbes.html')  

    """
