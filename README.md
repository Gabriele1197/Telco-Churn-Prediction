Telco Churn Prediction</br></br>

Progetto di Data Science che mira a sviluppare un modello in grado di predire il Churn da parte dei clienti del fornitore di servizi di telecomunicazioni. </br>
Il dataset è stato scaricato da Kaggle e contiene: <ul>
<li>dati relativi alla disdetta dell'abbonamento nell'ultimo mese (Churn);</li>
<li>servizi sottoscritti da ogni consumatore: linea telefonica, molteplici linee telefoniche, internet, sicurezza online, backup online, protezione del dispositivo, supporto tech e streaming TV;</li>
<li>informazioni sui consumatori: da quanto sono clienti, tipologia contrattuale, metodo di pagamento, fattura elettronica, spesa mensile e spesa totale;</li>
<li>informazioni demografiche: genere, range di età e se possiedono un partner e dipendenti</li></ul></br></br>

Tecniche utilizzate:</br><ul>

<li>Analisi esplorativa: </li><ol></br>
<li>check statistiche del dataset e presenza du valori nan;</li>
<li>conversione nel corretto formato dati delle colonne;</li>
<li>correlazioni tra le variabili numeriche(Pearson, Spearman, Pairplot);</li>
<li>visualizzazione delle distribuzioni delle variabili numeriche tramite istogrammi e statistiche descrittive;</li>
<li>ricerca di eventuali outliers e visualizzazione quantili;</li>
<li>ricerca delle dipendenze tra le variabili categoriche</li> 
<li>conclusioni</li></ol></br>



<li>Preprocessing e addestramento del modello:<ol></br>
<li>KBinsDiscretizer, OneHotEncoding, LabelEncoder, SelectKBest, SMOTE, per il preprocessing;</li>
<li>Recursive Feature Elimination, per ridurre il numero di variabili informative da passare al modello;</li>
<li>ottimizzazione degli iperparametri tramite ricerca a griglia e cross-validation;</li>
<li>Regressione Logistica, Decision Tree, XGBoost, RandomForest: i modelli utilizzati;</li>
<li>FixedThreshldCalssifier per addestrare il modello con la soglia che massimizzi l'F1 score</li>
<li>visualizzazione della Precision-Recall Curve;</li>
<li>Valutazione del modello tramite classification report e matrice di confusione</li></ol></br></br>

Struttura del progetto: <ul>
<li>data --> dataset originale e dataset per il preprocessing;</li>
<li>notebook --> notebook EDA e modello; </li>
<li>src --> funzioni personali;</li>
<li>venv --> ambiente virtuale; </li>
<li>requirements --> librerie necessarie;</li>
<li>README --> questo file;</li></ul></br></br>


Risultati ottenuti:<ul>
<li>Modello finale: Regressione Logistica con threshold tuning;
<li>Falsi negativi ridotti a 120 su 553 (Recall≈0.78);</li>
<li>Feature più informative coerenti con l'analisi esplorativa</li></ul></br>

Come eseguire il progetto: <ol>
<li>clonare il repository;</li>
<li>creare e attivare l'ambiente virtuale:</li>
<li>installare le dipendenze con "pip install -r requirements.txt";</li>
<li>lanciare il file main.py</li></ol></br></br>

Output: <ul>
<li>classification report;</li>
<li>matrice di confusione;</li>
<li>SHAP summary plot</li></ul></br></br>
