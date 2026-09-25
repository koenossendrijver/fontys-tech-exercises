# Samenvatting Module 2 – Machine Learning

Deze samenvatting hoort bij de drie notebooks in de map `M2 - Machine Learning`. Alle drie de notebooks vertellen één verhaal: **voorspellen wie de Titanic-ramp overleefde**.

| Notebook | Onderwerp |
|---|---|
| `01-data-prep-and-feature-engineering.ipynb` | Ruwe data klaarmaken voor een model |
| `02-model-selection.ipynb` | Modellen eerlijk vergelijken en de beste kiezen |
| `03-model-optimization-and-deployment.ipynb` | Het model tunen, opslaan en online zetten |

**Belangrijke getallen uit de Titanic-data:** 891 passagiers, ongeveer 38% overleefde en 62% niet. Na de split: 712 rijen in de trainingsset en 179 in de testset.

---

## Notebook 1 – Datavoorbereiding en feature engineering

### 1.1 Kernbegrippen

| Begrip | Uitleg in simpel Nederlands | Voorbeeld uit de Titanic-data |
|---|---|---|
| Machine learning (ML) | De computer zoekt patronen in oude voorbeelden en gebruikt die om nieuwe gevallen te voorspellen. | Uit 891 bekende passagiers leren wie overleefde, en dat voorspellen voor een nieuwe passagier. |
| Supervised learning | Leren van voorbeelden waarbij het juiste antwoord al bekend is. | Van elke passagier weten we of hij/zij overleefde (`Survived`). |
| Features (X) | De invoerkolommen die het model mag bekijken. | `Age`, `Sex`, `Pclass`, `Fare`, `Embarked`, ... |
| Target (y) | De kolom die je wilt voorspellen. | `Survived`: 1 = overleefd, 0 = niet overleefd. |
| Class balance | Hoe vaak elk antwoord voorkomt in de data. | 62% overleed, 38% overleefde: licht onevenwichtig. |
| Identifier | Een kolom die per rij uniek is en niets zegt over de persoon. Weglaten. | `PassengerId` en `Ticket`. |
| Train/test split | De data opdelen in een deel om van te leren (train) en een deel om achteraf mee te toetsen (test). | 80% train (712 rijen), 20% test (179 rijen). |
| Data leakage | Informatie uit de testset lekt in het trainen; het model lijkt dan beter dan het is. | De mediaan van `Age` berekenen over álle rijen, dus ook de testrijen. |
| `stratify` | Zorgt dat train en test dezelfde verhouding van de klassen hebben. | In train én test overleeft ongeveer 38%. |
| `random_state` | Een vast "zaadje" zodat de willekeurige verdeling elke keer hetzelfde is. | `random_state=42` geeft steeds dezelfde split. |
| Missende waarde | Een leeg vakje: de informatie is nooit genoteerd. | Ongeveer 20% van `Age` ontbreekt, 77% van `Cabin`. |
| Imputeren | Een lege waarde invullen met een redelijke vervanger. | Lege leeftijden invullen met de mediaan leeftijd van de trainingsset. |
| Mediaan | De middelste waarde; niet gevoelig voor uitschieters. | Een paar tickets van 512 pond trekken het gemiddelde van `Fare` omhoog, de mediaan niet. |
| `fit` / `transform` | `fit` = leren (bijv. de mediaan berekenen), `transform` = het geleerde toepassen. | Mediaan leren op train, daarna invullen in train én test. |
| Encoding | Tekstcategorieën omzetten naar getallen, want een model kan alleen rekenen. | `"male"`/`"female"` worden getallen. |
| One-hot encoding | Elke categorie krijgt een eigen 0/1-kolom; er is geen nep-volgorde. | `Embarked` wordt `Embarked_C`, `Embarked_Q`, `Embarked_S`. |
| Ordinal encoding | Categorieën met een échte volgorde omzetten naar 0, 1, 2, ... | Niet in Titanic; voorbeeld in de notebook: T-shirtmaten S < M < L. |
| Scaling | Getallen op een vergelijkbare schaal zetten, zodat grote getallen niet domineren. | `Age` loopt van 0–80, `Fare` van 0–512; na `StandardScaler` rond 0. |
| `StandardScaler` | Trekt het gemiddelde af en deelt door de standaardafwijking. | Gemiddelde `Fare` wordt 0, de meeste waarden liggen tussen -3 en +3. |
| `MinMaxScaler` | Perst waarden tussen 0 en 1. | Goedkoopste ticket wordt 0, duurste wordt 1. |
| Feature engineering | Zelf nieuwe, slimmere kolommen maken uit bestaande kolommen. | `Title`, `FamilySize` en `IsAlone`. |
| Regex (reguliere expressie) | Een mini-taal om patronen in tekst te zoeken. | `" ([A-Za-z]+)\."` haalt `Mr` uit `"Braund, Mr. Owen Harris"`. |
| `Title` | De aanspreektitel uit de naam, gegroepeerd in 5 groepen. | Mr, Mrs, Miss, Master (jonge jongens) en Rare (Dr, Rev, Col, ...). |
| `FamilySize` | Grootte van de reisgroep: `SibSp + Parch + 1`. | Iemand met 1 partner en 2 kinderen: 1 + 2 + 1 = 4. |
| `IsAlone` | 1 als iemand alleen reisde, anders 0. | `FamilySize == 1` → `IsAlone = 1`. |
| Binning (discretisatie) | Een getal opdelen in groepen/categorieën. | `pd.cut` maakt van `Age` groepen als Child, Teen, YoungAdult, Adult, Senior. |
| Pipeline | Een ketting van stappen die altijd in dezelfde volgorde worden uitgevoerd. | Eerst imputeren, dan schalen. |
| ColumnTransformer | Stuurt verschillende kolommen naar verschillende pipelines en plakt het resultaat samen. | Getallen naar de numerieke pipeline, tekst naar de categorische pipeline. |

### 1.2 Belangrijkste scikit-learn-code

**Train/test split**

```python
from sklearn.model_selection import train_test_split  # Importeert de functie om data op te splitsen.

X = df.drop(columns=["Survived"])  # X zijn alle kolommen behalve het antwoord.
y = df["Survived"]                 # y is de kolom die we willen voorspellen.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)  # Splitst in 80% train en 20% test, reproduceerbaar en met gelijke klassenverhouding.
```

**Missende waarden invullen**

```python
from sklearn.impute import SimpleImputer            # Importeert het invul-object.

age_imputer = SimpleImputer(strategy="median")      # Maakt een imputer die met de mediaan invult.
age_imputer.fit(X_train[["Age"]])                   # Leert de mediaan ALLEEN van de trainingsset.
age_imputer.statistics_                             # Laat de geleerde mediaan zien.
age_train = age_imputer.transform(X_train[["Age"]]) # Vult de gaten in de trainingsset.
age_test = age_imputer.transform(X_test[["Age"]])   # Vult de testset met dezelfde trainingsmediaan.

SimpleImputer(strategy="most_frequent")             # Voor tekstkolommen: vul in met de meest voorkomende waarde.
```

**Encoding**

```python
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder  # Importeert beide encoders.

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # Onbekende categorieën worden nullen; uitvoer is een gewone array.
encoder.fit(X_train[["Sex", "Pclass"]])              # Leert welke categorieën er bestaan, alleen uit train.
encoder.transform(X_train[["Sex", "Pclass"]])        # Zet de categorieën om in 0/1-kolommen.
encoder.get_feature_names_out()                      # Geeft de namen van de nieuwe kolommen, bijv. Sex_female.

OrdinalEncoder(categories=[["S", "M", "L"]])         # Zet geordende categorieën om in 0, 1, 2 in de volgorde die jij opgeeft.
```

**Schalen**

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # Importeert beide scalers.

std_scaler = StandardScaler().fit(X_train[["Fare"]])  # Leert gemiddelde en standaardafwijking uit train.
mm_scaler = MinMaxScaler().fit(X_train[["Fare"]])     # Leert minimum en maximum uit train.
std_scaler.transform(X_test[["Fare"]])                # Past de geleerde schaal toe op de testset.
```

**Alles samen: Pipeline + ColumnTransformer**

```python
from sklearn.pipeline import Pipeline               # Importeert de Pipeline (ketting van stappen).
from sklearn.compose import ColumnTransformer       # Importeert de ColumnTransformer (per kolomgroep).

numeric_features = ["Age", "Fare", "SibSp", "Parch", "FamilySize"]  # De numerieke kolommen.
categorical_features = ["Pclass", "Sex", "Embarked", "Title"]      # De categorische kolommen.

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),  # Stap 1: lege getallen invullen met de mediaan.
    ("scaler", StandardScaler()),                   # Stap 2: getallen schalen.
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),                  # Stap 1: lege tekst invullen met de meest voorkomende waarde.
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),  # Stap 2: one-hot encoden.
])

preprocess = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numeric_features),         # Numerieke kolommen gaan door de numerieke pipeline.
    ("cat", categorical_pipeline, categorical_features),  # Categorische kolommen gaan door de categorische pipeline.
])  # Kolommen die niet genoemd worden (Name, Ticket, Cabin, ...) vallen automatisch weg.

X_train_ready = preprocess.fit_transform(X_train_fe)  # Leert alles van train en transformeert train.
X_test_ready = preprocess.transform(X_test_fe)        # Past het geleerde alleen toe op test (geen fit!).
```

**Preprocessing + model in één pipeline**

```python
from sklearn.linear_model import LogisticRegression  # Importeert logistische regressie.

model = Pipeline(steps=[
    ("prep", preprocess),                                          # Eerst de complete voorbewerking.
    ("clf", LogisticRegression(max_iter=1000, random_state=42)),   # Daarna het model.
])
model.fit(X_train_fe, y_train)       # Traint voorbewerking én model in één keer op de trainingsset.
model.score(X_test_fe, y_test)       # Geeft de accuracy op de testset (ongeveer 0,84).
```

---

## Notebook 2 – Modelselectie

### 2.1 Kernbegrippen

| Begrip | Uitleg in simpel Nederlands | Voorbeeld uit de Titanic-data |
|---|---|---|
| Model | Een functie met instelbare "knoppen" die tijdens het trainen automatisch worden afgesteld. | Een functie die van leeftijd, klasse, geslacht, ... een overlevingskans maakt. |
| Classificatie | Een categorie voorspellen. | Overleefd: ja of nee. |
| Regressie | Een getal voorspellen. | Niet bij Titanic; bijvoorbeeld een huisprijs. |
| Dummy classifier | Een nepmodel dat altijd het meest voorkomende antwoord geeft. Dient als ondergrens (baseline). | Zegt altijd "overleden" en scoort zo ongeveer 62% accuracy. |
| Baseline | De minimale score die elk echt model moet verslaan. | 62% van de dummy. |
| Logistic Regression | Een gewogen scorekaart: elke feature krijgt een gewicht, samen wordt dat een kans. | Vrouw zijn telt sterk positief, 3e klasse telt negatief. Won het toernooi nipt. |
| K-Nearest Neighbors (KNN) | Kijkt naar de 5 meest vergelijkbare passagiers en laat die stemmen. Schalen is hier cruciaal. | Zonder schalen zou `Fare` (0–512) de afstand volledig bepalen. |
| Decision Tree | Een stroomschema van ja/nee-vragen. | "Is het een man? Zo ja, is de fare lager dan 26?" |
| Random Forest | Honderden bomen die elk op een willekeurig deel van de data leren en dan stemmen. | 200 bomen stemmen samen over overleven. |
| Gradient Boosting | Kleine bomen na elkaar, waarbij elke nieuwe boom de fouten van de vorige probeert te herstellen. | Vaak zeer nauwkeurig, maar trager en lastiger uit te leggen. |
| Overfitting | Het model leert de trainingsdata uit het hoofd (inclusief ruis) en doet het slecht op nieuwe data. | Een boom zonder dieptegrens haalt bijna 100% op train maar veel minder in cross-validatie. |
| Underfitting | Het model is te simpel om het patroon te zien; zowel train- als testscore zijn laag. | Een boom met `max_depth=1` mag maar één vraag stellen. |
| Cross-validatie (CV) | De trainingsdata in K stukken (folds) verdelen; K keer trainen, steeds met een ander stuk als toets. Daarna het gemiddelde nemen. | 5-fold CV op de 712 trainingsrijen. |
| Standaardafwijking (std) over folds | Hoeveel de score "wiebelt" tussen de folds. Minder wiebelen = betrouwbaarder. | Bij gelijke gemiddelden kies je het model met de kleinste std. |
| Confusion matrix | Een 2x2-tabel met alle soorten goede en foute voorspellingen. | Rijen = werkelijkheid (overleden/overleefd), kolommen = voorspelling. |
| True Positive (TP) | Voorspeld "overleefd" en klopt. | Een vrouw uit 1e klasse die overleefde en zo voorspeld werd. |
| True Negative (TN) | Voorspeld "overleden" en klopt. | Een man uit 3e klasse die overleed en zo voorspeld werd. |
| False Positive (FP) | Voorspeld "overleefd", maar persoon overleed (vals alarm). | Model zegt "overleeft", passagier overleed. |
| False Negative (FN) | Voorspeld "overleden", maar persoon overleefde (gemist). | Model zegt "overlijdt", passagier overleefde toch. |
| Accuracy | Aandeel juiste voorspellingen: (TP + TN) / alles. | Ongeveer 0,84 op de testset. |
| Precision | Van iedereen die we "overleefd" noemden: hoeveel klopte? TP / (TP + FP). | Weinig valse alarmen = hoge precision. |
| Recall | Van alle echte overlevenden: hoeveel vonden we? TP / (TP + FN). | Weinig gemiste overlevenden = hoge recall. |
| F1-score | Het compromis (harmonisch gemiddelde) tussen precision en recall. | De dummy heeft F1 = 0, want hij voorspelt nooit "overleefd". |
| Imbalanced data | Eén klasse komt veel vaker voor; accuracy kan dan misleiden. | Titanic is licht onevenwichtig (62/38). |
| ROC-curve | Laat voor elke drempelwaarde zien hoe recall en het aantal valse alarmen tegen elkaar inruilen. | Drempel 50% kun je verschuiven naar 30% of 70%. |
| AUC | Oppervlakte onder de ROC-curve: kans dat het model een willekeurige overlevende hoger rangschikt dan een willekeurige niet-overlevende. 0,5 = gokken, 1,0 = perfect. | AUC van 0,8+ is solide. |
| Bias | Fout doordat het model te simpel is. | Linkerkant van de diepte-grafiek: beide curves laag. |
| Variance | Fout doordat het model te flexibel is en op elk toevallig detail reageert. | Rechterkant: trainscore bijna 1,0, CV-score zakt. |
| Learning curve | Grafiek van train- en CV-score bij steeds meer trainingsdata. Vertelt of meer data helpt. | De CV-curve vlakt af: meer passagiers helpt weinig; betere features of tuning wel. |

### 2.2 Belangrijkste scikit-learn-code

**Cross-validatie**

```python
from sklearn.model_selection import cross_val_score, cross_validate  # Importeert de CV-functies.

cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, n_jobs=-1)  # Geeft 5 accuracy-scores, één per fold, met alle CPU-kernen.
cv_scores.mean()                                                       # Het gemiddelde is de eerlijke score.
cv_scores.std()                                                        # De spreiding laat zien hoe stabiel het model is.

scoring = {"accuracy": "accuracy", "f1": make_scorer(f1_score, zero_division=0)}  # Twee metrieken tegelijk; F1 = 0 als er geen positieve voorspellingen zijn.
cv_res = cross_validate(pipe, X_train, y_train, cv=5, scoring=scoring)  # Voert CV uit en meet beide metrieken.
cv_res["test_accuracy"].mean()                                          # Gemiddelde accuracy over de folds.
```

**De modellen (het "toernooi")**

```python
from sklearn.dummy import DummyClassifier                     # Het nepmodel voor de baseline.
from sklearn.linear_model import LogisticRegression           # Gewogen scorekaart.
from sklearn.neighbors import KNeighborsClassifier            # Stemmen van de dichtstbijzijnde buren.
from sklearn.tree import DecisionTreeClassifier               # Eén beslisboom.
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # Bossen van bomen.

DummyClassifier(strategy="most_frequent")                     # Voorspelt altijd de meest voorkomende klasse.
LogisticRegression(max_iter=1000, random_state=42)            # Genoeg iteraties zodat het model convergeert.
KNeighborsClassifier(n_neighbors=5)                           # Laat de 5 meest gelijkende passagiers stemmen.
DecisionTreeClassifier(max_depth=3, random_state=42)          # Boom met maximaal 3 vragen per pad.
RandomForestClassifier(n_estimators=200, random_state=42)     # Bos van 200 bomen.
GradientBoostingClassifier(random_state=42)                   # Bomen die van elkaars fouten leren.

pipe = Pipeline([("prep", preprocessor), ("model", model)])   # Preprocessing BINNEN de pipeline, zodat CV niet lekt.
```

**Evaluatie op de testset**

```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             classification_report, roc_auc_score, RocCurveDisplay)  # Importeert alle metrieken.

champion_pipe.fit(X_train, y_train)                    # Traint het winnende model op de hele trainingsset.
y_pred = champion_pipe.predict(X_test)                 # Voorspelt 0/1 voor de testset.
accuracy_score(y_test, y_pred)                         # Aandeel juiste voorspellingen.
precision_score(y_test, y_pred)                        # Hoeveel voorspelde overlevenden echt overleefden.
recall_score(y_test, y_pred)                           # Hoeveel echte overlevenden gevonden werden.
f1_score(y_test, y_pred)                               # Compromis tussen precision en recall.
cm = confusion_matrix(y_test, y_pred)                  # Maakt de 2x2-tabel.
tn, fp, fn, tp = cm.ravel()                            # Haalt de vier getallen eruit, in de volgorde TN, FP, FN, TP.
ConfusionMatrixDisplay(confusion_matrix=cm).plot()     # Tekent de confusion matrix.
print(classification_report(y_test, y_pred))          # Precision, recall en F1 per klasse in één overzicht.
proba = champion_pipe.predict_proba(X_test)[:, 1]      # Kans op "overleefd" voor elke testpassagier.
roc_auc_score(y_test, proba)                           # Berekent de AUC uit die kansen.
RocCurveDisplay.from_estimator(champion_pipe, X_test, y_test)  # Tekent de ROC-curve.
```

**Learning curve**

```python
from sklearn.model_selection import learning_curve    # Importeert de learning-curve-functie.

train_sizes, lc_train, lc_cv = learning_curve(
    champion_pipe, X_train, y_train,
    train_sizes=np.linspace(0.1, 1.0, 5), cv=5, shuffle=True, random_state=42,
)  # Traint op 10% t/m 100% van de data en geeft train- en CV-scores per grootte terug.
```

---

## Notebook 3 – Optimalisatie en deployment

### 3.1 Kernbegrippen

| Begrip | Uitleg in simpel Nederlands | Voorbeeld uit de Titanic-data |
|---|---|---|
| Parameter | Een waarde die het model zelf leert uit de data. | De splitsregels in de bomen van het Random Forest. |
| Hyperparameter | Een instelling die jij kiest vóór het trainen (zoals de oventemperatuur bij het bakken). | `n_estimators`, `max_depth`, `min_samples_leaf`. |
| Tunen | Verschillende hyperparameter-waarden uitproberen en de beste houden. | Zoeken naar de beste combinatie voor het Random Forest. |
| `n_estimators` | Aantal bomen in het bos. | Getest: 100 en 300. |
| `max_depth` | Hoe diep elke boom mag groeien; `None` = geen limiet. | Getest: 4, 8 en None. |
| `min_samples_leaf` | Minimaal aantal passagiers in een eindpunt van een boom; groter = gladder, minder uit het hoofd leren. | Getest: 1, 3 en 5. |
| `max_features` | Hoeveel kolommen een boom per splitsing mag bekijken. | `"sqrt"`, `"log2"` of `None`. |
| Baseline (voor tunen) | De score met standaardinstellingen: je "voor"-foto. | Random Forest met alle defaults, gemeten met 5-fold CV. |
| Grid search | Álle combinaties van een rooster aan waarden proberen, elk met CV. | 2 x 3 x 3 = 18 combinaties x 5 folds = 90 trainingen. |
| Random search | Een vast aantal willekeurige combinaties proberen uit (bredere) bereiken. | `n_iter=15`: 15 willekeurige combinaties. |
| `model__` prefix | Twee underscores vertellen scikit-learn bij welke pipeline-stap een instelling hoort. | `model__max_depth` hoort bij de stap `"model"`. |
| `best_params_` / `best_score_` | De beste gevonden instellingen en hun gemiddelde CV-score. | Bijv. `{'model__max_depth': 8, ...}`. |
| `best_estimator_` | Het beste model, al opnieuw getraind (refit) op de hele trainingsset. | Direct bruikbaar om te voorspellen en op te slaan. |
| Testset één keer gebruiken | Pas helemaal aan het eind naar de testset kijken; anders wordt het stiekem trainingsdata. | De 179 testpassagiers pas na al het tunen gebruiken. |
| Productie | Het model draait ergens waar echte gebruikers of programma's het kunnen bereiken. | Een web-app waarin je passagiergegevens invult. |
| Productie-loop | Opslaan → laden → verpakken (interface) → hosten → monitoren. | `.joblib`-bestand → Gradio-app → Hugging Face Spaces. |
| joblib | Bibliotheek om Python-objecten (zoals een model) als bestand op te slaan en terug te laden. | `titanic_model_v1.joblib`. |
| Hele pipeline opslaan | Sla preprocessing + model samen op, zodat nieuwe data precies zo wordt schoongemaakt als de trainingsdata. | Mediaan, schaal en one-hot-kolommen gaan mee in het bestand. |
| Versiebeheer | Elk opgeslagen model een versie of datum geven, zodat je kunt terugrollen. | `titanic_model_v1.joblib`, `titanic_model_v2_2026-09-25.joblib`. |
| `predict` vs `predict_proba` | `predict` geeft 0 of 1; `predict_proba` geeft de kans per klasse. | 1e-klasse vrouw van 30: hoge overlevingskans; 3e-klasse man van 28: lage kans. |
| Data drift | De wereld verandert, waardoor een model langzaam verouderd raakt. | Nieuwe passagiers met heel andere leeftijden of prijzen dan in de trainingsdata. |
| Gradio | Bibliotheek die een Python-functie omzet in een webpagina met knoppen en schuifjes. | Schuifjes voor leeftijd en fare, dropdown voor klasse en haven. |
| Hugging Face Spaces | Gratis hosting voor kleine ML-apps met een vaste publieke URL. | Upload `app.py`, model, `requirements.txt` en `README.md`. |
| Versie vastpinnen | In `requirements.txt` exact de scikit-learn-versie noemen waarmee getraind is. | `scikit-learn==<trainingsversie>`, anders laadt het model misschien niet. |
| API (FastAPI) | Een webadres waar andere programma's data naartoe sturen en een voorspelling terugkrijgen. | POST naar `/predict` → `{"survival_probability": 0.87}`. |

### 3.2 Belangrijkste scikit-learn-code

**Baseline meten**

```python
from sklearn.ensemble import RandomForestClassifier     # Importeert het Random Forest.

champion = Pipeline([
    ("prep", preprocessor),                             # Dezelfde preprocessing als notebook 1.
    ("model", RandomForestClassifier(random_state=42)), # Random Forest met standaardinstellingen.
])
baseline_score = cross_val_score(champion, X_train, y_train, cv=5).mean()  # De "voor"-score om tuning mee te vergelijken.
```

**Grid search**

```python
from sklearn.model_selection import GridSearchCV   # Importeert grid search met cross-validatie.

param_grid = {
    "model__n_estimators": [100, 300],             # Aantal bomen om te proberen.
    "model__max_depth": [4, 8, None],              # Maximale boomdiepte om te proberen.
    "model__min_samples_leaf": [1, 3, 5],          # Minimale bladgrootte om te proberen.
}
grid_search = GridSearchCV(champion, param_grid, cv=5, n_jobs=-1)  # Probeert alle 18 combinaties met 5-fold CV.
grid_search.fit(X_train, y_train)                  # Voert de zoektocht uit, alleen op de trainingsset.
grid_search.best_params_                           # De beste combinatie van instellingen.
grid_search.best_score_                            # De gemiddelde CV-score van die combinatie.
pd.DataFrame(grid_search.cv_results_)              # Scores van álle combinaties als tabel.
```

**Random search**

```python
from sklearn.model_selection import RandomizedSearchCV  # Importeert random search.
from scipy.stats import randint                         # Om willekeurige gehele getallen te trekken.

param_distributions = {
    "model__n_estimators": randint(100, 401),           # Elk geheel getal van 100 t/m 400.
    "model__max_depth": [3, 4, 5, 6, 8, 10, 12, None],  # Lijst waaruit willekeurig gekozen wordt.
    "model__min_samples_leaf": randint(1, 10),          # Elk geheel getal van 1 t/m 9.
    "model__max_features": ["sqrt", "log2", None],      # Hoeveel kolommen per splitsing.
}
random_search = RandomizedSearchCV(
    champion, param_distributions, n_iter=15, cv=5, n_jobs=-1, random_state=42
)  # Probeert 15 willekeurige combinaties, reproduceerbaar.
random_search.fit(X_train, y_train)                     # Voert de zoektocht uit.
best_model = random_search.best_estimator_              # Het beste model, al opnieuw getraind op alle trainingsdata.
```

**Eindtoets op de testset (één keer!)**

```python
y_pred = best_model.predict(X_test)                    # Voorspelt de testset met het getunede model.
accuracy_score(y_test, y_pred)                         # Accuracy op onbekende passagiers.
f1_score(y_test, y_pred)                               # F1 op onbekende passagiers.
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)  # Tekent de confusion matrix direct uit de voorspellingen.
```

**Opslaan, laden en voorspellen**

```python
import joblib                                            # Importeert joblib voor opslaan/laden.

joblib.dump(best_model, "outputs/titanic_model_v1.joblib")  # Slaat de HELE pipeline op als bestand.
loaded_model = joblib.load("outputs/titanic_model_v1.joblib")  # Laadt de pipeline weer in, zoals een app zou doen.
loaded_model.predict(new_passengers)                     # Geeft 0 of 1 per nieuwe passagier.
loaded_model.predict_proba(new_passengers)               # Geeft per passagier de kans op 0 en op 1.
survived_col = list(loaded_model.classes_).index(1)      # Zoekt op welke kolom bij klasse 1 (overleefd) hoort.

from sklearn.base import clone                           # Importeert clone.
clone(champion).set_params(**grid_search.best_params_)   # Maakt een verse, ongetrainde kopie met de beste instellingen.
```

---

## 10 oefenvragen met antwoorden

**1. Waarom moet je de data splitsen in train en test vóórdat je missende waarden invult of gaat schalen?**
> Om **data leakage** te voorkomen. Als je bijvoorbeeld de mediaan van `Age` over alle rijen berekent, heeft de testset die waarde beïnvloed. Het model "ziet" dan al een stukje van de toets, waardoor de testscore te optimistisch wordt.

**2. Wat is het verschil tussen `fit`, `transform` en `fit_transform`? Op welke dataset gebruik je welke?**
> `fit` leert iets uit de data (bijv. mediaan, categorieën, gemiddelde). `transform` past het geleerde toe. `fit_transform` doet beide in één keer. Je gebruikt `fit`/`fit_transform` **alleen op de trainingsset** en `transform` op de testset (en op nieuwe data).

**3. Waarom gebruik je voor `Embarked` (C, Q, S) one-hot encoding en niet C=1, Q=2, S=3?**
> Havens hebben geen volgorde. Met 1, 2, 3 denkt het model dat S "drie keer" C is en dat Q "tussen" C en S ligt. One-hot encoding maakt aparte 0/1-kolommen (`Embarked_C`, `Embarked_Q`, `Embarked_S`) zonder nep-volgorde. `OrdinalEncoder` gebruik je alleen bij een echte volgorde, zoals S < M < L.

**4. Waarom is `pd.get_dummies` gevaarlijk bij machine learning, en wat doet `handle_unknown="ignore"`?**
> `get_dummies` maakt kolommen op basis van de categorieën die toevallig in de data zitten. Train en test kunnen dan verschillende kolommen krijgen, waardoor het model crasht of kolommen verkeerd leest. `OneHotEncoder` legt de kolommen vast op basis van de trainingsset; met `handle_unknown="ignore"` wordt een onbekende categorie in de testset gewoon allemaal nullen in plaats van een foutmelding.

**5. Een beslisboom zonder dieptegrens haalt bijna 100% accuracy op de trainingsdata, maar veel minder met 5-fold cross-validatie. Hoe heet dit verschijnsel en wat kun je eraan doen?**
> **Overfitting**: de boom heeft de trainingspassagiers uit het hoofd geleerd in plaats van het algemene patroon. Oplossingen: de boom beperken (`max_depth`, `min_samples_leaf`), een Random Forest gebruiken, of meer data verzamelen. Kies de instelling waarbij de CV-score het hoogst is.

**6. Leg uit hoe 5-fold cross-validatie werkt en waarom het beter is dan één validatieset.**
> De trainingsdata wordt in 5 gelijke stukken (folds) verdeeld. Er wordt 5 keer getraind; elke keer is een ander stuk de toets en de andere 4 zijn training. De eindscore is het gemiddelde. Voordelen: elke rij wordt één keer getoetst, de score hangt niet af van één toevallige split, en de standaardafwijking laat zien hoe stabiel het model is.

**7. Een model heeft deze confusion matrix op de testset: TN = 100, FP = 10, FN = 20, TP = 49. Bereken accuracy, precision en recall.**
> - Accuracy = (TP + TN) / totaal = (49 + 100) / 179 = 149 / 179 ≈ **0,83**
> - Precision = TP / (TP + FP) = 49 / 59 ≈ **0,83**
> - Recall = TP / (TP + FN) = 49 / 69 ≈ **0,71**

**8. Waarom is de F1-score van de `DummyClassifier(strategy="most_frequent")` precies 0, terwijl zijn accuracy ongeveer 62% is?**
> De dummy voorspelt altijd "overleden" (0). Hij heeft dus nul true positives, waardoor precision en recall voor "overleefd" allebei 0 zijn, en F1 dus ook. De 62% accuracy komt alleen doordat 62% van de passagiers overleed. Dat laat zien dat accuracy misleidend kan zijn en dat je altijd eerst een baseline moet meten.

**9. Wat is het verschil tussen een parameter en een hyperparameter? Noem bij een Random Forest van elk een voorbeeld.**
> Een **parameter** leert het model zelf uit de data, bijvoorbeeld de splitsregels in de bomen. Een **hyperparameter** kies je zelf vóór het trainen, bijvoorbeeld `n_estimators` (aantal bomen), `max_depth` of `min_samples_leaf`. Vergelijk: de oventemperatuur (hyperparameter) stel je in, hoe de cake bakt (parameters) gebeurt vanzelf.

**10. Je hebt je getunede Random Forest klaar voor een web-app. Wat sla je op, hoe, en waarom? En waarom mag je tijdens het tunen niet naar de testscore kijken?**
> Je slaat de **hele pipeline** (preprocessing + model) op met `joblib.dump(best_model, "titanic_model_v1.joblib")`, met een versienummer in de naam. Zo wordt nieuwe invoer in de app precies zo schoongemaakt als de trainingsdata (zelfde mediaan, schaal en one-hot-kolommen), en kun je terugrollen naar een oudere versie. Tijdens het tunen gebruik je alleen cross-validatie op de trainingsset: als je naar de testscore kijkt en daarna instellingen aanpast, wordt de testset stiekem trainingsdata en is de eindscore niet meer eerlijk.
