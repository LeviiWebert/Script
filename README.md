# Scansante

Script qui permet de récupérer les données publiées à l'adresse https://www.scansante.fr/applications/casemix_ghm_cmd

## Fonctionnement

L'utilisateur fourni un ou plusieurs FINESS au script. Ce dernier va interroger scansante pour chaque code finess et
extraire les données du HTML. Enfin, il exporte les données formatées soit dans une table mysql soit dans un fichier
csv.

Le script patiente entre 15 et 60 secondes entre chaque appel à scansante.fr

## Installation

Python: 3.12 (doit fonctionner avec toutes les versions depuis la 3.10)

Dependances:

* beautifulsoup4
* requests
* click
* mysqlclient


## Utiliser le script

* Déclencher le script : `python scansante.py`

Options :
* --finess xxxxxx : fourni un code FINESS au script
* --finess_file /path/to/file.txt : fourni un fichier avec une liste de code FINESS
* --db_url: chaîne de connexion à une base de données MySQL (mysql://user:pw@host:port/dbname)
* --file_ouput /path/to/file.csv : chemin vers le fichier de sorti (peut ne pas exister)

Pour que le script fonctionne correctement, il faut :
* lui fournir au moins un finess.
* lui fournir au moins un output (soit un fichier soit une base de données)

Exemples :

```bash
python scansante.py --finess 060780723 --db_url mysql://root:example@127.0.0.1:3306/quotes
python scansante.py --finess 060780723 --file_ouput ./output.csv
python scansante.py --finess_file ~/all_finess.txt --db_url mysql://root:example@127.0.0.1:3306/quotes
```

## Miscellanous

Vous pouvez créer la table d'output (appelée scansante) dans la base de données avec :
`python scansante.py --db_url mysql://root:example@127.0.0.1:3306/quotes --db_init`

Attention, elle sera droppée si elle existait précédemment.
