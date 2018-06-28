# cuda_classification

Voor het gebruik van deze classificatie zijn bepaalde stappen nodig:

Alle gebruikte afbeeldingen moeten in een map staan (bij deze scripten is er gebruik gemaakt van de ChinaSet van “Tuberculose - RIVM.” n.d. Accessed April 12, 2018.  https://www.rivm.nl/Onderwerpen/T/Tuberculose.)

in de commandline bij het uitvoeren van preprocessing.py moeten de volgende argumenten meegegeven worden:
--path_target en --path_source

een voorbeeld:
python preprocessing.py --path_target D:\cnn --path_source D:\cnn\ChinaSet_AllFiles\CXR_png

De cnn1.py script moet in dezelfde map staan als --path_target
