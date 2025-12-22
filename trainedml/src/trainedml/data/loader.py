"""
Module de chargement de données publiques en ligne pour trainedml.
Ce module permet de récupérer automatiquement des jeux de données depuis des sources open data ou des URLs documentées.
"""


import pandas as pd
import pooch

class DataLoader:
    """
    Classe responsable du chargement et de l'abstraction des jeux de données publics.
    Les autres modules ne connaissent pas la source des données.
    """
    def __init__(self):
        pass

    def load_csv_from_url(self, url: str, known_hash=None, sep=",") -> pd.DataFrame:
        """
        Télécharge un fichier CSV depuis une URL (avec cache local) et le charge dans un DataFrame pandas.
        Args:
            url (str): Lien direct vers le fichier CSV.
            known_hash (str, optional): Hash du fichier pour vérification (voir doc pooch).
        Returns:
            pd.DataFrame: Données chargées.
        """
        try:
            fname = pooch.retrieve(
                url=url,
                known_hash=known_hash or None,
                progressbar=True
            )
            return pd.read_csv(fname, sep=sep)
        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement des données depuis {url} : {e}")


    def load_dataset(self, name=None, url=None, target=None, sep=None):
        """
        Charge un dataset par nom connu ou URL, et retourne X, y.
        Args:
            name (str): nom du dataset ("iris", "wine", etc.)
            url (str): URL d'un CSV distant
            target (str): nom de la colonne cible (si url)
        Returns:
            X (pd.DataFrame), y (pd.Series)
        """
        if name == "iris":
            url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
            df = self.load_csv_from_url(url)
            X = df.drop(columns=["species"])
            y = df["species"]
            return X, y
        elif name == "wine":
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
            cols = ["class","alcohol","malic_acid","ash","alcalinity_of_ash","magnesium","total_phenols","flavanoids","nonflavanoid_phenols","proanthocyanins","color_intensity","hue","od280/od315_of_diluted_wines","proline"]
            df = pd.read_csv(url, header=None, names=cols)
            X = df.drop(columns=["class"])
            y = df["class"]
            return X, y
        elif url is not None and target is not None:
            # Si le CSV est winequality, utiliser sep=';'
            sep_to_use = sep
            if sep_to_use is None:
                if "winequality" in url:
                    sep_to_use = ";"
                else:
                    sep_to_use = ","
            df = self.load_csv_from_url(url, sep=sep_to_use)
            X = df.drop(columns=[target])
            y = df[target]
            return X, y
        else:
            raise ValueError("Spécifiez un nom de dataset connu ou une url+target.")

    # Ajouter ici d'autres méthodes pour charger d'autres datasets publics (INSEE, data.gouv.fr, etc.)
