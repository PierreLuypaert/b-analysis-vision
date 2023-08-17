import cv2
import numpy as np

# Définir les paramètres de la vidéo
largeur, hauteur = 640, 480
fps = 30
nom_fichier_sortie = 'ma_video.mp4'

# Créer l'objet VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(nom_fichier_sortie, fourcc, fps, (largeur, hauteur))

# Boucle pour écrire des images dans la vidéo
for i in range(300):  # Par exemple, nous allons créer une vidéo de 300 images
    # Créer une image/frame aléatoire (ici, une image avec des pixels aléatoires)
    frame = np.random.randint(0, 256, (hauteur, largeur, 3), dtype=np.uint8)

    # Ajouter la frame à la vidéo
    video_writer.write(frame)

# Libérer les ressources
video_writer.release()