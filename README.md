##Projet de creation bibliothèque similaires a Numpyy mais en c++

-Gestion de tableau n-dimensionnels
Les fonctionnalités finies:
.Creation tableau de 1D-3D
.np.zeros
.np.ones
.np.full
.np.eye
.arr.shape
.arr.size
.indexation ou acces tableau
.arr.reshape
.arr.flatten
.np.arange
.np.linspace
.np.concatenate([arr1, arr2], axis=0) → Concaténation sur une dimension
. np.hstack([arr1, arr2]) → Concaténation horizontale
. np.vstack([arr1, arr2]) → Concaténation verticale
.np.random.rand(3,3)
.np.randoom.randint(0, 10, (2,3))
.arr[1] → Accès à l'élément d’index 1
.arr[1, 2] → Accès à l’élément (1,2)
.arr[0:3] → Extraction des trois premiers éléments
.arr[:, 1] → Extraction de la deuxième colonne
.arr[1:, :2] → Extraction de sous-matrices
.np.add(arr1, arr2) ou arr1 + arr2 → Addition
.np.subtract(arr1, arr2) ou arr1 - arr2 → Soustraction
.np.multiply(arr1, arr2) ou arr1 * arr2 → Multiplication élément par élément
.np.dot(arr1, arr2) → Produit matriciel
.np.divide(arr1, arr2) ou arr1 / arr2 → Division
J'ai ajouter aussi l'operation avec le scalaire.
