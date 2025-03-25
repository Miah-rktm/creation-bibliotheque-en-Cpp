#include "ndarray.h"
#include <iomanip>
#include <random>
template <typename T>
NDarray<T>::NDarray(std::initializer_list<size_t> dims, T value)
{
    shape = dims;
    size_t tsize = 1;
    for (size_t d : shape)
        tsize *= d;
    tab.resize(tsize, value); // Redimensionnement du tableau avec les valeurs données
}
template <typename T>
size_t NDarray<T>::getSize() const
{
    size_t tsize = 1;
    for (size_t d : shape)
        tsize *= d; // Calcul taille en fonction dimension
    return tsize;
}
template <typename T>
void NDarray<T>::print() const
{
    if (shape.size() == 1)
    {
        std::cout << "[";
        for (size_t i = 0; i < shape[0]; i++) // shape[0] est la taille du tab 1D
        {
            std::cout << tab[i] << " ";
        }
        std::cout << "]" << std::endl;
    }
    else if (shape.size() == 2)
    {
        for (size_t i = 0; i < shape[0]; i++) // shape[0] nombre de lignes,et shape[1] nbre de colonnes
        {
            std::cout << "[";
            for (size_t j = 0; j < shape[1]; j++)
            {
                std::cout << std::setw(4) << tab[i * shape[1] + j] << " ";
            }
            std::cout << "]" << std::endl;
        }
    }
    else
    {
        size_t planeSize = shape[1] * shape[2]; // calcul de la taille d'un "plan" dans le tableau 3D
        for (size_t k = 0; k < shape[0]; k++)   // shape[0] c'est le nombre de tranches
        {
            std::cout << "Slice " << k << ":\n";  // affiche le num de tranche
            for (size_t i = 0; i < shape[1]; i++) // Parcours chaque ligne dans cette tranche
            {
                std::cout << "[";
                for (size_t j = 0; j < shape[2]; j++) // Parcours chaque élément dans la ligne
                {
                    std::cout << std::setw(4) << tab[k * planeSize + i * shape[2] + j] << " "; // affichage des éléments
                }
                std::cout << "]" << std::endl;
            }
            std::cout << std::endl;
        }
    }
}
template <typename T>
T &NDarray<T>::at(std::initializer_list<size_t> indices)
{
    // Si le tableau est 1D
    if (shape.size() == 1 && indices.size() == 1)
    {
        return tab[*indices.begin()]; // retourne la valeur à l'indice spécifié
    }

    // Sinon pour les n-dimensionnels:calcul des index appropriés
    size_t index = 0;
    size_t multiplier = 1; // variable permettant d'ajuster l'index en fonction de nbre de colonnes
    for (size_t i = indices.size(); i-- > 0;)
    {
        index += *(indices.begin() + i) * multiplier;
        multiplier *= this->shape[i]; // calcul de l'indice en fonction de dimensions
    }
    return this->tab[index]; // retourne la valeur à l'indice calculé
}

template <typename T>
NDarray<T> NDarray<T>::zeros(std::initializer_list<size_t> dims)
{
    return NDarray<T>(dims, 0);
}
template <typename T>
NDarray<T> NDarray<T>::ones(std::initializer_list<size_t> dims)
{
    return NDarray<T>(dims, 1);
}
template <typename T>
NDarray<T> NDarray<T>::full(std::initializer_list<size_t> dims, T value)
{
    return NDarray<T>(dims, value);
}
template <typename T>
NDarray<T> NDarray<T>::eye(size_t n)
{
    NDarray<T> result({n, n}, 0); // tab a n*n initialisé à 0
    for (size_t i = 0; i < n; i++)
    {
        result.at({i, i}) = 1; // Met les éléments diagonaux à 1
    }
    return result;
}
template <typename T>
NDarray<T> NDarray<T>::arange(T start, T stop, T step)
{
    std::vector<T> values; // vecteur temporaire pour stocker les valeurs
    for (T i = start; i < stop; i += step)
    {
        values.push_back(i);
    }
    // Crée le tableau à une dimension avec la taille appropriée
    NDarray<T> arr({values.size()});
    arr.tab = values; // Remplir le tableau avec les valeurs générées
    return arr;
}

template <typename T>
NDarray<T> NDarray<T>::linspace(T start, T stop, size_t num)
{
    std::vector<T> values;               // vecteur temporaire pour stocker les valeurs
    T step = (stop - start) / (num - 1); // Calcul de pas entre chaque valeurs
    for (size_t i = 0; i < num; ++i)
    {
        values.push_back(start + i * step);
    }
    NDarray<T> arr({values.size()});
    arr.tab = values; // Remplir le tableau avec les valeurs générées
    return arr;
}
template <typename T>
void NDarray<T>::reshape(std::initializer_list<size_t> new_shape)
{
    size_t new_size = 1; // Nouvelle taille du tableau
    for (size_t d : new_shape)
    {
        new_size *= d;
    }

    // Vérification si la taille reste la même
    if (new_size == getSize())
    {
        shape = new_shape; // Mise à jour de la forme
    }
    else
    {
        std::cout << "Erreur : nouvelle forme incompatible avec la taille actuelle du tableau." << std::endl;
    }
}
template <typename T>
NDarray<T> NDarray<T>::flatten() const
{
    NDarray<T> flat({getSize()}, 0);       // Crée un tableau 1D avec la même taille que le tab original
    for (size_t i = 0; i < getSize(); ++i) // Copie de tous les éléments dans le tableau 1D
    {
        flat.tab[i] = tab[i];
    }
    return flat;
}
template <typename T>
NDarray<T> NDarray<T>::concatenate(const NDarray<T> &other, size_t axis)
{
    std::vector<size_t> shape1 = this->getShape();
    std::vector<size_t> shape2 = other.getShape();

    if (shape1.size() != shape2.size())
    {
        throw std::runtime_error("Les tableaux doivent avoir le même nombre de dimensions");
    }

    std::vector<size_t> new_shape = shape1;
    new_shape[axis] += shape2[axis]; // Agrandit la dimension spécifiée par axis

    NDarray<T> result(new_shape, T()); // Utilisation du nouveau constructeur:creation tableau avec la nouvelle forme
    size_t offset = 0;

    for (size_t i = 0; i < this->getSize(); i++) // Copier les éléments du tableau courant
    {
        result.tab[offset++] = this->tab[i];
    }

    for (size_t i = 0; i < other.getSize(); i++) // Ajouter les éléments de l'autre tableau
    {
        result.tab[offset++] = other.tab[i];
    }

    return result;
}

template <typename T>
NDarray<T> NDarray<T>::hstack(const NDarray<T> &other)
{
    std::vector<size_t> shape1 = this->getShape();
    std::vector<size_t> shape2 = other.getShape();

    // si les deux tableaux sont 1D, concatène sur l’axe 0
    if (shape1.size() == 1 && shape2.size() == 1)
    {
        return this->concatenate(other, 0);
    }

    // Vérifie la compatibilité pour une concaténation horizontale
    if (shape1[0] != shape2[0])
    {
        throw std::runtime_error("Les tableaux doivent avoir le même nombre de lignes pour hstack");
    }

    return this->concatenate(other, 1); // Concatène sur l’axe des colonnes (axe 1)
}

template <typename T>
NDarray<T> NDarray<T>::vstack(const NDarray<T> &other)
{
    std::vector<size_t> shape1 = this->getShape();
    std::vector<size_t> shape2 = other.getShape();

    NDarray<T> arr1_2d = *this; // Copie du tableau courant
    NDarray<T> arr2_2d = other; // Copie de l'autre tableau

    // Convertit un tableau 1D en 2D (1 ligne) si nécessaire
    if (shape1.size() == 1)
    {
        arr1_2d.reshape({1, shape1[0]});
    }
    if (shape2.size() == 1)
    {
        arr2_2d.reshape({1, shape2[0]});
    }

    // Vérifie que les tableaux ont le même nombre de colonnes
    if (arr1_2d.getShape()[1] != arr2_2d.getShape()[1])
    {
        throw std::runtime_error("Les tableaux doivent avoir le même nombre de colonnes pour vstack");
    }

    return arr1_2d.concatenate(arr2_2d, 0); // Concatène verticalement (axe 0)
}
template <>
NDarray<double> NDarray<double>::rand(std::initializer_list<size_t> dims)
{
    size_t tsize = 1;
    for (size_t d : dims)
        tsize *= d; // Calcule la taille totale du tableau

    // Initialisation du générateur de nombres aléatoires
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0); // Distribution uniforme entre 0 et 1

    NDarray<double> result(dims, 0.0); // Crée un tableau initialisé à 0
    for (size_t i = 0; i < tsize; i++)
    {
        result.tab[i] = dis(gen); // Remplit avec des valeurs aléatoires
    }
    return result;
}
